"""
================================================================================
MySQL 会话冷持久化仓库 (配合 Redis 热缓存实现“双写持久化 + 过期冷回源”)
================================================================================
核心工程原则：
1. 【绝不在 SSE 流式输出期间持有 MySQL 连接】：
   - 仅在 Redis Cache Miss 时打开短连接读取历史并立即关闭；
   - 仅在整轮 SSE 生成完毕后打开短事务一次性落库并立即提交关闭。
2. 【不依赖 Redis TTL 过期回调写库】：
   - Redis 过期通知既不保证可靠送达，也拿不到已被删除的 Value；
   - 因此每轮对话结束即同步持久化到 MySQL（Source of Truth），Redis 只充当 2 小时活跃热缓存。
3. 【TTL 过期后的无损冷恢复】：
   - 用户隔天或 2 小时后回到同一 `session_id` 继续对话时，Redis 虽已自动淘汰释放内存，
     但系统会自动回源查 MySQL `agent_threads` + `agent_messages` 重建上下文并回填 Redis。
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import desc, select

from app.legacy_java_migrated.models_db import (
    AgentMessageEntity,
    AgentThreadEntity,
    AsyncSessionLocal,
)


async def load_thread_from_db(session_id: str, limit_messages: int = 40) -> Optional[Dict[str, Any]]:
    """
    【冷恢复回源】：当 Redis 中的会话热状态因 TTL 过期被删除时，
    从 MySQL 读取线程元状态与最近 N 条消息，重建完整的 LangGraph State。
    """
    try:
        async with AsyncSessionLocal() as db:
            thread_res = await db.execute(
                select(AgentThreadEntity).where(
                    AgentThreadEntity.thread_id == session_id,
                    AgentThreadEntity.status == 1,
                )
            )
            thread = thread_res.scalars().first()
            if thread is None:
                return None

            msg_res = await db.execute(
                select(AgentMessageEntity)
                .where(AgentMessageEntity.thread_id == session_id)
                .order_by(desc(AgentMessageEntity.id))
                .limit(limit_messages)
            )
            recent_rows = list(reversed(msg_res.scalars().all()))

        restored_msgs: List[BaseMessage] = []
        for row in recent_rows:
            if row.role == "user":
                restored_msgs.append(HumanMessage(content=row.content))
            else:
                restored_msgs.append(AIMessage(content=row.content))

        return {
            "messages": restored_msgs,
            "user_query": "",
            "new_asset_ids": [],
            "active_assets": json.loads(thread.active_assets_json)
            if thread.active_assets_json
            else [],
            "sticky_agent": thread.sticky_agent,
            "clinical_slots": json.loads(thread.clinical_slots_json)
            if thread.clinical_slots_json
            else {"positive_symptoms": [], "negative_symptoms": []},
            "require_tcm_joint_consult": True,
            "latest_brain_mcp_output": json.loads(thread.latest_brain_output_json)
            if thread.latest_brain_output_json
            else None,
            "sse_events": [],
        }
    except Exception:
        # 若本地未启动 MySQL 服务，安全返回 None，不影响降级学习体验
        return None


async def persist_turn_to_db(
    session_id: str,
    uid: Optional[int],
    user_query: str,
    assistant_reply: str,
    artifacts: List[Dict[str, Any]],
    final_state: Dict[str, Any],
) -> bool:
    """
    【流结束后短事务落库】：
    在整轮 SSE 吐字完毕后调用，单次事务同时完成：
    1. Upsert `agent_threads` 表（更新 sticky_agent、临床否定症状槽位、active_assets）；
    2. Insert `agent_messages` 表（追加本轮 user 提问与 assistant 回复及画布 _artifact）。
    """
    try:
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(AgentThreadEntity).where(AgentThreadEntity.thread_id == session_id)
            )
            thread = res.scalars().first()
            now = datetime.now()

            slots_str = json.dumps(
                final_state.get("clinical_slots") or {}, ensure_ascii=False
            )
            assets_str = json.dumps(
                final_state.get("active_assets") or [], ensure_ascii=False
            )
            brain_out = final_state.get("latest_brain_mcp_output")
            brain_str = json.dumps(brain_out, ensure_ascii=False) if brain_out else None

            if thread is None:
                thread = AgentThreadEntity(
                    thread_id=session_id,
                    uid=uid,
                    title=user_query[:60],
                    sticky_agent=final_state.get("sticky_agent"),
                    clinical_slots_json=slots_str,
                    active_assets_json=assets_str,
                    latest_brain_output_json=brain_str,
                    status=1,
                    created_at=now,
                    updated_at=now,
                )
                db.add(thread)
            else:
                if uid is not None and thread.uid is None:
                    thread.uid = uid
                thread.sticky_agent = final_state.get("sticky_agent")
                thread.clinical_slots_json = slots_str
                thread.active_assets_json = assets_str
                if brain_str is not None:
                    thread.latest_brain_output_json = brain_str
                thread.updated_at = now

            db.add(
                AgentMessageEntity(
                    thread_id=session_id,
                    role="user",
                    content=user_query,
                    artifacts_json=None,
                    created_at=now,
                )
            )
            db.add(
                AgentMessageEntity(
                    thread_id=session_id,
                    role="assistant",
                    content=assistant_reply,
                    artifacts_json=json.dumps(artifacts, ensure_ascii=False) if artifacts else None,
                    created_at=now,
                )
            )
            await db.commit()
        return True
    except Exception:
        return False


async def get_full_thread_history_from_db(session_id: str) -> Dict[str, Any]:
    """查询 MySQL 中永久保存的完整会话明细与结构化槽位（供历史回溯接口查看）"""
    try:
        async with AsyncSessionLocal() as db:
            t_res = await db.execute(
                select(AgentThreadEntity).where(AgentThreadEntity.thread_id == session_id)
            )
            thread = t_res.scalars().first()
            if thread is None:
                return {"found": False, "session_id": session_id, "messages": []}

            m_res = await db.execute(
                select(AgentMessageEntity)
                .where(AgentMessageEntity.thread_id == session_id)
                .order_by(AgentMessageEntity.id.asc())
            )
            rows = m_res.scalars().all()

        return {
            "found": True,
            "session_id": thread.thread_id,
            "uid": thread.uid,
            "title": thread.title,
            "sticky_agent": thread.sticky_agent,
            "clinical_slots": json.loads(thread.clinical_slots_json)
            if thread.clinical_slots_json
            else {},
            "active_assets": json.loads(thread.active_assets_json)
            if thread.active_assets_json
            else [],
            "updated_at": thread.updated_at.isoformat() if thread.updated_at else None,
            "messages": [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "artifacts": json.loads(r.artifacts_json) if r.artifacts_json else [],
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
        }
    except Exception as e:
        return {"found": False, "session_id": session_id, "error": str(e), "messages": []}
