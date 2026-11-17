"""
================================================================================
LangGraph 多轮对话双级存储引擎 (Redis 热缓存 + MySQL 冷恢复回源)
================================================================================
数据流转全景：
1. 【热读取 (Redis Cache Hit, <1ms)】：
   - 2 小时内活跃的对话直接命中 `dao_c:agent:session:{session_id}`，同时刷新 TTL；
2. 【过期冷回源 (Redis Cache Miss -> MySQL Recovery -> Cache Warmup)】：
   - 若用户隔天再来聊（Redis Key 已因 TTL 过期淘汰），自动回源查 MySQL `agent_threads` +
     `agent_messages` 重建 `sticky_agent` / 临床否定症状槽位 / 最近 40 条消息，并回填 Redis；
3. 【双写落库 (流结束后异步短事务)】：
   - 由 `main.py` 在每轮 SSE 吐字完毕后调用 `persist_turn_to_db(...)` 永久落入 MySQL，
     再调用 `save_state(...)` 更新 Redis 热状态（不依赖不可靠的 Redis 过期事件写库）。
"""
import json
from typing import Any, Dict, Tuple

from langchain_core.messages import messages_from_dict, messages_to_dict

from app.cache.redis_client import build_key, redis_manager
from app.config import settings
from app.storage.session_repository import load_thread_from_db


def default_session_state() -> Dict[str, Any]:
    """构造全新会话的初始 LangGraph 状态"""
    return {
        "messages": [],
        "user_query": "",
        "new_asset_ids": [],
        "active_assets": [],
        "sticky_agent": None,
        "clinical_slots": {"positive_symptoms": [], "negative_symptoms": []},
        "require_tcm_joint_consult": True,
        "latest_brain_mcp_output": None,
        "sse_events": [],
    }


class AgentSessionRedisStore:
    """基于 Redis 热缓存 + MySQL 冷恢复的 LangGraph 状态管理器"""

    @staticmethod
    def _session_key(session_id: str) -> str:
        return build_key("agent", "session", session_id)

    async def load_state_with_source(self, session_id: str) -> Tuple[Dict[str, Any], str]:
        """
        读取会话状态并返回来源标记：
        - "redis_hit"       : 命中 Redis 活跃热缓存
        - "mysql_recovered" : Redis TTL 已过期，自动从 MySQL 冷回源重建并预热 Redis
        - "new_session"     : 全新会话
        """
        client = await redis_manager.get_client()
        key = self._session_key(session_id)
        raw = await client.get(key)

        if raw:
            try:
                payload = json.loads(raw)
                serialized_msgs = payload.get("messages") or []
                restored_msgs = messages_from_dict(serialized_msgs) if serialized_msgs else []

                # 滑动续期 TTL，保持活跃对话不掉出热缓存
                await client.expire(key, settings.REDIS_SESSION_TTL_SECONDS)

                return {
                    "messages": restored_msgs,
                    "user_query": "",
                    "new_asset_ids": [],
                    "active_assets": payload.get("active_assets") or [],
                    "sticky_agent": payload.get("sticky_agent"),
                    "clinical_slots": payload.get("clinical_slots")
                    or {"positive_symptoms": [], "negative_symptoms": []},
                    "require_tcm_joint_consult": payload.get("require_tcm_joint_consult", True),
                    "latest_brain_mcp_output": payload.get("latest_brain_mcp_output"),
                    "sse_events": [],
                }, "redis_hit"
            except json.JSONDecodeError:
                pass

        # Redis 未命中（如 2 小时 TTL 过期或服务重启）-> 回源 MySQL 永久账本
        recovered_state = await load_thread_from_db(session_id)
        if recovered_state is not None:
            # 预热写回 Redis，下一轮对话恢复 1ms 热读取
            await self.save_state(session_id, recovered_state)
            return recovered_state, "mysql_recovered"

        return default_session_state(), "new_session"

    async def load_state(self, session_id: str) -> Dict[str, Any]:
        """兼容接口：直接返回状态字典"""
        state, _ = await self.load_state_with_source(session_id)
        return state

    async def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        将更新后的 LangGraph 热状态写入 Redis，并设置 TTL 自动淘汰。
        注：Redis 热缓存仅保留最近 40 条消息控制内存上限，全量历史永久保存在 MySQL `agent_messages` 表。
        """
        client = await redis_manager.get_client()
        msgs = list(state.get("messages") or [])[-40:]

        serializable_payload = {
            "messages": messages_to_dict(msgs),
            "active_assets": state.get("active_assets") or [],
            "sticky_agent": state.get("sticky_agent"),
            "clinical_slots": state.get("clinical_slots")
            or {"positive_symptoms": [], "negative_symptoms": []},
            "require_tcm_joint_consult": state.get("require_tcm_joint_consult", True),
            "latest_brain_mcp_output": state.get("latest_brain_mcp_output"),
        }
        await client.setex(
            self._session_key(session_id),
            settings.REDIS_SESSION_TTL_SECONDS,
            json.dumps(serializable_payload, ensure_ascii=False),
        )

    async def delete_state(self, session_id: str) -> int:
        """仅清除 Redis 热缓存（可用于模拟 TTL 过期，测试从 MySQL 冷回源重建效果）"""
        client = await redis_manager.get_client()
        return await client.delete(self._session_key(session_id))

    async def inspect_raw_state(self, session_id: str) -> Dict[str, Any]:
        """学习/调试辅助接口：查看当前 Redis 中存储的原始 JSON 与剩余 TTL"""
        client = await redis_manager.get_client()
        key = self._session_key(session_id)
        raw = await client.get(key)
        ttl = await client.ttl(key)
        return {
            "redis_key": key,
            "backend": "redis" if redis_manager.is_redis_live else "in_memory_fallback",
            "ttl_seconds": ttl,
            "state": json.loads(raw) if raw else None,
        }


session_redis_store = AgentSessionRedisStore()
