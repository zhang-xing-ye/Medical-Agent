"""
================================================================================
DAO-C 多模态医学智能体平台统一网关入口 (dao_c_modelend_agent/app/main.py)
================================================================================
本服务整合了五大核心能力：
1. 【Java 侧代码完整迁移】：挂载 `routers_java_compat`，替代原 Spring Boot (`dao_c_backend`)
   的 `/user/*`、`/category/*`、`/model/*`、`/session/*` 接口与 JWT/AES 鉴权，并集成 Redis 旁路缓存与注销黑名单。
2. 【大文件凭据中心 (`AssetHub`)】：挂载 `/api/v1/assets/*`，上传 `.nii.gz`/`.npy` 生成短凭据 `asset_id` 写入 Redis。
3. 【总控 DeepSeek + 中医针灸 32B Sub-Agent Handoff + 脑病 MCP 调度】：
   暴露 `/api/v1/agent/chat/stream`，使用 Redis 分布式锁 + Lua 滑动窗口限流。
4. 【Redis 热缓存 + MySQL 冷持久双轨制】：
   流前命中 Redis（TTL 过期则自动从 MySQL 冷回源预热）-> 无 DB 连接流式吐字 -> 流结束后短事务永久落入 MySQL。
5. 【长耗时任务 Redis Pub/Sub 订阅总线】：暴露 `/api/v1/tasks/{task_id}/events` 实时推送 GPU 任务进度。
"""
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agents.orchestrator_graph import AgentPlatformState, medical_agent_graph
from app.agents.tcm_acupuncture_agent import stream_tcm_specialist_handoff
from app.asset_hub.asset_service import router as asset_router
from app.cache.auth_and_ratelimit import check_rate_limit
from app.cache.redis_client import redis_manager
from app.cache.session_state_store import session_redis_store
from app.cache.task_progress_bus import get_task_snapshot, subscribe_task_events
from app.config import settings
from app.legacy_java_migrated.models_db import init_db_tables
from app.legacy_java_migrated.routers_java_compat import router as java_compat_router
from app.mcp_client.brain_mcp_client import brain_mcp_client
from app.storage.session_repository import (
    get_full_thread_history_from_db,
    persist_turn_to_db,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理全局异步 Redis 连接池与 MySQL 智能体持久化表初始化"""
    await redis_manager.init()
    await init_db_tables()
    yield
    await redis_manager.close()


app = FastAPI(
    title="DAO-C Multi-Modal Medical Agent Platform (Full Python Stack + Redis + MySQL)",
    description=(
        "整合原 Java 业务后端、Redis 热缓存与限流总线、MySQL 永久历史存储、"
        "总控 DeepSeek (LangGraph)、中医针灸 32B 领域大模型与脑病预测 FastMCP 服务"
    ),
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 挂载原 Java (dao_c_backend) 迁移过来的全部兼容路由
app.include_router(java_compat_router)

# 2. 挂载大文件凭据中心 (AssetHub) 路由
app.include_router(asset_router)


class AgentChatRequest(BaseModel):
    session_id: str = Field(default="default_session", description="多轮会话唯一标识")
    uid: Optional[int] = Field(default=None, description="当前登录用户 ID (用于 MySQL 归档关联)")
    query: str = Field(..., description="用户自然语言需求（如中医问诊或脑部影像筛查指令）")
    asset_ids: List[str] = Field(
        default_factory=list,
        description="本轮附带的文件凭据 ID 列表（如通过 /api/v1/assets/upload 获得的 ['ast_901']）",
    )


def _format_sse(event_type: str, payload: Any) -> str:
    """构造标准 Server-Sent Events (SSE) 数据帧"""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/v1/agent/chat/stream", tags=["Agent-Orchestrator (总控流式对话)"])
async def agent_chat_stream(req: AgentChatRequest):
    """
    统一智能体 SSE 流式入口（Redis 热缓存 + MySQL 冷持久 + 防连接池耗尽三段式生命周期）：
    1. Redis Lua 滑动窗口限流校验；
    2. 在进入流式吐字前，持有会话分布式锁加载状态（Redis 命中或从 MySQL 冷回源重建），随后立即释放 DB 连接；
    3. 流式生成期间零数据库连接占用；
    4. 生成完毕后：短事务写入 MySQL 永久保存 + 刷新 Redis 热状态（TTL=2h）+ 释放分布式锁。
    """
    allowed, current_count = await check_rate_limit(req.session_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"请求过于频繁（{settings.RATE_LIMIT_WINDOW_SECONDS} 秒内已请求 {current_count} 次），"
                "请稍后再试。"
            ),
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        # 对同一 session_id 加 Redis 分布式互斥锁，防止并发连点导致会话状态互相覆盖
        async with redis_manager.distributed_lock(
            f"session:{req.session_id}", ttl_seconds=120, wait_timeout=5.0
        ) as acquired:
            if not acquired:
                yield _format_sse(
                    "error",
                    {"code": "SESSION_BUSY", "message": "当前会话仍有推理任务正在执行，请稍候。"},
                )
                return

            # 1. 优先从 Redis 读取热缓存；若 TTL 已过期则自动回源 MySQL 重建并预热 Redis
            prev_state, state_source = await session_redis_store.load_state_with_source(
                req.session_id
            )

            input_state: AgentPlatformState = {
                **prev_state,
                "messages": [*prev_state["messages"], HumanMessage(content=req.query)],
                "user_query": req.query,
                "new_asset_ids": req.asset_ids,
                "sse_events": [],
            }

            # 2. 运行 LangGraph 状态机完成路由、MCP 工具执行或 Sub-Agent Handoff 移交
            final_state: AgentPlatformState = await medical_agent_graph.ainvoke(input_state)

            stream_from_tcm_subagent = False
            collected_artifacts: List[Dict[str, Any]] = []
            non_subagent_tokens: List[str] = []

            for evt in final_state.get("sse_events") or []:
                if evt["event"] == "subagent_stream_ready":
                    stream_from_tcm_subagent = True
                else:
                    if evt["event"] == "artifact":
                        collected_artifacts.append(evt["data"])
                    elif evt["event"] == "token":
                        non_subagent_tokens.append(str(evt["data"].get("delta", "")))
                    yield _format_sse(evt["event"], evt["data"])

            # 3. 若触发了 Sub-Agent Handoff 到中医针灸 32B 专家节点，则由 vLLM 直接向前端流式吐出 Token
            if stream_from_tcm_subagent:
                collected_tokens: List[str] = []
                async for chunk_evt in stream_tcm_specialist_handoff(
                    windowed_messages=final_state["messages"],
                    clinical_slots=final_state["clinical_slots"],
                    latest_user_query=req.query,
                ):
                    if chunk_evt["event"] == "token":
                        collected_tokens.append(chunk_evt["data"]["delta"])
                    yield _format_sse(chunk_evt["event"], chunk_evt["data"])

                full_reply = "".join(collected_tokens)
                final_state["messages"] = [*final_state["messages"], AIMessage(content=full_reply)]
            else:
                full_reply = "".join(non_subagent_tokens)
                if not full_reply and final_state.get("messages"):
                    last_msg = final_state["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        full_reply = str(last_msg.content)

            # 4. 【流结束后双写】：
            #    (a) 短事务永久写入 MySQL (agent_threads + agent_messages)
            #    (b) 刷新 Redis 热状态缓存与 TTL (供下一次对话 1ms 命中)
            persisted_to_mysql = await persist_turn_to_db(
                session_id=req.session_id,
                uid=req.uid,
                user_query=req.query,
                assistant_reply=full_reply,
                artifacts=collected_artifacts,
                final_state=final_state,
            )
            await session_redis_store.save_state(req.session_id, final_state)

            yield _format_sse(
                "done",
                {
                    "session_id": req.session_id,
                    "state_source": state_source,
                    "persisted_to_mysql": persisted_to_mysql,
                    "sticky_agent": final_state.get("sticky_agent"),
                    "clinical_slots": final_state.get("clinical_slots"),
                    "active_assets": [a["asset_id"] for a in final_state.get("active_assets") or []],
                    "redis_backend": "redis" if redis_manager.is_redis_live else "in_memory_fallback",
                },
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# 长耗时异步任务进度查询与 Pub/Sub SSE 订阅端点
# ============================================================================
@app.get("/api/v1/tasks/{task_id}", tags=["Redis-TaskBus (异步任务状态)"])
async def get_async_task_status(task_id: str):
    """查询 Redis Hash 中保存的任务状态快照"""
    snapshot = await get_task_snapshot(task_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"未找到任务记录: {task_id}")
    return snapshot


@app.get("/api/v1/tasks/{task_id}/events", tags=["Redis-TaskBus (异步任务状态)"])
async def stream_async_task_events(task_id: str):
    """通过 Redis Pub/Sub 订阅长耗时 GPU/MCP 任务的实时进度流 (SSE)"""

    async def task_sse_gen() -> AsyncGenerator[str, None]:
        async for evt in subscribe_task_events(task_id):
            yield _format_sse("task_progress", evt)

    return StreamingResponse(task_sse_gen(), media_type="text/event-stream")


# ============================================================================
# Redis 热缓存与 MySQL 永久历史对照学习接口
# ============================================================================
@app.get("/api/v1/redis/inspect/{session_id}", tags=["Storage-Debug (Redis与MySQL对照学习)"])
async def inspect_redis_session(session_id: str):
    """查看指定会话在 Redis 中存储的热缓存 Key、剩余 TTL、粘性路由及临床槽位"""
    return await session_redis_store.inspect_raw_state(session_id)


@app.delete("/api/v1/redis/session/{session_id}", tags=["Storage-Debug (Redis与MySQL对照学习)"])
async def clear_redis_session(session_id: str):
    """清除 Redis 热缓存（用于模拟 2 小时 TTL 过期，观察下一轮如何自动从 MySQL 冷回源重建）"""
    deleted = await session_redis_store.delete_state(session_id)
    return {"session_id": session_id, "redis_deleted": deleted}


@app.get("/api/v1/sessions/{session_id}/history", tags=["Storage-Debug (Redis与MySQL对照学习)"])
async def get_mysql_session_history(session_id: str):
    """查看 MySQL `agent_threads` + `agent_messages` 中永久保存的完整会话历史与画布记录"""
    return await get_full_thread_history_from_db(session_id)


@app.get("/api/v1/mcp/tools", tags=["MCP-Host (已挂载MCP工具自省)"])
async def inspect_registered_mcp_tools():
    """查询当前已连接的脑病预测 FastMCP 服务 (`api_new_mcp`) 暴露的工具列表"""
    try:
        tools = await brain_mcp_client.list_available_tools()
        return {"status": "connected", "mcp_endpoint": settings.BRAIN_AD_MCP_URL, "tools": tools}
    except Exception as e:
        return {"status": "offline", "mcp_endpoint": settings.BRAIN_AD_MCP_URL, "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)
