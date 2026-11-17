"""
================================================================================
长耗时 MCP/GPU 异步任务进度总线 (Redis Hash + Pub/Sub)
================================================================================
适用场景：
对于全脑高精度分割 (BMS)、大批量 TAP 预测等耗时数十秒甚至数分钟的 GPU 任务：
- Worker 侧调用 `publish_task_progress(...)`：
  1. 用 Redis Hash (`HSET dao_c:task:status:{task_id}`) 记录最新快照状态，供前端随时刷新轮询；
  2. 用 Redis Pub/Sub (`PUBLISH dao_c:task:events:{task_id}`) 实时广播百分比进度；
- FastAPI SSE 网关侧调用 `subscribe_task_events(task_id)`：
  订阅频道并零轮询实时推送进度给前端，收到 `succeeded` / `failed` 终态后自动退订。
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional

from app.cache.redis_client import build_key, redis_manager


def _status_key(task_id: str) -> str:
    return build_key("task", "status", task_id)


def _channel_name(task_id: str) -> str:
    return build_key("task", "events", task_id)


# 本地降级用的进程内事件队列
_FALLBACK_QUEUES: Dict[str, asyncio.Queue] = {}


async def publish_task_progress(
    task_id: str,
    status: str,
    progress: int,
    message: str,
    result_asset_id: Optional[str] = None,
) -> None:
    """
    更新任务快照状态 (Redis Hash) 并向 Pub/Sub 频道广播进度事件。
    status 取值: queued | running | succeeded | failed
    """
    event_payload = {
        "task_id": task_id,
        "status": status,
        "progress": progress,
        "message": message,
        "result_asset_id": result_asset_id or "",
        "updated_at": str(int(time.time())),
    }
    client = await redis_manager.get_client()
    s_key = _status_key(task_id)
    await client.hset(s_key, mapping={k: str(v) for k, v in event_payload.items()})
    await client.expire(s_key, 3600)

    msg_json = json.dumps(event_payload, ensure_ascii=False)
    if redis_manager.is_redis_live:
        await client.publish(_channel_name(task_id), msg_json)
    else:
        q = _FALLBACK_QUEUES.setdefault(task_id, asyncio.Queue())
        await q.put(event_payload)


async def get_task_snapshot(task_id: str) -> Dict[str, str]:
    """查询任务当前状态快照（断线重连或轮询接口使用）"""
    client = await redis_manager.get_client()
    return await client.hgetall(_status_key(task_id))


async def subscribe_task_events(
    task_id: str,
    timeout_seconds: float = 120.0,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    订阅指定 `task_id` 的实时进度事件流，用于转接至 FastAPI SSE `StreamingResponse`。
    """
    # 先发送当前 Hash 中已有的最新快照，防止订阅前任务已完成或错失首包
    snapshot = await get_task_snapshot(task_id)
    if snapshot:
        yield snapshot
        if snapshot.get("status") in ("succeeded", "failed"):
            return

    if redis_manager.is_redis_live:
        client = await redis_manager.get_client()
        pubsub = client.pubsub()
        channel = _channel_name(task_id)
        await pubsub.subscribe(channel)
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        try:
            while asyncio.get_event_loop().time() < deadline:
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if msg and msg.get("type") == "message":
                    payload = json.loads(msg["data"])
                    yield payload
                    if payload.get("status") in ("succeeded", "failed"):
                        break
                else:
                    await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
    else:
        q = _FALLBACK_QUEUES.setdefault(task_id, asyncio.Queue())
        deadline = asyncio.get_event_loop().time() + timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=1.0)
                yield payload
                if payload.get("status") in ("succeeded", "failed"):
                    break
            except asyncio.TimeoutError:
                continue
