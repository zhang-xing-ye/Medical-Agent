"""
================================================================================
Redis 异步连接池与分布式锁基座 (redis.asyncio)
================================================================================
工程设计要点：
1. 【全异步非阻塞】：采用 `redis.asyncio` 连接池（`max_connections=50`），与 FastAPI / LangGraph
   共享同一个 `asyncio` 事件循环，绝不阻塞 SSE 流式输出。
2. 【统一 Key 命名空间】：所有 Key 均通过 `build_key(...)` 加上 `dao_c:` 前缀，
   防止与同 Redis 实例下其他实验项目冲突。
3. 【安全分布式锁】：加锁使用 `SET key token NX EX ttl`，解锁使用 Lua 脚本校验 `token`
   原子比对后删除，防止“A 线程超时后误删 B 线程持有的锁”。
4. 【本地降级兜底】：若开发环境未启动 Redis 服务，提供轻量内存兜底容器保证逻辑可阅读与降级可用。
"""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import settings

# Lua 原子解锁脚本：只有锁的持有者 (ARGV[1] == token) 才能删除锁
_RELEASE_LOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def build_key(*parts: str) -> str:
    """构造规范化的 Redis Key，例如 build_key('agent', 'session', 's1') -> 'dao_c:agent:session:s1'"""
    clean_parts = [str(p).strip(":") for p in parts if p]
    return ":".join([settings.REDIS_KEY_PREFIX, *clean_parts])


class InMemoryFallbackStore:
    """当本地未启动 Redis 服务时的降级内存存储（仅用于单进程开发调试兜底）"""

    def __init__(self) -> None:
        self._kv: Dict[str, tuple[str, Optional[float]]] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._zsets: Dict[str, list[tuple[str, float]]] = {}

    def _evict_if_expired(self, key: str) -> None:
        item = self._kv.get(key)
        if item and item[1] is not None and time.time() > item[1]:
            self._kv.pop(key, None)

    async def get(self, key: str) -> Optional[str]:
        self._evict_if_expired(key)
        item = self._kv.get(key)
        return item[0] if item else None

    async def set(
        self, key: str, value: str, ex: Optional[int] = None, nx: bool = False
    ) -> bool:
        self._evict_if_expired(key)
        if nx and key in self._kv:
            return False
        expire_at = (time.time() + ex) if ex else None
        self._kv[key] = (value, expire_at)
        return True

    async def setex(self, key: str, time_sec: int, value: str) -> bool:
        return await self.set(key, value, ex=time_sec)

    async def delete(self, *keys: str) -> int:
        deleted = 0
        for k in keys:
            if k in self._kv:
                del self._kv[k]
                deleted += 1
            if k in self._hashes:
                del self._hashes[k]
                deleted += 1
        return deleted

    async def exists(self, key: str) -> int:
        self._evict_if_expired(key)
        return 1 if key in self._kv else 0

    async def hset(self, name: str, mapping: Dict[str, str]) -> int:
        bucket = self._hashes.setdefault(name, {})
        bucket.update({k: str(v) for k, v in mapping.items()})
        return len(mapping)

    async def hgetall(self, name: str) -> Dict[str, str]:
        return dict(self._hashes.get(name, {}))

    async def expire(self, key: str, seconds: int) -> bool:
        if key in self._kv:
            val, _ = self._kv[key]
            self._kv[key] = (val, time.time() + seconds)
            return True
        return key in self._hashes

    async def ttl(self, key: str) -> int:
        self._evict_if_expired(key)
        item = self._kv.get(key)
        if not item:
            return -2
        if item[1] is None:
            return -1
        return max(0, int(item[1] - time.time()))


class RedisManager:
    """管理全局异步 Redis 连接池生命周期与常用并发原语"""

    def __init__(self) -> None:
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._fallback = InMemoryFallbackStore()
        self._using_fallback: bool = False

    async def init(self) -> None:
        """在 FastAPI lifespan 启动时初始化异步连接池并执行 PING 探活"""
        if self._client is not None:
            return
        try:
            self._pool = aioredis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=50,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=3.0,
            )
            client = aioredis.Redis(connection_pool=self._pool)
            await client.ping()
            self._client = client
            self._using_fallback = False
        except (RedisError, OSError):
            # 若本地未启动 Redis，降级为内存字典模式，不阻断网关启动
            self._client = None
            self._using_fallback = True

    async def close(self) -> None:
        """应用关闭时优雅释放连接池资源"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

    @property
    def is_redis_live(self) -> bool:
        return self._client is not None and not self._using_fallback

    async def get_client(self) -> Any:
        if self._client is None and not self._using_fallback:
            await self.init()
        return self._client if self._client is not None else self._fallback

    @asynccontextmanager
    async def distributed_lock(
        self,
        lock_name: str,
        ttl_seconds: int = 30,
        wait_timeout: float = 3.0,
    ) -> AsyncGenerator[bool, None]:
        """
        分布式互斥锁上下文管理器：
        用于保护同一 `session_id` 的并发请求（如用户连点发送两次），防止 LangGraph 热状态覆盖写。
        """
        client = await self.get_client()
        lock_key = build_key("lock", lock_name)
        token = uuid.uuid4().hex
        acquired = False
        deadline = time.monotonic() + wait_timeout

        try:
            while time.monotonic() < deadline:
                ok = await client.set(lock_key, token, ex=ttl_seconds, nx=True)
                if ok:
                    acquired = True
                    break
                await asyncio.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                if self.is_redis_live:
                    await client.eval(_RELEASE_LOCK_LUA, 1, lock_key, token)
                else:
                    current = await client.get(lock_key)
                    if current == token:
                        await client.delete(lock_key)


redis_manager = RedisManager()
