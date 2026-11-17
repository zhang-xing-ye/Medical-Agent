"""
================================================================================
Redis 鉴权黑名单、滑动窗口限流与 Cache-Aside 旁路缓存
================================================================================
集中演示后端高频考察的三大经典 Redis 工程模式：
1. 【JWT 主动吊销黑名单 (Logout Revocation)】：
   - 无状态 JWT 一旦签发默认无法提前失效；用户退出登录或封号时，
     将 Token 的 SHA256 摘要存入 Redis `dao_c:auth:blacklist:{hash}`，
     TTL 设为 Token 剩余有效期（到期自动清理，不留垃圾数据）。
2. 【Lua 原子滑动窗口限流 (Sliding Window Rate Limiter)】：
   - 保护昂贵的 GPU 推理与 DeepSeek API 接口 (`/api/v1/agent/chat/stream`)；
   - 使用 Redis `ZSET` + 原子 Lua 脚本：剔除窗口外旧时间戳 -> 统计窗口内请求数 ->
     未超限则 `ZADD` 当前毫秒时间戳并刷新过期时间，杜绝“先查后写”并发竞态。
3. 【Cache-Aside 旁路缓存 + 防击穿锁】：
   - 缓存低频变更、高频读取的菜单分类树 `/category/list/tree` (`dao_c:legacy:category_tree`)，
     降低 MySQL 查询压力；通过互斥锁防止热点 Key 过期瞬间大量请求击穿数据库。
"""
import hashlib
import json
import time
import uuid
from typing import Any, Optional, Tuple

from app.cache.redis_client import build_key, redis_manager
from app.config import settings

# Lua 原子滑动窗口限流脚本
# KEYS[1]: 限流 ZSET Key
# ARGV[1]: 当前毫秒时间戳 (now_ms)
# ARGV[2]: 窗口起始毫秒时间戳 (window_start_ms)
# ARGV[3]: 最大允许请求次数 (max_requests)
# ARGV[4]: 窗口过期秒数 (window_seconds)
# ARGV[5]: 本次请求唯一成员名 (member)
_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now_ms = tonumber(ARGV[1])
local window_start = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local ttl_sec = tonumber(ARGV[4])
local member = ARGV[5]

redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
local current_count = redis.call('ZCARD', key)
if current_count >= max_requests then
    return {0, current_count}
end

redis.call('ZADD', key, now_ms, member)
redis.call('EXPIRE', key, ttl_sec)
return {1, current_count + 1}
"""

# 内存降级用的滑动窗口时间戳记录
_FALLBACK_RATE_BUCKETS: dict[str, list[float]] = {}


def _token_digest(token: str) -> str:
    clean = token.replace("Bearer ", "").strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:32]


# ============================================================================
# 1. JWT 主动吊销黑名单
# ============================================================================
async def blacklist_jwt_token(token: str, exp_timestamp: Optional[int] = None) -> None:
    """将退出登录的 JWT 写入 Redis 黑名单，TTL 与 Token 剩余生存时间对齐"""
    if not token:
        return
    now_sec = int(time.time())
    ttl = max(60, (exp_timestamp - now_sec)) if exp_timestamp else (settings.JWT_EXPIRE_DAYS * 86400)
    key = build_key("auth", "blacklist", _token_digest(token))
    client = await redis_manager.get_client()
    await client.setex(key, ttl, "1")


async def is_jwt_blacklisted(token: str) -> bool:
    """在鉴权拦截器中校验 Token 是否已被注销拉黑"""
    if not token:
        return False
    key = build_key("auth", "blacklist", _token_digest(token))
    client = await redis_manager.get_client()
    return bool(await client.exists(key))


# ============================================================================
# 2. 基于 Redis ZSET + Lua 的滑动窗口限流器
# ============================================================================
async def check_rate_limit(
    subject_id: str,
    max_requests: int = settings.RATE_LIMIT_MAX_REQUESTS,
    window_seconds: int = settings.RATE_LIMIT_WINDOW_SECONDS,
) -> Tuple[bool, int]:
    """
    校验 `subject_id` (如 session_id 或 user_id) 在最近 `window_seconds` 秒内是否超过请求配额。
    返回: (是否放行 allowed, 当前窗口已使用次数 current_count)
    """
    key = build_key("ratelimit", "chat", subject_id)
    now_ms = int(time.time() * 1000)
    window_start_ms = now_ms - window_seconds * 1000

    if redis_manager.is_redis_live:
        client = await redis_manager.get_client()
        member = f"{now_ms}:{uuid.uuid4().hex[:6]}"
        res = await client.eval(
            _SLIDING_WINDOW_LUA,
            1,
            key,
            now_ms,
            window_start_ms,
            max_requests,
            window_seconds + 1,
            member,
        )
        allowed = int(res[0]) == 1
        count = int(res[1])
        return allowed, count

    # 降级单机内存滑动窗口
    bucket = [t for t in _FALLBACK_RATE_BUCKETS.get(key, []) if t > window_start_ms]
    if len(bucket) >= max_requests:
        _FALLBACK_RATE_BUCKETS[key] = bucket
        return False, len(bucket)
    bucket.append(now_ms)
    _FALLBACK_RATE_BUCKETS[key] = bucket
    return True, len(bucket)


# ============================================================================
# 3. Cache-Aside 旁路缓存 (用于三级分类树 /category/list/tree)
# ============================================================================
CATEGORY_TREE_CACHE_KEY = build_key("legacy", "category_tree")


async def get_cached_category_tree() -> Optional[Any]:
    client = await redis_manager.get_client()
    raw = await client.get(CATEGORY_TREE_CACHE_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def set_cached_category_tree(tree_data: Any) -> None:
    client = await redis_manager.get_client()
    await client.setex(
        CATEGORY_TREE_CACHE_KEY,
        settings.REDIS_CATEGORY_CACHE_TTL_SECONDS,
        json.dumps(tree_data, ensure_ascii=False),
    )


async def invalidate_category_tree_cache() -> None:
    """当管理员增删改模型或分类时，执行 Cache-Aside 写后删缓存"""
    client = await redis_manager.get_client()
    await client.delete(CATEGORY_TREE_CACHE_KEY)
