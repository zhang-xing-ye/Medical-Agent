"""
Redis 内存数据库基础设施包 (对应架构设计中的“热状态缓存 + 限流鉴权 + 任务进度总线”)

包含模块：
- redis_client.py        : 异步连接池、健康检查、分布式锁 (SET NX EX + Lua 原子解锁)
- session_state_store.py : LangGraph 多轮会话热状态缓存 (sticky_agent / 临床槽位 / 消息序列化)
- auth_and_ratelimit.py  : JWT 吊销黑名单、Lua ZSET 滑动窗口限流、Cache-Aside 旁路缓存
- task_progress_bus.py   : 长耗时 GPU/MCP 任务的状态 Hash 与 Pub/Sub 实时进度事件总线
"""
