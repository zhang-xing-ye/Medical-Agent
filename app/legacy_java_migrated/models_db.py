"""
对应原 Java MyBatis-Plus 实体类:
- User.java       -> Table("user")
- Category.java   -> Table("category")
- Model.java      -> Table("model")
- Session.java    -> Table("session")
使用 SQLAlchemy 2.0 Async 异步引擎，全面替代原 MyBatis-Plus Mapper。
"""
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import settings

db_url = settings.SQLITE_URL if settings.USE_SQLITE_FALLBACK else settings.MYSQL_URL
engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class UserEntity(Base):
    """对应原 com.dao.c.backend.entity.User"""

    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[Optional[str]] = mapped_column(String(100))
    nickname: Mapped[Optional[str]] = mapped_column(String(100))
    phone: Mapped[Optional[str]] = mapped_column(String(255))
    email: Mapped[Optional[str]] = mapped_column(String(255))
    avatar: Mapped[Optional[str]] = mapped_column(String(500))
    status: Mapped[int] = mapped_column(Integer, default=1)
    openid: Mapped[Optional[str]] = mapped_column(String(255))
    openid_hash: Mapped[Optional[str]] = mapped_column(String(255))
    createtime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    updatetime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    password: Mapped[Optional[str]] = mapped_column(String(255))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "nickname": self.nickname,
            "phone": self.phone,
            "email": self.email,
            "avatar": self.avatar,
            "status": self.status,
            "createtime": self.createtime.isoformat() if self.createtime else None,
            "updatetime": self.updatetime.isoformat() if self.updatetime else None,
        }


class CategoryEntity(Base):
    """对应原 com.dao.c.backend.entity.Category"""

    __tablename__ = "category"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_name: Mapped[Optional[str]] = mapped_column(String(100))
    category_url: Mapped[Optional[str]] = mapped_column(String(255))
    category_level: Mapped[int] = mapped_column(Integer, default=1)
    created_time: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    status: Mapped[int] = mapped_column(Integer, default=1)

    def to_camel_dict(self) -> dict:
        """保持与原 Java Jackson 驼峰序列化字段名一致"""
        return {
            "id": self.id,
            "categoryName": self.category_name,
            "categoryUrl": self.category_url,
            "categoryLevel": self.category_level,
            "createdTime": self.created_time.isoformat() if self.created_time else None,
            "status": self.status,
            "children": [],
        }


class ModelEntity(Base):
    """对应原 com.dao.c.backend.entity.Model"""

    __tablename__ = "model"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    server_url: Mapped[Optional[str]] = mapped_column(String(255))
    url: Mapped[Optional[str]] = mapped_column(String(255))
    api: Mapped[Optional[str]] = mapped_column(String(255))
    modelname: Mapped[Optional[str]] = mapped_column(String(100))
    intro: Mapped[Optional[str]] = mapped_column(Text)
    avatar: Mapped[Optional[str]] = mapped_column(String(500))
    createtime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    updatetime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    status: Mapped[int] = mapped_column(Integer, default=1)
    served_model_name: Mapped[Optional[str]] = mapped_column(String(100))
    category_id: Mapped[Optional[int]] = mapped_column(Integer)

    def to_public_camel_dict(self) -> dict:
        """对应原 ModelServiceImpl 排除 url / api / status 后的驼峰字段"""
        return {
            "id": self.id,
            "serverUrl": self.server_url,
            "modelname": self.modelname,
            "intro": self.intro,
            "avatar": self.avatar,
            "createtime": self.createtime.isoformat() if self.createtime else None,
            "updatetime": self.updatetime.isoformat() if self.updatetime else None,
            "servedModelName": self.served_model_name,
            "categoryId": self.category_id,
        }


class SessionEntity(Base):
    """对应原 com.dao.c.backend.entity.Session（旧版表单会话表，保留 100% 兼容）"""

    __tablename__ = "session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[Optional[int]] = mapped_column(Integer)
    mid: Mapped[Optional[int]] = mapped_column(Integer)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    content: Mapped[Optional[str]] = mapped_column(Text)
    createtime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    updatetime: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.now)
    status: Mapped[int] = mapped_column(Integer, default=1)


# ============================================================================
# 新增：Multi-Agent 智能体持久化表（MySQL 冷存储总账，配合 Redis 热缓存使用）
# ============================================================================
class AgentThreadEntity(Base):
    """
    智能体会话线程主表 (MySQL 永久保存)：
    - 当 Redis 中活跃会话状态 (TTL=2h) 过期被淘汰后，系统通过此表无损恢复
      `sticky_agent`、`clinical_slots_json` 和 `active_assets_json`。
    """

    __tablename__ = "agent_threads"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    uid: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sticky_agent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    clinical_slots_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active_assets_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latest_brain_output_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


class AgentMessageEntity(Base):
    """
    智能体多轮对话消息明细表 (MySQL 永久保存)：
    - Redis 热缓存仅保留最近 40 条用于快速推理；
    - 本表永久保存全量问答文本与前端画布 `_artifact` JSON，支持长期历史回溯与科研审计。
    """

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    artifacts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


async def init_db_tables() -> bool:
    """
    自动创建新增的 agent_threads / agent_messages 表（若表已存在则幂等跳过）。
    若本地未启动 MySQL，安全捕获异常以便降级仅用 Redis/内存学习阅读。
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                Base.metadata.create_all,
                tables=[AgentThreadEntity.__table__, AgentMessageEntity.__table__],
            )
        return True
    except Exception:
        return False


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
