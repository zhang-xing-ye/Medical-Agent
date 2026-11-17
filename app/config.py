"""
全局配置中心 (替代原 Java application.properties 与 Python .env)
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"请在 .env 或环境变量中配置 {name}，参考 .env.example")
    return value


class Settings:
    # 1. 服务端口 (默认 8081，直接无缝替代原 Spring Boot 后端)
    PORT: int = int(os.getenv("PORT", "8081"))

    # 2. 数据库配置 (兼容原 MySQL dao_c_demo，可显式选择 SQLite 供单机调试)
    MYSQL_URL: str = os.getenv(
        "DATABASE_URL",
        "mysql+aiomysql://medical_agent:change-me@localhost:3306/dao_c_demo?charset=utf8mb4",
    )
    USE_SQLITE_FALLBACK: bool = os.getenv("USE_SQLITE_FALLBACK", "false").lower() == "true"
    SQLITE_URL: str = "sqlite+aiosqlite:///./dao_c_demo_local.db"

    # 2.5 Redis 内存数据库配置 (对齐原 Java application.properties: spring.redis.database=3)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/3")
    REDIS_KEY_PREFIX: str = os.getenv("REDIS_KEY_PREFIX", "dao_c")
    # 会话热状态过期时间 (默认 2 小时，每次对话自动续期)
    REDIS_SESSION_TTL_SECONDS: int = int(os.getenv("REDIS_SESSION_TTL_SECONDS", "7200"))
    # 资产凭据元数据缓存时间 (默认 24 小时)
    REDIS_ASSET_TTL_SECONDS: int = int(os.getenv("REDIS_ASSET_TTL_SECONDS", "86400"))
    # 分类菜单旁路缓存时间 (默认 10 分钟)
    REDIS_CATEGORY_CACHE_TTL_SECONDS: int = int(os.getenv("REDIS_CATEGORY_CACHE_TTL_SECONDS", "600"))
    # LLM/Agent 滑动窗口限流配置 (默认每用户/会话每 60 秒最多 20 次请求)
    RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "20"))
    RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    # 3. 安全配置由部署环境提供；接入旧系统时使用原系统的密钥以保持兼容
    JWT_SIGNATURE: str = _required_env("SIGNATURE")
    JWT_EXPIRE_DAYS: int = 10
    AES_SECRET_KEY_BASE64: str = _required_env("AES_SECRET_KEY")

    # 4. 统一资产中心 (AssetHub) 存储路径 (用于存放 .nii.gz / .npy 等大体积医学文件)
    SHARED_ASSET_DIR: str = os.getenv(
        "SHARED_ASSET_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "storage", "assets")),
    )

    # 5. 总控 DeepSeek 模型配置 (Supervisor / Central Orchestrator)
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    SUPERVISOR_MODEL: str = os.getenv("SUPERVISOR_MODEL", "deepseek-chat")

    # 6. 中医针灸 32B 领域微调大模型配置 (由 vLLM OpenAI-Compatible Server 部署)
    # 若本地调试暂未启动 vLLM，可回退至 DeepSeek API 模拟运行
    TCM_VLLM_BASE_URL: str = os.getenv("TCM_VLLM_BASE_URL", "http://localhost:8000/v1")
    TCM_VLLM_API_KEY: str = os.getenv("TCM_VLLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", "EMPTY"))
    TCM_MODEL_NAME: str = os.getenv(
        "TCM_MODEL_NAME", "DeepSeek-R1-Acupuncture-32B-LoRA"
    )

    # 7. 脑病预测 FastMCP 服务地址 (对应重构后的 api_new_mcp)
    BRAIN_AD_MCP_URL: str = os.getenv("BRAIN_AD_MCP_URL", "http://localhost:8101/mcp")

    # 8. 上下文工程：历史消息动态滑动窗口 Token 预算
    MAX_HISTORY_TOKEN_BUDGET: int = int(os.getenv("MAX_HISTORY_TOKEN_BUDGET", "3000"))


settings = Settings()
os.makedirs(settings.SHARED_ASSET_DIR, exist_ok=True)
