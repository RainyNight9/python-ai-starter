import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# ==========================================
# 配置文件 (集中管理环境变量)
# ==========================================

# 调用 load_dotenv() 会自动去项目根目录寻找 .env 文件，并将其加载到系统的环境变量中
load_dotenv()

# 使用 Pydantic 的 BaseSettings 可以非常方便地进行配置管理和类型检查
class Settings(BaseSettings):
    PROJECT_NAME: str = "Python AI Starter"
    
    # AI 相关的配置：默认从环境变量中取，如果没有则使用第二个参数的默认值
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    # 数据库连接 URL，这里使用本地 SQLite 数据库文件 (ai_platform.db)
    DATABASE_URL: str = "sqlite:///./ai_platform.db"

    # RAG / Embedding 相关配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# 实例化配置对象，其他文件可以直接 from app.core.config import settings 引入使用
settings = Settings()
