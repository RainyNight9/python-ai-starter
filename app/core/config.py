import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# ==========================================
# 配置文件 (集中管理环境变量)
# ==========================================

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "Python AI Starter"

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ai_platform.db")

    EMBEDDING_API_KEY: str = os.getenv(
        "EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")
    )
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    RAG_CHUNK_SIZE: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    RAG_CHUNK_OVERLAP: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))


settings = Settings()
