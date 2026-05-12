from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

# ==========================================
# 数据库核心配置 (SQLAlchemy 引擎与会话)
# ==========================================

# 1. 创建数据库引擎 (Engine)
# Engine 负责与数据库建立底层连接。
# check_same_thread=False 是 SQLite 特有的配置，允许在多个线程中共享数据库连接，这在 FastAPI 异步框架中是必需的。
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# 2. 创建会话工厂 (SessionLocal)
# Session 负责执行具体的增删改查操作。每次有请求来时，我们都会通过它创建一个新的 Session 实例。
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 3. 创建 ORM 基类 (Base)
# 我们在 models.py 中定义的数据表类都需要继承这个 Base 类，SQLAlchemy 会根据这些类自动生成对应的 SQL 语句。
Base = declarative_base()

# 4. 获取数据库会话的依赖函数 (Dependency)
# 这是一个生成器函数。在 FastAPI 中，每次处理请求时通过 Depends(get_db) 调用它。
# 它的作用是：请求来时创建连接，请求处理完毕后（无论是否报错）通过 finally 确保连接被关闭。
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
