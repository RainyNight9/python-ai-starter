from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.db.database import engine
from app.db.models import Base

# ==========================================
# FastAPI 主入口文件 (程序的起点)
# ==========================================

# 初始化数据库表：
# SQLAlchemy 会检查数据库（我们配置的 sqlite:///./ai_platform.db）中是否已经有了表结构
# 如果没有，它会自动根据 models.py 中的定义创建表
Base.metadata.create_all(bind=engine)

# 创建 FastAPI 应用实例
app = FastAPI(title="Python AI Platform", description="从 0 到 1 搭建的 AI 调用平台")

# 添加 CORS (跨域资源共享) 中间件
# 如果你的前端和后端不在同一个端口（比如前端跑在 3000，后端跑在 8000）
# 浏览器会默认阻止跨域请求。加上这个中间件允许所有来源访问，解决开发时的跨域报错。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有源（开发环境方便，生产环境建议写死具体域名）
    allow_credentials=True,
    allow_methods=["*"], # 允许所有 HTTP 方法 (GET, POST 等)
    allow_headers=["*"], # 允许所有请求头
)

# 挂载 API 路由
# 将 endpoints.py 中定义的路由（如 /chat, /history）挂载到应用上
# prefix="/api" 表示这些接口统一加一个前缀，实际访问就是 /api/chat
app.include_router(router, prefix="/api")

# 挂载静态文件（前端页面）
# 这样访问 http://127.0.0.1:8000/ 时，FastAPI 会自动返回 static/index.html
app.mount("/", StaticFiles(directory="static", html=True), name="static")
