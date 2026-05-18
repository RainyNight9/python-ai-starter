# 部署指南

## 快速开始

### 1. 环境准备

确保已安装：
- Python 3.8+
- Node.js 18+ 和 pnpm

### 2. 后端启动

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 启动后端
make backend
# 或
python3 -m uvicorn app.main:app --reload
```

后端将运行在 `http://127.0.0.1:8000`

访问 `http://127.0.0.1:8000/docs` 查看 API 文档。

### 3. 前端启动

```bash
cd frontend
pnpm install
pnpm dev
```

前端将运行在 `http://localhost:5173`

### 4. 生产构建

```bash
# 前端构建
cd frontend
pnpm build

# 构建产物在 frontend/dist/
# 可以复制到 static/ 目录供后端直接服务
```

## 配置说明

### 必需配置

在 `.env` 中配置：

```env
# 聊天模型配置
OPENAI_API_KEY=your_chat_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

# RAG 向量化配置
EMBEDDING_API_KEY=your_embedding_key  # 可选，不填则使用 OPENAI_API_KEY
EMBEDDING_MODEL=text-embedding-v3
```

### 可选配置

```env
# RAG 检索参数
RAG_TOP_K=3
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50

# 数据库
DATABASE_URL=sqlite:///./ai_platform.db
```

## Docker 部署（可选）

创建 `Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建并运行：

```bash
docker build -t python-ai-starter .
docker run -p 8000:8000 -v $(pwd)/uploads:/app/uploads python-ai-starter
```

## 生产部署建议

### 1. 使用生产级 ASGI 服务器

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 2. 反向代理

使用 Nginx 或 Caddy 作为反向代理：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. 持久化存储

确保以下目录持久化：
- `uploads/` - 上传的文档
- `app/rag/vectorstore/` - 向量库索引
- `ai_platform.db` - SQLite 数据库

### 4. 环境变量安全

生产环境不要提交 `.env` 文件，使用环境变量或密钥管理服务。

## 常见问题

### 1. 向量库检索失败

确保 `EMBEDDING_API_KEY` 对 DashScope Embedding API 有效。

### 2. Agent 工具调用失败

确认模型支持 function calling（如 gpt-4、qwen-plus 等）。

### 3. 前端跨域问题

开发环境已配置代理，生产环境确保后端 CORS 配置正确。

## 验收检查

运行以下命令验证部署：

```bash
# 后端编译检查
make test

# 前端构建检查
cd frontend && pnpm build

# 完整检查
make check
```

访问 `http://127.0.0.1:8000/docs` 测试以下流程：
1. 创建会话
2. 上传文档
3. 流式问答（查看 RAG 引用）
4. Agent 问答（查看工具调用步骤）
5. 查看历史记录
