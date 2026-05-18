# Python AI Starter 🚀

一个从零开始的 AI 全栈入门项目，帮助你快速掌握 AI 应用开发的核心技术。

## 📖 项目简介

本项目是一个完整的 AI 对话平台，包含：

- **后端**：FastAPI + SQLite + OpenAI 兼容接口
- **前端**：React + TypeScript + Vite
- **核心功能**：
  - 💬 多会话管理
  - 🌊 流式对话输出
  - 📚 RAG 文档问答（检索增强生成）
  - 🤖 Agent 工具调用
  - 📊 完整的引用追踪

## ✨ 当前版本功能 (v3.3+)

### 基础功能
- ✅ 多会话隔离管理
- ✅ 流式对话（打字机效果）
- ✅ 系统提示词自定义
- ✅ 历史记录管理
- ✅ Token 智能截断

### RAG 功能
- ✅ PDF/TXT 文档上传
- ✅ 自动文档切片与向量化
- ✅ 语义检索与引用展示
- ✅ 文档管理（列表、删除、重建）
- ✅ 可配置的检索参数

### Agent 功能
- ✅ 计算器工具
- ✅ 时间查询工具
- ✅ 文档检索工具
- ✅ 会话导出工具
- ✅ 完整的工具调用轨迹

### 前端功能
- ✅ 现代化 UI 设计
- ✅ 响应式布局
- ✅ 实时流式显示
- ✅ RAG 引用卡片
- ✅ Agent 步骤展开
- ✅ 知识库可视化管理

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- Node.js 18+
- pnpm（推荐）或 npm

### 2. 安装依赖

```bash
# 后端依赖
pip install -r requirements.txt

# 前端依赖
cd frontend
pnpm install
```

### 3. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的配置
# 必需配置：
# - OPENAI_API_KEY: 你的 API 密钥
# - OPENAI_BASE_URL: API 网关地址
# - MODEL_NAME: 模型名称
```

### 4. 启动服务

```bash
# 启动后端（终端 1）
make backend
# 或
python3 -m uvicorn app.main:app --reload

# 启动前端（终端 2）
make frontend
# 或
cd frontend && pnpm dev
```

### 5. 访问应用

- 前端界面：http://localhost:5173
- API 文档：http://127.0.0.1:8000/docs

## 📚 文档导航

### 新手入门
- **[TESTING.md](TESTING.md)** - 完整的测试流程，带你体验所有功能
- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - 零基础学习指南，详细讲解每个概念

### 开发文档
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 部署指南，包含 Docker 和生产环境配置
- **[.env.example](.env.example)** - 配置说明，所有环境变量的详细解释

### 代码结构
```
python-ai-starter/
├── app/                    # 后端代码
│   ├── api/               # API 路由
│   │   └── endpoints.py   # 所有接口定义（已添加详细注释）
│   ├── agent/             # Agent 工具
│   │   ├── tools.py       # 工具实现（已添加详细注释）
│   │   └── tool_definitions.py  # 工具声明
│   ├── core/              # 核心配置
│   │   └── config.py      # 环境变量配置
│   ├── db/                # 数据库
│   │   ├── models.py      # 数据模型
│   │   ├── database.py    # 数据库连接
│   │   └── migrate.py     # 数据库迁移
│   ├── rag/               # RAG 功能
│   │   └── document_processor.py  # 文档处理（已添加详细注释）
│   ├── services/          # 业务逻辑
│   │   ├── llm.py         # LLM 调用
│   │   └── agent_service.py  # Agent 编排
│   └── main.py            # 应用入口
├── frontend/              # 前端代码
│   ├── src/
│   │   ├── App.tsx        # 主应用组件（已添加详细注释）
│   │   ├── App.css        # 样式文件
│   │   └── main.tsx       # 入口文件
│   └── vite.config.ts     # Vite 配置
├── uploads/               # 上传的文档
├── static/                # 静态文件（教学版前端）
├── .env.example           # 配置模板
├── requirements.txt       # Python 依赖
├── Makefile              # 快捷命令
└── README.md             # 本文件
```

## 🎯 核心概念

### 1. 多会话管理

每个会话独立存储聊天记录，互不干扰：

```python
# 数据库模型
ChatSession:
  - id: 会话 ID
  - title: 会话标题
  - created_at: 创建时间

ChatMessage:
  - id: 消息 ID
  - session_id: 所属会话
  - role: user/assistant
  - content: 消息内容
```

### 2. RAG（检索增强生成）

工作流程：
1. 用户上传文档 → 文档切片 → 向量化 → 存入向量库
2. 用户提问 → 问题向量化 → 检索相似片段 → 拼接到 prompt
3. LLM 基于检索内容回答 → 返回答案和引用信息

### 3. Agent 工具调用

Agent 可以自主决定调用哪些工具：

```
用户: "现在几点？帮我算一下到晚上8点还有多久"
  ↓
Agent 思考: 需要先查询时间，再进行计算
  ↓
调用工具 1: get_server_time() → "14:30"
  ↓
调用工具 2: calculate("(20-14)*60 + (0-30)") → "330分钟"
  ↓
返回: "现在是14:30，距离晚上8点还有330分钟（5.5小时）"
```

### 4. 流式输出

使用 Server-Sent Events (SSE) 实现打字机效果：

```
客户端发送请求
  ↓
服务端首行返回: ::META::{"rag_citations":[...]}
  ↓
服务端逐字返回: "根" "据" "参" "考" "资" "料" "..."
  ↓
客户端实时显示
```

## 🔧 配置说明

### 必需配置

```env
# 聊天模型配置
OPENAI_API_KEY=sk-xxx              # 你的 API 密钥
OPENAI_BASE_URL=https://api.openai.com/v1  # API 网关
MODEL_NAME=gpt-4o-mini             # 模型名称
```

### 可选配置

```env
# RAG 向量化配置（不填则使用 OPENAI_API_KEY）
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_MODEL=text-embedding-v3

# RAG 检索参数
RAG_TOP_K=3                        # 检索返回的文档片段数
RAG_CHUNK_SIZE=500                 # 文档切片大小（字符）
RAG_CHUNK_OVERLAP=50               # 切片重叠大小（字符）

# 数据库
DATABASE_URL=sqlite:///./ai_platform.db
```

## 📖 使用示例

### 示例 1：基础对话

```
用户: "Python 有哪些特点？"
系统提示词: "你是一个 Python 专家"

AI: "Python 的主要特点包括：
1. 简洁易读的语法
2. 动态类型系统
3. 丰富的标准库
..."
```

### 示例 2：RAG 文档问答

```
1. 上传文档: "Python教程.pdf"
2. 提问: "这个教程讲了什么内容？"

AI: "根据参考资料，这个教程主要讲解了..."

引用信息:
- 文件: Python教程.pdf
- 页码: 第 1 页
- 相似度: 0.234
- 片段: "本教程将介绍 Python 的基础语法..."
```

### 示例 3：Agent 工具调用

```
用户: "帮我计算 (100 + 200) * 3，然后从文档中查找相关内容"

Agent 执行步骤:
1. 调用 calculate("(100 + 200) * 3") → "900"
2. 调用 search_uploaded_documents("计算") → 检索结果
3. 综合回答: "计算结果是 900。根据文档..."
```

## 🧪 测试流程

详细的测试步骤请查看 [TESTING.md](TESTING.md)，包括：

1. ✅ 后端 API 测试（16 项）
2. ✅ 前端功能测试（13 项）
3. ✅ 集成场景测试（4 项）
4. ✅ 常见问题排查

## 🎓 学习路径

### 第一步：跑通项目（30 分钟）
1. 按照"快速开始"配置环境
2. 启动后端和前端
3. 完成 TESTING.md 中的基础测试

### 第二步：理解代码（2-3 小时）
1. 阅读 LEARNING_GUIDE.md
2. 按顺序阅读核心文件（已添加详细注释）：
   - `app/main.py` - 应用入口
   - `app/api/endpoints.py` - API 接口
   - `app/services/llm.py` - LLM 调用
   - `app/rag/document_processor.py` - RAG 实现
   - `app/agent/tools.py` - Agent 工具
   - `frontend/src/App.tsx` - 前端主组件

### 第三步：动手实践（1-2 天）
1. 修改系统提示词，定制 AI 角色
2. 添加自己的工具函数
3. 调整 RAG 参数，优化检索效果
4. 美化前端界面

### 第四步：进阶开发（持续）
1. 接入更多模型提供商
2. 实现流式 Agent
3. 添加更多工具（搜索、天气等）
4. 部署到生产环境

## 🔍 核心 API 接口

### 会话管理
- `GET /api/sessions` - 获取会话列表
- `POST /api/sessions` - 创建新会话
- `DELETE /api/sessions/{id}` - 删除会话

### 聊天接口
- `POST /api/chat/stream` - 流式对话（支持 RAG）
- `POST /api/chat/agent` - Agent 对话（支持工具调用）
- `POST /api/chat` - 非流式对话（简化版）

### 文档管理
- `POST /api/upload` - 上传文档
- `GET /api/documents` - 文档列表
- `DELETE /api/documents/{filename}` - 删除文档
- `POST /api/documents/rebuild` - 重建向量库

### 历史记录
- `GET /api/history?session_id=1` - 获取历史
- `DELETE /api/history?session_id=1` - 清空历史

### RAG 配置
- `GET /api/rag/settings` - 查看 RAG 配置

## 🛠️ 开发工具

### Makefile 命令

```bash
make backend          # 启动后端
make frontend         # 启动前端
make frontend-build   # 构建前端
make test            # 运行测试
make check           # 完整检查（测试+构建）
```

### API 文档

访问 http://127.0.0.1:8000/docs 查看交互式 API 文档，可以直接测试所有接口。

## 🐛 常见问题

### Q1: 后端启动失败
**A:** 检查 Python 版本和依赖安装，使用 `python3 -m uvicorn app.main:app --reload`

### Q2: 文档上传失败
**A:** 确认 `EMBEDDING_API_KEY` 配置正确，检查文件格式（只支持 PDF/TXT）

### Q3: RAG 检索无结果
**A:** 确认文档已成功上传，检查 `app/rag/vectorstore/` 目录是否存在

### Q4: Agent 工具不调用
**A:** 确认模型支持 function calling（如 gpt-4、qwen-plus）

### Q5: 前端无法连接后端
**A:** 确认后端正在运行，检查 `vite.config.ts` 中的代理配置

更多问题请查看 [TESTING.md](TESTING.md) 的"常见问题排查"章节。

## 📦 部署

### 开发环境
```bash
# 后端
make backend

# 前端
make frontend
```

### 生产环境

详细部署方案请查看 [DEPLOYMENT.md](DEPLOYMENT.md)，包括：
- Docker 部署
- Nginx 反向代理
- 环境变量管理
- 持久化存储配置

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

贡献方式：
1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 发起 Pull Request

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

## 🙏 致谢

感谢所有贡献者和使用者！

## 📞 联系方式

- 提交 Issue：[GitHub Issues](https://github.com/your-repo/issues)
- 讨论交流：[GitHub Discussions](https://github.com/your-repo/discussions)

---

**开始你的 AI 全栈之旅吧！** 🚀

如有问题，请先查看：
1. [TESTING.md](TESTING.md) - 测试流程
2. [LEARNING_GUIDE.md](LEARNING_GUIDE.md) - 学习指南
3. [DEPLOYMENT.md](DEPLOYMENT.md) - 部署指南
