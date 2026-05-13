# 📖 Python AI 平台 - 源码学习指南

本文档是对本项目的深度拆解，旨在帮助你更好地理解代码逻辑。  
我们在主要的 `.py` 和 `.html` 源码文件中添加了**中文注释**，建议对照本指南阅读源码。

**版本说明**：以下内容以当前仓库 **v3.0** 行为为准（流式对话、系统提示词、Token 历史、RAG 上传与检索）；若你本地代码较旧，请先与远程仓库同步。

---

## 一、技术栈全景图

本项目采用轻量级、易本地跑通的现代 Web + AI 栈：

| 层级 | 技术 | 作用 |
|------|------|------|
| Web 框架 | **FastAPI** | 异步友好，适合 AI 接口与流式响应。 |
| ORM / 数据库 | **SQLAlchemy** + **SQLite** | 聊天记录持久化，零配置本地文件库。 |
| 聊天模型 | **OpenAI Python SDK**（`AsyncOpenAI`） | 调用兼容 OpenAI Chat Completions 的网关（由 `OPENAI_BASE_URL` 决定）。 |
| RAG | **LangChain**（`langchain_community` 等）+ **FAISS**（`faiss-cpu`） | 文档加载、切块、本地向量索引与相似度检索。 |
| 向量化（Embedding） | **DashScope HTTP API** + **`requests`** | 将文本转为向量；实现见 `app/rag/document_processor.py`。Bearer 与聊天客户端**共用**配置项 `OPENAI_API_KEY`（未单独拆分「向量专用 Key」）。 |
| 历史 Token 估算 | **tiktoken**（`cl100k_base`） | 在流式对话路径中控制上下文长度，见 `app/api/endpoints.py`。 |
| 前端 | **单文件 HTML/CSS/JS** | 无构建步骤，便于理解请求与流式渲染。 |

---

## 二、项目核心流程解析（与当前前端一致）

默认页面加载后，**发送消息**走的是 **`POST /api/chat/stream`**，并会带上 **系统提示词**；若已上传文档并完成向量化，同一路径会 **自动做 RAG 检索**。下面按时间顺序说明。

### 步骤 1：前端发起请求 (`static/index.html`)

1. 用户点击发送，执行 `sendMessage()`。
2. 读取输入框与 **「角色设定 (System Prompt)」**，组装 JSON：`{ "message": "...", "system_prompt": "..." }`（`system_prompt` 可为空字符串）。
3. 使用 `fetch('POST', '/api/chat/stream', ...)` 发起请求。
4. 使用 `res.body.getReader()` 与 `TextDecoder` **逐块读取响应体**，拼接到新建的助手气泡中（打字机效果）。

> **说明**：路由层使用 `StreamingResponse` 且 `media_type` 为 `text/event-stream`；生成器按块 `yield` 模型输出的**原始文本片段**。前端按字节流解码拼接即可，**未使用**浏览器 `EventSource` 解析标准 SSE `data:` 帧格式。

### 步骤 2：后端接收与校验 (`app/api/endpoints.py`)

1. 进入 `chat_with_ai_stream`，FastAPI 用 Pydantic 模型 **`ChatRequest`** 校验 JSON（`message` 必填，`system_prompt` 默认 `""`）。
2. `db: Session = Depends(get_db)` 注入数据库会话。

### 步骤 3：写入用户消息

1. 构造 `models.ChatMessage(role="user", content=request.message)`，`db.add` 后 `commit`。

### 步骤 4：RAG 检索（若有向量库）

1. 调用 `retrieve_relevant_context(request.message)`（定义在 `app/rag/document_processor.py`）。
2. 若 `app/rag/vectorstore` 目录不存在（尚未上传过文档），返回空字符串，对话行为与「无知识库」一致。
3. 若有检索结果，将其拼接到 **`final_system_prompt`** 末尾（带「请基于以下参考资料…」等说明），再与请求里的 `system_prompt` 合并。

### 步骤 5：组装历史上下文（Token 截断）

1. 查询全部 `ChatMessage`，按 `id` **降序**取出（最新在前）。
2. **跳过第一条**：即刚写入的当前用户消息，从更早的消息开始累加。
3. 对每条历史调用 `get_messages_token_count`（基于 **tiktoken `cl100k_base`**），在不超过 **`MAX_HISTORY_TOKENS`（代码中为 2000）** 的前提下从近到远追加；再 **反转为时间正序**，得到 `history` 列表。
4. 将 `history` 与当前用户句一起交给 `generate_ai_response_stream`（当前句在 `llm.py` 里再次作为最后一条 `user` 消息追加，与历史拼接方式以源码为准）。

> **与固定「最近 10 条」的区别**：非流式接口 `POST /api/chat` 仍使用「最近 10 条」逻辑，**且不调用 RAG**、不使用请求体中的 `system_prompt`。这是刻意保留的简化路径，便于对比学习。

### 步骤 6：流式调用大模型 (`app/services/llm.py`)

1. `generate_ai_response_stream` 组装 `messages`：**若有 `system_prompt`（已含 RAG 拼接结果）则放在最前**，再 `extend(history)`，最后追加当前用户 `user` 消息。
2. `await client.chat.completions.create(..., stream=True)`，`async for chunk` 解析 `delta.content` 并 `yield` 给上层。

### 步骤 7：流结束后的持久化

1. `endpoints.py` 中 `event_generator` 在流式迭代结束后，将完整回复写入 **`ChatMessage(role="assistant", ...)`** 并 `commit`。

### 步骤 8：上传知识库（可选路径）

1. 用户选择 PDF 或 TXT，前端 `POST /api/upload`，`multipart/form-data` 字段名 **`file`**。
2. 后端保存到 `uploads/`，调用 `process_and_store_document`：加载 → **RecursiveCharacterTextSplitter** 切块 → **DashScopeEmbeddings** 向量化 → **FAISS** 写入或合并到 `app/rag/vectorstore` 目录。

### 步骤 9：历史列表与清空

1. **`GET /api/history`**：按 `id` 升序返回全部消息，供页面初次渲染。
2. **`DELETE /api/history`**：删除表中所有聊天行（**不删除**已上传文件与向量库；若需「清空知识库」需另行实现或手动删目录）。

---

## 三、行业延伸：上下文与记忆（与代码对照）

当前流式路径已实现 **「滑动窗口 + Token 计数」** 的一种形式（按条累加直至超过阈值）。更进阶的做法仍可扩展：

1. **按模型真实上下文上限调参**：将 `MAX_HISTORY_TOKENS` 与所用聊天模型的窗口、预留输出 Token 一并考虑。  
2. **对话摘要**：历史过长时用廉价模型生成摘要，以 system 或单独消息注入。  
3. **长期记忆 RAG**：把历史对话也 embedding 入库，按问题检索相关旧话（与本仓库「文档 RAG」可并存为不同集合）。

详见 `app/api/endpoints.py` 中 `get_messages_token_count` 与循环截断逻辑。

---

## 四、推荐的源码阅读顺序

建议按以下顺序打开文件（**结合文件内注释**）：

1. **`app/main.py`**：应用创建、CORS、`/api` 路由、静态资源挂载。  
2. **`app/core/config.py`**：环境变量与默认值（含 `EMBEDDING_MODEL`）。  
3. **`app/db/database.py`** 与 **`app/db/models.py`**：引擎、`get_db`、`ChatMessage` 表结构。  
4. **`app/api/endpoints.py`**：`ChatRequest` / `ChatResponse`；**重点** `chat_with_ai_stream`、`upload_document`；对照 `chat_with_ai`（非流式）。  
5. **`app/services/llm.py`**：流式与非流式 Chat Completions。  
6. **`app/rag/document_processor.py`**：Embedding、切块、FAISS 入库与 `retrieve_relevant_context`。  
7. **`static/index.html`**：系统提示词、上传、流式 `fetch`、历史加载与清空。

---

## 五、开发工具推荐：查看 SQLite 数据库

项目使用 SQLite，数据文件为项目根目录下的 **`ai_platform.db`**（由 `DATABASE_URL` 配置）。

在 VS Code 等编辑器中查看表数据，可安装轻量插件（例如搜索 **SQLite Viewer**），在资源管理器中单击 `ai_platform.db` 以表格方式浏览 **`chat_messages`** 表。

向量数据不在 SQLite 中，而在 **`app/rag/vectorstore/`** 目录（FAISS 本地文件）；上传的原始文件在 **`uploads/`**。

---

## 六、接口一览（便于你对照抓包或写客户端）

| 方法 | 路径 | 作用 |
|------|------|------|
| `POST` | `/api/chat/stream` | 流式对话；支持 `system_prompt`；内置 RAG 检索；历史按 Token 截断。 |
| `POST` | `/api/chat` | 非流式对话；最近 10 条历史；无 RAG；无请求体 `system_prompt`。 |
| `POST` | `/api/upload` | 上传 PDF/TXT，解析并向量化入库。 |
| `GET` | `/api/history` | 返回全部聊天消息。 |
| `DELETE` | `/api/history` | 清空聊天消息表。 |

---

祝你学习愉快。看不懂的地方可以把具体文件与行号发给 AI 助手，请它结合本指南讲解。
