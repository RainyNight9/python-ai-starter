# Python AI Starter 🚀

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991.svg)](https://openai.com/)

欢迎来到 **Python AI 全栈入门项目**！本项目专为你从 0 搭建自己的 AI 调用平台而设计，涵盖了「后端接口 + AI 调用 + 数据库存储 + 极简前端」。当前默认体验已对齐 **v3.2**：在 **v3.1**（流式、系统提示词、Token 历史、RAG、Agent）基础上，增加 **多会话**（`chat_sessions` + 请求体 `session_id`）与 **RAG 引用可观测**（检索片段的文件名、页码、距离、摘要预览；流式首行 `::META::` JSON，Agent 响应字段 `rag_citations`）。更细的小白向说明见 **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)**。

通过这个项目，你将经历：**从工程入手 → 跑通 → 理解 → 升级 → 变成 AI 全栈** 的完整学习路径。

## 📑 目录 (Table of Contents)
- [当前版本已实现能力 (v3.2)](#-当前版本已实现能力-v32)
- [🟢 第一阶段：工程入手与跑通 (Run)](#-第一阶段工程入手与跑通-run)
- [🔵 第二阶段：理解核心代码 (Understand)](#-第二阶段理解核心代码-understand)
- [🟠 第三阶段：内置能力说明与仍可做的挑战 (Upgrade)](#-第三阶段内置能力说明与仍可做的挑战-upgrade)
- [🔴 第四阶段：变成 AI 全栈 (Become Full-Stack)](#-第四阶段变成-ai-全栈-become-full-stack)
- [🛠️ 常见问题与避坑指南 (Troubleshooting)](#-常见问题与避坑指南-troubleshooting)
- [🤝 参与贡献 (Contributing)](#-参与贡献-contributing)

---

## ✨ 当前版本已实现能力 (v3.2)

| 能力 | 说明 |
|------|------|
| **流式对话** | 前端默认（未勾选 Agent）调用 `POST /api/chat/stream`；首行固定为 `::META::` + JSON（`rag_citations`，可为空），随后为模型文本流。 |
| **系统提示词** | 请求体字段 `system_prompt`；与 RAG 拼接后一并注入 system。 |
| **多会话** | 表 `chat_sessions`；聊天请求体带 `session_id`（默认 `1`）；`GET/POST /api/sessions`、`DELETE /api/sessions/{id}`；`GET/DELETE /api/history?session_id=` 按会话隔离。 |
| **历史与 Token** | 流式与 Agent 接口内用 `tiktoken`（`cl100k_base`）控制历史消息总 Token。 |
| **RAG + 引用** | 检索使用 `similarity_search_with_score`；返回 `source`（文件名）、`page`、`distance`、`snippet_preview`；前端在黄色信息条展示。 |
| **Agent / 工具调用** | `POST /api/chat/agent`：返回 `reply` + `steps` + `rag_citations`；需网关支持 `tools` / `tool_calls`。 |
| **非流式兼容** | `POST /api/chat`：最近 10 条、无 RAG、无 `system_prompt`；同样支持 `session_id`。 |
| **历史 API** | `GET /api/history?session_id=`；`DELETE /api/history?session_id=`。 |

**配置要点：**

- 聊天模型：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MODEL_NAME`（见 `app/core/config.py` 与 `.env.example`）。
- RAG 向量化：当前实现通过 **DashScope** 的 HTTP Embedding API；**与聊天共用** `OPENAI_API_KEY` 作为 Bearer。启用 RAG 时请确认该 Key 对 DashScope Embedding 有效，并理解其与聊天网关是否为同一套凭证（详见上表与 `app/rag/document_processor.py` 内注释）。
- 向量模型：`EMBEDDING_MODEL`（如 `text-embedding-v3`），见 `.env.example`。

---

## 🟢 第一阶段：工程入手与跑通 (Run)

### 1. 环境准备
确保你已经安装了 Python 3.8 或以上版本。推荐使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
1. 复制 `.env.example` 文件并重命名为 `.env`：
   ```bash
   cp .env.example .env
   ```
2. 打开 `.env` 文件，至少配置：
   - `OPENAI_API_KEY`：**同一变量**在代码中被两处使用——`AsyncOpenAI` 聊天鉴权，以及 RAG 里 DashScope Embedding 的 `Bearer`（见 `document_processor.py`）。因此在你**不改代码**的前提下，若既要 RAG 又要聊天，需保证该 Key 对 **两处服务** 都可用；常见做法是聊天与向量均走 **DashScope**（或同一兼容网关），或仅使用不需要 DashScope 的聊天路径且暂时不用 RAG。
   - `OPENAI_BASE_URL`、`MODEL_NAME`：按你的聊天模型提供方填写。
   - `EMBEDDING_MODEL`：使用 RAG 时，与 DashScope 文档一致即可（默认 `text-embedding-v3`）。
   > 若你希望「聊天用 A 厂商 Key、向量用 B 厂商 Key」，需要把配置拆成两个环境变量并在代码里分别传入；当前仓库未拆分。

### 4. 启动服务
```bash
uvicorn app.main:app --reload
```
若终端找不到 `uvicorn`，可使用：
```bash
python3 -m uvicorn app.main:app --reload
```

启动成功后，打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)  
在页面中可设置角色、上传 PDF/TXT 后再对话（默认走流式接口）。

---

## 🔵 第二阶段：理解核心代码 (Understand)

了解项目是怎么跑起来的，建议按下面顺序阅读（**零基础长文手册**见 [LEARNING_GUIDE.md](LEARNING_GUIDE.md)；代码已迭代至 **v3.2**）：

1. **`app/main.py`**（程序入口）  
   挂载 API（`/api`）、静态前端（`static/`）、CORS。

2. **`app/db/models.py`**、`app/db/migrate.py`（v3.2 多会话）  
   `ChatSession` / `ChatMessage.session_id`；启动时 SQLite 轻量迁移。

3. **`app/core/config.py`**（配置）  
   从环境变量读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`MODEL_NAME`、`EMBEDDING_MODEL` 等。

4. **`app/services/llm.py`**（聊天模型调用）  
   `generate_ai_response_stream`：流式 + `system_prompt` + `history`；`generate_ai_response`：非流式，供 `/api/chat` 使用。

5. **`app/api/endpoints.py`**（HTTP 接口）  
   - `POST /chat/stream`：按 `session_id` 存消息 → RAG 检索（含引用元数据）→ Token 截断历史 → 流式输出（首行 `::META::`）。  
   - `POST /chat/agent`：同上，响应含 `steps` 与 `rag_citations`。  
   - `GET/POST /api/sessions`、`DELETE /api/sessions/{id}`；`GET/DELETE /history?session_id=`。  
   - `POST /upload`、`POST /chat`：见上表。

6. **`app/services/agent_service.py`**、`app/agent/`（Agent）  
   工具声明与实现、`run_tool_agent` 多轮 `tool_calls` 编排与 `steps` 轨迹。

7. **`app/rag/document_processor.py`**（RAG）  
   文档加载、切块、DashScope Embedding、FAISS；`retrieve_relevant_context_with_citations` 返回正文与引用列表。

8. **`app/db/database.py`**（持久化）  
   引擎、`get_db`、会话工厂。

9. **`static/index.html`**（前端）  
   会话切换/新建/删除、系统提示词、上传、Agent 勾选、流式解析 `::META::` 与 RAG 引用展示、历史按会话加载。

---

## 🟠 第三阶段：内置能力说明与仍可做的挑战 (Upgrade)

下列能力 **已在当前代码中实现**，适合对照源码阅读，而不是从零重做一遍：

- **流式输出**：`/api/chat/stream` + `llm.generate_ai_response_stream`；首行 `::META::` 承载 `rag_citations`。
- **系统提示词**：`ChatRequest.system_prompt`，流式与 Agent 链路均可使用（非流式 `/api/chat` 除外）。
- **多会话**：`ChatSession` + `session_id`；`/api/sessions` 与按会话的 `/api/history`。
- **RAG 引用可观测**：`retrieve_relevant_context_with_citations` + 前端黄色引用条；Agent 响应字段 `rag_citations`。
- **Agent 工具调用**：`/api/chat/agent` + `agent_service.run_tool_agent` + `app/agent/`；前端展示 `steps`。
- **换模型 / 换网关**：改 `.env` 中的 `OPENAI_BASE_URL`、`MODEL_NAME`、`OPENAI_API_KEY`；无需改 `llm.py` 里的 URL（客户端从 `settings` 读取）。

你仍可尝试的 **进阶挑战**（仓库尚未实现或仅部分涉及）：

1. **会话增强**：重命名会话、导出某会话为 Markdown、会话级「清空向量库」等。  
2. **统一两条聊天接口**：让 `/api/chat` 也支持 `system_prompt` 与 RAG，或明确在文档中保留「简化版 vs 完整版」的教学分工（当前为后者）。  
3. **RAG 增强**：混合检索、rerank、按文档删除索引、重建向量库等。  
4. **流式 Agent / 更多工具**：在现有 Agent 循环上增加 SSE、搜索、天气、业务 API 等（注意工具安全与白名单）。

---

## 🔴 第四阶段：变成 AI 全栈 (Become Full-Stack)

在掌握 v3.2 代码路径基础上，可继续深入：

1. **RAG 工程化**：更大规模的切片策略、评测集、幻觉与引用格式规范；向量库可对比 Chroma / Milvus 与当前 FAISS 本地方案的差异。  
2. **Agent 进阶**：流式工具回合、并行工具、更强错误恢复与观测（日志 / OpenTelemetry）。  
3. **前端工程化**：在保留本仓库「单文件可读」的前提下，增加 React/Vue 等示例目录或独立小项目。

---

## 🛠️ 常见问题与避坑指南 (Troubleshooting)

在初次跑通项目的过程中，如果你遇到了环境或依赖安装问题，可以参考以下解决办法：

### 1. `pip install` 报错（如 `pydantic-core` 编译失败）
- **原因**：如果你使用的是较新的 Python 版本（如 Python 3.14），旧版本的依赖包（带写死版本号）可能还没提供预编译的 wheel 包，导致系统尝试从 C/Rust 源码编译安装时由于 API 不兼容而报错。
- **解决办法**：
  1. 去掉 `requirements.txt` 里的版本号锁定（例如把 `pydantic==2.5.3` 改成 `pydantic`），让系统自动拉取适配你当前 Python 版本的最新包。
  2. 使用明确的模块调用方式来安装：
     ```bash
     python3 -m pip install -r requirements.txt
     ```

### 2. 终端提示 `zsh: command not found: uvicorn`
- **原因**：当 `pip` 发现当前用户对全局 Python 环境没有写入权限时，会自动把包安装到用户独立的目录下（如 `~/.local/bin` 或 `~/Library/Python/3.14/bin`）。如果这个目录没有被配置到你的系统变量（PATH）里，终端就找不到 `uvicorn` 命令。
- **解决办法**：
  不需要修改环境变量，直接通过 Python 模块的方式调用 Uvicorn 即可：
  ```bash
  python3 -m uvicorn app.main:app --reload
  ```

### 3. RAG 上传成功但对话似乎「用不上文档」
- **检查**：`OPENAI_API_KEY` 是否对 **DashScope Embedding** 请求有效（与聊天是否同一厂商需结合你的配置判断）；`app/rag/vectorstore` 是否在首次上传后生成；对话是否走 **`/api/chat/stream`** 或 **`/api/chat/agent`**（二者均会注入检索结果）。非流式 `/api/chat` **不会**调用 RAG。

---

## 🤝 参与贡献 (Contributing)

作为一个开源的入门级项目，我们非常欢迎任何形式的贡献，包括但不限于：
- 修复代码或文档中的错别字/Bug
- 提交「第三阶段」进阶挑战的实现代码（可以提交到 `examples/` 目录下）
- 优化前端界面的 UI

**贡献流程：**
1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 发起 Pull Request

## 📄 开源协议 (License)

本项目采用 [MIT License](LICENSE) 开源协议。你可以自由地使用、修改和分发本项目代码。

> **开始你的代码之旅吧！如有问题随时呼叫我。**
