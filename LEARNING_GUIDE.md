# Python AI Starter 零基础学习手册（v3.1）

本文档面向**几乎没有 Python 或 AI 工程经验**的读者：先用白话讲清「Python 你要会什么」「AI 概念是什么」，再按**当前仓库真实代码（v3.1）**把整条链路讲透。阅读时建议打开对应 `.py` / `static/index.html` 对照源码中的中文注释。

**v3.1 相对 v3.0 的变化**：在保留「流式对话 + 系统提示词 + Token 历史 + RAG」的基础上，新增 **Agent 模式**（`POST /api/chat/agent`）：模型可通过 **工具调用（Function Calling）** 让后端执行计算器、服务器时间、字符串反转等演示工具，并返回 **`steps` 轨迹**便于学习；前端增加「Agent 模式」勾选框。

---

## 如何使用本手册

| 你的情况 | 建议读法 |
|----------|----------|
| 完全新手 | 按顺序从「第一部分」读到「第五部分」，再动手跑 README 里的启动命令。 |
| 会一点 Python | 可跳过第一部分部分小节，从第二部分 AI 概念开始。 |
| 只想查接口/文件 | 直接看「第七部分 接口一览」「第六部分 关键文件索引」。 |

---

## 第一部分：读本项目前，你需要哪些 Python 基础

### 1.1 Python 是什么

Python 是一门**解释型**编程语言：你把代码写在 `.py` 文件里，用 `python` 命令运行，解释器**一行行执行**。本项目要求 **Python 3.8+**（见 `README.md`）。

### 1.2 虚拟环境与依赖

- **虚拟环境（venv）**：把当前项目的第三方包装在独立目录里，避免和系统里其他项目冲突。  
  典型命令：`python -m venv venv`，然后 `source venv/bin/activate`（Mac/Linux）。
- **pip**：安装依赖的工具。本项目依赖写在 `requirements.txt`，执行 `pip install -r requirements.txt`。

### 1.3 本项目会遇到的语法（速查）

**变量与基础类型**

```python
name = "Alice"      # 字符串 str
count = 3           # 整数 int
ratio = 0.5         # 浮点数 float
items = [1, 2, 3]   # 列表 list：有序，可重复，用下标 items[0]
user = {"role": "user", "content": "你好"}  # 字典 dict：键值对，用 user["role"]
```

**函数与返回值**

```python
def add(a: int, b: int) -> int:
    return a + b
```

`-> int` 是**类型提示**，帮助 IDE 检查错误，运行时不强制。

**异步 `async` / `await`（本项目核心）**

- 在 **FastAPI** 里，很多接口处理函数写成 `async def`，里面用 `await` 等待网络 I/O（例如等大模型 API 返回），这样**同一线程**可以在等待时去处理其他请求，**并发**更好。
- 规则很简单：在 `async def` 里，遇到「可能要等一会儿」的异步函数，前面加 `await`。
- 本项目中：`generate_ai_response_stream`、`run_tool_agent`、`chat_with_ai_stream` 等都涉及异步。

**导入模块**

```python
from app.core.config import settings   # 从包 app.core.config 导入对象 settings
```

`app` 是源码包，运行时要保证在项目根目录执行 `uvicorn app.main:app`，这样 Python 才能找到 `app`。

**列表与字典的组合（对话历史）**

大模型 API 普遍使用「消息列表」：

```python
messages = [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "1+1=?"},
]
```

本项目里 `history`、 `messages` 大量出现这种结构。

---

## 第二部分：AI 与大模型相关概念（白话）

### 2.1 大模型（LLM）在做什么

可以把 **大语言模型** 想象成一个**读过海量文本的预测机器**：根据你给的上下文，**预测下一个词**（实际是多 token），连起来就是一句完整回复。它**不会天然记住**你的上传文件或昨天聊过什么——除非你把内容**再次放进本次请求**里。

### 2.2 本项目如何「跟模型说话」

我们通过 **兼容 OpenAI 的 Chat Completions HTTP API**（用官方 `openai` 库里的 `AsyncOpenAI`）把 `messages` 发给网关，网关再转发给具体模型。网关地址由环境变量 **`OPENAI_BASE_URL`** 决定，模型名由 **`MODEL_NAME`** 决定。

### 2.3 消息里的 `role`：system / user / assistant / tool

| 角色 | 含义 |
|------|------|
| `system` | 系统级说明：人设、规则、注入的知识摘要等。 |
| `user` | 用户说的话。 |
| `assistant` | 模型上一轮或历史轮次的回复。 |
| `tool` | **仅在有工具调用时出现**：后端把工具执行结果以该角色发回模型，让模型继续推理。 |

### 2.4 流式 vs 非流式

- **非流式**：一次 HTTP 响应返回**整段**答案，等待时间长，但实现简单。  
- **流式（stream）**：服务器边生成边推**小块文本**，前端边收边显示「打字机效果」，首字更快。  
  本仓库默认聊天走 **`POST /api/chat/stream`**，响应类型为 `text/event-stream`；前端用 `fetch` + `ReadableStream` 读字节流（**不是**浏览器 `EventSource` 解析标准 SSE 帧）。

### 2.5 Token 与历史截断

模型一次能看的上下文长度有限，且通常按 **Token**（可理解为「词块」）计费。本项目用 **tiktoken**（`cl100k_base`）估算历史消息的 Token 数，在流式与 Agent 路径中把历史限制在约 **`MAX_HISTORY_TOKENS = 2000`** 以内（见 `app/api/endpoints.py`）。  
非流式的旧接口 **`POST /api/chat`** 仍用「最近 10 条」简化策略，**便于对比学习**。

### 2.6 RAG（检索增强生成）是什么

**RAG** = 先把文档切成小块 → 用 **Embedding 模型**把每块变成向量 → 存进 **向量库**；用户提问时，把问题也变成向量，做**相似度检索**，把最相关的几段文字取出来，**拼进 system 侧说明**里再让模型回答。  
这样模型「看到」了文档内容，而不是凭空编造。本仓库向量库为本地 **FAISS**（目录 `app/rag/vectorstore/`），Embedding 走 **DashScope HTTP API**（与聊天共用配置项 `OPENAI_API_KEY` 作为 Bearer，详见 `app/rag/document_processor.py` 注释）。

### 2.7 Agent 与工具调用（Function Calling）是什么

**普通聊天**：模型只能输出文本，**不能**真的在你电脑上运行代码。  
**Agent（工具调用）**：我们把一组**工具定义**（名字、描述、参数 JSON Schema）随请求发给模型；模型若判断「需要算一下 / 查一下时间」，会返回 **`tool_calls`**；**后端**根据名字执行对应 Python 函数，把**字符串结果**以 `role="tool"` 的消息发回；模型再读结果，决定继续调工具还是给出最终自然语言答案。  
本仓库的编排逻辑在 **`app/services/agent_service.py`**，工具声明在 **`app/agent/tool_definitions.py`**，实现在 **`app/agent/tools.py`**。

---

## 第三部分：技术栈与目录结构（对照仓库）

| 层级 | 技术 | 作用 |
|------|------|------|
| Web | **FastAPI** | 提供 REST API、依赖注入、异步路由。 |
| 数据库 | **SQLAlchemy** + **SQLite** | 表 `chat_messages` 存 user/assistant 文本。 |
| 聊天 | **OpenAI Python SDK**（`AsyncOpenAI`） | 调用兼容 OpenAI 的 Chat Completions。 |
| RAG | **LangChain** 组件 + **FAISS** | 加载 PDF/TXT、切块、向量索引与检索。 |
| Embedding | **DashScope**（`requests` 调用） | 文本转向量。 |
| 历史长度 | **tiktoken** | Token 估算。 |
| 前端 | **单文件** `static/index.html` | 无构建步骤；流式 + Agent 双路径。 |

**建议目录心智模型**

```
app/
  main.py              # 入口：建表、CORS、挂载 /api 与 static
  core/config.py       # 环境变量与 settings
  api/endpoints.py     # 所有 HTTP 接口
  services/llm.py      # 纯聊天：流式 / 非流式
  services/agent_service.py  # Agent 多轮工具循环
  agent/               # 工具声明 + 实现
  db/                  # SQLite 连接与 ORM 模型
  rag/document_processor.py    # RAG 入库与检索
static/index.html      # 页面
uploads/               # 上传的原始文件
ai_platform.db         # SQLite 数据库文件（默认路径）
```

---

## 第四部分：v3.1 已实现能力总览

| 能力 | 说明 |
|------|------|
| **流式对话** | `POST /api/chat/stream`；`llm.generate_ai_response_stream`；前端默认路径。 |
| **系统提示词** | 请求体 `system_prompt`；与 RAG 拼接后的内容一并作为 system 发给模型。 |
| **Token 历史** | 流式与 Agent 路径均按 Token 从近到远截取历史（上限约 2000 tokens）。 |
| **RAG** | `POST /api/upload` 上传 PDF/TXT → 切块 → DashScope Embedding → FAISS；对话前 `retrieve_relevant_context`。 |
| **非流式简化接口** | `POST /api/chat`：最近 10 条、**无** `system_prompt`、**无** RAG。 |
| **Agent + 工具** | `POST /api/chat/agent`：Function Calling 多轮；返回 `reply` + `steps`；可与 RAG 并存；落库仍为 user/assistant 各一条（不存 tool 中间消息）。 |
| **历史 API** | `GET /api/history`、`DELETE /api/history`。 |

**前端行为（`static/index.html`）**

- 未勾选 **Agent 模式**：`POST /api/chat/stream`（流式 + RAG）。  
- 勾选 **Agent 模式**：`POST /api/chat/agent`（非流式 JSON）；界面展示可折叠的 **`steps`** 便于对照学习。

---

## 第五部分：从请求到响应的完整路径

### 5.1 路径 A：默认流式聊天（含 RAG 与 Token 历史）

1. 用户在前端输入并发送 → `sendMessage()` → `fetch('/api/chat/stream', { message, system_prompt })`。  
2. **`chat_with_ai_stream`**（`endpoints.py`）：校验 `ChatRequest` → 写入 user 消息到 SQLite。  
3. **`retrieve_relevant_context`**：若有向量库则检索，把结果拼到 `final_system_prompt`。  
4. 从数据库按 id 倒序取历史，**跳过刚写入的当前 user**，按条累加 Token，不超过上限后反转为时间正序 → `history`。  
5. **`generate_ai_response_stream`**（`llm.py`）：`messages = [system?] + history + 当前 user`。  
6. `AsyncOpenAI.chat.completions.create(..., stream=True)`，`async for chunk` 把 `delta.content` 逐块 `yield`。  
7. `StreamingResponse` 把 chunk 写给浏览器；生成器结束后将**完整拼接**的 assistant 文本写入数据库。

### 5.2 路径 B：Agent 模式（工具调用 + steps）

1. 前端勾选 Agent → `POST /api/chat/agent`，请求体同样是 `ChatRequest`。  
2. 与流式路径一致：写 user → RAG 检索拼 system → **相同** Token 历史策略得到 `history`。  
3. **`run_tool_agent`**（`agent_service.py`）：组装 `messages`，附带 **`tools=get_openai_style_tools()`**，`tool_choice="auto"`。  
4. 若响应含 **`tool_calls`**：把 assistant 消息（含 `tool_calls`）追加到 `messages`，对每个调用 **`dispatch_tool_call`**（`tools.py`），将 `role="tool"` 的结果追加，再请求模型，最多 **`_MAX_AGENT_TOOL_ROUNDS = 8`** 轮。  
5. 若无 `tool_calls`：取 `content` 为最终 `reply`，同时累积 **`trace`**（`model_tool_calls` / `tool_result` / `model_final` / `agent_abort` 等）。  
6. 返回 **`AgentChatResponse(reply, steps)`**；`endpoints` 将 `reply` 作为 assistant 落库。  
7. **注意**：中间 tool 轮次**不写入** SQLite，避免把「过程」当成用户可见对话。

### 5.3 路径 C：上传文档（RAG 入库）

1. 前端 `FormData` 字段名 **`file`** → `POST /api/upload`。  
2. 保存到 `uploads/`，调用 **`process_and_store_document`**：加载 PDF/TXT → **RecursiveCharacterTextSplitter** 切块 → DashScope 批量 Embedding → **FAISS** 写入或合并 `app/rag/vectorstore/`。

### 5.4 路径 D：非流式简化 `/api/chat`

写 user → 取最近 10 条 → 构造 history（去掉最后一条当前 user）→ **`generate_ai_response`**（无 system、无 RAG）→ 写 assistant → 返回 `ChatResponse`。

---

## 第六部分：内置工具与安全设计（学习 Agent 必读）

声明见 **`app/agent/tool_definitions.py`**，实现见 **`app/agent/tools.py`**：

| 工具名 | 作用 | 实现要点 |
|--------|------|----------|
| `calculate` | 数学表达式求值 | **禁止 `eval`**：用 `ast` 解析 + 白名单运算符，正则限制字符集。 |
| `get_server_time` | 返回服务器本地时间 ISO 字符串 | `timezone_hint` 为教学占位，**未做**完整时区换算。 |
| `reverse_text` | 字符串反转 | 演示非数学类工具。 |

**安全原则**：模型生成的参数会在你的服务器上执行——必须校验、限长、避免任意代码执行与路径遍历；扩展工具时沿用此思路。

---

## 第七部分：HTTP 接口一览

| 方法 | 路径 | 作用 |
|------|------|------|
| `POST` | `/api/chat/stream` | 流式对话；`system_prompt`；RAG；Token 截断历史。 |
| `POST` | `/api/chat/agent` | 非流式 Agent；`system_prompt`；RAG；同策略历史；返回 `reply` + `steps`。 |
| `POST` | `/api/chat` | 非流式；最近 10 条；无 RAG；请求体中的 `system_prompt` 不使用。 |
| `POST` | `/api/upload` | 上传 PDF/TXT，解析并向量化入库。 |
| `GET` | `/api/history` | 全部聊天消息。 |
| `DELETE` | `/api/history` | 清空聊天表（不删向量库与 `uploads/`）。 |

---

## 第八部分：配置与环境变量

复制 `.env.example` 为 `.env`，至少关注：

- **`OPENAI_API_KEY`**：聊天客户端鉴权；RAG 的 DashScope Embedding 也用它作 Bearer（变量名历史原因，含义是「Bearer Token」）。若既要聊天又要 RAG，需保证该 Key 对**两边**都可用，或按 README 说明改用同一兼容网关。  
- **`OPENAI_BASE_URL`**、**`MODEL_NAME`**：聊天网关与模型名。  
- **`EMBEDDING_MODEL`**：如 `text-embedding-v3`，与 DashScope 文档一致。  
- **`DATABASE_URL`**：默认 `sqlite:///./ai_platform.db`（见 `config.py`）。

**Agent 额外要求**：你的网关与模型需支持 **`tools` / `tool_calls`**；若不支持，调用会报错，此时可仅用流式接口或更换模型。

---

## 第九部分：推荐源码阅读顺序

1. `app/main.py`：应用创建、CORS、`/api`、静态目录。  
2. `app/core/config.py`：`Settings` 与环境变量。  
3. `app/db/database.py`、`app/db/models.py`：`get_db`、`ChatMessage`。  
4. `app/api/endpoints.py`：重点 `chat_with_ai_stream`、`chat_with_ai_agent`、`upload_document`、`chat_with_ai`。  
5. `app/services/llm.py`：流式与非流式 Chat Completions。  
6. `app/services/agent_service.py`：工具循环与 trace。  
7. `app/agent/tool_definitions.py`、`app/agent/tools.py`。  
8. `app/rag/document_processor.py`：入库与检索。  
9. `static/index.html`：Agent 勾选、流式 reader、`/upload`。

---

## 第十部分：调试与学习技巧

- **浏览器开发者工具 → Network**：对比 `/api/chat/stream` 与 `/api/chat/agent` 的请求与响应体。  
- **SQLite**：用 VS Code 插件等打开项目根目录 **`ai_platform.db`**，查看表 **`chat_messages`**。  
- **向量文件**：在 **`app/rag/vectorstore/`**，不是数据库里的表。  
- **对照学习**：同一问题分别用「仅流式」与「Agent」提问，观察 `steps` 里模型是否选择调用工具。

---

## 第十一部分：常见问题（与 README 互补）

1. **RAG 似乎没用上**  
   确认已上传成功、向量目录存在；对话走 **`/api/chat/stream`** 或 **`/api/chat/agent`**（两者都会 RAG）；`/api/chat` 不会 RAG。

2. **Agent 报错 500**  
   多为网关不支持 `tools` 或模型名错误；查看终端堆栈与 Network 响应。

3. **`uvicorn` 找不到**  
   使用 `python3 -m uvicorn app.main:app --reload`。

---

## 第十二部分：你还可以自己扩展的方向（本仓库未实现）

- 多会话 `session_id`、流式 Agent、引用来源（页码/文件名）展示、删除/重建向量库、更多业务工具（HTTP 查询、数据库只读查询等）、对 `/api/chat` 与流式路径的能力对齐。

---

祝你学习顺利。若某段代码看不懂，把**文件路径 + 行号**发给助手，并说明「走的是流式还是 Agent」，排查会更快。
