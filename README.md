# Python AI Starter 🚀

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991.svg)](https://openai.com/)

欢迎来到 **Python AI 全栈入门项目**！本项目专为你从 0 搭建自己的 AI 调用平台而设计，涵盖了“后端接口 + AI 调用 + 数据库存储 + 极简前端”。

通过这个项目，你将经历：**从工程入手 → 跑通 → 理解 → 升级 → 变成 AI 全栈** 的完整学习路径。

## 📑 目录 (Table of Contents)
- [🟢 第一阶段：工程入手与跑通 (Run)](#-第一阶段工程入手与跑通-run)
- [🔵 第二阶段：理解核心代码 (Understand)](#-第二阶段理解核心代码-understand)
- [🟠 第三阶段：项目升级与魔改 (Upgrade)](#-第三阶段项目升级与魔改-upgrade)
- [🔴 第四阶段：变成 AI 全栈 (Become Full-Stack)](#-第四阶段变成-ai-全栈-become-full-stack)
- [🛠️ 常见问题与避坑指南 (Troubleshooting)](#-常见问题与避坑指南-troubleshooting)
- [🤝 参与贡献 (Contributing)](#-参与贡献-contributing)

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
2. 打开 `.env` 文件，填入你的 OpenAI API Key（如果没有，也可以使用支持 OpenAI 格式的第三方中转 API Key）。
   > *注意：如果你使用的是第三方中转 API，你可能还需要在 `app/services/llm.py` 中修改 `base_url` 参数。*

### 4. 启动服务
```bash
uvicorn app.main:app --reload
```
启动成功后，打开浏览器访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)
你就可以直接在极简前端页面中与你的 AI 助手聊天了！

---

## 🔵 第二阶段：理解核心代码 (Understand)

了解项目是怎么跑起来的，你需要看以下几个核心文件：

1. **`app/main.py`** (程序入口)
   - 它是整个 FastAPI 后端的启动文件。
   - 负责挂载 API 路由 (`/api`) 和前端静态文件 (`static/index.html`)。
2. **`app/services/llm.py`** (AI 大脑)
   - 这里封装了与 OpenAI 通信的代码。
   - 看看 `generate_ai_response` 函数，它是如何把用户的对话历史拼接起来发送给大模型的。
3. **`app/db/`** (记忆系统)
   - `models.py`: 定义了数据库长什么样（我们用 SQLite 存了 `role` 和 `content`）。
   - `database.py`: 数据库的连接配置。
4. **`app/api/endpoints.py`** (接口层)
   - 这里是前端和后端的桥梁。
   - 重点看 `@router.post("/chat")`：它接收前端的消息 -> 存入数据库 -> 读取历史 -> 调 AI -> 存入 AI 回复 -> 返回给前端。
5. **`static/index.html`** (门面)
   - 包含 HTML/CSS/JS 的单文件前端，展示了如何用 `fetch` 调用后端的 `/api/chat` 接口。

---

## 🟠 第三阶段：项目升级与魔改 (Upgrade)

现在的平台只是一个“雏形”，你可以尝试以下升级挑战来提升你的工程能力：

### 挑战 1：换用开源模型 / 国内大模型
- **目标**：不使用 OpenAI，改用国内大模型（如智谱、百川、DeepSeek）或本地部署的开源模型（Ollama）。
- **做法**：修改 `app/services/llm.py`。大多数国内模型兼容 OpenAI SDK，你只需要改 `base_url` 和 `api_key`，以及 `model` 名称即可。

### 挑战 2：实现流式输出 (Streaming)
- **目标**：像 ChatGPT 一样，让 AI 的回复一个字一个字地蹦出来，而不是转圈等半天。
- **做法**：
  1. 后端：在 `llm.py` 中开启 `stream=True`，并使用 FastAPI 的 `StreamingResponse` 返回异步生成器。
  2. 前端：使用 `fetch` 的 `ReadableStream` 或者 `EventSource` (SSE) 接收数据并逐步渲染。

### 挑战 3：给 AI 加上系统提示词 (System Prompt)
- **目标**：让 AI 扮演特定角色（比如“毒舌程序员”、“翻译官”、“心理医生”）。
- **做法**：在传给 AI 的 `messages` 列表的最前面，强行插入一条 `{"role": "system", "content": "你是一个资深 Python 工程师..."}`。

### 挑战 4：多会话管理 (Sessions)
- **目标**：现在所有人都在同一个聊天记录里，如何实现“新建对话”功能？
- **做法**：在数据库模型 `ChatMessage` 中增加一个字段 `session_id`。前端生成一个 UUID，每次发消息带上这个 ID，后端按 ID 查询历史记录。

---

## 🔴 第四阶段：变成 AI 全栈 (Become Full-Stack)

当你完成了上述挑战，恭喜你已经具备了 AI 后端工程师的基础！下一步的进阶方向：

1. **RAG (检索增强生成)**：结合 LangChain / LlamaIndex 和向量数据库（如 Milvus, Chroma），让 AI 能够读取你的本地 PDF/Word 文档并回答问题。
2. **Agent (智能体)**：给 AI 接入工具（如联网搜索、代码执行、调用外部 API 查询天气等），让 AI 帮你干活而不仅仅是聊天。
3. **前端工程化**：抛弃单文件 HTML，使用 React/Vue/Next.js 等现代前端框架，结合 TailwindCSS，写出真正媲美 ChatGPT 官网的漂亮界面。

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
- **原因**：当 `pip` 发现当前用户对全局 Python 环境没有写入权限时，会自动把包安装到用户独立的目录下（如 `~/.local/bin` 或 `~/Library/Python/3.14/bin`）。如果这个目录没有被配置到你的系统环境变量（PATH）里，终端就找不到 `uvicorn` 命令。
- **解决办法**：
  不需要修改环境变量，直接通过 Python 模块的方式调用 Uvicorn 即可：
  ```bash
  python3 -m uvicorn app.main:app --reload
  ```

---

## 🤝 参与贡献 (Contributing)

作为一个开源的入门级项目，我们非常欢迎任何形式的贡献，包括但不限于：
- 修复代码或文档中的错别字/Bug
- 提交上述“第三阶段”升级挑战的实现代码（可以提交到 `examples/` 目录下）
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