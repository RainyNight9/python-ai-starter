"""
app.services.llm —— 与大模型对话的最小封装层（不含 Agent / 工具循环）。

学习路径建议：
1. 先读本文件：理解「messages 如何组装」与「流式 chunk 如何解析」。
2. 再读 app/services/agent_service.py：在 messages 之上叠加 tools 与多轮 tool 结果。

本文件刻意保持「薄」：复杂业务（RAG、Token 截断、落库）放在 api/endpoints.py，
避免所有逻辑堆在一处难以维护。
"""

from openai import AsyncOpenAI
from app.core.config import settings

# ==========================================
# AI 核心服务 (封装与大模型的交互逻辑)
# ==========================================

# 初始化异步的 OpenAI 客户端 (AsyncOpenAI)
# 相比同步版本，异步版本可以在等待 AI 返回结果时，让 FastAPI 去处理其他用户的请求，提高并发能力。
# base_url 指向「兼容 OpenAI Chat Completions」的网关；模型名由 MODEL_NAME 控制。
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

async def generate_ai_response_stream(prompt: str, history: list = None, system_prompt: str = None):
    """
    【V2.0 新增】流式调用大模型获取回复 (打字机效果)
    :param prompt: 用户当前的输入
    :param history: 历史对话记录
    :param system_prompt: 系统角色设定
    """
    # messages 的顺序很重要：system 永远在最前；接着是「已发生的对话」；
    # 最后是本轮 user。模型会基于整段上下文续写 assistant。
    messages = []

    # 1. 注入灵魂：如果有系统提示词，强制放在对话的最开始
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 2. 拼接历史记录和当前问题
    # history 元素形如 {"role":"user"|"assistant","content":"..."}，由上层 endpoints 负责截断。
    messages.extend(history or [])
    messages.append({"role": "user", "content": prompt})

    try:
        # 发起流式请求 (stream=True)
        # 流式响应是一个异步迭代器：每收到一块网络数据就可能产出一个 chunk，
        # 因此首字节更快，适合聊天 UI；代价是上层需要自行拼接完整答案用于落库。
        response = await client.chat.completions.create(
            model=settings.MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True  # 核心参数：开启流式传输
        )
        
        # 异步遍历模型返回的数据块 (chunks)
        async for chunk in response:
            # 提取每个 chunk 里的文字增量并 yield (产出) 出去
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n\n[⚠️ AI 调用出错: {str(e)}]"

# 保留原有的非流式方法，兼容老接口
async def generate_ai_response(prompt: str, history: list = None) -> str:
    """
    调用大模型获取回复
    :param prompt: 用户当前的输入 (例如："你好")
    :param history: 历史对话记录（可选），格式为字典列表：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    :return: AI 返回的字符串文本
    """
    # 1. 拼接上下文
    # AI 本身是没有记忆的，我们必须把之前的对话记录 (history) 和用户当次的新问题 (prompt) 拼接在一起，一起发给 AI
    messages = history or []
    messages.append({"role": "user", "content": prompt})

    try:
        # 2. 发起 API 请求
        response = await client.chat.completions.create(
            model=settings.MODEL_NAME, # 根据配置动态使用模型，例如 qwen
            messages=messages,         # 将拼接好的对话列表传过去
            temperature=0.7,           # 创造力参数，0.0 比较死板严谨，1.0 比较发散和具创造性
            max_tokens=1000            # 限制 AI 单次回复的最大 token 数量，防止花费过多或者回复过长
        )
        # 3. 解析结果并返回
        # 模型的回复包裹在 response 对象的 choices 数组里
        return response.choices[0].message.content
    except Exception as e:
        # 错误处理：如果 API Key 错误、网络不通等，这里会捕获异常并返回错误信息给前端，防止后端崩溃
        return f"AI 调用出错: {str(e)}"
