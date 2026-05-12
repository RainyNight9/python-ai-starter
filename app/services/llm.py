from openai import AsyncOpenAI
from app.core.config import settings

# ==========================================
# AI 核心服务 (封装与大模型的交互逻辑)
# ==========================================

# 初始化异步的 OpenAI 客户端 (AsyncOpenAI)
# 相比同步版本，异步版本可以在等待 AI 返回结果时，让 FastAPI 去处理其他用户的请求，提高并发能力。
client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL
)

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
