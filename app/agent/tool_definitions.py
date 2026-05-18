# -*- coding: utf-8 -*-
"""
工具声明（Tool Declarations）—— 告诉大模型：「你可以调用哪些函数」。

---------------------------------------------------------------------------
知识点：OpenAI 兼容 API 中的 tools 参数
---------------------------------------------------------------------------
Chat Completions 接口支持传入 `tools` 列表，每一项描述：
- type：通常为 "function"
- function.name：函数名（后端路由到具体 Python 函数时常用这个名字做键）
- function.description：自然语言说明，帮助模型判断**何时**该调用
- function.parameters：JSON Schema，描述**参数名、类型、是否必填**

模型返回的 tool_calls 里会带上 name + arguments（JSON 字符串），
后端解析后执行对应 Python 函数，再把字符串结果以 role="tool" 的消息发回。

本文件只放「声明」，不放具体算法实现，避免单文件过长、职责混乱。
"""

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Python 类型提示（typing）小课堂：
# List[Dict[str, Any]] 表示「由若干字典组成的列表」，字典的键是 str，
# 值可以是任意类型 Any。这样写能让 IDE / 静态检查器更好地提示错误。
# ---------------------------------------------------------------------------


def get_openai_style_tools() -> List[Dict[str, Any]]:
    """
    返回传给 `AsyncOpenAI.chat.completions.create(..., tools=...)` 的工具列表。

    说明：
    - 结构需符合 OpenAI SDK / 兼容网关的约定；多数「OpenAI 兼容」厂商同理。
    - 若你的网关较旧不支持 tools，调用会报错，此时可改用不支持 Agent 的
      普通聊天接口，或升级网关/模型。
    """
    return [
        {
            "type": "function",
            "function": {
                # 模型在 tool_calls 里会用这个名字；后端必须与之一致
                "name": "calculate",
                "description": (
                    "对只包含数字与 + - * / 括号 的数学表达式求值。"
                    "当用户明确需要数值计算、或口算容易出错时使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": '例如 "((12 + 3) * 4) / 2"',
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_server_time",
                "description": (
                    "返回本服务器当前日期与时间（ISO 格式）。"
                    "当用户问「现在几点」「今天是几号」等与真实时间相关的问题时使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        # 无参工具也建议给空对象 properties，部分网关对 schema 较严格
                        "timezone_hint": {
                            "type": "string",
                            "description": (
                                "可选。用户提到的时区或城市，如 Asia/Shanghai；"
                                "若为空则使用服务器本地时区。"
                            ),
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reverse_text",
                "description": (
                    "将一段文本按字符顺序完全反转。用于演示「非数学类」工具，"
                    "或用户明确要求反转字符串时。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "需要被反转的原始文本",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_uploaded_documents",
                "description": (
                    "从用户已经上传并入库的 PDF/TXT 文档中检索相关片段。"
                    "当用户的问题需要基于知识库、资料、文件内容回答时使用。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "要检索的自然语言问题或关键词",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回片段数量，建议 3，最大 8",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "export_session_markdown",
                "description": "把当前会话聊天记录导出为 Markdown 文本。用户要求导出、整理或保存会话时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "integer",
                            "description": "可选。要导出的会话 ID；不传则使用当前会话。",
                        }
                    },
                    "required": [],
                },
            },
        },
    ]
