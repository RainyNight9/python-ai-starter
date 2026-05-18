# -*- coding: utf-8 -*-
"""
Agent 编排服务：实现「模型 ⟷ 工具」多轮协作的主循环（ReAct 的工程化子集）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.agent.tool_definitions import get_openai_style_tools
from app.agent.tools import clear_tool_context, dispatch_tool_call, set_tool_context
from app.core.config import settings

_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)

_MAX_AGENT_TOOL_ROUNDS = 8


def _assistant_message_to_dict(msg: Any) -> Dict[str, Any]:
    """将 SDK 返回的 assistant 消息对象转换为可再次发给 API 的 dict。"""
    raw_content = getattr(msg, "content", None)
    d: Dict[str, Any] = {
        "role": "assistant",
        "content": raw_content if raw_content is not None else "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            serialized.append(
                {
                    "id": getattr(tc, "id", ""),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", "") if fn else "",
                        "arguments": getattr(fn, "arguments", "") if fn else "",
                    },
                }
            )
        d["tool_calls"] = serialized
    return d


async def run_tool_agent(
    *,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    session_id: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """执行带工具调用能力的完整对话回合，返回最终回复和工具轨迹。"""
    tools = get_openai_style_tools()
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    final_text = ""
    set_tool_context(session_id=session_id)

    try:
        for round_idx in range(_MAX_AGENT_TOOL_ROUNDS):
            response = await _client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2000,
            )

            choice = response.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)

            if tool_calls:
                trace.append(
                    {
                        "kind": "model_tool_calls",
                        "round": round_idx,
                        "detail": f"模型请求 {len(tool_calls)} 个工具",
                    }
                )
                messages.append(_assistant_message_to_dict(msg))

                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    name = getattr(fn, "name", "") if fn else ""
                    args = getattr(fn, "arguments", "") if fn else ""
                    tc_id = getattr(tc, "id", "")

                    result = await dispatch_tool_call(name, args or "{}")
                    trace.append(
                        {
                            "kind": "tool_result",
                            "round": round_idx,
                            "tool_name": name,
                            "tool_call_id": tc_id,
                            "arguments": args,
                            "result_preview": result[:500]
                            + ("…(截断)" if len(result) > 500 else ""),
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": result,
                        }
                    )
                continue

            final_text = (msg.content or "").strip()
            trace.append(
                {
                    "kind": "model_final",
                    "round": round_idx,
                    "detail": "模型返回最终文本（无 tool_calls）",
                }
            )
            return final_text, trace

        trace.append(
            {
                "kind": "agent_abort",
                "detail": f"已达最大工具回合数 {_MAX_AGENT_TOOL_ROUNDS}，停止继续调用模型",
            }
        )
        return (
            final_text
            or "抱歉：在限定轮次内未能完成回答；请简化问题或检查模型是否支持工具调用。",
            trace,
        )
    finally:
        clear_tool_context()
