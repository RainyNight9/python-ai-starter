# -*- coding: utf-8 -*-
"""
Agent 编排服务：实现「模型 ⟷ 工具」多轮协作的主循环（ReAct 的工程化子集）。

---------------------------------------------------------------------------
核心概念（学习 AI Agent 时务必建立的心智模型）
---------------------------------------------------------------------------
1. **Message 列表**：与大模型交互的基本单位是「消息数组」messages。
   常见 role：system / user / assistant / tool。
2. **第一轮**：把 system + 历史 + 当前 user 发给模型，并附上 tools 定义。
3. **模型两种输出**：
   - 纯文本 content：通常表示模型认为可以直接回答，无需工具。
   - tool_calls：模型请求执行一个或多个工具；每项含 id、function.name、
     function.arguments（JSON 字符串）。
4. **工具回合**：后端按 id 执行工具，把结果以 role="tool"、且带
   tool_call_id 的消息追加回 messages，再次请求模型。
5. **终止条件**：模型不再返回 tool_calls；或达到最大轮数防止死循环；
   或网关报错。

本模块刻意不操作数据库：由 API 层负责「落库 / RAG / 历史截断」，
保持单一职责，便于单元测试与阅读。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from app.agent.tool_definitions import get_openai_style_tools
from app.agent.tools import dispatch_tool_call
from app.core.config import settings

# 与 OpenAI 客户端一致：复用 llm.py 的 base_url / api_key 配置
_client = AsyncOpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
)

# 防止模型反复请求工具导致无限循环；数值可按业务调大/调小
_MAX_AGENT_TOOL_ROUNDS = 8


def _assistant_message_to_dict(msg: Any) -> Dict[str, Any]:
    """
    将 SDK 返回的 assistant 消息对象转换为「可再次发给 API」的 dict。

    说明：不同版本 SDK 对象字段可能略有差异，因此用 getattr 做温和降级。
    """
    raw_content = getattr(msg, "content", None)
    d: Dict[str, Any] = {
        "role": "assistant",
        # 部分网关在同时返回 tool_calls 时 content 可能为 null；用空串提升兼容性
        "content": raw_content if raw_content is not None else "",
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        serialized = []
        for tc in tool_calls:
            # tc 通常为 ChatCompletionMessageFunctionToolCall
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
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    执行「带工具调用能力」的完整对话回合，返回 (最终回复文本, 轨迹列表)。

    参数：
    - user_message：本轮用户输入（自然语言）。
    - history：可选，OpenAI 格式的历史消息列表，仅含 user/assistant 文本轮次。
      API 层应保证不包含「当前这条 user」（本函数会在末尾追加）。
    - system_prompt：可选系统提示；若 API 层已拼接 RAG，可一并传入。

    返回值 trace 中每一项建议结构：
    {"kind": "...", "detail": "..."} 便于前端或日志展示。

    异常策略：
    - 不在此函数吞掉所有异常：若网关/鉴权失败，抛出给 FastAPI 变成 500，
      便于开发阶段排查；生产环境可在 endpoints 层统一转换为友好文案。
    """
    tools = get_openai_style_tools()
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 历史只放「已发生」的对话；当前用户句单独追加，避免与流式接口语义混淆
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    final_text: str = ""

    for round_idx in range(_MAX_AGENT_TOOL_ROUNDS):
        # temperature 略低：工具场景更希望稳定遵循格式，而非天马行空
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

        # ---------- 分支 A：模型请求执行工具 ----------
        if tool_calls:
            trace.append(
                {
                    "kind": "model_tool_calls",
                    "round": round_idx,
                    "detail": f"模型请求 {len(tool_calls)} 个工具",
                }
            )

            # 必须把 assistant 消息（含 tool_calls）原样追加，API 才接受后续 tool 消息
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

            # 执行完本回合所有工具后，继续 while/for 下一轮，让模型读取 tool 结果
            continue

        # ---------- 分支 B：模型返回最终自然语言 ----------
        final_text = (msg.content or "").strip()
        trace.append(
            {
                "kind": "model_final",
                "round": round_idx,
                "detail": "模型返回最终文本（无 tool_calls）",
            }
        )
        return final_text, trace

    # 若多轮后仍在请求工具，强制结束，避免无限循环扣费
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
