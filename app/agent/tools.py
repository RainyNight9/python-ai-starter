# -*- coding: utf-8 -*-
"""
工具实现（Tool Implementations）—— 模型「想调用」的函数，在这里真正执行。

---------------------------------------------------------------------------
Python 知识点：为什么要把「声明」和「实现」分开？
---------------------------------------------------------------------------
- 声明（JSON Schema）是给模型看的「说明书」。
- 实现（Python 函数）是给程序看的「行为」。
分离后：你可以只改说明书、或只改实现，互不影响；单元测试也更容易针对
tools.py 编写（不依赖大模型）。

---------------------------------------------------------------------------
安全提示（非常重要）
---------------------------------------------------------------------------
任何会被大模型触发的工具，本质上都是「由模型生成的参数 → 在你的服务器上执行」。
因此必须：
- 校验输入（白名单字符、长度上限、禁止路径遍历等）
- 禁止直接把用户/模型字符串交给 eval / os.system / 任意 SQL 拼接

下面的 calculate 使用 ast 抽象语法树 + 白名单运算符，避免 eval 的安全隐患。
"""

from __future__ import annotations

import ast
import json
import operator
import re
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from app.rag.document_processor import retrieve_relevant_context_with_citations

# ---------------------------------------------------------------------------
# 安全计算器：仅允许数字与 + - * / 括号，以及一元正负号
# ---------------------------------------------------------------------------

# ast 运算符节点类型 → Python 内置 operator 函数
# 我们只开放「算术」相关节点，杜绝 ast.Call（函数调用）等危险结构
_ALLOWED_BIN_OPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS: Dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast_expr(node: ast.AST) -> Any:
    """
    递归计算 ast 表达式节点；遇到不允许的节点类型直接抛 ValueError。

    这是典型的「访问者模式」简化版：根据 node 的类型分派到不同逻辑。
    """
    # 数字字面量（Python 3.8+ 用 ast.Constant；旧版可能是 ast.Num）
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("常量类型不允许（仅支持 int/float）")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"不支持的二元运算符: {op_type.__name__}")
        left = _eval_ast_expr(node.left)
        right = _eval_ast_expr(node.right)
        return _ALLOWED_BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
        operand = _eval_ast_expr(node.operand)
        return _ALLOWED_UNARY_OPS[op_type](operand)

    raise ValueError(f"不支持的语法结构: {type(node).__name__}")


def safe_calculate(expression: str) -> str:
    """
    对数学表达式求值，返回字符串形式的结果（便于统一作为 tool 消息 content）。

    步骤：
    1. 白名单正则：拒绝字母、方括号、函数名等非算术字符
    2. ast.parse(..., mode='eval')：解析为表达式语法树
    3. 递归求值，任何异常都转成可读错误信息字符串（模型可据此重试或解释）
    """
    expr = expression.strip()
    if not expr:
        return "错误：表达式为空"

    # 仅允许数字、空白、四则运算与括号、小数点
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "错误：表达式包含不允许的字符（仅支持 0-9 + - * / ( ) 与空格）"

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast_expr(tree.body)
        return str(value)
    except ZeroDivisionError:
        return "错误：除数为零"
    except Exception as e:
        return f"错误：无法求值（{e.__class__.__name__}: {e}）"


def get_server_time(timezone_hint: Optional[str] = None) -> str:
    """
    返回服务器本地当前时间（ISO 8601 字符串）。

    参数 timezone_hint：
    - 教学用占位：完整时区转换需要第三方库（如 zoneinfo / pytz），
      为减少依赖，这里只在返回字符串里附带「用户提到的提示」，
      真实时区仍以服务器为准——注释写明，避免误以为已做精确时区换算。
    """
    now = datetime.now()
    base = now.isoformat(timespec="seconds")
    if timezone_hint:
        return f"{base} （服务器本地时间；用户提示: {timezone_hint}）"
    return f"{base} （服务器本地时间）"


def reverse_text(text: str) -> str:
    """字符串反转；切片 [::-1] 是 Python 惯用写法。"""
    return text[::-1]


def search_uploaded_documents(query: str, top_k: int = 3) -> str:
    """从已上传文档的向量库中检索相关片段。"""
    if not query.strip():
        return "错误：检索问题为空"

    try:
        context, citations = retrieve_relevant_context_with_citations(
            query=query,
            top_k=max(1, min(int(top_k), 8)),
        )
    except Exception as e:
        return f"错误：文档检索失败（{e.__class__.__name__}: {e}）"

    if not context:
        return "没有检索到相关文档片段。"

    citation_lines = []
    for idx, citation in enumerate(citations, 1):
        page = citation.get("page")
        page_text = f"第 {page + 1} 页" if isinstance(page, int) else "页码未知"
        citation_lines.append(
            f"[{idx}] {citation.get('source', '未知来源')}，{page_text}，"
            f"距离 {citation.get('distance')}"
        )

    return "相关片段：\n" + context + "\n\n引用：\n" + "\n".join(citation_lines)


def export_session_markdown(session_id: Optional[int] = None) -> str:
    """导出当前会话为 Markdown 文本，由 Agent 运行上下文提供 session_id。"""
    active_session_id = session_id or _TOOL_CONTEXT.get("session_id")
    if not active_session_id:
        return "错误：缺少 session_id，无法导出会话。"

    try:
        from app.db.database import SessionLocal
        from app.db import models

        db = SessionLocal()
        records = (
            db.query(models.ChatMessage)
            .filter(models.ChatMessage.session_id == int(active_session_id))
            .order_by(models.ChatMessage.id.asc())
            .all()
        )
        if not records:
            return f"# 会话 {active_session_id}\n\n暂无聊天记录。"

        lines = [f"# 会话 {active_session_id}", ""]
        for record in records:
            role = "用户" if record.role == "user" else "助手"
            lines.append(f"## {role} · {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            lines.append(record.content)
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"错误：导出会话失败（{e.__class__.__name__}: {e}）"
    finally:
        try:
            db.close()
        except Exception:
            pass


_TOOL_CONTEXT: Dict[str, Any] = {}


def set_tool_context(**kwargs: Any) -> None:
    """设置本轮 Agent 工具运行上下文。"""
    _TOOL_CONTEXT.clear()
    _TOOL_CONTEXT.update(kwargs)


def clear_tool_context() -> None:
    """清空本轮 Agent 工具运行上下文。"""
    _TOOL_CONTEXT.clear()


# ---------------------------------------------------------------------------
# 工具分发器：根据模型给出的 function.name 调用对应 Python 函数
# ---------------------------------------------------------------------------

# Callable[..., str] 表示「任意参数签名，但返回值应是 str」的简化标注
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "calculate": lambda **kwargs: safe_calculate(kwargs.get("expression", "")),
    "get_server_time": lambda **kwargs: get_server_time(
        kwargs.get("timezone_hint") or None
    ),
    "reverse_text": lambda **kwargs: reverse_text(kwargs.get("text", "")),
    "search_uploaded_documents": lambda **kwargs: search_uploaded_documents(
        kwargs.get("query", ""), kwargs.get("top_k", 3)
    ),
    "export_session_markdown": lambda **kwargs: export_session_markdown(
        kwargs.get("session_id")
    ),
}


async def dispatch_tool_call(name: str, arguments_json: str) -> str:
    """
    根据工具名与参数 JSON 字符串执行工具，返回字符串结果。

    设计为 async 函数：即使当前工具都是同步的，未来若接入「查 HTTP API」
    等 I/O 密集型工具，可直接在函数内使用 await，而无需改动 endpoints 层
    的调用方式（统一 await dispatch_tool_call）。

    参数 arguments_json：
    - 来自模型 tool_calls[i].function.arguments，类型为 str
    - 内容应是 JSON 对象，例如 '{"expression":"1+2"}'
    """
    if name not in TOOL_REGISTRY:
        return f"错误：未知工具名称 {name!r}"

    try:
        parsed: Any = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"错误：工具参数不是合法 JSON（{e}）"

    if not isinstance(parsed, dict):
        return "错误：工具参数必须是 JSON 对象（顶层应为 dict）"

    # 使用 **parsed 将字典展开为关键字参数；各工具 lambda 用 kwargs.get 取值
    try:
        fn = TOOL_REGISTRY[name]
        # 同步调用包装在 async 函数里：若未来改异步 IO，可在此 await
        return fn(**parsed)
    except TypeError as e:
        return f"错误：参数不匹配（{e}）"
    except Exception as e:
        return f"错误：工具执行异常（{e.__class__.__name__}: {e}）"
