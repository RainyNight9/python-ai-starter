from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import tiktoken
import os
import shutil

from app.db.database import get_db
from app.db import models
from app.services.llm import generate_ai_response, generate_ai_response_stream
from app.services.agent_service import run_tool_agent
from app.rag.document_processor import process_and_store_document, retrieve_relevant_context

# ==========================================
# API 路由与接口 (处理 HTTP 请求)
# ==========================================

# 初始化 tiktoken 编码器（使用 gpt-3.5/gpt-4 通用的 cl100k_base 词表）
encoding = tiktoken.get_encoding("cl100k_base")

def get_messages_token_count(messages: list) -> int:
    """计算一段对话历史的总 Token 数量"""
    num_tokens = 0
    for message in messages:
        # 每条消息都会有一些格式上的开销（如 <|im_start|> 等）
        num_tokens += 4 
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2  # 加上最后的 <|im_start|>assistant 开销
    return num_tokens

# 创建路由对象，相当于应用的一个子模块
router = APIRouter()

# ----------------- 数据验证模型 (Pydantic) -----------------
# 自动验证前端传过来的 JSON 数据格式，不符合会直接报错 422
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = ""  # V2.0 新增：支持系统角色设定

class ChatResponse(BaseModel):
    reply: str
    
class MessageDTO(BaseModel):
    role: str
    content: str
    created_at: str


class AgentChatResponse(BaseModel):
    """
    Agent 模式专用响应：除了最终自然语言 reply，还返回 steps 轨迹，
    便于你在前端或学习笔记里对照「模型何时决定调用工具、参数是什么、
    工具返回了什么（预览）」。
    """

    reply: str
    # 使用 Field(default_factory=list) 而不是默认值 []，避免可变默认参数坑
    steps: List[Dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------

@router.post("/chat/stream")
async def chat_with_ai_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """
    【V3.0 新增 RAG】流式对话接口 (Server-Sent Events)
    """
    # 1. 保存用户的消息到数据库
    user_msg = models.ChatMessage(role="user", content=request.message)
    db.add(user_msg)
    db.commit()
    
    # 2. 【RAG 核心逻辑】：去向量数据库里搜相关的文档内容
    relevant_context = retrieve_relevant_context(request.message)
    
    # 动态组装 System Prompt
    final_system_prompt = request.system_prompt
    if relevant_context:
        rag_prompt = f"\n\n请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。\n\n[参考资料开始]\n{relevant_context}\n[参考资料结束]"
        final_system_prompt = final_system_prompt + rag_prompt if final_system_prompt else rag_prompt
    
    # 3. 获取并智能截断历史记录 (Token 管理)
    MAX_HISTORY_TOKENS = 2000  # 我们允许历史记录占用的最大 Token 数
    
    # 先把所有的历史记录取出来
    all_history_records = db.query(models.ChatMessage).order_by(models.ChatMessage.id.desc()).all()
    
    # 动态组装历史记录，确保总 Token 数不超过限制
    history = []
    current_tokens = 0
    
    # 因为查出来是倒序的（最新的在前面），我们要跳过第一条（刚刚存入的用户新消息）
    for msg in all_history_records[1:]:
        msg_dict = {"role": msg.role, "content": msg.content}
        msg_tokens = get_messages_token_count([msg_dict])
        
        # 如果加上这条消息就超了，说明前面的记忆已经装不下了，停止追加
        if current_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
            
        history.append(msg_dict)
        current_tokens += msg_tokens
        
    # 把截取出来的历史记录反转回时间正序
    history.reverse()

    # 4. 构造流式生成器函数
    async def event_generator():
        full_reply = ""
        # 逐块接收 AI 吐出来的字
        async for chunk in generate_ai_response_stream(request.message, history, final_system_prompt):
            full_reply += chunk
            yield chunk  # 立刻把这个字发送给前端
            
        # 5. 当流式输出完全结束后，把它存入数据库
        if full_reply:
            assistant_msg = models.ChatMessage(role="assistant", content=full_reply)
            db.add(assistant_msg)
            db.commit()

    # 使用 StreamingResponse 返回流式数据
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/agent", response_model=AgentChatResponse)
async def chat_with_ai_agent(request: ChatRequest, db: Session = Depends(get_db)):
    """
    【Agent 教学接口】非流式：在对话链路中启用「工具调用 / Function Calling」。

    与普通 /chat/stream 的差异（学习时请对照思考）：
    - /chat/stream：模型只能生成文本；知识来自 RAG 注入的 system 侧文字。
    - /chat/agent：模型可多次请求后端工具（计算、时间、字符串处理等），
      形成「推理 → 行动 → 再推理」的闭环；仍可将 RAG 拼进 system，二者可并存。

    持久化策略：与普通聊天一致，最终仍写入 user / assistant 各一条，
    不把 tool 中间消息写入 SQLite（那是「过程」，不是用户可见对话）。
    """
    # 1) 记录用户消息
    user_msg = models.ChatMessage(role="user", content=request.message)
    db.add(user_msg)
    db.commit()

    # 2) RAG：与流式接口保持一致，把检索结果并入 system，便于对比两种路径
    relevant_context = retrieve_relevant_context(request.message)
    final_system_prompt = request.system_prompt
    if relevant_context:
        rag_prompt = (
            "\n\n请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。\n\n"
            f"[参考资料开始]\n{relevant_context}\n[参考资料结束]"
        )
        final_system_prompt = (
            final_system_prompt + rag_prompt if final_system_prompt else rag_prompt
        )

    # 3) 历史：复用与流式接口相同的「按 Token 预算从近到远」策略，保证行为一致
    MAX_HISTORY_TOKENS = 2000
    all_history_records = db.query(models.ChatMessage).order_by(models.ChatMessage.id.desc()).all()
    history: List[Dict[str, str]] = []
    current_tokens = 0
    for msg in all_history_records[1:]:
        msg_dict = {"role": msg.role, "content": msg.content}
        msg_tokens = get_messages_token_count([msg_dict])
        if current_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
        history.append(msg_dict)
        current_tokens += msg_tokens
    history.reverse()

    # 4) 调用 Agent 编排层（纯异步，不持有 db Session）
    reply, steps = await run_tool_agent(
        user_message=request.message,
        history=history,
        system_prompt=final_system_prompt or None,
    )

    # 5) 落库助手最终回复
    assistant_msg = models.ChatMessage(role="assistant", content=reply)
    db.add(assistant_msg)
    db.commit()

    return AgentChatResponse(reply=reply, steps=steps)


# ----------------- V3.0 新增：文档上传与处理接口 -----------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """接收前端上传的文档，并送入向量数据库"""
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 或 TXT 文件")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    # 1. 保存文件到本地
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. 调用 RAG 处理逻辑：解析 -> 切块 -> 向量化 -> 存库
        chunks_count = process_and_store_document(file_path)
        return {"status": "success", "message": f"文件 {file.filename} 处理成功，共切分出 {chunks_count} 个知识块。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
# --------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    """
    处理用户的聊天请求。
    参数中的 db: Session = Depends(get_db) 会自动调用 database.py 里的 get_db，注入数据库连接。
    """
    # 1. 保存用户的消息到数据库
    user_msg = models.ChatMessage(role="user", content=request.message)
    db.add(user_msg)
    db.commit() # 必须 commit，数据才会真正写入硬盘
    
    # 2. 获取历史记录作为 AI 的上下文（这里取最近的 10 条，避免上下文过长超出 Token 限制）
    history_records = db.query(models.ChatMessage).order_by(models.ChatMessage.id.desc()).limit(10).all()
    history_records.reverse() # 因为是 desc() 倒序取的，所以需要反转一下，恢复成时间正序
    
    # 构造传给 AI 的历史格式（剔除掉最后一条，即刚刚存入的用户新消息，因为它会在 llm.py 中被 append 进去）
    history = [{"role": msg.role, "content": msg.content} for msg in history_records[:-1]]

    # 3. 调用我们封装好的 AI 接口（这是一个 await 异步调用）
    ai_reply = await generate_ai_response(request.message, history)
    
    # 4. 把 AI 的回复也存到数据库里
    assistant_msg = models.ChatMessage(role="assistant", content=ai_reply)
    db.add(assistant_msg)
    db.commit()
    
    # 5. 返回给前端（会自动按照 response_model=ChatResponse 转换为 JSON）
    return ChatResponse(reply=ai_reply)

@router.get("/history", response_model=List[MessageDTO])
async def get_history(db: Session = Depends(get_db)):
    """
    获取全部历史聊天记录（前端页面加载时会调用）
    """
    records = db.query(models.ChatMessage).order_by(models.ChatMessage.id.asc()).all()
    # 转换为 DTO 对象返回给前端
    return [
        MessageDTO(
            role=r.role, 
            content=r.content, 
            # 把 datetime 对象格式化为更容易阅读的字符串
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S")
        ) for r in records
    ]

@router.delete("/history")
async def clear_history(db: Session = Depends(get_db)):
    """
    清空聊天记录
    """
    db.query(models.ChatMessage).delete()
    db.commit()
    return {"status": "success", "message": "History cleared"}
