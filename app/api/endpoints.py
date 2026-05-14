from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Any, Dict, List
import json
import tiktoken
import os
import shutil

from app.db.database import get_db
from app.db import models
from app.services.llm import generate_ai_response, generate_ai_response_stream
from app.services.agent_service import run_tool_agent
from app.rag.document_processor import (
    process_and_store_document,
    retrieve_relevant_context_with_citations,
)

# ==========================================
# API 路由与接口 (处理 HTTP 请求)
# ==========================================

encoding = tiktoken.get_encoding("cl100k_base")


def get_messages_token_count(messages: list) -> int:
    """计算一段对话历史的总 Token 数量"""
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2
    return num_tokens


router = APIRouter()


def _require_session(db: Session, session_id: int) -> models.ChatSession:
    s = (
        db.query(models.ChatSession)
        .filter(models.ChatSession.id == session_id)
        .first()
    )
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    return s


# ----------------- 数据验证模型 (Pydantic) -----------------
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = ""
    session_id: int = 1


class ChatResponse(BaseModel):
    reply: str


class MessageDTO(BaseModel):
    role: str
    content: str
    created_at: str


class AgentChatResponse(BaseModel):
    reply: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    rag_citations: List[Dict[str, Any]] = Field(default_factory=list)


class SessionCreate(BaseModel):
    title: str = "新会话"


class SessionDTO(BaseModel):
    id: int
    title: str
    created_at: str


# ----------------- 会话管理 -----------------
@router.get("/sessions", response_model=List[SessionDTO])
def list_sessions(db: Session = Depends(get_db)):
    rows = (
        db.query(models.ChatSession)
        .order_by(models.ChatSession.id.asc())
        .all()
    )
    return [
        SessionDTO(
            id=r.id,
            title=r.title,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for r in rows
    ]


@router.post("/sessions", response_model=SessionDTO)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    title = (body.title or "").strip() or "新会话"
    s = models.ChatSession(title=title)
    db.add(s)
    db.commit()
    db.refresh(s)
    return SessionDTO(
        id=s.id,
        title=s.title,
        created_at=s.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    )


@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)):
    if db.query(models.ChatSession).count() <= 1:
        raise HTTPException(status_code=400, detail="至少保留一个会话")
    _require_session(db, session_id)
    db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).delete(synchronize_session=False)
    db.query(models.ChatSession).filter(models.ChatSession.id == session_id).delete(
        synchronize_session=False
    )
    db.commit()
    return {"status": "success", "message": "会话已删除"}


# ----------------- 聊天 -----------------
@router.post("/chat/stream")
async def chat_with_ai_stream(request: ChatRequest, db: Session = Depends(get_db)):
    _require_session(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user",
        content=request.message,
        session_id=request.session_id,
    )
    db.add(user_msg)
    db.commit()

    relevant_context, rag_citations = retrieve_relevant_context_with_citations(
        request.message
    )

    final_system_prompt = request.system_prompt
    if relevant_context:
        rag_prompt = (
            f"\n\n请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。\n\n"
            f"[参考资料开始]\n{relevant_context}\n[参考资料结束]"
        )
        final_system_prompt = (
            final_system_prompt + rag_prompt if final_system_prompt else rag_prompt
        )

    MAX_HISTORY_TOKENS = 2000
    all_history_records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == request.session_id)
        .order_by(models.ChatMessage.id.desc())
        .all()
    )

    history = []
    current_tokens = 0
    for msg in all_history_records[1:]:
        msg_dict = {"role": msg.role, "content": msg.content}
        msg_tokens = get_messages_token_count([msg_dict])
        if current_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
        history.append(msg_dict)
        current_tokens += msg_tokens
    history.reverse()

    async def event_generator():
        meta = json.dumps({"rag_citations": rag_citations}, ensure_ascii=False)
        yield "::META::" + meta + "\n"
        full_reply = ""
        async for chunk in generate_ai_response_stream(
            request.message, history, final_system_prompt
        ):
            full_reply += chunk
            yield chunk
        if full_reply:
            assistant_msg = models.ChatMessage(
                role="assistant",
                content=full_reply,
                session_id=request.session_id,
            )
            db.add(assistant_msg)
            db.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/agent", response_model=AgentChatResponse)
async def chat_with_ai_agent(request: ChatRequest, db: Session = Depends(get_db)):
    _require_session(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user",
        content=request.message,
        session_id=request.session_id,
    )
    db.add(user_msg)
    db.commit()

    relevant_context, rag_citations = retrieve_relevant_context_with_citations(
        request.message
    )
    final_system_prompt = request.system_prompt
    if relevant_context:
        rag_prompt = (
            "\n\n请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。\n\n"
            f"[参考资料开始]\n{relevant_context}\n[参考资料结束]"
        )
        final_system_prompt = (
            final_system_prompt + rag_prompt if final_system_prompt else rag_prompt
        )

    MAX_HISTORY_TOKENS = 2000
    all_history_records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == request.session_id)
        .order_by(models.ChatMessage.id.desc())
        .all()
    )
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

    reply, steps = await run_tool_agent(
        user_message=request.message,
        history=history,
        system_prompt=final_system_prompt or None,
    )

    assistant_msg = models.ChatMessage(
        role="assistant",
        content=reply,
        session_id=request.session_id,
    )
    db.add(assistant_msg)
    db.commit()

    return AgentChatResponse(
        reply=reply, steps=steps, rag_citations=rag_citations
    )


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 或 TXT 文件")

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_count = process_and_store_document(file_path)
        return {
            "status": "success",
            "message": f"文件 {file.filename} 处理成功，共切分出 {chunks_count} 个知识块。",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    _require_session(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user",
        content=request.message,
        session_id=request.session_id,
    )
    db.add(user_msg)
    db.commit()

    history_records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == request.session_id)
        .order_by(models.ChatMessage.id.desc())
        .limit(10)
        .all()
    )
    history_records.reverse()

    history = [
        {"role": msg.role, "content": msg.content} for msg in history_records[:-1]
    ]

    ai_reply = await generate_ai_response(request.message, history)

    assistant_msg = models.ChatMessage(
        role="assistant",
        content=ai_reply,
        session_id=request.session_id,
    )
    db.add(assistant_msg)
    db.commit()

    return ChatResponse(reply=ai_reply)


@router.get("/history", response_model=List[MessageDTO])
async def get_history(
    session_id: int = Query(1, ge=1), db: Session = Depends(get_db)
):
    _require_session(db, session_id)
    records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.id.asc())
        .all()
    )
    return [
        MessageDTO(
            role=r.role,
            content=r.content,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for r in records
    ]


@router.delete("/history")
async def clear_history(
    session_id: int = Query(1, ge=1), db: Session = Depends(get_db)
):
    _require_session(db, session_id)
    db.query(models.ChatMessage).filter(
        models.ChatMessage.session_id == session_id
    ).delete(synchronize_session=False)
    db.commit()
    return {"status": "success", "message": "History cleared"}
