import json
import os
import shutil
from typing import Any, Dict, List, Optional

import tiktoken
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import models
from app.db.database import get_db
from app.core.config import settings
from app.rag.document_processor import (
    clear_vector_store,
    delete_document_and_rebuild,
    list_uploaded_documents,
    process_and_store_document,
    rebuild_vector_store_from_uploads,
    retrieve_relevant_context_with_citations,
)
from app.services.agent_service import run_tool_agent
from app.services.llm import generate_ai_response, generate_ai_response_stream

# ==========================================
# API 路由与接口 (处理 HTTP 请求)
# ==========================================

# 初始化 tiktoken 编码器（使用 gpt-3.5/gpt-4 通用的 cl100k_base 词表）
encoding = tiktoken.get_encoding("cl100k_base")
MAX_HISTORY_TOKENS = 2000


def get_messages_token_count(messages: list) -> int:
    """计算一段对话历史的总 Token 数量"""
    num_tokens = 0
    for message in messages:
        # 每条消息都会有一些格式上的开销（如 <|im_start|> 等）
        num_tokens += 4
        for value in message.values():
            num_tokens += len(encoding.encode(str(value)))
    num_tokens += 2
    return num_tokens


# 创建路由对象，相当于应用的一个子模块
router = APIRouter()


# ----------------- 数据验证模型 (Pydantic) -----------------
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = ""
    session_id: int = 1


class ChatResponse(BaseModel):
    reply: str


class CitationDTO(BaseModel):
    source: str
    page: Optional[int] = None
    distance: float
    snippet_preview: str


class AgentResponse(BaseModel):
    reply: str
    steps: List[Dict[str, Any]]
    rag_citations: List[CitationDTO]


class MessageDTO(BaseModel):
    role: str
    content: str
    created_at: str


class SessionDTO(BaseModel):
    id: int
    title: str
    created_at: str


class CreateSessionRequest(BaseModel):
    title: str = "新会话"


class UploadedDocumentDTO(BaseModel):
    filename: str
    size_bytes: int
    modified_at: int


class RagSettingsDTO(BaseModel):
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


# --------------------------------------------------------


def _format_datetime(dt) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_default_session(db: Session) -> models.ChatSession:
    session = db.query(models.ChatSession).filter(models.ChatSession.id == 1).first()
    if session:
        return session

    session = models.ChatSession(id=1, title="默认会话")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def _get_session_or_404(db: Session, session_id: int) -> models.ChatSession:
    if session_id == 1:
        return _ensure_default_session(db)

    session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    return session


def _build_rag_system_prompt(system_prompt: str, relevant_context: str) -> str:
    final_system_prompt = system_prompt or ""
    if not relevant_context:
        return final_system_prompt

    rag_prompt = (
        "\n\n请基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请明确说明。"
        f"\n\n[参考资料开始]\n{relevant_context}\n[参考资料结束]"
    )
    return final_system_prompt + rag_prompt if final_system_prompt else rag_prompt


def _load_token_limited_history(
    db: Session, session_id: int, skip_latest: bool = True
) -> List[Dict[str, str]]:
    records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.id.desc())
        .all()
    )

    if skip_latest:
        records = records[1:]

    history = []
    current_tokens = 0
    for msg in records:
        msg_dict = {"role": msg.role, "content": msg.content}
        msg_tokens = get_messages_token_count([msg_dict])
        if current_tokens + msg_tokens > MAX_HISTORY_TOKENS:
            break
        history.append(msg_dict)
        current_tokens += msg_tokens

    history.reverse()
    return history


def _citation_dicts_to_dtos(citations: List[Dict[str, Any]]) -> List[CitationDTO]:
    return [CitationDTO(**citation) for citation in citations]


# ----------------- V3.2 多会话接口 -----------------
@router.get("/sessions", response_model=List[SessionDTO])
async def list_sessions(db: Session = Depends(get_db)):
    """获取全部会话"""
    _ensure_default_session(db)
    sessions = db.query(models.ChatSession).order_by(models.ChatSession.id.asc()).all()
    return [
        SessionDTO(id=s.id, title=s.title, created_at=_format_datetime(s.created_at))
        for s in sessions
    ]


@router.post("/sessions", response_model=SessionDTO)
async def create_session(request: CreateSessionRequest, db: Session = Depends(get_db)):
    """创建新会话"""
    title = request.title.strip() or "新会话"
    session = models.ChatSession(title=title[:200])
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionDTO(
        id=session.id,
        title=session.title,
        created_at=_format_datetime(session.created_at),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    """删除指定会话及其聊天记录"""
    session = _get_session_or_404(db, session_id)
    db.query(models.ChatMessage).filter(models.ChatMessage.session_id == session.id).delete()
    db.delete(session)
    db.commit()
    return {"status": "success", "message": f"Session {session_id} deleted"}


# ----------------- V3.2 流式聊天 + RAG 引用 -----------------
@router.post("/chat/stream")
async def chat_with_ai_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """流式对话接口：支持多会话、RAG 引用和系统提示词"""
    session = _get_session_or_404(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user", content=request.message, session_id=session.id
    )
    db.add(user_msg)
    db.commit()

    relevant_context, citations = retrieve_relevant_context_with_citations(
        request.message, top_k=settings.RAG_TOP_K
    )
    final_system_prompt = _build_rag_system_prompt(request.system_prompt, relevant_context)
    history = _load_token_limited_history(db, session.id, skip_latest=True)

    async def event_generator():
        meta = {"rag_citations": citations}
        yield f"::META::{json.dumps(meta, ensure_ascii=False)}\n"

        full_reply = ""
        async for chunk in generate_ai_response_stream(
            request.message, history, final_system_prompt
        ):
            full_reply += chunk
            yield chunk

        if full_reply:
            assistant_msg = models.ChatMessage(
                role="assistant", content=full_reply, session_id=session.id
            )
            db.add(assistant_msg)
            db.commit()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ----------------- V3.0 文档上传与处理接口 -----------------
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """接收前端上传的文档，并送入向量数据库"""
    filename = file.filename or ""
    if not filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="只支持上传 PDF 或 TXT 文件")

    os.makedirs("uploads", exist_ok=True)
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("uploads", safe_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_count = process_and_store_document(file_path)
        return {
            "status": "success",
            "message": f"文件 {safe_filename} 处理成功，共切分出 {chunks_count} 个知识块。",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")


# ----------------- V3.3 RAG 文档管理接口 -----------------
@router.get("/documents", response_model=List[UploadedDocumentDTO])
async def list_documents():
    """列出已上传的 PDF/TXT 文档。"""
    return [UploadedDocumentDTO(**doc) for doc in list_uploaded_documents()]


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    """删除某个上传文档，并重建向量库。"""
    try:
        deleted = delete_document_and_rebuild(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建向量库失败: {str(e)}")

    if not deleted:
        raise HTTPException(status_code=404, detail=f"文档 {filename} 不存在")
    return {"status": "success", "message": f"Document {filename} deleted"}


@router.post("/documents/rebuild")
async def rebuild_documents():
    """基于 uploads 目录重建整个向量库。"""
    try:
        chunks_count = rebuild_vector_store_from_uploads()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建向量库失败: {str(e)}")
    return {
        "status": "success",
        "message": f"向量库重建完成，共写入 {chunks_count} 个知识块。",
    }


@router.delete("/documents")
async def clear_documents_vector_store():
    """清空本地向量库，但保留 uploads 目录下的原始文件。"""
    clear_vector_store()
    return {"status": "success", "message": "Vector store cleared"}


@router.get("/rag/settings", response_model=RagSettingsDTO)
async def get_rag_settings():
    """查看当前 RAG 配置。"""
    return RagSettingsDTO(
        embedding_model=settings.EMBEDDING_MODEL,
        top_k=settings.RAG_TOP_K,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )


# ----------------- V3.2 Agent 聊天接口 -----------------
@router.post("/chat/agent", response_model=AgentResponse)
async def chat_with_agent(request: ChatRequest, db: Session = Depends(get_db)):
    """Agent 对话接口：支持多会话、RAG 引用、系统提示词和工具调用轨迹"""
    session = _get_session_or_404(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user", content=request.message, session_id=session.id
    )
    db.add(user_msg)
    db.commit()

    relevant_context, citations = retrieve_relevant_context_with_citations(
        request.message, top_k=settings.RAG_TOP_K
    )
    final_system_prompt = _build_rag_system_prompt(request.system_prompt, relevant_context)
    history = _load_token_limited_history(db, session.id, skip_latest=True)

    reply, steps = await run_tool_agent(
        user_message=request.message,
        history=history,
        system_prompt=final_system_prompt,
        session_id=session.id,
    )

    assistant_msg = models.ChatMessage(
        role="assistant", content=reply, session_id=session.id
    )
    db.add(assistant_msg)
    db.commit()

    return AgentResponse(
        reply=reply,
        steps=steps,
        rag_citations=_citation_dicts_to_dtos(citations),
    )


# ----------------- 兼容旧版非流式聊天接口 -----------------
@router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, db: Session = Depends(get_db)):
    """非流式聊天接口：保留简化版逻辑，并支持 session_id"""
    session = _get_session_or_404(db, request.session_id)

    user_msg = models.ChatMessage(
        role="user", content=request.message, session_id=session.id
    )
    db.add(user_msg)
    db.commit()

    history_records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session.id)
        .order_by(models.ChatMessage.id.desc())
        .limit(10)
        .all()
    )
    history_records.reverse()
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_records[:-1]
    ]

    ai_reply = await generate_ai_response(request.message, history)

    assistant_msg = models.ChatMessage(
        role="assistant", content=ai_reply, session_id=session.id
    )
    db.add(assistant_msg)
    db.commit()

    return ChatResponse(reply=ai_reply)


@router.get("/history", response_model=List[MessageDTO])
async def get_history(session_id: int = 1, db: Session = Depends(get_db)):
    """获取指定会话的历史聊天记录"""
    session = _get_session_or_404(db, session_id)
    records = (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session.id)
        .order_by(models.ChatMessage.id.asc())
        .all()
    )
    return [
        MessageDTO(
            role=r.role,
            content=r.content,
            created_at=_format_datetime(r.created_at),
        )
        for r in records
    ]


@router.delete("/history")
async def clear_history(session_id: int = 1, db: Session = Depends(get_db)):
    """清空指定会话的聊天记录"""
    session = _get_session_or_404(db, session_id)
    db.query(models.ChatMessage).filter(models.ChatMessage.session_id == session.id).delete()
    db.commit()
    return {"status": "success", "message": f"History cleared for session {session.id}"}
