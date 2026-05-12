from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

from app.db.database import get_db
from app.db import models
from app.services.llm import generate_ai_response, generate_ai_response_stream

# ==========================================
# API 路由与接口 (处理 HTTP 请求)
# ==========================================

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
# --------------------------------------------------------

@router.post("/chat/stream")
async def chat_with_ai_stream(request: ChatRequest, db: Session = Depends(get_db)):
    """
    【V2.0 新增】流式对话接口 (Server-Sent Events)
    """
    # 1. 保存用户的消息到数据库
    user_msg = models.ChatMessage(role="user", content=request.message)
    db.add(user_msg)
    db.commit()
    
    # 2. 获取历史记录作为上下文
    history_records = db.query(models.ChatMessage).order_by(models.ChatMessage.id.desc()).limit(10).all()
    history_records.reverse()
    history = [{"role": msg.role, "content": msg.content} for msg in history_records[:-1]]

    # 3. 构造流式生成器函数
    async def event_generator():
        full_reply = ""
        # 逐块接收 AI 吐出来的字
        async for chunk in generate_ai_response_stream(request.message, history, request.system_prompt):
            full_reply += chunk
            yield chunk  # 立刻把这个字发送给前端
            
        # 4. 当流式输出完全结束后，我们才得到完整的句子，这时把它存入数据库
        if full_reply:
            assistant_msg = models.ChatMessage(role="assistant", content=full_reply)
            db.add(assistant_msg)
            db.commit()

    # 使用 StreamingResponse 返回流式数据
    return StreamingResponse(event_generator(), media_type="text/event-stream")

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
