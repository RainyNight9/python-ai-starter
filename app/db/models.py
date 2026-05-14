from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from datetime import datetime
from sqlalchemy.orm import relationship

from app.db.database import Base

# ==========================================
# 数据库模型 (定义数据表结构)
# ==========================================


class ChatSession(Base):
    """对话会话：多条 ChatMessage 归属同一 session，实现多会话隔离。"""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), default="新会话")
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    role = Column(String(50))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    session = relationship("ChatSession", back_populates="messages")
