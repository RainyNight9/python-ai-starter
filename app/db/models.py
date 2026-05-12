from sqlalchemy import Column, Integer, String, DateTime, Text
from datetime import datetime
from app.db.database import Base

# ==========================================
# 数据库模型 (定义数据表结构)
# ==========================================

# ChatMessage 继承自 Base，SQLAlchemy 会将其映射为数据库中的表
class ChatMessage(Base):
    # __tablename__ 定义了数据库中真实的表名
    __tablename__ = "chat_messages"

    # Column 定义了表中的每一列字段
    # id: 主键，自增整数，设置了 index=True 会为其建立索引，加快查询速度
    id = Column(Integer, primary_key=True, index=True)
    
    # role: 记录是谁发的消息，"user" 代表用户，"assistant" 代表 AI
    role = Column(String(50)) 
    
    # content: 消息的具体文本内容，使用 Text 类型可以存储很长的字符串
    content = Column(Text)
    
    # created_at: 记录消息产生的时间。默认值为 datetime.utcnow (即数据插入时的 UTC 时间)
    created_at = Column(DateTime, default=datetime.utcnow)
