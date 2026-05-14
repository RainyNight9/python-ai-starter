"""
SQLite 轻量迁移：为已有库增加 chat_sessions 与 chat_messages.session_id。

不使用 Alembic，便于入门项目零额外依赖；新库由 Base.metadata.create_all 直接建全表。
"""

from __future__ import annotations

from sqlalchemy import inspect, text

from app.db import models


def run_startup_migrations(engine) -> None:
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if "chat_sessions" not in tables:
        models.ChatSession.__table__.create(bind=engine, checkfirst=True)

    if "chat_messages" not in tables:
        return

    cols = {c["name"] for c in inspector.get_columns("chat_messages")}
    if "session_id" not in cols:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE chat_messages ADD COLUMN session_id INTEGER")
            )

    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = Session()
    try:
        if db.query(models.ChatSession).count() == 0:
            db.add(models.ChatSession(id=1, title="默认会话"))
            db.commit()
        db.query(models.ChatMessage).filter(
            models.ChatMessage.session_id.is_(None)
        ).update({models.ChatMessage.session_id: 1}, synchronize_session=False)
        db.commit()
    finally:
        db.close()
