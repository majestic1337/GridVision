import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class ChatRecord:
    id: str
    title: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class MessageRecord:
    id: str
    chat_id: str
    role: str
    content: str
    created_at: str
    sources: Optional[List[Dict[str, Any]]]
    attachments: Optional[List[Dict[str, Any]]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sources_json TEXT,
                attachments_json TEXT,
                FOREIGN KEY(chat_id) REFERENCES chats(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_chat(db_path: Path, title: Optional[str] = None) -> ChatRecord:
    chat_id = uuid4().hex
    now = _utc_now()
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, title, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return ChatRecord(id=chat_id, title=title, created_at=now, updated_at=now)


def list_chats(db_path: Path) -> List[ChatRecord]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        return [ChatRecord(**dict(r)) for r in rows]
    finally:
        conn.close()


def get_chat(db_path: Path, chat_id: str) -> Optional[ChatRecord]:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
        return ChatRecord(**dict(row)) if row else None
    finally:
        conn.close()


def update_chat_title_if_empty(db_path: Path, chat_id: str, title: str) -> None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if row and (row["title"] is None or not str(row["title"]).strip()):
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))
            conn.commit()
    finally:
        conn.close()


def touch_chat(db_path: Path, chat_id: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_utc_now(), chat_id))
        conn.commit()
    finally:
        conn.close()


def add_message(
    db_path: Path,
    chat_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> MessageRecord:
    message_id = uuid4().hex
    now = _utc_now()
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, created_at, sources_json, attachments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                chat_id,
                role,
                content,
                now,
                json.dumps(sources) if sources else None,
                json.dumps(attachments) if attachments else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    touch_chat(db_path, chat_id)
    return MessageRecord(
        id=message_id,
        chat_id=chat_id,
        role=role,
        content=content,
        created_at=now,
        sources=sources,
        attachments=attachments,
    )


def list_messages(db_path: Path, chat_id: str) -> List[MessageRecord]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, chat_id, role, content, created_at, sources_json, attachments_json
            FROM messages
            WHERE chat_id = ?
            ORDER BY created_at ASC
            """,
            (chat_id,),
        ).fetchall()
        messages: List[MessageRecord] = []
        for row in rows:
            sources = json.loads(row["sources_json"]) if row["sources_json"] else None
            attachments = json.loads(row["attachments_json"]) if row["attachments_json"] else None
            messages.append(
                MessageRecord(
                    id=row["id"],
                    chat_id=row["chat_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                    sources=sources,
                    attachments=attachments,
                )
            )
        return messages
    finally:
        conn.close()


def delete_chat(db_path: Path, chat_id: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
    finally:
        conn.close()


def as_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return asdict(obj)
    return dict(obj)
