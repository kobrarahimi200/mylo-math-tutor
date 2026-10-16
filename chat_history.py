"""Durable local conversation history, shared by the web API and CLI."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@contextmanager
def database():
    path = Path(os.getenv("CHAT_HISTORY_DB", str(Path(__file__).parent / "data" / "chat_history.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                updated_at TEXT NOT NULL, level TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                question TEXT NOT NULL, response TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS exchange_conversation ON exchanges(conversation_id, id);
        """)
        with connection:
            yield connection
    finally:
        connection.close()


def list_conversations(limit: int | None = None) -> list[dict[str, Any]]:
    with database() as connection:
        rows = connection.execute("SELECT id, title, updated_at AS updatedAt, level FROM conversations ORDER BY updated_at DESC, id LIMIT ?", (limit if limit is not None else -1,)).fetchall()
        return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with database() as connection:
        conversation = connection.execute("SELECT id, title, updated_at AS updatedAt, level FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if conversation is None:
            return None
        messages = []
        for row in connection.execute("SELECT id, question, response FROM exchanges WHERE conversation_id = ? ORDER BY id", (conversation_id,)):
            response = json.loads(row["response"])
            messages.extend([
                {"id": f"{row['id']}-user", "role": "user", "content": row["question"]},
                {"id": f"{row['id']}-assistant", "role": "assistant", "content": response["answer"],
                 "language": response.get("metadata", {}).get("language", "en"),
                 "metadata": response.get("metadata", {}),
                 "visualization": next(iter(response.get("visualizations", [])), None)},
            ])
        return {**dict(conversation), "messages": messages}


def recent_context(conversation_id: str, max_turns: int = 12) -> list[dict[str, str]]:
    with database() as connection:
        rows = connection.execute("SELECT question, response FROM exchanges WHERE conversation_id = ? ORDER BY id DESC LIMIT ?", (conversation_id, (max_turns + 1) // 2)).fetchall()
        turns = []
        for row in reversed(rows):
            turns.extend([{"role": "user", "content": row["question"]}, {"role": "assistant", "content": json.loads(row["response"])["answer"]}])
        return turns[-max_turns:]


def save_exchange(conversation_id: str, question: str, response: dict[str, Any], level: str, *, replace_latest: bool = False) -> None:
    now = datetime.now(timezone.utc).isoformat()
    title = " ".join(question.split())[:72]
    with database() as connection:
        connection.execute("INSERT INTO conversations(id, title, updated_at, level) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at, level = excluded.level", (conversation_id, title, now, level))
        if replace_latest:
            previous = connection.execute("SELECT id, question FROM exchanges WHERE conversation_id = ? ORDER BY id DESC LIMIT 1", (conversation_id,)).fetchone()
            if previous is not None and previous["question"] == question:
                connection.execute("UPDATE exchanges SET response = ? WHERE id = ?", (json.dumps(response, ensure_ascii=False), previous["id"]))
                return
        connection.execute("INSERT INTO exchanges(conversation_id, question, response) VALUES (?, ?, ?)", (conversation_id, question, json.dumps(response, ensure_ascii=False)))
