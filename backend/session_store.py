from __future__ import annotations

import sqlite3
from pathlib import Path


class SessionStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _columns(self, conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(messages)").fetchall()
        return {row[1] for row in rows}

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            columns = self._columns(conn)
            if "model_id" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN model_id TEXT NOT NULL DEFAULT ''")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, model_id, id)
                """
            )

    def get_recent_messages(self, session_id: str, model_id: str, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ? AND model_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, model_id, limit),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def append(self, session_id: str, model_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(session_id, model_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, model_id, role, content),
            )

    def clear(self, session_id: str, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND model_id = ?",
                (session_id, model_id),
            )
