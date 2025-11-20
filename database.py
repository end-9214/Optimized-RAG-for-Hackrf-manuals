import sqlite3
import threading
import uuid
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class SessionStore:
    def __init__(self, db_path: str = "sessions.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            self._conn.commit()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id) VALUES (?)",
                (session_id,),
            )
            self._conn.commit()
        return session_id

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None

    def list_sessions(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [row["session_id"] for row in rows]

    def get_history_messages(self, session_id: str) -> List[BaseMessage]:
        rows = self._conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

        history: List[BaseMessage] = []
        for row in rows:
            role = row["role"]
            content = row["content"]
            if role == "human":
                history.append(HumanMessage(content=content))
            elif role == "ai":
                history.append(AIMessage(content=content))
            else:
                history.append(SystemMessage(content=content))
        return history

    def append_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        if not messages:
            return
        payload = [
            (session_id, self._message_role(msg), msg.content) for msg in messages
        ]
        with self._lock:
            cur = self._conn.cursor()
            cur.executemany(
                "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
                payload,
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "human"
        if isinstance(message, AIMessage):
            return "ai"
        return "system"