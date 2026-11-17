"""Tiny SQLite store for chat history (no accounts: one local file next to the app)."""
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).with_name("chats.db")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL, text TEXT NOT NULL, links TEXT, created_at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
        """)


def create_chat(title):
    now = time.time()
    with _conn() as c:
        return c.execute("INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
                         (title.strip()[:60] or "New chat", now, now)).lastrowid


def add_message(chat_id, role, text, links=None):
    now = time.time()
    with _conn() as c:
        c.execute("INSERT INTO messages (chat_id, role, text, links, created_at) VALUES (?, ?, ?, ?, ?)",
                  (chat_id, role, text, json.dumps(links) if links else None, now))
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))


def list_chats():
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC")]


def get_messages(chat_id):
    with _conn() as c:
        return [{"role": r["role"], "text": r["text"], "links": json.loads(r["links"]) if r["links"] else []}
                for r in c.execute("SELECT role, text, links FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))]


def delete_chat(chat_id):
    with _conn() as c:
        c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


init()
