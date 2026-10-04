"""database.py - SQLite storage for analysis history and feedback.
We store summaries only: never raw log contents, never API keys. All SQL uses ? placeholders.
"""
import sqlite3
from contextlib import closing

import config


def _connect():
    return sqlite3.connect(config.DB_PATH)


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                file_name TEXT,
                category TEXT,
                severity TEXT,
                confidence REAL,
                summary TEXT,
                method TEXT,
                seconds REAL,
                feedback TEXT)"""
        )


def save_analysis(report: dict) -> int:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO analyses (created_at, file_name, category, severity, confidence, summary, method, seconds) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (report["generated_at"], report["file_name"], report["category"], report["severity"],
             report["confidence"], report["executive_summary"][:500], report["source"], report["analysis_seconds"]),
        )
        return cur.lastrowid


def save_feedback(analysis_id: int, value: str) -> None:
    if value not in ("helpful", "not_helpful"):
        raise ValueError("feedback must be 'helpful' or 'not_helpful'")
    with closing(_connect()) as conn, conn:
        conn.execute("UPDATE analyses SET feedback = ? WHERE id = ?", (value, analysis_id))


def get_history(limit: int = 50) -> list:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def category_counts() -> dict:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT category, COUNT(*) FROM analyses GROUP BY category").fetchall()
    return {c: n for c, n in rows}


def feedback_stats() -> dict:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT feedback, COUNT(*) FROM analyses WHERE feedback IS NOT NULL GROUP BY feedback").fetchall()
    return {f: n for f, n in rows}
