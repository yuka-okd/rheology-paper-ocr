from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class LocalRunStore:
    """SQLite metadata for local browser sessions and reviewer decisions."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = data_dir / "sessions.sqlite3"
        self._initialize()

    def create_run(self, name: str | None) -> dict:
        run_id = uuid4().hex[:12]
        created_at = _now()
        run_name = name.strip() if name and name.strip() else f"Run {created_at[:16].replace('T', ' ')}"
        input_dir = self.data_dir / "runs" / run_id / "input"
        output_dir = self.data_dir / "runs" / run_id / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (id, name, created_at, status, input_dir, output_dir)
                VALUES (?, ?, ?, 'ready', ?, ?)
                """,
                (run_id, run_name, created_at, str(input_dir), str(output_dir)),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def list_runs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def update_status(self, run_id: str, status: str, error: str | None = None) -> None:
        completed_at = _now() if status in {"completed", "failed", "blocked"} else None
        with self._connect() as connection:
            connection.execute(
                "UPDATE runs SET status = ?, error = ?, completed_at = COALESCE(?, completed_at) WHERE id = ?",
                (status, error, completed_at, run_id),
            )

    def save_decision(self, run_id: str, curve_id: str, figure_id: str | None, decision: str, note: str | None) -> dict:
        updated_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO reviewer_decisions (run_id, curve_id, figure_id, decision, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, curve_id, figure_id)
                DO UPDATE SET decision = excluded.decision, note = excluded.note, updated_at = excluded.updated_at
                """,
                (run_id, curve_id, figure_id or "", decision, note or "", updated_at),
            )
        return {"curve_id": curve_id, "figure_id": figure_id, "decision": decision, "note": note or "", "updated_at": updated_at}

    def decisions_for_run(self, run_id: str) -> dict[tuple[str, str | None], dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM reviewer_decisions WHERE run_id = ?", (run_id,)).fetchall()
        return {
            (row["curve_id"], row["figure_id"] or None): {
                "decision": row["decision"],
                "note": row["note"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    input_dir TEXT NOT NULL,
                    output_dir TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviewer_decisions (
                    run_id TEXT NOT NULL,
                    curve_id TEXT NOT NULL,
                    figure_id TEXT NOT NULL DEFAULT '',
                    decision TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, curve_id, figure_id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
