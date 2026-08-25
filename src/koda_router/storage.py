"""SQLite persistence for routing configuration and usage tracking."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from koda_router.models import RoutingConfig, UsageRecord


class RoutingStorage:
    """Stores routing config and usage records for cost tracking."""

    def __init__(self, db_path: str = "data/routing.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS routing_config (
                id      TEXT PRIMARY KEY DEFAULT 'default',
                config  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS usage_records (
                id              TEXT PRIMARY KEY,
                model_id        TEXT NOT NULL,
                provider        TEXT NOT NULL,
                model           TEXT NOT NULL,
                input_tokens    INTEGER DEFAULT 0,
                output_tokens   INTEGER DEFAULT 0,
                cost            REAL DEFAULT 0.0,
                latency_ms      REAL DEFAULT 0.0,
                complexity      TEXT DEFAULT '',
                conversation_id TEXT,
                timestamp       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON usage_records(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_usage_provider
                ON usage_records(provider);
            CREATE INDEX IF NOT EXISTS idx_usage_model
                ON usage_records(model_id);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_config(self) -> RoutingConfig:
        row = self._conn.execute(
            "SELECT config FROM routing_config WHERE id = 'default'"
        ).fetchone()
        if not row:
            return RoutingConfig()
        return RoutingConfig.model_validate_json(row["config"])

    def save_config(self, config: RoutingConfig) -> None:
        self._conn.execute(
            """INSERT INTO routing_config (id, config) VALUES ('default', ?)
               ON CONFLICT(id) DO UPDATE SET config = excluded.config""",
            (config.model_dump_json(),),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Usage records
    # ------------------------------------------------------------------

    def record_usage(self, record: UsageRecord) -> str:
        if not record.id:
            record.id = str(uuid.uuid4())
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc)

        self._conn.execute(
            """INSERT INTO usage_records
               (id, model_id, provider, model, input_tokens, output_tokens,
                cost, latency_ms, complexity, conversation_id, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.model_id,
                record.provider,
                record.model,
                record.input_tokens,
                record.output_tokens,
                record.cost,
                record.latency_ms,
                record.complexity,
                record.conversation_id,
                record.timestamp.isoformat(),
            ),
        )
        self._conn.commit()
        return record.id

    def get_usage(
        self,
        since: Optional[str] = None,
        provider: Optional[str] = None,
        limit: int = 100,
    ) -> list[UsageRecord]:
        clauses: list[str] = []
        params: list = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if provider:
            clauses.append("provider = ?")
            params.append(provider)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM usage_records {where} ORDER BY timestamp DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_cost_summary(self, since: Optional[str] = None) -> dict:
        """Get aggregated cost summary."""
        where = ""
        params: list = []
        if since:
            where = "WHERE timestamp >= ?"
            params.append(since)

        # Total cost
        total_row = self._conn.execute(
            f"SELECT COALESCE(SUM(cost), 0) as total, "
            f"COALESCE(SUM(input_tokens), 0) as input_tokens, "
            f"COALESCE(SUM(output_tokens), 0) as output_tokens, "
            f"COUNT(*) as request_count "
            f"FROM usage_records {where}",
            params,
        ).fetchone()

        # By provider
        by_provider_rows = self._conn.execute(
            f"SELECT provider, COALESCE(SUM(cost), 0) as cost, COUNT(*) as count "
            f"FROM usage_records {where} GROUP BY provider",
            params,
        ).fetchall()

        # By model
        by_model_rows = self._conn.execute(
            f"SELECT model_id, COALESCE(SUM(cost), 0) as cost, COUNT(*) as count "
            f"FROM usage_records {where} GROUP BY model_id ORDER BY cost DESC LIMIT 10",
            params,
        ).fetchall()

        return {
            "total_cost": round(total_row["total"], 6),
            "total_input_tokens": total_row["input_tokens"],
            "total_output_tokens": total_row["output_tokens"],
            "request_count": total_row["request_count"],
            "by_provider": {r["provider"]: {"cost": round(r["cost"], 6), "count": r["count"]} for r in by_provider_rows},
            "by_model": {r["model_id"]: {"cost": round(r["cost"], 6), "count": r["count"]} for r in by_model_rows},
        }

    def get_daily_cost(self) -> float:
        """Get today's total spend."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        row = self._conn.execute(
            "SELECT COALESCE(SUM(cost), 0) as total FROM usage_records WHERE timestamp >= ?",
            (today,),
        ).fetchone()
        return row["total"]

    def _row_to_record(self, row: sqlite3.Row) -> UsageRecord:
        return UsageRecord(
            id=row["id"],
            model_id=row["model_id"],
            provider=row["provider"],
            model=row["model"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cost=row["cost"],
            latency_ms=row["latency_ms"],
            complexity=row["complexity"],
            conversation_id=row["conversation_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def close(self) -> None:
        self._conn.close()
