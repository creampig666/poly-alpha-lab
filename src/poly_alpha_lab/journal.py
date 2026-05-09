"""SQLite-backed research journal and paper-trading ledger."""

from __future__ import annotations

import csv
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

JournalSide = Literal["YES", "NO", "NONE"]
ProbabilitySource = Literal[
    "manual",
    "external_model",
    "ai_estimate",
    "weather_threshold_model",
    "unknown",
]
JournalStatus = Literal["OPEN", "RESOLVED", "VOID", "SKIPPED"]
CandidateGrade = Literal["A", "B", "C", "SKIP"]

EXPORT_FIELDS = [
    "id",
    "created_at",
    "updated_at",
    "market_id",
    "slug",
    "question",
    "category",
    "candidate_score",
    "candidate_grade",
    "side",
    "fair_yes_probability",
    "probability_source",
    "entry_price",
    "entry_size",
    "fee_per_share",
    "expected_value_per_share",
    "expected_profit",
    "status",
    "resolution_value",
    "realized_pnl",
    "brier_score",
    "rationale",
    "notes",
]


class JournalEntry(BaseModel):
    id: int
    created_at: str
    updated_at: str
    market_id: str
    slug: str | None = None
    question: str
    category: str | None = None
    end_date: str | None = None
    candidate_score: float | None = None
    candidate_grade: CandidateGrade | None = None
    side: JournalSide = "NONE"
    fair_yes_probability: float | None = Field(default=None, ge=0, le=1)
    probability_source: ProbabilitySource = "unknown"
    entry_price: float | None = Field(default=None, ge=0, le=1)
    entry_size: float | None = Field(default=None, ge=0)
    fee_per_share: float | None = Field(default=None, ge=0)
    expected_value_per_share: float | None = None
    expected_profit: float | None = None
    rationale: str | None = None
    status: JournalStatus = "OPEN"
    resolution_value: int | None = None
    realized_pnl: float | None = None
    brier_score: float | None = None
    notes: str | None = None


class ResearchJournal:
    """Research journal for manual probability notes and paper trades."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def create_entry(
        self,
        *,
        market_id: str,
        question: str,
        slug: str | None = None,
        category: str | None = None,
        end_date: str | None = None,
        candidate_score: float | None = None,
        candidate_grade: str | None = None,
        side: str = "NONE",
        fair_yes_probability: float | None = None,
        probability_source: str = "unknown",
        entry_price: float | None = None,
        entry_size: float | None = None,
        fee_per_share: float | None = None,
        expected_value_per_share: float | None = None,
        expected_profit: float | None = None,
        rationale: str | None = None,
        status: str | None = None,
        notes: str | None = None,
    ) -> JournalEntry:
        """Create a journal entry. This never places a real order."""

        side = _normalize_side(side)
        probability_source = _normalize_probability_source(probability_source)
        candidate_grade = _normalize_grade(candidate_grade)
        status = _normalize_status(status or ("SKIPPED" if side == "NONE" else "OPEN"))
        _validate_probability(fair_yes_probability)
        _validate_price(entry_price, "entry_price")
        _validate_non_negative(entry_size, "entry_size")
        _validate_non_negative(fee_per_share, "fee_per_share")

        auto_expected_value_per_share, auto_expected_profit = self._expected_values(
            side=side,
            fair_yes_probability=fair_yes_probability,
            entry_price=entry_price,
            entry_size=entry_size,
            fee_per_share=fee_per_share,
        )
        expected_value_per_share = (
            auto_expected_value_per_share
            if expected_value_per_share is None
            else expected_value_per_share
        )
        expected_profit = auto_expected_profit if expected_profit is None else expected_profit
        now = _utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO journal_entries (
                    created_at, updated_at, market_id, slug, question, category, end_date,
                    candidate_score, candidate_grade, side, fair_yes_probability,
                    probability_source, entry_price, entry_size, fee_per_share,
                    expected_value_per_share, expected_profit, rationale, status, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    now,
                    market_id,
                    slug,
                    question,
                    category,
                    end_date,
                    candidate_score,
                    candidate_grade,
                    side,
                    fair_yes_probability,
                    probability_source,
                    entry_price,
                    entry_size,
                    fee_per_share,
                    expected_value_per_share,
                    expected_profit,
                    rationale,
                    status,
                    notes,
                ),
            )
            entry_id = int(cursor.lastrowid)
        return self.get_entry(entry_id)

    def list_entries(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
        category: str | None = None,
        grade: str | None = None,
    ) -> list[JournalEntry]:
        """List recent entries with optional filters."""

        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(_normalize_status(status))
        if category:
            clauses.append("category = ?")
            params.append(category)
        if grade:
            clauses.append("candidate_grade = ?")
            params.append(_normalize_grade(grade))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM journal_entries {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def delete_entry(self, entry_id: int) -> None:
        """Hard-delete one journal entry."""

        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM journal_entries WHERE id = ?", (entry_id,))
            if cursor.rowcount == 0:
                raise LookupError(f"journal entry not found: {entry_id}")

    def export_csv(
        self,
        output_path: str | Path,
        *,
        status: str | None = None,
        category: str | None = None,
        grade: str | None = None,
    ) -> int:
        """Export matching entries to CSV. Returns number of exported rows."""

        entries = self._query_entries(status=status, category=category, grade=grade)
        path = Path(output_path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=EXPORT_FIELDS)
            writer.writeheader()
            for entry in entries:
                data = entry.model_dump()
                writer.writerow({field: data.get(field) for field in EXPORT_FIELDS})
        return len(entries)

    def get_entry(self, entry_id: int) -> JournalEntry:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM journal_entries WHERE id = ?", (entry_id,)).fetchone()
        if row is None:
            raise LookupError(f"journal entry not found: {entry_id}")
        return _entry_from_row(row)

    def _query_entries(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        grade: str | None = None,
    ) -> list[JournalEntry]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(_normalize_status(status))
        if category:
            clauses.append("category = ?")
            params.append(category)
        if grade:
            clauses.append("candidate_grade = ?")
            params.append(_normalize_grade(grade))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM journal_entries {where} ORDER BY id ASC",
                params,
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def update_resolution(self, *, entry_id: int, resolution_value: int, notes: str | None = None) -> JournalEntry:
        """Resolve an entry and compute realized PnL and Brier score."""

        if resolution_value not in {0, 1}:
            raise ValueError("resolution_value must be 0 or 1")
        entry = self.get_entry(entry_id)
        realized_pnl = self.compute_entry_pnl(entry, resolution_value=resolution_value)
        brier_score = self.compute_brier_score(entry, resolution_value=resolution_value)
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE journal_entries
                SET updated_at = ?, status = 'RESOLVED', resolution_value = ?,
                    realized_pnl = ?, brier_score = ?, notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (now, resolution_value, realized_pnl, brier_score, notes, entry_id),
            )
        return self.get_entry(entry_id)

    def compute_entry_pnl(self, entry: JournalEntry, *, resolution_value: int | None = None) -> float | None:
        """Compute paper-trade PnL for an entry."""

        if entry.side == "NONE":
            return None
        if resolution_value is None:
            resolution_value = entry.resolution_value
        if resolution_value not in {0, 1}:
            return None
        if entry.entry_price is None or entry.entry_size is None:
            return None
        fee = entry.fee_per_share or 0.0
        if entry.side == "YES":
            return (resolution_value - entry.entry_price - fee) * entry.entry_size
        return ((1 - resolution_value) - entry.entry_price - fee) * entry.entry_size

    def compute_brier_score(
        self,
        entry: JournalEntry,
        *,
        resolution_value: int | None = None,
    ) -> float | None:
        """Compute Brier score for the fair YES probability."""

        if resolution_value is None:
            resolution_value = entry.resolution_value
        if resolution_value not in {0, 1} or entry.fair_yes_probability is None:
            return None
        return (entry.fair_yes_probability - resolution_value) ** 2

    def summarize_performance(self) -> dict[str, Any]:
        """Summarize journal performance and calibration."""

        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM journal_entries ORDER BY id ASC").fetchall()
        entries = [_entry_from_row(row) for row in rows]
        resolved = [entry for entry in entries if entry.status == "RESOLVED"]
        open_entries = [entry for entry in entries if entry.status == "OPEN"]
        realized = [entry.realized_pnl for entry in resolved if entry.realized_pnl is not None]
        briers = [entry.brier_score for entry in resolved if entry.brier_score is not None]
        wins = [pnl for pnl in realized if pnl > 0]

        return {
            "total_entries": len(entries),
            "open_entries": len(open_entries),
            "resolved_entries": len(resolved),
            "total_realized_pnl": sum(realized) if realized else 0.0,
            "average_pnl": _average(realized),
            "average_brier_score": _average(briers),
            "win_rate": (len(wins) / len(realized)) if realized else None,
            "by_category": _group_summary(entries, "category"),
            "by_candidate_grade": _group_summary(entries, "candidate_grade"),
            "by_probability_source": _group_summary(entries, "probability_source"),
        }

    def _expected_values(
        self,
        *,
        side: str,
        fair_yes_probability: float | None,
        entry_price: float | None,
        entry_size: float | None,
        fee_per_share: float | None,
    ) -> tuple[float | None, float | None]:
        if side == "NONE" or fair_yes_probability is None or entry_price is None:
            return None, None
        fee = fee_per_share or 0.0
        fair_side_probability = fair_yes_probability if side == "YES" else 1 - fair_yes_probability
        ev_per_share = fair_side_probability - entry_price - fee
        expected_profit = ev_per_share * entry_size if entry_size is not None else None
        return ev_per_share, expected_profit

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    slug TEXT,
                    question TEXT NOT NULL,
                    category TEXT,
                    end_date TEXT,
                    candidate_score REAL,
                    candidate_grade TEXT,
                    side TEXT NOT NULL,
                    fair_yes_probability REAL,
                    probability_source TEXT NOT NULL,
                    entry_price REAL,
                    entry_size REAL,
                    fee_per_share REAL,
                    expected_value_per_share REAL,
                    expected_profit REAL,
                    rationale TEXT,
                    status TEXT NOT NULL,
                    resolution_value INTEGER,
                    realized_pnl REAL,
                    brier_score REAL,
                    notes TEXT
                )
                """
            )


def _entry_from_row(row: sqlite3.Row) -> JournalEntry:
    return JournalEntry(**dict(row))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_side(value: str) -> JournalSide:
    side = value.upper()
    if side not in {"YES", "NO", "NONE"}:
        raise ValueError("side must be YES, NO, or NONE")
    return side  # type: ignore[return-value]


def _normalize_probability_source(value: str) -> ProbabilitySource:
    source = value.lower()
    valid = {"manual", "external_model", "ai_estimate", "weather_threshold_model", "unknown"}
    if source not in valid:
        raise ValueError(
            "probability_source must be manual, external_model, ai_estimate, "
            "weather_threshold_model, or unknown"
        )
    return source  # type: ignore[return-value]


def _normalize_status(value: str) -> JournalStatus:
    status = value.upper()
    if status not in {"OPEN", "RESOLVED", "VOID", "SKIPPED"}:
        raise ValueError("status must be OPEN, RESOLVED, VOID, or SKIPPED")
    return status  # type: ignore[return-value]


def _normalize_grade(value: str | None) -> CandidateGrade | None:
    if value is None:
        return None
    grade = value.upper()
    if grade not in {"A", "B", "C", "SKIP"}:
        raise ValueError("candidate_grade must be A, B, C, or SKIP")
    return grade  # type: ignore[return-value]


def _validate_probability(value: float | None) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError("fair_yes_probability must be between 0 and 1")


def _validate_price(value: float | None, name: str) -> None:
    if value is not None and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_non_negative(value: float | None, name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be non-negative")


def _average(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _group_summary(entries: list[JournalEntry], field_name: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[JournalEntry]] = {}
    for entry in entries:
        key = getattr(entry, field_name) or "unknown"
        groups.setdefault(str(key), []).append(entry)
    return {key: _summary_for_entries(group) for key, group in sorted(groups.items())}


def _summary_for_entries(entries: list[JournalEntry]) -> dict[str, Any]:
    resolved = [entry for entry in entries if entry.status == "RESOLVED"]
    realized = [entry.realized_pnl for entry in resolved if entry.realized_pnl is not None]
    briers = [entry.brier_score for entry in resolved if entry.brier_score is not None]
    wins = [pnl for pnl in realized if pnl > 0]
    return {
        "entries": len(entries),
        "resolved_entries": len(resolved),
        "total_realized_pnl": sum(realized) if realized else 0.0,
        "average_pnl": _average(realized),
        "average_brier_score": _average(briers),
        "win_rate": (len(wins) / len(realized)) if realized else None,
    }
