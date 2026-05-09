"""SQLite-backed weather forward replay and paper backtest dataset."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

BacktestStatus = Literal["OPEN", "RESOLVED", "VOID", "SKIPPED"]
PaperSide = Literal["YES", "NO", "NONE"]

EXPORT_FIELDS = [
    "id",
    "created_at",
    "as_of_time",
    "market_id",
    "slug",
    "question",
    "category",
    "location_name",
    "metric",
    "target_date",
    "threshold",
    "unit",
    "bucket_mode",
    "bucket_lower_bound",
    "bucket_upper_bound",
    "weather_model",
    "model_parameters",
    "distribution_assumption",
    "calibration_applied",
    "calibration_quality",
    "calibration_n",
    "calibration_min_samples_required",
    "calibration_bias_raw",
    "calibration_bias_shrunk",
    "calibration_std_error_raw",
    "calibration_std_error_used",
    "forecast_mean",
    "forecast_std",
    "forecast_issued_at",
    "forecast_source",
    "forecast_model",
    "std_method",
    "forecast_station_id",
    "resolution_station_id",
    "signal_status",
    "validation_warnings",
    "model_p_yes",
    "yes_breakeven",
    "no_upper_bound",
    "yes_model_edge",
    "no_model_edge",
    "suggested_paper_side",
    "entry_price",
    "fee_per_share",
    "entry_size",
    "expected_value_per_share",
    "expected_profit",
    "actual_value",
    "resolution_value",
    "realized_pnl",
    "brier_score",
    "status",
    "notes",
]


class WeatherBacktestSnapshot(BaseModel):
    id: int
    created_at: str
    as_of_time: str | None = None
    market_id: str
    slug: str | None = None
    question: str
    category: str | None = None
    location_name: str | None = None
    metric: str | None = None
    target_date: str | None = None
    threshold: float | None = None
    unit: str | None = None
    bucket_mode: str | None = None
    bucket_lower_bound: float | None = None
    bucket_upper_bound: float | None = None
    weather_model: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    distribution_assumption: str | None = None
    calibration_applied: bool = False
    calibration_quality: str | None = None
    calibration_n: int | None = None
    calibration_min_samples_required: int | None = None
    calibration_bias_raw: float | None = None
    calibration_bias_shrunk: float | None = None
    calibration_std_error_raw: float | None = None
    calibration_std_error_used: float | None = None
    forecast_mean: float | None = None
    forecast_std: float | None = None
    forecast_issued_at: str | None = None
    forecast_source: str | None = None
    forecast_model: str | None = None
    std_method: str | None = None
    forecast_station_id: str | None = None
    resolution_station_id: str | None = None
    signal_status: str | None = None
    validation_warnings: list[str] = Field(default_factory=list)
    model_p_yes: float | None = Field(default=None, ge=0, le=1)
    yes_breakeven: float | None = None
    no_upper_bound: float | None = None
    yes_model_edge: float | None = None
    no_model_edge: float | None = None
    suggested_paper_side: PaperSide = "NONE"
    entry_price: float | None = None
    fee_per_share: float | None = None
    entry_size: float | None = None
    expected_value_per_share: float | None = None
    expected_profit: float | None = None
    actual_value: float | None = None
    resolution_value: int | None = None
    realized_pnl: float | None = None
    brier_score: float | None = None
    status: BacktestStatus = "OPEN"
    notes: str | None = None


@dataclass
class AddFromSignalsResult:
    saved: list[WeatherBacktestSnapshot] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


class WeatherBacktestStore:
    """Forward replay dataset for weather paper alpha signals."""

    def __init__(self, db_path: str | Path = "data/weather_backtest.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_snapshot(
        self,
        signal: dict[str, Any],
        *,
        entry_size: float,
        status: str | None = None,
        notes: str | None = None,
    ) -> WeatherBacktestSnapshot:
        """Store one weather alpha signal snapshot. This never places a real order."""

        side = _normalize_side(str(signal.get("suggested_paper_side") or "NONE"))
        if side not in {"YES", "NO"}:
            raise ValueError("weather backtest snapshots require suggested_paper_side YES or NO")
        draft = signal.get("journal_draft_payload")
        if not isinstance(draft, dict):
            draft = {}
        entry_price = _optional_float(draft.get("entry_price"))
        fee_per_share = _optional_float(draft.get("fee_per_share"))
        edge = _side_edge(signal, side)
        expected_profit = edge * entry_size if edge is not None else None
        status = _normalize_status(status or "OPEN")
        created_at = _utc_now()
        values = {
            "created_at": created_at,
            "as_of_time": signal.get("as_of_time"),
            "market_id": str(signal.get("market_id") or ""),
            "slug": signal.get("slug"),
            "question": str(signal.get("question") or ""),
            "category": signal.get("category"),
            "location_name": signal.get("location_name"),
            "metric": signal.get("metric"),
            "target_date": signal.get("target_date"),
            "threshold": _optional_float(signal.get("threshold")),
            "unit": signal.get("unit"),
            "bucket_mode": signal.get("bucket_mode"),
            "bucket_lower_bound": _optional_float(signal.get("bucket_lower_bound")),
            "bucket_upper_bound": _optional_float(signal.get("bucket_upper_bound")),
            "weather_model": signal.get("weather_model"),
            "model_parameters": _json_object(signal.get("model_parameters")),
            "distribution_assumption": signal.get("distribution_assumption"),
            "calibration_applied": 1 if bool(signal.get("calibration_applied")) else 0,
            "calibration_quality": signal.get("calibration_quality"),
            "calibration_n": _optional_int(signal.get("calibration_n")),
            "calibration_min_samples_required": _optional_int(
                signal.get("calibration_min_samples_required")
            ),
            "calibration_bias_raw": _optional_float(signal.get("calibration_bias_raw")),
            "calibration_bias_shrunk": _optional_float(signal.get("calibration_bias_shrunk")),
            "calibration_std_error_raw": _optional_float(signal.get("calibration_std_error_raw")),
            "calibration_std_error_used": _optional_float(signal.get("calibration_std_error_used")),
            "forecast_mean": _optional_float(signal.get("forecast_mean")),
            "forecast_std": _optional_float(signal.get("forecast_std")),
            "forecast_issued_at": signal.get("forecast_issued_at"),
            "forecast_source": signal.get("forecast_source"),
            "forecast_model": signal.get("forecast_model"),
            "std_method": signal.get("std_method"),
            "forecast_station_id": signal.get("forecast_station_id"),
            "resolution_station_id": signal.get("resolution_station_id"),
            "signal_status": signal.get("signal_status"),
            "validation_warnings": _json_list(signal.get("validation_warnings")),
            "model_p_yes": _optional_float(signal.get("model_p_yes")),
            "yes_breakeven": _optional_float(signal.get("yes_breakeven")),
            "no_upper_bound": _optional_float(signal.get("no_upper_bound")),
            "yes_model_edge": _optional_float(signal.get("yes_model_edge")),
            "no_model_edge": _optional_float(signal.get("no_model_edge")),
            "suggested_paper_side": side,
            "entry_price": entry_price,
            "fee_per_share": fee_per_share,
            "entry_size": entry_size,
            "expected_value_per_share": edge,
            "expected_profit": expected_profit,
            "status": status,
            "notes": notes,
        }
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO weather_backtest_snapshots (
                    {", ".join(values)}
                ) VALUES (
                    {", ".join("?" for _ in values)}
                )
                """,
                tuple(values.values()),
            )
            snapshot_id = int(cursor.lastrowid)
        return self.get_snapshot(snapshot_id)

    def add_from_signals(
        self,
        signals_json_path: str | Path,
        *,
        entry_size: float,
        strict: bool = False,
        include_needs_review: bool = False,
        allow_unconfirmed_bucket: bool = False,
        allow_station_mismatch: bool = False,
        allow_sample_data: bool = False,
        status: str | None = None,
    ) -> AddFromSignalsResult:
        path = Path(signals_json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("signals JSON must contain a list")
        result = AddFromSignalsResult()
        for signal in data:
            if not isinstance(signal, dict):
                continue
            eligible, reasons = weather_signal_skip_reasons(
                signal,
                strict=strict,
                include_needs_review=include_needs_review,
                allow_unconfirmed_bucket=allow_unconfirmed_bucket,
                allow_station_mismatch=allow_station_mismatch,
                allow_sample_data=allow_sample_data,
            )
            label = f"{signal.get('market_id') or 'unknown'}:{','.join(reasons)}"
            if not eligible:
                result.skipped.append(label)
                continue
            snapshot_status = status
            notes = "forward replay dataset; not a historical Polymarket backtest"
            if not _strict_backtest_eligible(signal):
                notes = (
                    "not strict backtest eligible; requires manual validation; "
                    "forward replay dataset; not a historical Polymarket backtest"
                )
                snapshot_status = snapshot_status or "SKIPPED"
            result.saved.append(
                self.add_snapshot(
                    signal,
                    entry_size=entry_size,
                    status=snapshot_status,
                    notes=notes,
                )
            )
        return result

    def list_snapshots(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
        signal_status: str | None = None,
        side: str | None = None,
        location: str | None = None,
    ) -> list[WeatherBacktestSnapshot]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(_normalize_status(status))
        if signal_status:
            clauses.append("signal_status = ?")
            params.append(signal_status)
        if side:
            clauses.append("suggested_paper_side = ?")
            params.append(_normalize_side(side))
        if location:
            clauses.append("location_name = ?")
            params.append(location)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM weather_backtest_snapshots {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def get_snapshot(self, snapshot_id: int) -> WeatherBacktestSnapshot:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM weather_backtest_snapshots WHERE id = ?",
                (snapshot_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"weather backtest snapshot not found: {snapshot_id}")
        return _snapshot_from_row(row)

    def resolve_snapshot(
        self,
        *,
        snapshot_id: int,
        actual_value: float | None,
        resolution_value: int,
        notes: str | None = None,
    ) -> WeatherBacktestSnapshot:
        if resolution_value not in {0, 1}:
            raise ValueError("resolution_value must be 0 or 1")
        snapshot = self.get_snapshot(snapshot_id)
        realized_pnl = compute_snapshot_pnl(snapshot, resolution_value=resolution_value)
        brier_score = (
            (snapshot.model_p_yes - resolution_value) ** 2
            if snapshot.model_p_yes is not None
            else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE weather_backtest_snapshots
                SET actual_value = ?, resolution_value = ?, realized_pnl = ?,
                    brier_score = ?, status = 'RESOLVED', notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (actual_value, resolution_value, realized_pnl, brier_score, notes, snapshot_id),
            )
        return self.get_snapshot(snapshot_id)

    def summarize(self) -> dict[str, Any]:
        snapshots = self._query_snapshots()
        resolved = [snapshot for snapshot in snapshots if snapshot.status == "RESOLVED"]
        realized = [
            snapshot.realized_pnl for snapshot in resolved if snapshot.realized_pnl is not None
        ]
        briers = [snapshot.brier_score for snapshot in resolved if snapshot.brier_score is not None]
        wins = [pnl for pnl in realized if pnl > 0]
        return {
            "total_snapshots": len(snapshots),
            "open": sum(1 for snapshot in snapshots if snapshot.status == "OPEN"),
            "resolved": len(resolved),
            "skipped": sum(1 for snapshot in snapshots if snapshot.status == "SKIPPED"),
            "total_realized_pnl": sum(realized) if realized else 0.0,
            "average_pnl": _average(realized),
            "average_brier_score": _average(briers),
            "win_rate": (len(wins) / len(realized)) if realized else None,
            "by_signal_status": _group_summary(snapshots, "signal_status"),
            "by_location": _group_summary(snapshots, "location_name"),
            "by_metric": _group_summary(snapshots, "metric"),
            "by_bucket_mode": _group_summary(snapshots, "bucket_mode"),
            "by_weather_model": _group_summary(snapshots, "weather_model"),
            "by_edge_bucket": _edge_bucket_summary(snapshots),
        }

    def export_csv(self, output_path: str | Path) -> int:
        snapshots = self._query_snapshots()
        path = Path(output_path)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=EXPORT_FIELDS)
            writer.writeheader()
            for snapshot in snapshots:
                data = snapshot.model_dump()
                data["validation_warnings"] = json.dumps(
                    snapshot.validation_warnings,
                    ensure_ascii=False,
                )
                data["model_parameters"] = json.dumps(
                    snapshot.model_parameters,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                writer.writerow({field: data.get(field) for field in EXPORT_FIELDS})
        return len(snapshots)

    def _query_snapshots(self) -> list[WeatherBacktestSnapshot]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM weather_backtest_snapshots ORDER BY id ASC"
            ).fetchall()
        return [_snapshot_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weather_backtest_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    as_of_time TEXT,
                    market_id TEXT NOT NULL,
                    slug TEXT,
                    question TEXT NOT NULL,
                    category TEXT,
                    location_name TEXT,
                    metric TEXT,
                    target_date TEXT,
                    threshold REAL,
                    unit TEXT,
                    bucket_mode TEXT,
                    bucket_lower_bound REAL,
                    bucket_upper_bound REAL,
                    weather_model TEXT,
                    model_parameters TEXT NOT NULL DEFAULT '{}',
                    distribution_assumption TEXT,
                    calibration_applied INTEGER NOT NULL DEFAULT 0,
                    calibration_quality TEXT,
                    calibration_n INTEGER,
                    calibration_min_samples_required INTEGER,
                    calibration_bias_raw REAL,
                    calibration_bias_shrunk REAL,
                    calibration_std_error_raw REAL,
                    calibration_std_error_used REAL,
                    forecast_mean REAL,
                    forecast_std REAL,
                    forecast_issued_at TEXT,
                    forecast_source TEXT,
                    forecast_model TEXT,
                    std_method TEXT,
                    forecast_station_id TEXT,
                    resolution_station_id TEXT,
                    signal_status TEXT,
                    validation_warnings TEXT NOT NULL,
                    model_p_yes REAL,
                    yes_breakeven REAL,
                    no_upper_bound REAL,
                    yes_model_edge REAL,
                    no_model_edge REAL,
                    suggested_paper_side TEXT NOT NULL,
                    entry_price REAL,
                    fee_per_share REAL,
                    entry_size REAL,
                    expected_value_per_share REAL,
                    expected_profit REAL,
                    actual_value REAL,
                    resolution_value INTEGER,
                    realized_pnl REAL,
                    brier_score REAL,
                    status TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(weather_backtest_snapshots)").fetchall()
        existing = {row["name"] for row in rows}
        for column_name, column_type in {
            "weather_model": "TEXT",
            "model_parameters": "TEXT NOT NULL DEFAULT '{}'",
            "distribution_assumption": "TEXT",
            "calibration_applied": "INTEGER NOT NULL DEFAULT 0",
            "calibration_quality": "TEXT",
            "calibration_n": "INTEGER",
            "calibration_min_samples_required": "INTEGER",
            "calibration_bias_raw": "REAL",
            "calibration_bias_shrunk": "REAL",
            "calibration_std_error_raw": "REAL",
            "calibration_std_error_used": "REAL",
        }.items():
            if column_name not in existing:
                conn.execute(
                    f"ALTER TABLE weather_backtest_snapshots ADD COLUMN {column_name} {column_type}"
                )


def weather_signal_skip_reasons(
    signal: dict[str, Any],
    *,
    strict: bool,
    include_needs_review: bool = False,
    allow_unconfirmed_bucket: bool = False,
    allow_station_mismatch: bool = False,
    allow_sample_data: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    side = str(signal.get("suggested_paper_side") or "NONE").upper()
    signal_status = str(signal.get("signal_status") or "").upper()
    if side not in {"YES", "NO"}:
        reasons.append("no_paper_side")
    if signal_status != "VALID":
        if strict or signal_status != "NEEDS_MANUAL_REVIEW" or not include_needs_review:
            reasons.append("signal_status_not_valid")
    if _is_sample_or_manual(signal.get("forecast_source")) and not allow_sample_data:
        reasons.append("sample_or_manual_forecast")
    if strict and _normalized(signal.get("std_method")) in {
        "manual_assumption",
        "fallback_error_std",
    }:
        reasons.append("manual_std_method")
    forecast_issued_at = _parse_optional_datetime(signal.get("forecast_issued_at"))
    as_of_time = _parse_optional_datetime(signal.get("as_of_time"))
    if forecast_issued_at is None:
        reasons.append("missing_forecast_issued_at")
    if as_of_time is None:
        reasons.append("missing_as_of_time")
    if forecast_issued_at is not None and as_of_time is not None and forecast_issued_at > as_of_time:
        if _live_capture_tolerance_applied(signal):
            reasons.append("forecast_issued_within_live_capture_tolerance")
        else:
            reasons.append("forecast_after_as_of_time")
    resolution_station_id = _optional_string(signal.get("resolution_station_id"))
    forecast_station_id = _optional_string(signal.get("forecast_station_id"))
    if (
        resolution_station_id
        and resolution_station_id != forecast_station_id
        and not allow_station_mismatch
    ):
        reasons.append("station_not_matched")
    if not bool(signal.get("bucket_numeric_boundary_confirmed")) and not allow_unconfirmed_bucket:
        reasons.append("bucket_boundary_not_confirmed")
    if strict and bool(signal.get("calibration_applied")):
        quality = str(signal.get("calibration_quality") or "").upper()
        if quality not in {"MEDIUM", "HIGH"}:
            reasons.append("calibration_quality_insufficient")
        calibration_n = _optional_int(signal.get("calibration_n"))
        min_required = _optional_int(signal.get("calibration_min_samples_required"))
        if calibration_n is None or min_required is None or calibration_n < min_required:
            reasons.append("calibration_samples_too_low")
    return not reasons, _unique(reasons)


def _live_capture_tolerance_applied(signal: dict[str, Any]) -> bool:
    """Return True if the alpha layer accepted ``forecast_issued_at > as_of_time``
    via live capture tolerance.

    The alpha layer sets ``forecast_timing_tolerance_applied=True`` and adds
    ``forecast_issued_within_live_capture_tolerance`` to ``validation_warnings``
    when the gap is within ``--forecast-time-tolerance-seconds``. The backtest
    layer mirrors that decision so the daily-capture summary does not surface
    a misleading ``forecast_after_as_of_time`` reason for live captures.
    """

    if bool(signal.get("forecast_timing_tolerance_applied")):
        return True
    warnings = signal.get("validation_warnings") or []
    if not isinstance(warnings, list):
        return False
    return any(str(item) == "forecast_issued_within_live_capture_tolerance" for item in warnings)


def compute_snapshot_pnl(
    snapshot: WeatherBacktestSnapshot,
    *,
    resolution_value: int,
) -> float | None:
    if snapshot.suggested_paper_side not in {"YES", "NO"}:
        return None
    if snapshot.entry_price is None or snapshot.entry_size is None:
        return None
    fee = snapshot.fee_per_share or 0.0
    if snapshot.suggested_paper_side == "YES":
        return (resolution_value - snapshot.entry_price - fee) * snapshot.entry_size
    return ((1 - resolution_value) - snapshot.entry_price - fee) * snapshot.entry_size


def _snapshot_from_row(row: sqlite3.Row) -> WeatherBacktestSnapshot:
    data = dict(row)
    data["validation_warnings"] = json.loads(data["validation_warnings"] or "[]")
    data["model_parameters"] = json.loads(data.get("model_parameters") or "{}")
    return WeatherBacktestSnapshot(**data)


def _strict_backtest_eligible(signal: dict[str, Any]) -> bool:
    eligible, _ = weather_signal_skip_reasons(
        signal,
        strict=True,
        include_needs_review=False,
        allow_unconfirmed_bucket=False,
        allow_station_mismatch=False,
        allow_sample_data=False,
    )
    return eligible


def _side_edge(signal: dict[str, Any], side: str) -> float | None:
    key = "yes_model_edge" if side == "YES" else "no_model_edge"
    return _optional_float(signal.get(key))


def _json_list(value: Any) -> str:
    if not isinstance(value, list):
        value = []
    return json.dumps([str(item) for item in value], ensure_ascii=False)


def _json_object(value: Any) -> str:
    if not isinstance(value, dict):
        value = {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_side(value: str) -> PaperSide:
    side = value.upper()
    if side not in {"YES", "NO", "NONE"}:
        raise ValueError("side must be YES, NO, or NONE")
    return side  # type: ignore[return-value]


def _normalize_status(value: str) -> BacktestStatus:
    status = value.upper()
    if status not in {"OPEN", "RESOLVED", "VOID", "SKIPPED"}:
        raise ValueError("status must be OPEN, RESOLVED, VOID, or SKIPPED")
    return status  # type: ignore[return-value]


def _optional_string(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip().upper()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_sample_or_manual(value: Any) -> bool:
    return _normalized(value) in {"sample", "manual"}


def _normalized(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip().casefold()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_summary(
    snapshots: list[WeatherBacktestSnapshot],
    field_name: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[WeatherBacktestSnapshot]] = {}
    for snapshot in snapshots:
        key = getattr(snapshot, field_name) or "unknown"
        groups.setdefault(str(key), []).append(snapshot)
    return {key: _summary_for_group(group) for key, group in sorted(groups.items())}


def _edge_bucket_summary(
    snapshots: list[WeatherBacktestSnapshot],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[WeatherBacktestSnapshot]] = {}
    for snapshot in snapshots:
        groups.setdefault(_edge_bucket(snapshot), []).append(snapshot)
    return {key: _summary_for_group(group) for key, group in sorted(groups.items())}


def _edge_bucket(snapshot: WeatherBacktestSnapshot) -> str:
    edge = (
        snapshot.yes_model_edge
        if snapshot.suggested_paper_side == "YES"
        else snapshot.no_model_edge
    )
    edge = max(edge or 0.0, 0.0)
    if edge < 0.05:
        return "0-5%"
    if edge < 0.10:
        return "5-10%"
    if edge < 0.20:
        return "10-20%"
    return "20%+"


def _summary_for_group(snapshots: list[WeatherBacktestSnapshot]) -> dict[str, Any]:
    resolved = [snapshot for snapshot in snapshots if snapshot.status == "RESOLVED"]
    realized = [
        snapshot.realized_pnl for snapshot in resolved if snapshot.realized_pnl is not None
    ]
    briers = [snapshot.brier_score for snapshot in resolved if snapshot.brier_score is not None]
    wins = [pnl for pnl in realized if pnl > 0]
    return {
        "snapshots": len(snapshots),
        "resolved": len(resolved),
        "total_realized_pnl": sum(realized) if realized else 0.0,
        "average_pnl": _average(realized),
        "average_brier_score": _average(briers),
        "win_rate": len(wins) / len(realized) if realized else None,
    }
