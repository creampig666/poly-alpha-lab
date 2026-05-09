"""Daily forward replay capture for weather paper research.

This module is intentionally read-only. It never signs orders, places orders,
uses wallets, or calls authenticated trading endpoints.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.client import GammaClient
from poly_alpha_lab.clob_client import ClobClient
from poly_alpha_lab.filters import filter_markets
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.market_type_classifier import MarketType, classify_strategy_candidate
from poly_alpha_lab.models import Market, MarketStructureError, OrderBook, OrderLevel
from poly_alpha_lab.network_debug import mask_proxy_url
from poly_alpha_lab.resolution_analyzer import analyze_resolution
from poly_alpha_lab.strategy_runner import (
    StrategyCandidate,
    scan_strategy_candidates,
    write_strategy_candidates_json,
)
from poly_alpha_lab.weather_alpha import (
    WeatherAlphaScanResult,
    run_weather_alpha_scan,
    write_weather_alpha_signals_json,
)
from poly_alpha_lab.weather_backtest import WeatherBacktestStore
from poly_alpha_lab.weather_calibration import load_calibration_summaries
from poly_alpha_lab.weather_data import (
    CsvWeatherDataProvider,
    LocationResolver,
    OpenMeteoForecastProvider,
    WeatherDataProvider,
)

WeatherProviderName = Literal["csv", "open-meteo"]
WeatherModelName = Literal["normal", "student_t", "normal_mixture"]
BucketModeName = Literal["rounded", "floor"]
ReportLanguageName = Literal["zh", "en"]


class DailyWeatherCaptureConfig(BaseModel):
    """Configuration for one daily weather forward capture."""

    limit: int = 100
    min_liquidity: float = 1000
    size: float = 10
    weather_provider: WeatherProviderName = "csv"
    weather_data: str | None = None
    locations_file: str = "data/weather/locations.csv"
    calibration_json: str | None = None
    weather_model: WeatherModelName = "normal"
    bucket_mode: BucketModeName = "rounded"
    proxy: str | None = None
    trust_env: bool = True
    output_dir: str = "data/daily"
    snapshot_db: str = "data/polymarket_snapshots.sqlite"
    backtest_db: str = "data/weather_backtest.sqlite"
    entry_size: float = 10
    strict: bool = False
    include_needs_review: bool = False
    dry_run: bool = False
    sizes: list[float] = Field(default_factory=lambda: [10.0, 50.0, 100.0])
    forecast_time_tolerance_seconds: float = Field(default=120.0, ge=0)
    report_language: ReportLanguageName = "zh"
    write_markdown_report: bool = True


class DailyWeatherCaptureSummary(BaseModel):
    """JSON summary for a forward capture run."""

    captured_at: str
    alpha_as_of_time: str | None = None
    snapshot_captured_at: str | None = None
    forecast_time_tolerance_seconds: float = 120.0
    output_dir: str
    strategy_candidates_path: str | None = None
    weather_alpha_path: str | None = None
    summary_path: str | None = None
    markdown_report_path: str | None = None
    snapshot_db: str
    backtest_db: str
    strategy_candidates_count: int = 0
    weather_candidates_count: int = 0
    weather_alpha_signals_count: int = 0
    snapshots_attempted: int = 0
    snapshots_inserted: int = 0
    backtest_saved: int = 0
    backtest_skipped: int = 0
    skipped_reasons: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timing_warnings: list[str] = Field(default_factory=list)
    run_mode: str = "forward_capture"
    network_mode: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    exit_code: int = 0


@dataclass
class SnapshotCaptureResult:
    attempted: int = 0
    inserted: int = 0
    skipped_reasons: Counter[str] = field(default_factory=Counter)


class PolymarketSnapshotStore:
    """SQLite store for forward-captured public CLOB order books."""

    def __init__(self, db_path: str | Path = "data/polymarket_snapshots.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def insert_snapshot(self, record: dict[str, Any]) -> int:
        columns = list(record)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                INSERT INTO polymarket_orderbook_snapshots (
                    {", ".join(columns)}
                ) VALUES (
                    {", ".join("?" for _ in columns)}
                )
                """,
                tuple(record.values()),
            )
            return int(cursor.lastrowid)

    def count_rows(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM polymarket_orderbook_snapshots").fetchone()
        return int(row["count"])

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS polymarket_orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    slug TEXT,
                    question TEXT,
                    category TEXT,
                    yes_token_id TEXT,
                    no_token_id TEXT,
                    yes_orderbook_timestamp TEXT,
                    no_orderbook_timestamp TEXT,
                    yes_bids_json TEXT NOT NULL,
                    yes_asks_json TEXT NOT NULL,
                    no_bids_json TEXT NOT NULL,
                    no_asks_json TEXT NOT NULL,
                    yes_best_bid REAL,
                    yes_best_ask REAL,
                    no_best_bid REAL,
                    no_best_ask REAL,
                    yes_spread REAL,
                    no_spread REAL,
                    yes_executable_avg_buy_price_size_10 REAL,
                    no_executable_avg_buy_price_size_10 REAL,
                    yes_executable_avg_buy_price_size_50 REAL,
                    no_executable_avg_buy_price_size_50 REAL,
                    yes_executable_avg_buy_price_size_100 REAL,
                    no_executable_avg_buy_price_size_100 REAL,
                    avg_buy_prices_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(polymarket_orderbook_snapshots)").fetchall()
        existing = {row["name"] for row in rows}
        for column_name, column_type in {
            "avg_buy_prices_json": "TEXT NOT NULL DEFAULT '{}'",
        }.items():
            if column_name not in existing:
                conn.execute(
                    f"ALTER TABLE polymarket_orderbook_snapshots ADD COLUMN {column_name} {column_type}"
                )


StrategyScanFunc = Callable[
    [GammaClient, ClobClient, DailyWeatherCaptureConfig],
    tuple[list[Market], list[StrategyCandidate], dict[str, BinaryMarketLiquidity]],
]
AlphaScanFunc = Callable[
    [str | Path, WeatherDataProvider, DailyWeatherCaptureConfig, GammaClient, datetime],
    WeatherAlphaScanResult,
]


def run_daily_weather_capture(
    config: DailyWeatherCaptureConfig,
    *,
    gamma_client: GammaClient | None = None,
    clob_client: ClobClient | None = None,
    weather_provider: WeatherDataProvider | None = None,
    now: datetime | None = None,
    strategy_scan_func: StrategyScanFunc | None = None,
    alpha_scan_func: AlphaScanFunc | None = None,
) -> DailyWeatherCaptureSummary:
    """Run one read-only weather forward capture."""

    captured_at_dt = _aware_now(now)
    captured_at = captured_at_dt.isoformat().replace("+00:00", "Z")
    output_dir = Path(config.output_dir)
    network_mode = {
        "proxy": mask_proxy_url(config.proxy) if config.proxy else None,
        "trust_env": config.trust_env,
    }
    summary = DailyWeatherCaptureSummary(
        captured_at=captured_at,
        output_dir=str(output_dir),
        snapshot_db=config.snapshot_db,
        backtest_db=config.backtest_db,
        network_mode=network_mode,
        dry_run=config.dry_run,
        forecast_time_tolerance_seconds=config.forecast_time_tolerance_seconds,
    )

    if config.dry_run:
        summary.warnings.append("dry_run_no_external_api_no_files_written")
        return summary

    day_dir = output_dir / captured_at_dt.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = captured_at_dt.strftime("%Y%m%dT%H%M%SZ")
    strategy_path = day_dir / f"strategy_candidates_{timestamp}.json"
    alpha_path = day_dir / f"weather_alpha_{timestamp}.json"
    summary_path = day_dir / f"capture_summary_{timestamp}.json"
    report_path = day_dir / f"capture_report_{timestamp}_{config.report_language}.md"
    summary.strategy_candidates_path = str(strategy_path)
    summary.weather_alpha_path = str(alpha_path)
    summary.summary_path = str(summary_path)
    if config.write_markdown_report:
        summary.markdown_report_path = str(report_path)

    gamma_client = gamma_client or GammaClient(proxy=config.proxy, trust_env=config.trust_env)
    clob_client = clob_client or ClobClient(proxy=config.proxy, trust_env=config.trust_env)
    strategy_scan_func = strategy_scan_func or _default_strategy_scan
    alpha_scan_func = alpha_scan_func or _default_alpha_scan

    markets: list[Market] = []
    strategy_candidates: list[StrategyCandidate] = []
    try:
        markets, strategy_candidates, _ = strategy_scan_func(gamma_client, clob_client, config)
        summary.strategy_candidates_count = write_strategy_candidates_json(
            strategy_candidates,
            strategy_path,
        )
    except Exception as exc:
        summary.errors.append(f"strategy_scan_failed:{type(exc).__name__}:{exc}")
        summary.exit_code = 1
        finalize_capture_outputs(summary, summary_path, report_language=config.report_language)
        return summary

    weather_candidate_ids = _weather_candidate_ids(strategy_candidates)
    summary.weather_candidates_count = len(weather_candidate_ids)
    result: WeatherAlphaScanResult | None = None
    try:
        provider = weather_provider or build_weather_provider(config)
        alpha_as_of_dt = datetime.now(UTC)
        summary.alpha_as_of_time = _iso_z(alpha_as_of_dt)
        if alpha_as_of_dt < captured_at_dt:
            summary.timing_warnings.append("alpha_as_of_time_before_pipeline_captured_at")
        result = alpha_scan_func(strategy_path, provider, config, gamma_client, alpha_as_of_dt)
        summary.weather_candidates_count = result.weather_candidate_count
        summary.weather_alpha_signals_count = write_weather_alpha_signals_json(result, alpha_path)
        weather_candidate_ids.update(str(signal.market_id) for signal in result.signals)
    except Exception as exc:
        summary.errors.append(f"weather_alpha_failed:{type(exc).__name__}:{exc}")
        summary.warnings.append("weather_alpha_failed_snapshot_attempt_from_strategy_candidates")

    market_index = {market.id: market for market in markets}
    snapshot_markets = _markets_for_ids(weather_candidate_ids, market_index, gamma_client, summary)
    snapshot_store = PolymarketSnapshotStore(config.snapshot_db)
    snapshot_captured_at = _iso_z(datetime.now(UTC))
    summary.snapshot_captured_at = snapshot_captured_at
    snapshot_result = capture_orderbook_snapshots(
        snapshot_markets,
        clob_client=clob_client,
        store=snapshot_store,
        captured_at=snapshot_captured_at,
        sizes=config.sizes,
    )
    summary.snapshots_attempted = snapshot_result.attempted
    summary.snapshots_inserted = snapshot_result.inserted
    summary.skipped_reasons.update(dict(snapshot_result.skipped_reasons))

    if result is not None and alpha_path.exists():
        try:
            backtest_store = WeatherBacktestStore(config.backtest_db)
            add_result = backtest_store.add_from_signals(
                alpha_path,
                entry_size=config.entry_size,
                strict=config.strict,
                include_needs_review=config.include_needs_review,
            )
            summary.backtest_saved = len(add_result.saved)
            summary.backtest_skipped = len(add_result.skipped)
            _count_labels(add_result.skipped, summary.skipped_reasons)
        except Exception as exc:
            summary.errors.append(f"weather_backtest_add_failed:{type(exc).__name__}:{exc}")

    finalize_capture_outputs(summary, summary_path, report_language=config.report_language)
    return summary


def build_weather_provider(config: DailyWeatherCaptureConfig) -> WeatherDataProvider:
    if config.weather_provider == "csv":
        return CsvWeatherDataProvider(config.weather_data or "data/weather/weather_forecasts.csv")
    return OpenMeteoForecastProvider(
        location_resolver=LocationResolver(config.locations_file),
        cache_dir="data/weather/cache",
        refresh_cache=False,
        proxy=config.proxy,
        trust_env=config.trust_env,
    )


def capture_orderbook_snapshots(
    markets: list[Market],
    *,
    clob_client: ClobClient,
    store: PolymarketSnapshotStore,
    captured_at: str,
    sizes: list[float],
) -> SnapshotCaptureResult:
    result = SnapshotCaptureResult(attempted=len(markets))
    for market in markets:
        try:
            record = build_orderbook_snapshot_record(
                market,
                clob_client=clob_client,
                captured_at=captured_at,
                sizes=sizes,
            )
            store.insert_snapshot(record)
            result.inserted += 1
        except Exception as exc:
            reason = f"orderbook_snapshot_failed:{market.id}:{type(exc).__name__}"
            result.skipped_reasons[reason] += 1
    return result


def build_orderbook_snapshot_record(
    market: Market,
    *,
    clob_client: ClobClient,
    captured_at: str,
    sizes: list[float],
) -> dict[str, Any]:
    yes_token_id = market.yes_token_id
    no_token_id = market.no_token_id
    yes_book = clob_client.get_orderbook(yes_token_id)
    no_book = clob_client.get_orderbook(no_token_id)
    liquidity = BinaryMarketLiquidity(yes_book=yes_book, no_book=no_book)
    avg_prices = _avg_prices(liquidity, sizes)
    notes = ["forward_capture_not_historical_backtest"]
    if yes_book.token_id and yes_book.token_id != yes_token_id:
        notes.append("yes_token_id_mismatch")
    if no_book.token_id and no_book.token_id != no_token_id:
        notes.append("no_token_id_mismatch")
    notes.append("clob_orderbook_timestamp_not_provided_using_captured_at")
    return {
        "captured_at": captured_at,
        "market_id": market.id,
        "slug": market.slug,
        "question": market.question,
        "category": market.category,
        "yes_token_id": yes_token_id,
        "no_token_id": no_token_id,
        "yes_orderbook_timestamp": captured_at,
        "no_orderbook_timestamp": captured_at,
        "yes_bids_json": _levels_json(yes_book.bids),
        "yes_asks_json": _levels_json(yes_book.asks),
        "no_bids_json": _levels_json(no_book.bids),
        "no_asks_json": _levels_json(no_book.asks),
        "yes_best_bid": liquidity.yes_best_bid,
        "yes_best_ask": liquidity.yes_best_ask,
        "no_best_bid": liquidity.no_best_bid,
        "no_best_ask": liquidity.no_best_ask,
        "yes_spread": liquidity.yes_spread,
        "no_spread": liquidity.no_spread,
        "yes_executable_avg_buy_price_size_10": _fixed_avg(avg_prices, 10, "yes"),
        "no_executable_avg_buy_price_size_10": _fixed_avg(avg_prices, 10, "no"),
        "yes_executable_avg_buy_price_size_50": _fixed_avg(avg_prices, 50, "yes"),
        "no_executable_avg_buy_price_size_50": _fixed_avg(avg_prices, 50, "no"),
        "yes_executable_avg_buy_price_size_100": _fixed_avg(avg_prices, 100, "yes"),
        "no_executable_avg_buy_price_size_100": _fixed_avg(avg_prices, 100, "no"),
        "avg_buy_prices_json": json.dumps(avg_prices, ensure_ascii=False, sort_keys=True),
        "source": "clob_book",
        "notes": ";".join(notes),
    }


def write_capture_summary(
    summary: DailyWeatherCaptureSummary,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def finalize_capture_outputs(
    summary: DailyWeatherCaptureSummary,
    summary_path: str | Path,
    *,
    report_language: ReportLanguageName = "zh",
) -> None:
    """Write human and machine-readable capture outputs."""

    if summary.markdown_report_path:
        write_daily_capture_report(
            summary,
            summary.markdown_report_path,
            language=report_language,
        )
    write_capture_summary(summary, summary_path)


def write_daily_capture_report(
    summary: DailyWeatherCaptureSummary,
    output_path: str | Path,
    *,
    language: ReportLanguageName = "zh",
) -> None:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        daily_capture_report_zh(summary)
        if language == "zh"
        else daily_capture_summary_to_markdown(summary)
    )
    path.write_text(text, encoding="utf-8")


def daily_capture_summary_to_markdown(summary: DailyWeatherCaptureSummary) -> str:
    lines = [
        "# Daily Weather Forward Capture",
        "",
        "- Dataset type: `forward_replay_capture_not_historical_backtest`",
        "- Trading/auth/wallet operations: `none`",
        f"- captured_at: `{summary.captured_at}`",
        f"- alpha_as_of_time: `{summary.alpha_as_of_time or 'n/a'}`",
        f"- snapshot_captured_at: `{summary.snapshot_captured_at or 'n/a'}`",
        f"- forecast_time_tolerance_seconds: `{summary.forecast_time_tolerance_seconds:g}`",
        f"- dry_run: `{summary.dry_run}`",
        f"- output_dir: `{summary.output_dir}`",
        f"- summary_path: `{summary.summary_path or 'n/a'}`",
        f"- strategy_candidates_path: `{summary.strategy_candidates_path or 'n/a'}`",
        f"- weather_alpha_path: `{summary.weather_alpha_path or 'n/a'}`",
        f"- snapshot_db: `{summary.snapshot_db}`",
        f"- backtest_db: `{summary.backtest_db}`",
        f"- strategy_candidates_count: `{summary.strategy_candidates_count}`",
        f"- weather_candidates_count: `{summary.weather_candidates_count}`",
        f"- weather_alpha_signals_count: `{summary.weather_alpha_signals_count}`",
        f"- snapshots_attempted: `{summary.snapshots_attempted}`",
        f"- snapshots_inserted: `{summary.snapshots_inserted}`",
        f"- backtest_saved: `{summary.backtest_saved}`",
        f"- backtest_skipped: `{summary.backtest_skipped}`",
        f"- network_mode: `{json.dumps(summary.network_mode, ensure_ascii=False)}`",
    ]
    if summary.skipped_reasons:
        lines.extend(["", "## Skipped Reasons", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in sorted(summary.skipped_reasons.items()))
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in summary.warnings)
    if summary.timing_warnings:
        lines.extend(["", "## Timing Warnings", ""])
        lines.extend(f"- `{warning}`" for warning in summary.timing_warnings)
    if summary.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{error}`" for error in summary.errors)
    return "\n".join(lines)


def daily_capture_report_zh(summary: DailyWeatherCaptureSummary) -> str:
    """Render a human-oriented Chinese daily capture report."""

    lines = [
        "# Polymarket 天气策略每日记录报告",
        "",
        "## 1. 本次运行结论",
        "",
        f"- {_daily_capture_conclusion_zh(summary)}",
        "",
        "## 2. 运行基本信息",
        "",
        f"- 运行时间 captured_at：`{summary.captured_at}`",
        f"- alpha 计算时间 alpha_as_of_time：`{summary.alpha_as_of_time or 'n/a'}`",
        f"- 盘口快照时间 snapshot_captured_at：`{summary.snapshot_captured_at or 'n/a'}`",
        f"- 是否 dry-run：`{summary.dry_run}`",
        "- 数据类型：`forward replay capture`，不是 `historical backtest`",
        f"- 网络模式：`{json.dumps(summary.network_mode, ensure_ascii=False)}`",
        f"- 输出目录：`{summary.output_dir}`",
        f"- summary JSON 路径：`{summary.summary_path or 'n/a'}`",
        f"- strategy candidates JSON 路径：`{summary.strategy_candidates_path or 'n/a'}`",
        f"- weather alpha JSON 路径：`{summary.weather_alpha_path or 'n/a'}`",
        f"- snapshot DB 路径：`{summary.snapshot_db}`",
        f"- backtest DB 路径：`{summary.backtest_db}`",
        "",
        "## 3. 数量摘要",
        "",
        "| 指标 | 数值 | 解释 |",
        "|---|---:|---|",
        f"| 策略候选市场数 | {summary.strategy_candidates_count} | 本次 strategy scan 找到的候选市场数量 |",
        f"| 天气候选市场数 | {summary.weather_candidates_count} | 被识别为天气相关的市场数量 |",
        f"| 天气 alpha signals | {summary.weather_alpha_signals_count} | 生成了模型概率和 edge 的天气信号数量 |",
        f"| 尝试保存盘口数 | {summary.snapshots_attempted} | 尝试抓取 CLOB orderbook 的市场数 |",
        f"| 成功保存盘口数 | {summary.snapshots_inserted} | 成功写入 SQLite 的盘口快照数 |",
        f"| backtest 保存数 | {summary.backtest_saved} | 进入 forward replay paper dataset 的信号数 |",
        f"| backtest 跳过数 | {summary.backtest_skipped} | 被 strict gate 跳过的信号数 |",
        "",
        "## 4. 今日是否有可交易信号？",
        "",
        _trade_signal_status_zh(summary),
        "",
        "## 5. 跳过原因解释",
        "",
    ]
    if summary.skipped_reasons:
        lines.extend(
            [
                "| 跳过原因组合 | 次数 | 中文解释 |",
                "|---|---:|---|",
                *(
                    f"| `{reason}` | {count} | {_skip_reason_explanation_zh(reason)} |"
                    for reason, count in sorted(summary.skipped_reasons.items())
                ),
            ]
        )
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "## 6. 错误与警告",
            "",
            *_errors_and_warnings_zh(summary),
            "",
            "## 7. 数据质量判断",
            "",
            *_data_quality_judgment_zh(summary),
            "",
            "## 8. 下一步建议",
            "",
            _next_step_zh(summary),
            "",
        ]
    )
    return "\n".join(lines)


def daily_capture_terminal_summary_zh(summary: DailyWeatherCaptureSummary) -> str:
    status = _run_status_zh(summary)
    return "\n".join(
        [
            "",
            f"本次运行{status}。",
            f"- 天气候选市场：{summary.weather_candidates_count}",
            f"- 已保存盘口快照：{summary.snapshots_inserted}",
            f"- 天气 alpha signals：{summary.weather_alpha_signals_count}",
            f"- 写入 paper dataset：{summary.backtest_saved}",
            f"- 中文报告：{summary.markdown_report_path or 'n/a'}",
        ]
    )


def _run_status_zh(summary: DailyWeatherCaptureSummary) -> str:
    if summary.exit_code != 0:
        return "失败"
    if summary.errors:
        return "部分成功"
    return "成功"


def _daily_capture_conclusion_zh(summary: DailyWeatherCaptureSummary) -> str:
    if summary.exit_code != 0:
        reason = summary.errors[0] if summary.errors else "未知错误"
        return f"本次运行失败，原因是 {reason}。"
    if summary.errors:
        return (
            f"本次运行部分成功，已保存 {summary.snapshots_inserted} 个天气盘口快照，"
            f"但存在错误：{summary.errors[0]}。"
        )
    if summary.backtest_saved > 0:
        return (
            f"本次运行成功，并有 {summary.backtest_saved} 条 signal 写入 "
            "forward replay backtest。"
        )
    if summary.weather_alpha_signals_count > 0:
        return (
            f"本次运行成功，生成 {summary.weather_alpha_signals_count} 个天气 alpha signals，"
            "但均未通过 strict gate。"
        )
    return (
        f"本次运行成功，已保存 {summary.snapshots_inserted} 个天气盘口快照，"
        "但没有生成可入库的 paper trade。"
    )


def _trade_signal_status_zh(summary: DailyWeatherCaptureSummary) -> str:
    if summary.backtest_saved > 0:
        return "今天有 signal 进入 forward replay paper dataset。注意这仍然不是真实交易。"
    if summary.weather_alpha_signals_count > 0:
        return "今天有天气 alpha signal，但没有进入 strict paper dataset。主要原因见下一节。"
    return "今天没有生成天气 alpha signal。这不是错误，只表示没有市场满足当前模型和规则条件。"


def _errors_and_warnings_zh(summary: DailyWeatherCaptureSummary) -> list[str]:
    rows: list[str] = []
    if not summary.errors and not summary.warnings and not summary.timing_warnings:
        return ["本次没有记录到运行错误。"]
    if summary.errors:
        rows.extend(["错误：", *[f"- `{error}`" for error in summary.errors]])
    if summary.warnings:
        rows.extend(["警告：", *[f"- `{warning}`" for warning in summary.warnings]])
    if summary.timing_warnings:
        rows.extend(["时间警告：", *[f"- `{warning}`" for warning in summary.timing_warnings]])
    return rows


def _data_quality_judgment_zh(summary: DailyWeatherCaptureSummary) -> list[str]:
    rows: list[str] = []
    if summary.errors:
        rows.append("- 运行记录到错误，需要先排查数据管线。")
    if summary.snapshots_inserted > 0:
        rows.append("- 盘口快照记录成功。")
    else:
        rows.append("- 没有成功记录盘口，需要检查网络或市场数量。")
    if summary.backtest_saved > 0:
        rows.append("- 今天有 paper entry，后续需等待市场结算后 resolve。")
    else:
        rows.append("- 今天没有 strict paper entry。")
    return rows


def _next_step_zh(summary: DailyWeatherCaptureSummary) -> str:
    if summary.errors:
        return "先修数据管线，不要分析策略收益。"
    if summary.backtest_saved > 0:
        return "后续等市场结算后，使用 weather-backtest resolve 填入 actual/resolution，再看 PnL 和 Brier。"
    if summary.weather_alpha_signals_count > 0:
        return "建议累计 7 天后统计 skipped reasons，再决定是否调整 station mapping、bucket boundary 或 edge threshold。"
    return "明天继续记录即可。不要因为单日没有 signal 就调整策略。"


_SKIP_REASON_TRANSLATIONS: dict[str, str] = {
    "no_paper_side": "模型没有给出 YES/NO 方向，通常是 edge 没超过阈值。",
    "signal_status_not_valid": "信号状态不是 VALID，存在数据、规则或来源风险。",
    "forecast_issued_within_live_capture_tolerance": (
        "live capture 中 forecast 时间略晚于 as_of_time，但在容忍窗口内，不视为未来函数。"
    ),
    "forecast_after_as_of_time": "forecast 时间晚于 as_of_time，存在未来数据风险。",
    "forecast_issued_after_as_of_time": "forecast 时间晚于 as_of_time，存在未来数据风险。",
    "station_not_matched": "模型 forecast 站点与 Polymarket 结算站点不一致。",
    "bucket_boundary_not_confirmed": "市场是温度桶，但具体数值边界没有被规则明确确认。",
    "sample_or_manual_forecast": "forecast 数据是 sample/manual，不可作为严格数据源。",
    "manual_std_method": "forecast_std 是手动假设，不可作为严格校准。",
    "calibration_quality_insufficient": "历史校准样本质量不足。",
    "calibration_samples_too_low": "校准样本数太少。",
}


def _skip_reason_explanation_zh(reason: str) -> str:
    matches = [
        explanation
        for token, explanation in _SKIP_REASON_TRANSLATIONS.items()
        if token in reason
    ]
    if matches:
        return " ".join(dict.fromkeys(matches))
    if reason.startswith("orderbook_snapshot_failed"):
        return "盘口快照抓取失败，通常是 CLOB orderbook 暂时不可用或网络异常。"
    if reason.startswith("market_fetch_failed"):
        return "市场详情拉取失败，通常是 Gamma API 或网络异常。"
    if reason.startswith("market_token_missing"):
        return "市场缺少 YES/NO CLOB token，无法安全抓取盘口。"
    return "未配置中文解释，请查看原始 reason。"


def _default_strategy_scan(
    gamma_client: GammaClient,
    clob_client: ClobClient,
    config: DailyWeatherCaptureConfig,
) -> tuple[list[Market], list[StrategyCandidate], dict[str, BinaryMarketLiquidity]]:
    markets = gamma_client.list_markets(
        active=True,
        closed=False,
        limit=config.limit,
        order="volume",
        ascending=False,
    )
    candidates = filter_markets(markets, min_liquidity=config.min_liquidity)
    liquidities: dict[str, BinaryMarketLiquidity] = {}
    for market in candidates:
        try:
            yes_book = clob_client.get_orderbook(market.yes_token_id)
            no_book = clob_client.get_orderbook(market.no_token_id)
            liquidities[market.id] = BinaryMarketLiquidity(yes_book=yes_book, no_book=no_book)
        except Exception:
            continue
    strategy_candidates = scan_strategy_candidates(candidates, liquidities, size=config.size)
    return candidates, strategy_candidates, liquidities


def _default_alpha_scan(
    strategy_json_path: str | Path,
    provider: WeatherDataProvider,
    config: DailyWeatherCaptureConfig,
    gamma_client: GammaClient,
    alpha_as_of_time: datetime,
) -> WeatherAlphaScanResult:
    def resolution_lookup(candidate: dict[str, Any]):
        market_id = candidate.get("market_id")
        if market_id is None:
            return None
        return analyze_resolution(gamma_client.get_market(str(market_id)))

    calibration_summaries = (
        load_calibration_summaries(config.calibration_json)
        if config.calibration_json
        else None
    )
    return run_weather_alpha_scan(
        strategy_json_path,
        provider,
        bucket_mode=config.bucket_mode,
        weather_model=config.weather_model,
        as_of_time=alpha_as_of_time,
        resolution_lookup=resolution_lookup,
        calibration_summaries=calibration_summaries,
        use_calibrated_std=bool(calibration_summaries),
        use_calibrated_bias=bool(calibration_summaries),
        forecast_time_tolerance_seconds=config.forecast_time_tolerance_seconds,
    )


def _weather_candidate_ids(candidates: list[StrategyCandidate]) -> set[str]:
    ids: set[str] = set()
    for candidate in candidates:
        data = candidate.model_dump(mode="json")
        classification = classify_strategy_candidate(data)
        if classification.market_type in {
            MarketType.weather_temperature_threshold,
            MarketType.weather_temperature_exact_bucket,
        }:
            ids.add(candidate.market_id)
    return ids


def _markets_for_ids(
    market_ids: set[str],
    market_index: dict[str, Market],
    gamma_client: GammaClient,
    summary: DailyWeatherCaptureSummary,
) -> list[Market]:
    markets: list[Market] = []
    for market_id in sorted(market_ids):
        market = market_index.get(market_id)
        if market is None:
            try:
                market = gamma_client.get_market(market_id)
            except Exception as exc:
                summary.skipped_reasons[f"market_fetch_failed:{market_id}:{type(exc).__name__}"] = (
                    summary.skipped_reasons.get(f"market_fetch_failed:{market_id}:{type(exc).__name__}", 0)
                    + 1
                )
                continue
        try:
            _ = market.yes_token_id
            _ = market.no_token_id
        except MarketStructureError as exc:
            summary.skipped_reasons[f"market_token_missing:{market_id}"] = (
                summary.skipped_reasons.get(f"market_token_missing:{market_id}", 0) + 1
            )
            summary.warnings.append(str(exc))
            continue
        markets.append(market)
    return markets


def _avg_prices(liquidity: BinaryMarketLiquidity, sizes: list[float]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for size in sizes:
        key = _size_key(size)
        result[key] = {
            "yes": _safe_avg(lambda: liquidity.yes_avg_buy_price(size)),
            "no": _safe_avg(lambda: liquidity.no_avg_buy_price(size)),
        }
    return result


def _fixed_avg(avg_prices: dict[str, dict[str, float | None]], size: float, side: str) -> float | None:
    return avg_prices.get(_size_key(size), {}).get(side)


def _safe_avg(func: Callable[[], float]) -> float | None:
    try:
        return func()
    except ValueError:
        return None


def _levels_json(levels: list[OrderLevel]) -> str:
    return json.dumps([level.model_dump() for level in levels], ensure_ascii=False)


def _size_key(size: float) -> str:
    return str(int(size)) if float(size).is_integer() else f"{size:g}"


def _count_labels(labels: list[str], counter: dict[str, int]) -> None:
    for label in labels:
        reason = label.split(":", 1)[1] if ":" in label else label
        counter[reason] = counter.get(reason, 0) + 1


def _aware_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return _aware_now(value).isoformat().replace("+00:00", "Z")
