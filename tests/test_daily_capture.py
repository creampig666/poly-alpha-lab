import json
import sqlite3
from datetime import UTC, datetime

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.daily_capture import (
    DailyWeatherCaptureConfig,
    DailyWeatherCaptureSummary,
    PolymarketSnapshotStore,
    capture_orderbook_snapshots,
    run_daily_weather_capture,
    write_capture_summary,
)
from poly_alpha_lab.models import Market, OrderBook, OrderLevel
from poly_alpha_lab.strategy_runner import StrategyCandidate
from poly_alpha_lab.weather_alpha import WeatherAlphaScanResult


NOW = datetime(2026, 5, 9, 1, 0, tzinfo=UTC)


def market(**overrides):
    data = {
        "id": "weather-1",
        "question": "Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        "slug": "sao-paulo-high-24c",
        "category": "weather",
        "active": True,
        "closed": False,
        "archived": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "liquidity": 50_000,
        "volume": 100_000,
        "endDate": "2026-05-09T23:59:00+00:00",
        "feesEnabled": True,
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.4, 0.6],
        "clobTokenIds": ["yes-token", "no-token"],
    }
    data.update(overrides)
    return Market.model_validate(data)


def book(token_id, ask_size=100):
    return OrderBook(
        token_id=token_id,
        bids=[OrderLevel(price=0.39 if token_id.startswith("yes") else 0.59, size=100)],
        asks=[OrderLevel(price=0.40 if token_id.startswith("yes") else 0.60, size=ask_size)],
    )


class FakeClob:
    def __init__(self, *, fail_tokens=None, ask_size=100):
        self.fail_tokens = set(fail_tokens or [])
        self.ask_size = ask_size
        self.calls = []

    def get_orderbook(self, token_id):
        self.calls.append(token_id)
        if token_id in self.fail_tokens:
            raise RuntimeError("orderbook unavailable")
        return book(token_id, ask_size=self.ask_size)


class FakeGamma:
    def get_market(self, market_id):
        return market(id=market_id)


def strategy_candidate(**overrides):
    data = {
        "market_id": "weather-1",
        "slug": "sao-paulo-high-24c",
        "question": "Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        "category": "weather",
        "candidate_score": 80.0,
        "candidate_grade": "A",
        "resolution_risk_score": 20.0,
        "ambiguity_risk": "LOW",
        "dispute_risk": "LOW",
        "yes_executable_avg_buy_price": 0.4,
        "no_executable_avg_buy_price": 0.6,
        "yes_fee_per_share": 0.001,
        "no_fee_per_share": 0.001,
        "yes_breakeven_probability": 0.401,
        "no_breakeven_probability": 0.601,
        "yes_required_probability": 0.401,
        "no_required_yes_probability_upper_bound": 0.399,
        "yes_depth_score": 20,
        "no_depth_score": 20,
        "strategy_score": 75,
    }
    data.update(overrides)
    return StrategyCandidate(**data)


def test_daily_capture_dry_run_writes_no_files(tmp_path):
    output_dir = tmp_path / "daily"
    summary = run_daily_weather_capture(
        DailyWeatherCaptureConfig(output_dir=str(output_dir), dry_run=True),
        now=NOW,
    )

    assert summary.dry_run is True
    assert not output_dir.exists()
    assert "dry_run_no_external_api_no_files_written" in summary.warnings


def test_daily_capture_cli_dry_run_writes_no_files(tmp_path, capsys):
    output_dir = tmp_path / "daily"
    exit_code = cli.run(
        [
            "daily-capture",
            "weather",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "`True`" in captured
    assert not output_dir.exists()


def test_capture_summary_json_fields_complete(tmp_path):
    summary = DailyWeatherCaptureSummary(
        captured_at="2026-05-09T01:00:00Z",
        output_dir=str(tmp_path),
        snapshot_db=str(tmp_path / "snap.sqlite"),
        backtest_db=str(tmp_path / "backtest.sqlite"),
        strategy_candidates_path=str(tmp_path / "strategy.json"),
        weather_alpha_path=str(tmp_path / "alpha.json"),
        summary_path=str(tmp_path / "summary.json"),
    )
    write_capture_summary(summary, tmp_path / "summary.json")

    data = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    for key in [
        "captured_at",
        "output_dir",
        "strategy_candidates_path",
        "weather_alpha_path",
        "snapshot_db",
        "backtest_db",
        "strategy_candidates_count",
        "weather_candidates_count",
        "weather_alpha_signals_count",
        "snapshots_attempted",
        "snapshots_inserted",
        "backtest_saved",
        "backtest_skipped",
        "alpha_as_of_time",
        "snapshot_captured_at",
        "forecast_time_tolerance_seconds",
        "timing_warnings",
        "run_mode",
        "network_mode",
    ]:
        assert key in data


def test_daily_capture_live_mode_uses_alpha_start_as_of_time(tmp_path):
    alpha_as_of_values = []

    def strategy_scan(gamma_client, clob_client, config):
        return [market()], [strategy_candidate()], {}

    def alpha_scan(strategy_path, provider, config, gamma_client, alpha_as_of_time):
        alpha_as_of_values.append(alpha_as_of_time)
        return WeatherAlphaScanResult(weather_candidate_count=1, signals=[])

    summary = run_daily_weather_capture(
        DailyWeatherCaptureConfig(
            output_dir=str(tmp_path / "daily"),
            snapshot_db=str(tmp_path / "snapshots.sqlite"),
            backtest_db=str(tmp_path / "backtest.sqlite"),
            forecast_time_tolerance_seconds=120,
        ),
        gamma_client=FakeGamma(),
        clob_client=FakeClob(),
        weather_provider=object(),
        now=NOW,
        strategy_scan_func=strategy_scan,
        alpha_scan_func=alpha_scan,
    )

    assert alpha_as_of_values
    assert alpha_as_of_values[0] != NOW
    assert summary.captured_at == "2026-05-09T01:00:00Z"
    assert summary.alpha_as_of_time is not None
    assert summary.alpha_as_of_time != summary.captured_at
    assert summary.snapshot_captured_at is not None
    assert summary.forecast_time_tolerance_seconds == 120


def test_orderbook_snapshot_sqlite_writes_avg_prices(tmp_path):
    store = PolymarketSnapshotStore(tmp_path / "snapshots.sqlite")
    result = capture_orderbook_snapshots(
        [market()],
        clob_client=FakeClob(ask_size=100),
        store=store,
        captured_at="2026-05-09T01:00:00Z",
        sizes=[10, 50, 100],
    )

    assert result.inserted == 1
    with sqlite3.connect(tmp_path / "snapshots.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM polymarket_orderbook_snapshots").fetchone()
    assert row["yes_executable_avg_buy_price_size_10"] == pytest.approx(0.40)
    assert row["no_executable_avg_buy_price_size_10"] == pytest.approx(0.60)
    assert row["yes_executable_avg_buy_price_size_100"] == pytest.approx(0.40)
    assert row["no_executable_avg_buy_price_size_100"] == pytest.approx(0.60)


def test_orderbook_snapshot_insufficient_depth_is_null(tmp_path):
    store = PolymarketSnapshotStore(tmp_path / "snapshots.sqlite")
    capture_orderbook_snapshots(
        [market()],
        clob_client=FakeClob(ask_size=20),
        store=store,
        captured_at="2026-05-09T01:00:00Z",
        sizes=[10, 50, 100],
    )

    with sqlite3.connect(tmp_path / "snapshots.sqlite") as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM polymarket_orderbook_snapshots").fetchone()
    assert row["yes_executable_avg_buy_price_size_10"] == pytest.approx(0.40)
    assert row["yes_executable_avg_buy_price_size_50"] is None
    assert row["yes_executable_avg_buy_price_size_100"] is None


def test_single_orderbook_failure_does_not_block_other_market(tmp_path):
    good = market(id="good", clobTokenIds=["yes-good", "no-good"])
    bad = market(id="bad", clobTokenIds=["yes-bad", "no-bad"])
    store = PolymarketSnapshotStore(tmp_path / "snapshots.sqlite")

    result = capture_orderbook_snapshots(
        [good, bad],
        clob_client=FakeClob(fail_tokens={"yes-bad"}),
        store=store,
        captured_at="2026-05-09T01:00:00Z",
        sizes=[10],
    )

    assert result.attempted == 2
    assert result.inserted == 1
    assert any("bad" in key for key in result.skipped_reasons)


def test_alpha_scan_failure_still_writes_summary_error_and_snapshots(tmp_path):
    def strategy_scan(gamma_client, clob_client, config):
        candidate = strategy_candidate()
        return [market()], [candidate], {}

    def alpha_scan(*args, **kwargs):
        raise RuntimeError("alpha unavailable")

    summary = run_daily_weather_capture(
        DailyWeatherCaptureConfig(
            output_dir=str(tmp_path / "daily"),
            snapshot_db=str(tmp_path / "snapshots.sqlite"),
            backtest_db=str(tmp_path / "backtest.sqlite"),
        ),
        gamma_client=FakeGamma(),
        clob_client=FakeClob(),
        now=NOW,
        strategy_scan_func=strategy_scan,
        alpha_scan_func=alpha_scan,
    )

    assert any(error.startswith("weather_alpha_failed") for error in summary.errors)
    assert summary.snapshots_inserted == 1
    assert summary.summary_path is not None
    assert (tmp_path / "snapshots.sqlite").exists()


def test_daily_capture_does_not_expose_trading_auth_or_wallet_api():
    import poly_alpha_lab.daily_capture as daily_capture

    public_names = set(dir(daily_capture))
    assert "place_order" not in public_names
    assert "sign_order" not in public_names
    assert "private_key" not in public_names
