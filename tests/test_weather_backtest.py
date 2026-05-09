import csv
import json

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.weather_backtest import (
    EXPORT_FIELDS,
    WeatherBacktestStore,
    weather_signal_skip_reasons,
)


def signal(**overrides):
    data = {
        "market_id": "weather-1",
        "slug": "weather-market",
        "question": "Will the highest temperature in Milan be 24\u00b0C on May 8?",
        "category": "weather",
        "location_name": "Milan",
        "metric": "high_temperature",
        "target_date": "2026-05-08",
        "threshold": 24.0,
        "unit": "C",
        "bucket_mode": "rounded",
        "bucket_lower_bound": 23.5,
        "bucket_upper_bound": 24.5,
        "weather_model": "normal",
        "model_parameters": {},
        "distribution_assumption": "normal forecast error",
        "calibration_applied": False,
        "calibration_quality": None,
        "calibration_n": None,
        "calibration_min_samples_required": None,
        "calibration_bias_raw": None,
        "calibration_bias_shrunk": None,
        "calibration_std_error_raw": None,
        "calibration_std_error_used": None,
        "forecast_mean": 24.0,
        "forecast_std": 0.8,
        "forecast_issued_at": "2026-05-08T00:00:00Z",
        "as_of_time": "2026-05-08T01:00:00Z",
        "forecast_source": "open_meteo",
        "forecast_model": "open_meteo_current_forecast",
        "std_method": "historical_error",
        "forecast_station_id": "LIMC",
        "resolution_station_id": "LIMC",
        "signal_status": "VALID",
        "validation_warnings": [],
        "model_p_yes": 0.62,
        "yes_breakeven": 0.55,
        "no_upper_bound": 0.40,
        "yes_model_edge": 0.07,
        "no_model_edge": -0.22,
        "suggested_paper_side": "YES",
        "bucket_numeric_boundary_confirmed": True,
        "actual_value": 999.0,
        "journal_draft_payload": {
            "side": "YES",
            "entry_price": 0.54,
            "entry_size": 10,
            "fee_per_share": 0.01,
        },
    }
    data.update(overrides)
    return data


def write_signals(path, signals):
    path.write_text(json.dumps(signals), encoding="utf-8")


def make_store(tmp_path):
    return WeatherBacktestStore(tmp_path / "weather_backtest.sqlite")


def test_add_valid_signal_success(tmp_path) -> None:
    store = make_store(tmp_path)
    path = tmp_path / "signals.json"
    write_signals(path, [signal()])

    result = store.add_from_signals(path, entry_size=10, strict=True)

    assert len(result.saved) == 1
    assert result.skipped == []
    snapshot = result.saved[0]
    assert snapshot.market_id == "weather-1"
    assert snapshot.status == "OPEN"
    assert snapshot.expected_value_per_share == pytest.approx(0.07)
    assert snapshot.expected_profit == pytest.approx(0.7)
    assert snapshot.actual_value is None
    assert snapshot.weather_model == "normal"


def test_strict_skips_needs_manual_review(tmp_path) -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(signal_status="NEEDS_MANUAL_REVIEW"),
        strict=True,
    )

    assert eligible is False
    assert "signal_status_not_valid" in reasons


def test_strict_skips_sample_forecast_source(tmp_path) -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(forecast_source="sample"),
        strict=True,
    )

    assert eligible is False
    assert "sample_or_manual_forecast" in reasons


def test_strict_skips_manual_std_method() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(std_method="manual_assumption"),
        strict=True,
    )

    assert eligible is False
    assert "manual_std_method" in reasons


def test_strict_skips_missing_forecast_issued_at() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(forecast_issued_at=None),
        strict=True,
    )

    assert eligible is False
    assert "missing_forecast_issued_at" in reasons


def test_strict_skips_forecast_after_as_of_time() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(forecast_issued_at="2026-05-08T02:00:00Z"),
        strict=True,
    )

    assert eligible is False
    assert "forecast_after_as_of_time" in reasons
    assert "forecast_issued_within_live_capture_tolerance" not in reasons


def test_live_capture_tolerance_signal_does_not_emit_forecast_after_as_of_time() -> None:
    """Tolerance flag set by alpha layer must propagate to backtest skip reasons.

    When the alpha layer accepts ``forecast_issued_at > as_of_time`` via
    ``--forecast-time-tolerance-seconds`` (live capture race), the backtest
    summary must report ``forecast_issued_within_live_capture_tolerance``
    instead of the misleading ``forecast_after_as_of_time``.
    """

    eligible, reasons = weather_signal_skip_reasons(
        signal(
            signal_status="NEEDS_MANUAL_REVIEW",
            forecast_issued_at="2026-05-08T01:00:30Z",
            as_of_time="2026-05-08T01:00:00Z",
            forecast_timing_tolerance_applied=True,
            validation_warnings=["forecast_issued_within_live_capture_tolerance"],
        ),
        strict=False,
        include_needs_review=True,
    )

    assert eligible is False  # blocked by some other reason in real captures
    assert "forecast_after_as_of_time" not in reasons
    assert "forecast_issued_within_live_capture_tolerance" in reasons


def test_live_capture_tolerance_inferred_from_validation_warnings_only() -> None:
    """Tolerance can be expressed via validation_warnings when the boolean flag is absent."""

    eligible, reasons = weather_signal_skip_reasons(
        signal(
            forecast_issued_at="2026-05-08T01:00:30Z",
            as_of_time="2026-05-08T01:00:00Z",
            validation_warnings=["forecast_issued_within_live_capture_tolerance"],
        ),
        strict=True,
    )

    assert "forecast_after_as_of_time" not in reasons
    assert "forecast_issued_within_live_capture_tolerance" in reasons


def test_replay_stale_forecast_without_tolerance_still_emits_after_as_of_time() -> None:
    """Real STALE replay case (no tolerance flag, no warning) must keep blocking reason."""

    eligible, reasons = weather_signal_skip_reasons(
        signal(
            forecast_issued_at="2026-05-08T03:00:00Z",
            as_of_time="2026-05-08T01:00:00Z",
            forecast_timing_tolerance_applied=False,
            validation_warnings=["forecast_issued_after_as_of_time"],
        ),
        strict=True,
    )

    assert eligible is False
    assert "forecast_after_as_of_time" in reasons
    assert "forecast_issued_within_live_capture_tolerance" not in reasons


def test_strict_skips_station_mismatch() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(forecast_station_id=None, resolution_station_id="LIMC"),
        strict=True,
    )

    assert eligible is False
    assert "station_not_matched" in reasons


def test_strict_skips_unconfirmed_bucket_boundary() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(bucket_numeric_boundary_confirmed=False),
        strict=True,
    )

    assert eligible is False
    assert "bucket_boundary_not_confirmed" in reasons


def test_strict_skips_insufficient_calibration() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(
            calibration_applied=True,
            calibration_quality="INSUFFICIENT",
            calibration_n=3,
            calibration_min_samples_required=30,
        ),
        strict=True,
    )

    assert eligible is False
    assert "calibration_quality_insufficient" in reasons
    assert "calibration_samples_too_low" in reasons


def test_allow_unconfirmed_bucket_can_pass_bucket_check() -> None:
    eligible, reasons = weather_signal_skip_reasons(
        signal(bucket_numeric_boundary_confirmed=False),
        strict=True,
        allow_unconfirmed_bucket=True,
    )

    assert eligible is True
    assert "bucket_boundary_not_confirmed" not in reasons


def test_add_from_signals_aggregates_tolerance_reason_not_stale_reason(tmp_path) -> None:
    """End-to-end: live tolerance signal must not surface forecast_after_as_of_time
    in the aggregated skipped labels that daily-capture summary consumes."""

    store = make_store(tmp_path)
    path = tmp_path / "signals.json"
    write_signals(
        path,
        [
            signal(
                market_id="tol-1",
                signal_status="NEEDS_MANUAL_REVIEW",
                forecast_issued_at="2026-05-08T01:00:30Z",
                as_of_time="2026-05-08T01:00:00Z",
                forecast_timing_tolerance_applied=True,
                validation_warnings=["forecast_issued_within_live_capture_tolerance"],
            )
        ],
    )

    result = store.add_from_signals(path, entry_size=10, strict=True)

    assert result.saved == []
    assert len(result.skipped) == 1
    label = result.skipped[0]
    assert "forecast_after_as_of_time" not in label
    assert "forecast_issued_within_live_capture_tolerance" in label


def test_add_from_signals_aggregates_stale_reason_when_no_tolerance(tmp_path) -> None:
    """Replay/historical stale signal must keep the blocking forecast_after_as_of_time reason."""

    store = make_store(tmp_path)
    path = tmp_path / "signals.json"
    write_signals(
        path,
        [
            signal(
                market_id="stale-1",
                forecast_issued_at="2026-05-08T03:00:00Z",
                as_of_time="2026-05-08T01:00:00Z",
                forecast_timing_tolerance_applied=False,
                validation_warnings=["forecast_issued_after_as_of_time"],
            )
        ],
    )

    result = store.add_from_signals(path, entry_size=10, strict=True)

    assert result.saved == []
    assert len(result.skipped) == 1
    label = result.skipped[0]
    assert "forecast_after_as_of_time" in label
    assert "forecast_issued_within_live_capture_tolerance" not in label


def test_resolve_yes_pnl_correct(tmp_path) -> None:
    store = make_store(tmp_path)
    snapshot = store.add_snapshot(signal(), entry_size=10)

    resolved = store.resolve_snapshot(
        snapshot_id=snapshot.id,
        actual_value=24.1,
        resolution_value=1,
    )

    assert resolved.realized_pnl == pytest.approx((1 - 0.54 - 0.01) * 10)


def test_resolve_no_pnl_correct(tmp_path) -> None:
    store = make_store(tmp_path)
    no_signal = signal(
        suggested_paper_side="NO",
        no_model_edge=0.08,
        journal_draft_payload={"side": "NO", "entry_price": 0.30, "fee_per_share": 0.01},
    )
    snapshot = store.add_snapshot(no_signal, entry_size=10)

    resolved = store.resolve_snapshot(
        snapshot_id=snapshot.id,
        actual_value=24.1,
        resolution_value=0,
    )

    assert resolved.realized_pnl == pytest.approx(((1 - 0) - 0.30 - 0.01) * 10)


def test_brier_score_correct(tmp_path) -> None:
    store = make_store(tmp_path)
    snapshot = store.add_snapshot(signal(model_p_yes=0.62), entry_size=10)

    resolved = store.resolve_snapshot(snapshot_id=snapshot.id, actual_value=24.1, resolution_value=1)

    assert resolved.brier_score == pytest.approx((0.62 - 1) ** 2)


def test_summary_edge_bucket_correct(tmp_path) -> None:
    store = make_store(tmp_path)
    for index, edge in enumerate([0.04, 0.07, 0.15, 0.25], start=1):
        store.add_snapshot(
            signal(market_id=f"weather-{index}", yes_model_edge=edge),
            entry_size=10,
        )

    summary = store.summarize()

    assert summary["by_edge_bucket"]["0-5%"]["snapshots"] == 1
    assert summary["by_edge_bucket"]["5-10%"]["snapshots"] == 1
    assert summary["by_edge_bucket"]["10-20%"]["snapshots"] == 1
    assert summary["by_edge_bucket"]["20%+"]["snapshots"] == 1


def test_weather_backtest_add_from_signals_saves_weather_model(tmp_path) -> None:
    store = make_store(tmp_path)
    path = tmp_path / "signals.json"
    write_signals(
        path,
        [
            signal(
                weather_model="student_t",
                model_parameters={"student_t_df": 5},
                distribution_assumption="student_t forecast error with variance matched to forecast_std",
            )
        ],
    )

    result = store.add_from_signals(path, entry_size=10, strict=True)

    assert result.saved[0].weather_model == "student_t"
    assert result.saved[0].model_parameters["student_t_df"] == 5


def test_weather_backtest_summary_by_weather_model(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add_snapshot(signal(market_id="normal", weather_model="normal"), entry_size=10)
    store.add_snapshot(
        signal(
            market_id="student",
            weather_model="student_t",
            model_parameters={"student_t_df": 5},
        ),
        entry_size=10,
    )

    summary = store.summarize()

    assert summary["by_weather_model"]["normal"]["snapshots"] == 1
    assert summary["by_weather_model"]["student_t"]["snapshots"] == 1


def test_export_csv_success(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add_snapshot(signal(), entry_size=10)
    output = tmp_path / "weather_backtest.csv"

    row_count = store.export_csv(output)

    assert row_count == 1
    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == EXPORT_FIELDS
    assert rows[0]["market_id"] == "weather-1"


def test_non_strict_can_save_needs_review_with_manual_validation_notes(tmp_path) -> None:
    store = make_store(tmp_path)
    path = tmp_path / "signals.json"
    write_signals(
        path,
        [
            signal(
                signal_status="NEEDS_MANUAL_REVIEW",
                forecast_source="sample",
                std_method="manual_assumption",
                forecast_station_id=None,
                bucket_numeric_boundary_confirmed=False,
            )
        ],
    )

    result = store.add_from_signals(
        path,
        entry_size=10,
        include_needs_review=True,
        allow_sample_data=True,
        allow_station_mismatch=True,
        allow_unconfirmed_bucket=True,
    )

    assert len(result.saved) == 1
    assert result.saved[0].status == "SKIPPED"
    assert "not strict backtest eligible" in (result.saved[0].notes or "")
    assert "requires manual validation" in (result.saved[0].notes or "")


def test_cli_weather_backtest_add_from_signals(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli_weather_backtest.sqlite"
    monkeypatch.setattr(cli.settings, "weather_backtest_db_path", str(db_path))
    path = tmp_path / "signals.json"
    write_signals(path, [signal()])

    exit_code = cli.run(
        [
            "weather-backtest",
            "add-from-signals",
            "--signals-json",
            str(path),
            "--entry-size",
            "10",
            "--strict",
        ]
    )

    assert exit_code == 0
    snapshots = WeatherBacktestStore(db_path).list_snapshots(limit=5)
    assert len(snapshots) == 1
