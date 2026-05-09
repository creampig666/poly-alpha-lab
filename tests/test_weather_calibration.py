import csv
import json

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.market_type_classifier import classify_strategy_candidate
from poly_alpha_lab.weather_alpha import build_weather_alpha_signal
from poly_alpha_lab.weather_calibration import (
    CALIBRATION_CSV_FIELDS,
    WeatherCalibrationSummary,
    calibration_quality_for_n,
    fit_weather_calibration,
    horizon_bucket,
    load_calibration_summaries,
    load_forecast_error_samples,
    write_calibration_csv,
    write_calibration_json,
)
from poly_alpha_lab.weather_data import WeatherForecast


def history_csv(path, rows):
    fields = [
        "location",
        "station_id",
        "metric",
        "target_date",
        "target_datetime",
        "forecast_issued_at",
        "forecast_mean",
        "actual_value",
        "unit",
        "forecast_source",
        "forecast_model",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(**overrides):
    data = {
        "location": "Milan",
        "station_id": "LIMC",
        "metric": "high_temperature",
        "target_date": "2026-05-09",
        "target_datetime": "2026-05-09T00:00:00Z",
        "forecast_issued_at": "2026-05-08T00:00:00Z",
        "forecast_mean": "24",
        "actual_value": "25",
        "unit": "C",
        "forecast_source": "open_meteo",
        "forecast_model": "model-a",
    }
    data.update(overrides)
    return data


def candidate():
    return {
        "market_id": "weather-1",
        "slug": "milan-high-24c",
        "question": "Will the highest temperature in Milan be 24\u00b0C on May 8?",
        "category": "weather",
        "yes_breakeven_probability": 0.05,
        "no_required_yes_probability_upper_bound": 0.02,
        "journal_draft_payload_yes": {
            "entry_price": 0.04,
            "entry_size": 10,
            "fee_per_share": 0.001,
        },
    }


def forecast(**overrides):
    data = {
        "date": "2026-05-08",
        "location": "Milan",
        "metric": "high_temperature",
        "forecast_mean": 24.0,
        "forecast_std": 0.8,
        "unit": "C",
        "forecast_issued_at": "2026-05-08T00:00:00Z",
        "forecast_source": "sample",
        "forecast_model": "sample_model",
        "std_method": "manual_assumption",
    }
    data.update(overrides)
    return WeatherForecast(**data)


def summary(**overrides):
    data = {
        "group_key": "metric=high_temperature|horizon_bucket=12_24h",
        "metric": "high_temperature",
        "horizon_bucket": "12_24h",
        "n": 60,
        "bias": 1.0,
        "mean_error": 1.0,
        "std_error": 2.0,
        "mae": 1.0,
        "rmse": 1.0,
        "q05": -1.0,
        "q25": 0.0,
        "q50": 1.0,
        "q75": 2.0,
        "q95": 3.0,
        "min_error": -2.0,
        "max_error": 4.0,
        "tail_abs_1": 0.5,
        "tail_abs_2": 0.25,
        "tail_abs_3": 0.1,
        "calibration_quality": "MEDIUM",
        "min_samples_required": 30,
        "bias_shrinkage_applied": True,
        "bias_raw": 1.0,
        "bias_shrunk": 1.0,
        "std_error_raw": 2.0,
        "std_error_used": 2.0,
        "quality_warnings": [],
    }
    data.update(overrides)
    return WeatherCalibrationSummary(**data)


def test_error_equals_actual_minus_forecast_mean(tmp_path) -> None:
    path = tmp_path / "history.csv"
    history_csv(path, [row(forecast_mean="24", actual_value="25.5")])

    sample = load_forecast_error_samples(path)[0]

    assert sample.error == pytest.approx(1.5)


def test_calibration_summary_stats(tmp_path) -> None:
    path = tmp_path / "history.csv"
    history_csv(
        path,
        [
            row(forecast_mean="24", actual_value="25"),
            row(forecast_mean="24", actual_value="23"),
            row(forecast_mean="24", actual_value="26"),
            row(forecast_mean="24", actual_value="24"),
        ],
    )

    summaries = fit_weather_calibration(
        path,
        group_by=["metric", "horizon_bucket"],
        min_samples=1,
        bias_shrinkage_k=0,
    )
    fitted = summaries[0]

    assert fitted.n == 4
    assert fitted.bias == pytest.approx(0.5)
    assert fitted.mean_error == pytest.approx(0.5)
    assert fitted.std_error == pytest.approx((5 / 3) ** 0.5)
    assert fitted.mae == pytest.approx(1.0)
    assert fitted.rmse == pytest.approx((6 / 4) ** 0.5)


def test_calibration_quantiles(tmp_path) -> None:
    path = tmp_path / "history.csv"
    history_csv(
        path,
        [
            row(forecast_mean="24", actual_value="23"),
            row(forecast_mean="24", actual_value="24"),
            row(forecast_mean="24", actual_value="25"),
            row(forecast_mean="24", actual_value="26"),
        ],
    )

    fitted = fit_weather_calibration(path, min_samples=1, bias_shrinkage_k=0)[0]

    assert fitted.q05 == pytest.approx(-0.85)
    assert fitted.q25 == pytest.approx(-0.25)
    assert fitted.q50 == pytest.approx(0.5)
    assert fitted.q75 == pytest.approx(1.25)
    assert fitted.q95 == pytest.approx(1.85)


def test_horizon_hours_and_bucket() -> None:
    assert horizon_bucket(11.9) == "0_12h"
    assert horizon_bucket(12) == "12_24h"
    assert horizon_bucket(24) == "24_48h"
    assert horizon_bucket(48) == "48_72h"
    assert horizon_bucket(72) == "72h_plus"


def test_horizon_datetime_approximation_warning(tmp_path) -> None:
    path = tmp_path / "history.csv"
    item = row(target_datetime="")
    history_csv(path, [item])

    sample = load_forecast_error_samples(path)[0]

    assert sample.horizon_hours == pytest.approx(47.9833, rel=1e-3)
    assert sample.horizon_bucket == "24_48h"
    assert "target_datetime_approximated_from_date" in sample.warnings


def test_min_samples_marks_insufficient_quality(tmp_path) -> None:
    path = tmp_path / "history.csv"
    history_csv(path, [row(), row(actual_value="26")])

    fitted = fit_weather_calibration(path, min_samples=3)[0]

    assert fitted.n == 2
    assert fitted.calibration_quality == "INSUFFICIENT"
    assert "insufficient_calibration_samples" in fitted.quality_warnings


def test_calibration_quality_boundaries() -> None:
    assert calibration_quality_for_n(9, 10) == "INSUFFICIENT"
    assert calibration_quality_for_n(10, 10) == "LOW"
    assert calibration_quality_for_n(20, 10) == "MEDIUM"
    assert calibration_quality_for_n(50, 10) == "HIGH"


def test_bias_shrinkage_formula(tmp_path) -> None:
    path = tmp_path / "history.csv"
    history_csv(
        path,
        [
            row(forecast_mean="24", actual_value="25"),
            row(forecast_mean="24", actual_value="25"),
            row(forecast_mean="24", actual_value="25"),
        ],
    )

    fitted = fit_weather_calibration(path, min_samples=3, bias_shrinkage_k=30)[0]

    assert fitted.bias_raw == pytest.approx(1.0)
    assert fitted.bias_shrunk == pytest.approx(3 / 33)
    assert fitted.bias == pytest.approx(3 / 33)


def test_calibration_json_and_csv_output(tmp_path) -> None:
    summaries = [summary()]
    json_path = tmp_path / "calibration.json"
    csv_path = tmp_path / "calibration.csv"

    assert write_calibration_json(summaries, json_path) == 1
    assert write_calibration_csv(summaries, csv_path) == 1
    loaded = load_calibration_summaries(json_path)

    assert summaries[0].group_key in loaded
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == CALIBRATION_CSV_FIELDS
    assert rows[0]["group_key"] == summaries[0].group_key


def test_scan_weather_applies_calibrated_std() -> None:
    c = candidate()
    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={summary().group_key: summary()},
        use_calibrated_std=True,
    )

    assert signal.calibration_applied is True
    assert signal.forecast_std_raw == pytest.approx(0.8)
    assert signal.forecast_std_calibrated == pytest.approx(2.0)
    assert signal.std_method == "calibrated_historical_error"


def test_scan_weather_applies_calibrated_bias() -> None:
    c = candidate()
    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={summary().group_key: summary()},
        use_calibrated_bias=True,
    )

    assert signal.forecast_mean_raw == pytest.approx(24.0)
    assert signal.forecast_mean_adjusted == pytest.approx(25.0)
    assert signal.calibration_bias == pytest.approx(1.0)


def test_scan_weather_insufficient_calibration_not_applied() -> None:
    c = candidate()
    low_n_summary = summary(
        n=3,
        calibration_quality="LOW",
        min_samples_required=3,
        bias_raw=1.0,
        bias_shrunk=3 / 33,
        bias=3 / 33,
        std_error_raw=2.0,
        std_error_used=2.0,
    )

    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={low_n_summary.group_key: low_n_summary},
        use_calibrated_std=True,
        use_calibrated_bias=True,
        min_calibration_samples=30,
    )

    assert signal.calibration_n == 3
    assert signal.calibration_quality == "INSUFFICIENT"
    assert signal.calibration_applied is False
    assert signal.forecast_mean_adjusted == pytest.approx(24.0)
    assert signal.forecast_std_calibrated == pytest.approx(0.8)
    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "insufficient_calibration_samples" in signal.validation_warnings


def test_scan_weather_medium_quality_calibration_can_apply() -> None:
    c = candidate()
    med_summary = summary(n=60, calibration_quality="MEDIUM", min_samples_required=30)

    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={med_summary.group_key: med_summary},
        use_calibrated_std=True,
        use_calibrated_bias=True,
        min_calibration_samples=30,
    )

    assert signal.calibration_quality == "MEDIUM"
    assert signal.calibration_applied is True
    assert signal.forecast_mean_adjusted == pytest.approx(25.0)
    assert signal.forecast_std_calibrated == pytest.approx(2.0)


def test_missing_calibration_group_requires_manual_review() -> None:
    c = candidate()
    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={},
        use_calibrated_std=True,
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "missing_calibration_group" in signal.validation_warnings


def test_journal_rationale_contains_calibration_quality() -> None:
    c = candidate()
    med_summary = summary(n=60, calibration_quality="MEDIUM", min_samples_required=30)

    signal = build_weather_alpha_signal(
        c,
        classify_strategy_candidate(c),
        forecast(),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries={med_summary.group_key: med_summary},
        use_calibrated_std=True,
        use_calibrated_bias=True,
        min_calibration_samples=30,
    )

    assert "calibration_quality=MEDIUM" in signal.journal_draft_payload["rationale"]


def test_current_signal_actual_value_not_used_for_model_probability() -> None:
    c = candidate()
    classification = classify_strategy_candidate(c)
    summaries = {summary().group_key: summary()}
    first = build_weather_alpha_signal(
        c,
        classification,
        forecast(actual_value=10),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries=summaries,
        use_calibrated_std=True,
        use_calibrated_bias=True,
    )
    second = build_weather_alpha_signal(
        c,
        classification,
        forecast(actual_value=100),
        as_of_time="2026-05-08T01:00:00Z",
        calibration_summaries=summaries,
        use_calibrated_std=True,
        use_calibrated_bias=True,
    )

    assert first.model_p_yes == second.model_p_yes


def test_weather_calibration_fit_cli(tmp_path, capsys) -> None:
    input_path = tmp_path / "history.csv"
    output_json = tmp_path / "calibration.json"
    output_csv = tmp_path / "calibration.csv"
    history_csv(input_path, [row(), row(actual_value="26"), row(actual_value="24")])

    exit_code = cli.run(
        [
            "weather-calibration",
            "fit",
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--group-by",
            "metric,horizon_bucket",
            "--min-samples",
            "3",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Weather Calibration Fit" in captured
    assert output_json.exists()
    assert output_csv.exists()
    assert json.loads(output_json.read_text(encoding="utf-8"))[0]["n"] == 3
