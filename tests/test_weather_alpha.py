import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.market_type_classifier import classify_strategy_candidate
from poly_alpha_lab.models import Market
from poly_alpha_lab.weather_alpha import (
    build_weather_alpha_signal,
    run_weather_alpha_scan,
    weather_alpha_report,
    write_weather_alpha_signals_json,
)
from poly_alpha_lab.weather_data import CsvWeatherDataProvider, WeatherForecast
from poly_alpha_lab.resolution_analyzer import ResolutionAnalysis


TEST_AS_OF_TIME = "2026-05-08T01:00:00Z"


def strategy_candidate(**overrides):
    data = {
        "market_id": "weather-1",
        "slug": "sao-paulo-high-24c-or-below",
        "question": "Will the highest temperature in Sao Paulo be 24\u00b0C or below on May 9?",
        "category": "weather",
        "candidate_score": 82.0,
        "candidate_grade": "A",
        "strategy_score": 70.0,
        "yes_breakeven_probability": 0.4,
        "no_required_yes_probability_upper_bound": 0.3,
        "journal_draft_payload_yes": {
            "market_id": "weather-1",
            "question": "Will the highest temperature in Sao Paulo be 24\u00b0C or below on May 9?",
            "category": "weather",
            "end_date": "2026-05-09T23:59:00+00:00",
            "candidate_score": 82.0,
            "candidate_grade": "A",
            "side": "YES",
            "entry_price": 0.39,
            "entry_size": 10,
            "fee_per_share": 0.01,
        },
        "journal_draft_payload_no": {
            "market_id": "weather-1",
            "question": "Will the highest temperature in Sao Paulo be 24\u00b0C or below on May 9?",
            "category": "weather",
            "end_date": "2026-05-09T23:59:00+00:00",
            "candidate_score": 82.0,
            "candidate_grade": "A",
            "side": "NO",
            "entry_price": 0.69,
            "entry_size": 10,
            "fee_per_share": 0.01,
        },
    }
    data.update(overrides)
    return data


def forecast(**overrides):
    data = {
        "date": "2026-05-09",
        "location": "Sao Paulo",
        "metric": "high_temperature",
        "forecast_mean": 22.0,
        "forecast_std": 1.0,
        "actual_value": 31.0,
        "unit": "C",
        "forecast_issued_at": "2026-05-07T00:00:00Z",
        "latitude": -23.5505,
        "longitude": -46.6333,
        "forecast_source": "open_meteo",
        "forecast_model": "deterministic_plus_error",
        "std_method": "historical_error",
        "actual_source": "weather_station",
    }
    data.update(overrides)
    return WeatherForecast(**data)


def test_weather_alpha_yes_edge_calculation() -> None:
    candidate = strategy_candidate()
    classification = classify_strategy_candidate(candidate)

    signal = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(),
        edge_threshold=0.05,
        as_of_time=TEST_AS_OF_TIME,
    )

    assert signal.suggested_paper_side == "YES"
    assert signal.yes_model_edge == pytest.approx(signal.model_p_yes - signal.yes_breakeven)
    assert signal.journal_draft_payload["side"] == "YES"
    assert signal.journal_draft_payload["probability_source"] == "weather_threshold_model"


def test_weather_alpha_no_edge_calculation() -> None:
    candidate = strategy_candidate(
        question="Will the highest temperature in Paris be above 25\u00b0C on May 10?",
        slug="paris-high-above-25c",
        yes_breakeven_probability=0.4,
        no_required_yes_probability_upper_bound=0.6,
    )
    candidate["journal_draft_payload_yes"]["end_date"] = "2026-05-10T23:59:00+00:00"
    candidate["journal_draft_payload_no"]["end_date"] = "2026-05-10T23:59:00+00:00"
    classification = classify_strategy_candidate(candidate)

    signal = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(
            date="2026-05-10",
            location="Paris",
            forecast_mean=20,
            forecast_std=1,
        ),
        edge_threshold=0.05,
        as_of_time=TEST_AS_OF_TIME,
    )

    assert signal.suggested_paper_side == "NO"
    assert signal.no_model_edge == pytest.approx(signal.no_upper_bound - signal.model_p_yes)
    assert signal.journal_draft_payload["side"] == "NO"


def test_weather_alpha_side_none_when_edge_below_threshold() -> None:
    candidate = strategy_candidate(
        yes_breakeven_probability=0.48,
        no_required_yes_probability_upper_bound=0.52,
    )
    classification = classify_strategy_candidate(candidate)

    signal = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(forecast_mean=24, forecast_std=1),
        edge_threshold=0.05,
        as_of_time=TEST_AS_OF_TIME,
    )

    assert signal.suggested_paper_side == "NONE"
    assert signal.journal_draft_payload == {}


def test_weather_alpha_generates_signal_for_exact_bucket_candidate() -> None:
    candidate = strategy_candidate(
        question="Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        slug="sao-paulo-high-24c",
        yes_breakeven_probability=0.2,
        no_required_yes_probability_upper_bound=0.1,
    )
    classification = classify_strategy_candidate(candidate)

    signal = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(forecast_mean=24, forecast_std=1),
        edge_threshold=0.05,
        bucket_mode="rounded",
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.suggested_paper_side == "YES"
    assert signal.bucket_mode == "rounded"
    assert signal.bucket_lower_bound == pytest.approx(23.5)
    assert signal.bucket_upper_bound == pytest.approx(24.5)
    assert "integer bucket" in signal.journal_draft_payload["rationale"]
    assert "bucket_interval=[23.5, 24.5)C" in signal.journal_draft_payload["rationale"]
    assert "requires manual validation before paper entry" in signal.journal_draft_payload["rationale"]


def test_run_weather_alpha_scan_with_csv(tmp_path) -> None:
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(json.dumps([strategy_candidate()]), encoding="utf-8")
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-09,Sao Paulo,high_temperature,22,1,31,C,2026-05-07T00:00:00Z\n",
        encoding="utf-8",
    )

    result = run_weather_alpha_scan(
        strategy_path,
        CsvWeatherDataProvider(csv_path),
        as_of_time=TEST_AS_OF_TIME,
    )

    assert result.weather_candidate_count == 1
    assert result.threshold_candidate_count == 1
    assert result.exact_bucket_candidate_count == 0
    assert len(result.signals) == 1
    assert result.signals[0].location_name == "Sao Paulo"


def test_run_weather_alpha_scan_counts_exact_bucket_candidate(tmp_path) -> None:
    strategy_path = tmp_path / "strategy.json"
    candidate = strategy_candidate(
        question="Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        slug="sao-paulo-high-24c",
    )
    strategy_path.write_text(json.dumps([candidate]), encoding="utf-8")
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-09,Sao Paulo,high_temperature,24,1,31,C,2026-05-07T00:00:00Z\n",
        encoding="utf-8",
    )

    result = run_weather_alpha_scan(
        strategy_path,
        CsvWeatherDataProvider(csv_path),
        as_of_time=TEST_AS_OF_TIME,
    )

    assert result.weather_candidate_count == 1
    assert result.threshold_candidate_count == 0
    assert result.exact_bucket_candidate_count == 1
    assert len(result.signals) == 1


def test_weather_alpha_output_json_works(tmp_path) -> None:
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(json.dumps([strategy_candidate()]), encoding="utf-8")
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-09,Sao Paulo,high_temperature,22,1,31,C,2026-05-07T00:00:00Z\n",
        encoding="utf-8",
    )
    result = run_weather_alpha_scan(
        strategy_path,
        CsvWeatherDataProvider(csv_path),
        as_of_time=TEST_AS_OF_TIME,
    )
    output = tmp_path / "signals.json"

    count = write_weather_alpha_signals_json(result, output)

    assert count == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data[0]["suggested_paper_side"] == "YES"


def test_weather_alpha_report_uses_paper_side_not_buy_sell() -> None:
    candidate = strategy_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(),
        edge_threshold=0.05,
        as_of_time=TEST_AS_OF_TIME,
    )

    report = weather_alpha_report(type("Result", (), {"weather_candidate_count": 1, "signals": [signal], "skipped": []})())

    assert "suggested_paper_side" in report
    assert "BUY_YES" not in report
    assert "BUY_NO" not in report


def test_alpha_scan_weather_cli_runs_with_csv(tmp_path, capsys, monkeypatch) -> None:
    class FakeGammaClient:
        def get_market(self, market_id: str) -> Market:
            return Market.model_validate(
                {
                    "id": market_id,
                    "question": "Will the highest temperature in Sao Paulo be 24\u00b0C or below on May 9?",
                    "slug": "sao-paulo",
                    "category": "Weather",
                    "active": True,
                    "closed": False,
                    "enableOrderBook": True,
                    "outcomes": ["Yes", "No"],
                    "outcomePrices": [0.4, 0.6],
                    "clobTokenIds": ["yes-token", "no-token"],
                    "resolutionCriteria": (
                        "This market will resolve to Yes if the official NOAA website reports "
                        "a high temperature at or below 24C by 11:59 PM ET on May 9. "
                        "Otherwise, this market resolves to No."
                    ),
                }
            )

    monkeypatch.setattr(cli, "GammaClient", FakeGammaClient)
    strategy_path = tmp_path / "strategy.json"
    strategy_path.write_text(json.dumps([strategy_candidate()]), encoding="utf-8")
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-09,Sao Paulo,high_temperature,22,1,31,C,2026-05-07T00:00:00Z\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "signals.json"

    exit_code = cli.run(
        [
            "alpha",
            "scan-weather",
            "--strategy-json",
            str(strategy_path),
            "--weather-data",
            str(csv_path),
            "--output-json",
            str(output_path),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Weather alpha signals generated: `1`" in captured
    assert output_path.exists()


def resolution_analysis(**overrides):
    data = {
        "market_id": "weather-1",
        "slug": "weather",
        "question": "Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        "category": "weather",
        "has_resolution_text": True,
        "resolution_text_source": 'market.raw["description"]',
        "resolution_text_excerpt": (
            "This market will resolve to the temperature range that contains the highest "
            "temperature recorded at the Malpensa Intl Airport Station."
        ),
        "what_counts_as_yes": "This market can not resolve to Yes until data is finalized.",
        "what_counts_as_no": None,
        "resolution_source": (
            "The resolution source is Wunderground for the Malpensa Intl Airport Station, "
            "available at https://www.wunderground.com/history/daily/it/milan/LIMC."
        ),
        "deadline": "The highest temperature recorded on May 9.",
        "time_zone": None,
        "ambiguity_risk": "LOW",
        "dispute_risk": "LOW",
        "risk_score": 20.0,
        "critical_phrases": [],
        "missing_fields": [],
        "warnings": [],
        "research_notes": [],
    }
    data.update(overrides)
    return ResolutionAnalysis(**data)


def exact_bucket_candidate(**overrides):
    candidate = strategy_candidate(
        question="Will the highest temperature in Milan be 24\u00b0C on May 8?",
        slug="milan-high-24c",
        yes_breakeven_probability=0.2,
        no_required_yes_probability_upper_bound=0.1,
    )
    candidate["journal_draft_payload_yes"]["end_date"] = "2026-05-08T12:00:00+00:00"
    candidate["journal_draft_payload_no"]["end_date"] = "2026-05-08T12:00:00+00:00"
    candidate.update(overrides)
    return candidate


def test_forecast_issued_after_as_of_time_sets_stale_and_side_none() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_mean=24,
            forecast_std=1,
            forecast_issued_at="2026-05-08T02:00:00Z",
        ),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    assert signal.signal_status == "STALE_OR_FUTURE_FORECAST"
    assert signal.suggested_paper_side == "NONE"
    assert "forecast_issued_after_as_of_time" in signal.validation_warnings
    assert signal.journal_draft_payload["side"] == "NONE"


def test_forecast_issued_within_live_tolerance_is_not_stale() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_mean=24,
            forecast_std=1,
            forecast_issued_at="2026-05-08T01:00:30Z",
            actual_value=None,
        ),
        generated_at=datetime(2026, 5, 8, 1, 0, 0, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
        forecast_time_tolerance_seconds=120,
    )

    assert signal.signal_status != "STALE_OR_FUTURE_FORECAST"
    assert "forecast_issued_within_live_capture_tolerance" in signal.validation_warnings
    assert "forecast_issued_after_as_of_time" not in signal.validation_warnings
    assert signal.forecast_timing_tolerance_applied is True
    assert signal.run_mode == "live"


def test_forecast_issued_beyond_live_tolerance_stays_stale() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_mean=24,
            forecast_std=1,
            forecast_issued_at="2026-05-08T01:10:00Z",
            actual_value=None,
        ),
        generated_at=datetime(2026, 5, 8, 1, 0, 0, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
        forecast_time_tolerance_seconds=120,
    )

    assert signal.signal_status == "STALE_OR_FUTURE_FORECAST"
    assert "forecast_issued_after_as_of_time" in signal.validation_warnings
    assert signal.forecast_timing_tolerance_applied is False


def test_replay_mode_still_requires_forecast_issued_before_as_of_time() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_mean=24,
            forecast_std=1,
            forecast_issued_at="2026-05-08T01:00:30Z",
            actual_value=None,
        ),
        generated_at=datetime(2026, 5, 8, 1, 0, 5, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.run_mode == "replay"
    assert signal.signal_status == "STALE_OR_FUTURE_FORECAST"
    assert "forecast_issued_after_as_of_time" in signal.validation_warnings


def test_missing_forecast_issued_at_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", forecast_issued_at=None),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "missing_forecast_issued_at" in signal.validation_warnings


def test_missing_forecast_source_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_source=None,
            actual_value=None,
        ),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "missing_forecast_source" in signal.data_provenance_warnings


def test_missing_location_mapping_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", location_mapping_found=False),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "missing_location_mapping" in signal.validation_warnings


def test_sample_forecast_source_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            forecast_source="sample",
            actual_value=None,
        ),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "manual_or_sample_forecast_source" in signal.data_provenance_warnings


def test_missing_std_method_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", std_method=None, actual_value=None),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "missing_std_method" in signal.data_provenance_warnings


def test_manual_std_assumption_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            std_method="manual_assumption",
            actual_value=None,
        ),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "manual_std_assumption" in signal.data_provenance_warnings


def test_fallback_std_warning_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(
            location="Milan",
            date="2026-05-08",
            std_method="fallback_error_std",
            actual_value=None,
        ),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "fallback_std_used" in signal.data_provenance_warnings


def test_resolution_risk_medium_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", forecast_mean=24, forecast_std=1),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(ambiguity_risk="MEDIUM", risk_score=38.0),
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "resolution_risk_not_low" in signal.validation_warnings
    assert signal.confidence <= 0.6


def test_wunderground_station_source_without_csv_station_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", station_id=None),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert signal.forecast_station_id is None
    assert signal.resolution_station_id == "LIMC"
    assert signal.station_id is None
    assert "missing_forecast_station_id_for_resolution_station" in signal.validation_warnings
    assert "missing_station_id_for_resolution_source" in signal.validation_warnings
    assert "missing_station_id_for_station_resolution" in signal.data_provenance_warnings


def test_matching_csv_station_id_avoids_station_mismatch_warning() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", station_id="LIMC"),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    assert "missing_station_id_for_resolution_source" not in signal.validation_warnings
    assert "station_id_mismatch" not in signal.validation_warnings
    assert signal.forecast_station_id == "LIMC"
    assert signal.resolution_station_id == "LIMC"
    assert signal.station_id == "LIMC"


def test_exact_bucket_without_explicit_boundary_requires_manual_review() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", forecast_mean=24, forecast_std=1),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(resolution_text_excerpt="Temperature resolves by source."),
    )

    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "bucket_boundary_assumption_unconfirmed" in signal.validation_warnings


def test_exact_bucket_range_structure_without_numeric_boundary_is_split() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08"),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    assert signal.bucket_structure_confirmed is True
    assert signal.bucket_numeric_boundary_confirmed is False
    assert signal.bucket_boundary_confirmed is False
    assert "bucket_boundary_inferred_not_explicit" in signal.validation_warnings


def test_milan_like_case_report_does_not_mix_forecast_and_resolution_station() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", station_id=None),
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    report = weather_alpha_report(
        type("Result", (), {"weather_candidate_count": 1, "signals": [signal], "skipped": []})()
    )

    assert "- forecast_station_id: `n/a`" in report
    assert "- resolution_station_id: `LIMC`" in report


def test_invalid_data_has_no_yes_no_journal_draft() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08"),
        as_of_time="2026-05-09T01:00:00Z",
        resolution_analysis=resolution_analysis(),
    )

    assert signal.signal_status == "INVALID_DATA"
    assert signal.suggested_paper_side == "NONE"
    assert signal.journal_draft_payload["side"] == "NONE"
    assert "entry_price" not in signal.journal_draft_payload


def test_generated_at_and_as_of_time_are_written_to_json(tmp_path) -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08"),
        generated_at=datetime(2026, 5, 8, 1, 30, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
    )
    output = tmp_path / "signals.json"

    write_weather_alpha_signals_json(
        type(
            "Result",
            (),
            {"signals": [signal]},
        )(),
        output,
    )
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data[0]["generated_at"] == "2026-05-08T01:30:00Z"
    assert data[0]["as_of_time"] == "2026-05-08T01:00:00Z"


def test_run_mode_replay_with_as_of_after_generated_at_adds_warning() -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", forecast_issued_at="2026-05-07T00:00:00Z"),
        generated_at=datetime(2026, 5, 8, 0, 30, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert signal.run_mode == "replay"
    assert "as_of_time_after_generated_at_replay_mode" in signal.validation_warnings


def test_weather_alpha_json_contains_forecast_provenance_fields(tmp_path) -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08"),
        generated_at=datetime(2026, 5, 8, 1, 30, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
    )
    output = tmp_path / "signals.json"

    write_weather_alpha_signals_json(type("Result", (), {"signals": [signal]})(), output)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data[0]["forecast_source"] == "open_meteo"
    assert data[0]["std_method"] == "historical_error"
    assert "data_provenance_warnings" in data[0]


def test_weather_alpha_json_contains_weather_model_fields(tmp_path) -> None:
    candidate = exact_bucket_candidate()
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08"),
        generated_at=datetime(2026, 5, 8, 1, 30, tzinfo=UTC),
        as_of_time="2026-05-08T01:00:00Z",
        weather_model="student_t",
        student_t_df=5,
    )
    output = tmp_path / "signals.json"

    write_weather_alpha_signals_json(type("Result", (), {"signals": [signal]})(), output)
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data[0]["weather_model"] == "student_t"
    assert data[0]["model_parameters"]["student_t_df"] == 5
    assert "student_t forecast error" in data[0]["distribution_assumption"]


def test_weather_alpha_journal_draft_rationale_contains_weather_model() -> None:
    candidate = exact_bucket_candidate(yes_breakeven_probability=0.05)
    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        forecast(location="Milan", date="2026-05-08", forecast_mean=24, forecast_std=1),
        as_of_time="2026-05-08T01:00:00Z",
        weather_model="normal_mixture",
        mixture_tail_weight=0.1,
        mixture_tail_scale=2.5,
    )

    rationale = signal.journal_draft_payload["rationale"]
    assert "weather_model=normal_mixture" in rationale
    assert "mixture_tail_weight" in rationale
    assert "normal mixture forecast error" in rationale


def test_actual_value_still_not_used_in_alpha_probability() -> None:
    candidate = exact_bucket_candidate()
    classification = classify_strategy_candidate(candidate)
    first = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(location="Milan", date="2026-05-08", actual_value=10),
        as_of_time="2026-05-08T01:00:00Z",
    )
    second = build_weather_alpha_signal(
        candidate,
        classification,
        forecast(location="Milan", date="2026-05-08", actual_value=100),
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert first.model_p_yes == second.model_p_yes
    assert "actual_value_present_not_used" in first.data_provenance_warnings


def test_sample_milan_csv_row_remains_needs_manual_review() -> None:
    provider = CsvWeatherDataProvider(Path("data/weather/weather_forecasts.csv"))
    csv_forecast = provider.get_forecast("Milan", "2026-05-08", "high_temperature")
    assert csv_forecast is not None
    candidate = exact_bucket_candidate()

    signal = build_weather_alpha_signal(
        candidate,
        classify_strategy_candidate(candidate),
        csv_forecast,
        as_of_time="2026-05-08T01:00:00Z",
        resolution_analysis=resolution_analysis(ambiguity_risk="MEDIUM", dispute_risk="MEDIUM", risk_score=38),
    )

    assert signal.forecast_source == "sample"
    assert signal.std_method == "manual_assumption"
    assert signal.signal_status == "NEEDS_MANUAL_REVIEW"
    assert "manual_or_sample_forecast_source" in signal.data_provenance_warnings
    assert "manual_std_assumption" in signal.data_provenance_warnings
