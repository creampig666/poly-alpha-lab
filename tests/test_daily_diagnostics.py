import json

from poly_alpha_lab.daily_diagnostics import (
    diagnose_missing_forecasts,
    diagnose_weather_daily_captures,
    missing_forecasts_report_zh,
    write_missing_forecasts_csv,
    write_missing_forecasts_markdown,
    weather_diagnostics_report_zh,
    write_weather_diagnostics_json,
    write_weather_diagnostics_markdown,
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _strategy_candidate(market_id="weather-1", question=None):
    return {
        "market_id": market_id,
        "slug": f"{market_id}-slug",
        "question": question or "Will the highest temperature in Milan be 24°C on May 10?",
        "category": "weather",
        "yes_executable_avg_buy_price": 0.2,
        "no_executable_avg_buy_price": 0.8,
        "reasons": [],
    }


def _summary(day_dir, timestamp, **overrides):
    data = {
        "captured_at": f"{day_dir.name}T12:00:00Z",
        "strategy_candidates_path": str(day_dir / f"strategy_candidates_{timestamp}.json"),
        "weather_alpha_path": str(day_dir / f"weather_alpha_{timestamp}.json"),
        "strategy_candidates_count": 2,
        "weather_candidates_count": 1,
        "weather_alpha_signals_count": 0,
        "snapshots_inserted": 1,
        "backtest_saved": 0,
        "backtest_skipped": 0,
        "skipped_reasons": {},
        "errors": [],
        "warnings": [],
    }
    data.update(overrides)
    return data


def test_diagnostics_reads_multiple_capture_summaries(tmp_path):
    daily_dir = tmp_path / "daily"
    day1 = daily_dir / "2026-05-10"
    day2 = daily_dir / "2026-05-11"
    _write_json(day1 / "capture_summary_a.json", _summary(day1, "a"))
    _write_json(day1 / "strategy_candidates_a.json", [_strategy_candidate(), {"market_id": "non-weather", "question": "Will BTC hit 100k?"}])
    _write_json(day1 / "weather_alpha_a.json", [])
    _write_json(day2 / "capture_summary_b.json", _summary(day2, "b", strategy_candidates_count=1, weather_candidates_count=1))
    _write_json(day2 / "strategy_candidates_b.json", [_strategy_candidate("weather-2")])
    _write_json(day2 / "weather_alpha_b.json", [])

    result = diagnose_weather_daily_captures(daily_dir, days=7)

    assert len(result["runs"]) == 2
    assert result["totals"]["strategy_candidates"] == 3
    assert result["totals"]["weather_candidates"] == 2


def test_diagnostics_funnel_and_zero_alpha_report_do_not_fail(tmp_path):
    daily_dir = tmp_path / "daily"
    day = daily_dir / "2026-05-10"
    _write_json(day / "capture_summary_a.json", _summary(day, "a"))
    _write_json(day / "strategy_candidates_a.json", [_strategy_candidate(), {"market_id": "non-weather", "question": "Will BTC hit 100k?"}])
    _write_json(day / "weather_alpha_a.json", [])

    result = diagnose_weather_daily_captures(daily_dir, days=7)
    report = weather_diagnostics_report_zh(result)

    assert result["funnel"][0]["stage"] == "strategy candidates"
    assert result["failure_reason_counts"]["missing_forecast"] >= 1
    assert "没有 alpha signal，无法统计 edge" in report


def test_diagnostics_aggregates_skipped_reasons(tmp_path):
    daily_dir = tmp_path / "daily"
    day = daily_dir / "2026-05-10"
    _write_json(
        day / "capture_summary_a.json",
        _summary(
            day,
            "a",
            weather_alpha_signals_count=1,
            backtest_skipped=2,
            skipped_reasons={"no_paper_side": 1, "bucket_boundary_not_confirmed": 1},
        ),
    )
    _write_json(day / "strategy_candidates_a.json", [_strategy_candidate()])
    _write_json(
        day / "weather_alpha_a.json",
        [
            {
                "market_id": "weather-1",
                "question": "Will the highest temperature in Milan be 24°C on May 10?",
                "model_p_yes": 0.2,
                "yes_breakeven": 0.3,
                "no_upper_bound": 0.1,
                "yes_model_edge": -0.1,
                "no_model_edge": -0.1,
                "edge_threshold": 0.05,
                "suggested_paper_side": "NONE",
                "signal_status": "NEEDS_MANUAL_REVIEW",
                "validation_warnings": ["bucket_boundary_assumption_unconfirmed"],
                "bucket_numeric_boundary_confirmed": False,
            }
        ],
    )

    result = diagnose_weather_daily_captures(daily_dir, days=7)

    assert result["backtest_skipped_reasons"]["no_paper_side"] == 1
    assert result["backtest_skipped_reasons"]["bucket_boundary_not_confirmed"] == 1
    assert result["suggested_paper_side_counts"]["NONE"] == 1


def test_chinese_report_contains_required_sections(tmp_path):
    daily_dir = tmp_path / "daily"
    day = daily_dir / "2026-05-10"
    _write_json(day / "capture_summary_a.json", _summary(day, "a"))
    _write_json(day / "strategy_candidates_a.json", [_strategy_candidate()])
    _write_json(day / "weather_alpha_a.json", [])

    result = diagnose_weather_daily_captures(daily_dir, days=7)
    report = weather_diagnostics_report_zh(result)

    assert "漏斗分析" in report
    assert "主要瓶颈" in report
    assert "下一步建议" in report


def test_diagnostics_writes_markdown_and_json(tmp_path):
    daily_dir = tmp_path / "daily"
    day = daily_dir / "2026-05-10"
    _write_json(day / "capture_summary_a.json", _summary(day, "a"))
    _write_json(day / "strategy_candidates_a.json", [_strategy_candidate()])
    _write_json(day / "weather_alpha_a.json", [])
    result = diagnose_weather_daily_captures(daily_dir, days=7)

    output_json = tmp_path / "weather_diagnostics.json"
    output_md = tmp_path / "weather_diagnostics.md"
    write_weather_diagnostics_json(result, output_json)
    write_weather_diagnostics_markdown(result, output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["totals"]["weather_candidates"] == 1
    assert "Polymarket 天气 Daily Capture 诊断报告" in output_md.read_text(encoding="utf-8")


def test_missing_forecasts_report_writes_details_and_pending_locations(tmp_path):
    daily_dir = tmp_path / "daily"
    day = daily_dir / "2026-05-10"
    _write_json(day / "capture_summary_a.json", _summary(day, "a"))
    _write_json(
        day / "strategy_candidates_a.json",
        [
            _strategy_candidate(
                "weather-sbgr",
                "Will the highest temperature in Sao Paulo be 24掳C on May 10?",
            )
        ],
    )
    _write_json(day / "weather_alpha_a.json", [])
    locations = tmp_path / "locations.csv"
    locations.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes\n",
        encoding="utf-8",
    )
    pending = tmp_path / "locations_pending.csv"

    result = diagnose_missing_forecasts(
        daily_dir,
        days=7,
        locations_file=locations,
        pending_output=pending,
    )
    report = missing_forecasts_report_zh(result)

    assert result["records"][0]["reason"] == "location_not_in_locations_csv"
    assert "Top missing locations/stations" in report
    assert pending.exists()


def test_missing_forecasts_writes_markdown_and_csv(tmp_path):
    result = {
        "days": 7,
        "records": [
            {
                "date": "2026-05-10",
                "captured_at": "2026-05-10T12:00:00Z",
                "market_id": "weather-1",
                "slug": "weather-1",
                "question": "Will the highest temperature in Milan be 24掳C on May 10?",
                "category": "weather",
                "parsed_market_type": "weather_temperature_exact_bucket",
                "parsed_location_name": "Milan",
                "metric": "high_temperature",
                "target_date": "2026-05-10",
                "threshold": 24,
                "range_lower": None,
                "range_upper": None,
                "unit": "C",
                "resolution_source": "",
                "extracted_station_id": "LIMC",
                "extracted_station_name": "Milan Malpensa Intl Airport Station",
                "forecast_lookup_key": "LIMC:2026-05-10:high_temperature",
                "locations_csv_match": "no",
                "reason": "station_not_in_locations_csv",
            }
        ],
        "reason_counts": {"station_not_in_locations_csv": 1},
        "locations_pending_path": str(tmp_path / "locations_pending.csv"),
        "locations_pending_new_or_updated": 1,
        "top_missing_locations_or_stations": [
            {
                "location_or_station": "LIMC",
                "count": 1,
                "example_question": "Will the highest temperature in Milan be 24掳C on May 10?",
                "suggested_action": "人工补 station_id 对应机场站点经纬度",
            }
        ],
    }
    output_csv = tmp_path / "missing.csv"
    output_md = tmp_path / "missing.md"

    write_missing_forecasts_csv(result, output_csv)
    write_missing_forecasts_markdown(result, output_md)

    assert "station_not_in_locations_csv" in output_csv.read_text(encoding="utf-8")
    assert "天气 Forecast 缺失明细诊断" in output_md.read_text(encoding="utf-8")
