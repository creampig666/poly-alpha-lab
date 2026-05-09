import csv
import json
import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.network_debug import (
    classify_network_error,
    mask_proxy_url,
    run_network_debug,
)
from poly_alpha_lab.weather_calibration import fit_weather_calibration, load_forecast_error_samples
from poly_alpha_lab.weather_data import LocationMapping
from poly_alpha_lab.weather_dataset_builder import (
    DATASET_CSV_FIELDS,
    OpenMeteoHistoricalDatasetProvider,
    build_provider_semantics_audit,
    build_one_sample,
    build_weather_dataset,
    collect_weather_dataset_samples,
    debug_open_meteo_provider,
    validate_manual_forecast_actual_csv,
    write_manual_forecast_actual_template,
    write_provider_semantics_audit,
    weather_dataset_cache_key,
)


def locations_csv(path, rows=None):
    rows = rows or [
        {
            "location_name": "Milan",
            "latitude": "45.6306",
            "longitude": "8.7281",
            "station_id": "LIMC",
            "source_location_name": "Malpensa Intl Airport Station",
            "timezone": "Europe/Rome",
            "notes": "test",
            "default_forecast_std": "0.8",
            "std_source": "test",
        }
    ]
    fields = [
        "location_name",
        "latitude",
        "longitude",
        "station_id",
        "source_location_name",
        "timezone",
        "notes",
        "default_forecast_std",
        "std_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def open_meteo_raw(value, *, source=None, date="2026-04-02"):
    raw = {
        "daily": {
            "time": [date],
            "temperature_2m_max": [value],
            "temperature_2m_min": [value - 8 if value is not None else None],
        }
    }
    if source is not None:
        raw["source"] = source
    return raw


def provider_with(fetcher, tmp_path):
    return OpenMeteoHistoricalDatasetProvider(cache_dir=tmp_path / "cache", fetcher=fetcher)


def mapping(**overrides):
    data = {
        "location_name": "Milan",
        "latitude": 45.6306,
        "longitude": 8.7281,
        "station_id": "LIMC",
        "source_location_name": "Malpensa Intl Airport Station",
        "timezone": "Europe/Rome",
    }
    data.update(overrides)
    return LocationMapping(**data)


def test_builder_does_not_use_actual_as_forecast(tmp_path) -> None:
    def fetcher(request_type, params):
        if request_type == "forecast":
            return open_meteo_raw(24, source="open_meteo_archive", date=params["start_date"])
        return open_meteo_raw(25, date=params["start_date"])

    sample, reason = build_one_sample(
        provider=provider_with(fetcher, tmp_path),
        mapping=mapping(),
        metric="high_temperature",
        target_datetime=datetime(2026, 4, 2, 12, tzinfo=UTC),
        forecast_issued_at=datetime(2026, 4, 1, 12, tzinfo=UTC),
    )

    assert sample is None
    assert reason == "forecast_response_looks_like_actual_archive"


def test_missing_forecast_mean_is_skipped(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(
            None if request_type == "forecast" else 25,
            date=params["start_date"],
        )

    sample, reason = build_one_sample(
        provider=provider_with(fetcher, tmp_path),
        mapping=mapping(),
        metric="high_temperature",
        target_datetime=datetime(2026, 4, 2, 12, tzinfo=UTC),
        forecast_issued_at=datetime(2026, 4, 1, 12, tzinfo=UTC),
    )

    assert sample is None
    assert reason == "missing_forecast_mean"


def test_missing_actual_value_is_skipped(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(
            24 if request_type == "forecast" else None,
            date=params["start_date"],
        )

    sample, reason = build_one_sample(
        provider=provider_with(fetcher, tmp_path),
        mapping=mapping(),
        metric="high_temperature",
        target_datetime=datetime(2026, 4, 2, 12, tzinfo=UTC),
        forecast_issued_at=datetime(2026, 4, 1, 12, tzinfo=UTC),
    )

    assert sample is None
    assert reason == "missing_actual_value"


def test_horizon_hours_calculated_correctly(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    issued_at = datetime(2026, 4, 1, 12, tzinfo=UTC)
    sample, reason = build_one_sample(
        provider=provider_with(fetcher, tmp_path),
        mapping=mapping(),
        metric="high_temperature",
        target_datetime=issued_at + timedelta(hours=36),
        forecast_issued_at=issued_at,
    )

    assert reason is None
    assert sample is not None
    assert sample.horizon_hours == pytest.approx(36)


def test_forecast_issued_at_after_target_is_skipped(tmp_path) -> None:
    sample, reason = build_one_sample(
        provider=provider_with(lambda request_type, params: open_meteo_raw(24, date=params["start_date"]), tmp_path),
        mapping=mapping(),
        metric="high_temperature",
        target_datetime=datetime(2026, 4, 2, 12, tzinfo=UTC),
        forecast_issued_at=datetime(2026, 4, 2, 12, tzinfo=UTC),
    )

    assert sample is None
    assert reason == "forecast_issued_at_not_before_target_datetime"


def test_cache_key_contains_request_type_and_forecast_issued_at() -> None:
    key = weather_dataset_cache_key(
        provider="open_meteo",
        location="Milan",
        metric="high_temperature",
        target_date="2026-04-02",
        forecast_issued_at="2026-04-01T12:00:00Z",
        horizon_hours=24,
        request_type="forecast",
    )

    assert "forecast" in key
    assert "2026-04-01t12-00-00-00-00" in key


def test_output_csv_fields_complete(tmp_path) -> None:
    output = tmp_path / "history.csv"
    locs = tmp_path / "locations.csv"
    locations_csv(locs)

    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    summary = build_weather_dataset(
        locations_file=locs,
        output_path=output,
        provider=provider_with(fetcher, tmp_path),
        start_date="2026-04-01",
        end_date="2026-04-01",
        metrics=["high_temperature"],
        forecast_issue_hours=[0],
        horizons=[24],
    )

    assert summary.samples_generated == 1
    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == DATASET_CSV_FIELDS
    assert rows[0]["forecast_source"] == "open_meteo"
    assert rows[0]["actual_source"] == "open_meteo_archive"
    assert rows[0]["forecast_source"] != rows[0]["actual_source"]


def test_mock_open_meteo_response_generates_samples(tmp_path) -> None:
    locs = tmp_path / "locations.csv"
    locations_csv(locs)

    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    result = collect_weather_dataset_samples(
        locations_file=locs,
        provider=provider_with(fetcher, tmp_path),
        start_date="2026-04-01",
        end_date="2026-04-02",
        metrics=["high_temperature"],
        forecast_issue_hours=[0, 12],
        horizons=[12, 24],
    )

    assert len(result.samples) == 8
    assert not result.skipped_reasons


def test_generated_csv_can_be_read_by_calibration_fit(tmp_path) -> None:
    output = tmp_path / "history.csv"
    locs = tmp_path / "locations.csv"
    locations_csv(locs)

    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    build_weather_dataset(
        locations_file=locs,
        output_path=output,
        provider=provider_with(fetcher, tmp_path),
        start_date="2026-04-01",
        end_date="2026-04-03",
        metrics=["high_temperature"],
        forecast_issue_hours=[0],
        horizons=[24],
    )

    samples = load_forecast_error_samples(output)
    summaries = fit_weather_calibration(output, min_samples=30)

    assert len(samples) == 3
    assert summaries[0].n == 3
    assert summaries[0].calibration_quality == "INSUFFICIENT"


def test_dataset_build_cli_runs_with_mocked_provider(tmp_path, monkeypatch, capsys) -> None:
    locs = tmp_path / "locations.csv"
    output = tmp_path / "history.csv"
    locations_csv(locs)

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        def get_forecast_mean(self, **kwargs):
            return 24.0, "open_meteo", "open_meteo_historical_forecast", "forecast-key"

        def get_actual_value(self, **kwargs):
            return 25.0, "open_meteo_archive", "actual-key"

    monkeypatch.setattr(cli, "OpenMeteoHistoricalDatasetProvider", FakeProvider)

    exit_code = cli.run(
        [
            "weather-dataset",
            "build",
            "--locations-file",
            str(locs),
            "--output",
            str(output),
            "--provider",
            "open-meteo",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-01",
            "--metrics",
            "high_temperature",
            "--forecast-issue-hours",
            "0",
            "--horizons",
            "24",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "Weather Dataset Build" in captured
    assert "samples_generated: `1`" in captured
    assert output.exists()


def test_debug_provider_mock_success_outputs_metadata(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    report = debug_open_meteo_provider(
        location="Milan",
        latitude=45.4642,
        longitude=9.19,
        target_date="2026-04-01",
        forecast_issued_at="2026-03-31T12:00:00Z",
        metric="high_temperature",
        horizon=24,
        cache_dir=tmp_path / "cache",
        fetcher=fetcher,
    )

    assert report.status == "SUCCESS"
    assert report.forecast_mean_extracted == pytest.approx(24)
    assert report.actual_value_extracted == pytest.approx(25)
    assert report.response_top_level_keys
    assert "historical_forecast_semantics_uncertain" in report.warnings


def test_debug_provider_network_failure_outputs_status(tmp_path) -> None:
    def fetcher(request_type, params):
        raise httpx.ConnectError("network unavailable")

    report = debug_open_meteo_provider(
        location="Milan",
        latitude=45.4642,
        longitude=9.19,
        target_date="2026-04-01",
        forecast_issued_at="2026-03-31T12:00:00Z",
        metric="high_temperature",
        horizon=24,
        cache_dir=tmp_path / "cache",
        fetcher=fetcher,
    )

    assert report.status == "NETWORK_FAILED"
    assert report.exception_type == "ConnectError"


def test_debug_provider_accepts_proxy_and_trust_env(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    report = debug_open_meteo_provider(
        location="Milan",
        latitude=45.4642,
        longitude=9.19,
        target_date="2026-04-01",
        forecast_issued_at="2026-03-31T12:00:00Z",
        metric="high_temperature",
        horizon=24,
        cache_dir=tmp_path / "cache",
        proxy="http://user:secret@127.0.0.1:7890",
        trust_env=False,
        fetcher=fetcher,
    )

    assert report.status == "SUCCESS"
    assert report.trust_env is False
    assert report.proxy_used == "http://user:***@127.0.0.1:7890"


def test_open_meteo_provider_accepts_proxy_and_trust_env(tmp_path) -> None:
    provider = OpenMeteoHistoricalDatasetProvider(
        cache_dir=tmp_path / "cache",
        fetcher=lambda request_type, params: open_meteo_raw(24, date=params["start_date"]),
        proxy="http://proxy.example:8080",
        trust_env=False,
    )

    assert provider.proxy == "http://proxy.example:8080"
    assert provider.trust_env is False


def test_build_network_failure_does_not_leave_real_smoke_csv(tmp_path, monkeypatch, capsys) -> None:
    locs = tmp_path / "locations.csv"
    output = tmp_path / "real_smoke.csv"
    locations_csv(locs)

    class FailingProvider:
        def __init__(self, *args, **kwargs):
            pass

        def get_forecast_mean(self, **kwargs):
            raise httpx.ConnectError("network unavailable")

    monkeypatch.setattr(cli, "OpenMeteoHistoricalDatasetProvider", FailingProvider)

    exit_code = cli.run(
        [
            "weather-dataset",
            "build",
            "--locations-file",
            str(locs),
            "--output",
            str(output),
            "--provider",
            "open-meteo",
            "--start-date",
            "2026-04-01",
            "--end-date",
            "2026-04-01",
            "--metrics",
            "high_temperature",
            "--forecast-issue-hours",
            "0",
            "--horizons",
            "24",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--audit-output",
            str(tmp_path / "audit.json"),
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "smoke_status: `NETWORK_FAILED`" in captured
    assert not output.exists()


def test_debug_provider_sample_response_not_marked_real(tmp_path) -> None:
    def fetcher(request_type, params):
        return open_meteo_raw(
            24 if request_type == "forecast" else 25,
            source="sample" if request_type == "forecast" else "open_meteo_archive",
            date=params["start_date"],
        )

    report = debug_open_meteo_provider(
        location="Milan",
        latitude=45.4642,
        longitude=9.19,
        target_date="2026-04-01",
        forecast_issued_at="2026-03-31T12:00:00Z",
        metric="high_temperature",
        horizon=24,
        cache_dir=tmp_path / "cache",
        fetcher=fetcher,
    )

    assert report.forecast_source == "sample"
    assert "mock_or_sample_response_not_real" in report.warnings


def test_connection_refused_classification() -> None:
    exc = OSError("[WinError 10061] connection refused")

    assert classify_network_error(exc) == "CONNECTION_REFUSED"


def test_debug_network_dns_failure_mock() -> None:
    def bad_dns(*args, **kwargs):
        raise socket.gaierror("getaddrinfo failed")

    report = run_network_debug(
        url="https://historical-forecast-api.open-meteo.com/v1/forecast",
        getaddrinfo=bad_dns,
    )

    assert report.dns_status == "FAILED"
    assert report.error_classification == "DNS_FAILED"


def test_debug_network_tcp_refused_mock() -> None:
    def dns(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    def refused(*args, **kwargs):
        raise OSError("[WinError 10061] connection refused")

    report = run_network_debug(
        url="https://historical-forecast-api.open-meteo.com/v1/forecast",
        getaddrinfo=dns,
        create_connection=refused,
        trust_env=False,
    )

    assert report.tcp_status == "FAILED"
    assert report.error_classification == "CONNECTION_REFUSED"


def test_proxy_url_masking() -> None:
    assert mask_proxy_url("http://user:secret@127.0.0.1:7890") == "http://user:***@127.0.0.1:7890"


def test_debug_network_passes_trust_env_to_httpx() -> None:
    calls = []

    class Conn:
        def close(self):
            pass

    def dns(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

    def http_get(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(200, json={"daily": {"time": ["2026-04-01"], "temperature_2m_max": [24]}})

    run_network_debug(
        url="https://historical-forecast-api.open-meteo.com/v1/forecast",
        getaddrinfo=dns,
        create_connection=lambda *args, **kwargs: Conn(),
        http_get=http_get,
        trust_env=False,
        proxy="http://proxy.example:8080",
    )

    assert calls
    assert all(call["trust_env"] is False for call in calls)
    assert all(call["proxy"] == "http://proxy.example:8080" for call in calls)


def test_manual_template_and_validation_reject_missing_data(tmp_path) -> None:
    locs = tmp_path / "locations.csv"
    output = tmp_path / "manual_template.csv"
    locations_csv(locs)

    rows = write_manual_forecast_actual_template(
        output_path=output,
        locations_file=locs,
        rows_per_location=1,
    )
    summary = validate_manual_forecast_actual_csv(output)

    assert rows == 1
    assert output.exists()
    assert summary.valid_rows == 0
    assert summary.can_use_for_calibration is False
    assert summary.errors["missing_forecast_mean"] == 1
    assert summary.errors["missing_actual_value"] == 1


def test_provider_semantics_audit_json_records_fields(tmp_path) -> None:
    output = tmp_path / "history.csv"
    audit_path = tmp_path / "audit.json"
    locs = tmp_path / "locations.csv"
    cache_dir = tmp_path / "cache"
    locations_csv(locs)

    def fetcher(request_type, params):
        return open_meteo_raw(24 if request_type == "forecast" else 25, date=params["start_date"])

    summary = build_weather_dataset(
        locations_file=locs,
        output_path=output,
        provider=OpenMeteoHistoricalDatasetProvider(cache_dir=cache_dir, fetcher=fetcher),
        start_date="2026-04-01",
        end_date="2026-04-01",
        metrics=["high_temperature"],
        forecast_issue_hours=[0],
        horizons=[24],
    )
    audit = build_provider_semantics_audit(
        summary=summary,
        cache_dir=cache_dir,
        start_date="2026-04-01",
        end_date="2026-04-01",
    )
    write_provider_semantics_audit(audit, audit_path)
    data = json.loads(audit_path.read_text(encoding="utf-8"))

    assert audit.smoke_status == "SUCCESS"
    assert data["smoke_status"] == "SUCCESS"
    assert audit.forecast_request_supported is True
    assert audit.actual_request_supported is True
    assert audit.forecast_mean_field == "temperature_2m_max"
    assert audit.actual_value_field == "temperature_2m_max"
    assert "historical_forecast_semantics_uncertain" in audit.warnings
    assert audit_path.exists()


def test_provider_semantics_audit_network_failed(tmp_path) -> None:
    summary = type(
        "Summary",
        (),
        {
            "samples_generated": 0,
            "skipped_count": 2,
            "skipped_reasons": {"forecast_provider_error": 2},
        },
    )()

    audit = build_provider_semantics_audit(
        summary=summary,
        cache_dir=tmp_path / "empty-cache",
        start_date="2026-04-01",
        end_date="2026-04-03",
    )

    assert audit.smoke_status == "NETWORK_FAILED"
