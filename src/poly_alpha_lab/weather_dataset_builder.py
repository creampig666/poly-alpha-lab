"""Build historical weather forecast-vs-actual datasets for calibration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, Field

from poly_alpha_lab.network_debug import classify_network_error, mask_proxy_url
from poly_alpha_lab.weather_data import LocationMapping

DatasetProviderName = Literal["open-meteo"]
DatasetMetric = Literal["high_temperature", "low_temperature", "average_temperature"]
ProviderDebugStatus = Literal[
    "SUCCESS",
    "NETWORK_FAILED",
    "TIMEOUT",
    "DNS_FAILED",
    "TLS_FAILED",
    "HTTP_ERROR",
    "API_UNSUPPORTED",
    "CONNECTION_REFUSED",
    "PARAMETER_ERROR",
    "PARSE_ERROR",
    "UNKNOWN_NETWORK_ERROR",
]

DATASET_CSV_FIELDS = [
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
    "actual_source",
    "source_location_name",
    "latitude",
    "longitude",
    "timezone",
    "horizon_hours",
    "notes",
]

MANUAL_TEMPLATE_FIELDS = [
    "location",
    "station_id",
    "source_location_name",
    "latitude",
    "longitude",
    "metric",
    "target_date",
    "target_datetime",
    "forecast_issued_at",
    "forecast_mean",
    "actual_value",
    "unit",
    "forecast_source",
    "forecast_model",
    "actual_source",
    "timezone",
    "notes",
    "verification_url_forecast",
    "verification_url_actual",
]


class WeatherForecastActualSample(BaseModel):
    """One forecast-vs-actual row suitable for weather-calibration fit."""

    location: str
    station_id: str | None = None
    source_location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    metric: DatasetMetric
    target_date: str
    target_datetime: str
    forecast_issued_at: str
    forecast_mean: float
    actual_value: float
    unit: str = "C"
    forecast_source: str
    forecast_model: str
    actual_source: str
    horizon_hours: float
    timezone: str | None = None
    notes: str | None = None


class WeatherDatasetBuildSummary(BaseModel):
    locations_processed: int
    samples_generated: int
    skipped_count: int
    skipped_reasons: dict[str, int] = Field(default_factory=dict)
    date_range: str
    metrics: list[str]
    output_path: str


class ProviderSemanticsAudit(BaseModel):
    provider: str
    smoke_status: Literal["SUCCESS", "NETWORK_FAILED", "API_UNSUPPORTED", "PARTIAL"]
    actual_date_range_used: str
    forecast_request_supported: bool
    actual_request_supported: bool
    forecast_mean_field: str | None = None
    actual_value_field: str | None = None
    forecast_issued_at_construction_method: str
    target_datetime_construction_method: str
    timezone_handling: str
    historical_forecast_request_params: dict[str, Any] | None = None
    historical_forecast_response_key_fields: dict[str, Any] = Field(default_factory=dict)
    actual_archive_request_params: dict[str, Any] | None = None
    actual_archive_response_key_fields: dict[str, Any] = Field(default_factory=dict)
    daily_variable_names: list[str] = Field(default_factory=list)
    returned_dates: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_classification: str | None = None
    proxy_used: str | None = None
    trust_env: bool = True


class ProviderDebugReport(BaseModel):
    status: ProviderDebugStatus
    provider: str
    request_type: str
    endpoint_url: str | None = None
    actual_endpoint_url: str | None = None
    request_params: dict[str, Any] | None = None
    actual_request_params: dict[str, Any] | None = None
    http_status_code: int | None = None
    actual_http_status_code: int | None = None
    response_top_level_keys: list[str] = Field(default_factory=list)
    actual_response_top_level_keys: list[str] = Field(default_factory=list)
    returned_dates: list[str] = Field(default_factory=list)
    actual_returned_dates: list[str] = Field(default_factory=list)
    forecast_mean_extracted: float | None = None
    actual_value_extracted: float | None = None
    forecast_source: str | None = None
    actual_source: str | None = None
    forecast_issued_at_construction_method: str
    target_datetime_construction_method: str
    exception_type: str | None = None
    exception_message: str | None = None
    response_body_excerpt: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error_classification: str | None = None
    proxy_used: str | None = None
    trust_env: bool = True


class ManualCsvValidationSummary(BaseModel):
    input_path: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    errors: dict[str, int] = Field(default_factory=dict)
    warnings: dict[str, int] = Field(default_factory=dict)
    can_use_for_calibration: bool = False


@dataclass
class DatasetBuildResult:
    samples: list[WeatherForecastActualSample] = field(default_factory=list)
    skipped_reasons: Counter[str] = field(default_factory=Counter)
    locations_processed: int = 0

    def summary(
        self,
        *,
        output_path: str | Path,
        start_date: str,
        end_date: str,
        metrics: list[str],
    ) -> WeatherDatasetBuildSummary:
        return WeatherDatasetBuildSummary(
            locations_processed=self.locations_processed,
            samples_generated=len(self.samples),
            skipped_count=sum(self.skipped_reasons.values()),
            skipped_reasons=dict(sorted(self.skipped_reasons.items())),
            date_range=f"{start_date}..{end_date}",
            metrics=metrics,
            output_path=str(output_path),
        )


class OpenMeteoHistoricalDatasetProvider:
    """Open-Meteo historical forecast and archive actual provider.

    Forecasts are fetched from the historical forecast host. Actuals are fetched
    separately from the archive host. The builder never uses archive actuals as
    forecast_mean.
    """

    forecast_base_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    actual_base_url = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(
        self,
        *,
        cache_dir: str | Path = "data/weather/dataset_cache",
        refresh_cache: bool = False,
        fetcher: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        timeout_seconds: float = 15,
        print_request_url: bool = False,
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_cache = refresh_cache
        self.fetcher = fetcher or self._http_fetch
        self.timeout_seconds = timeout_seconds
        self.print_request_url = print_request_url
        self.proxy = proxy
        self.trust_env = trust_env
        self.disabled_request_types: set[str] = set()
        self.request_errors: Counter[str] = Counter()

    def get_forecast_mean(
        self,
        *,
        mapping: LocationMapping,
        metric: str,
        target_date: str,
        forecast_issued_at: datetime,
        horizon_hours: float,
    ) -> tuple[float | None, str | None, str | None, str | None]:
        variable = metric_to_open_meteo_daily(metric)
        params = _open_meteo_params(mapping, variable, target_date)
        cache_key = weather_dataset_cache_key(
            provider="open_meteo",
            location=mapping.location_name,
            metric=metric,
            target_date=target_date,
            forecast_issued_at=forecast_issued_at,
            horizon_hours=horizon_hours,
            request_type="forecast",
        )
        raw = self._fetch_cached("forecast", cache_key, params)
        source = str(raw.get("source") or raw.get("forecast_source") or "open_meteo")
        model = str(
            raw.get("forecast_model")
            or raw.get("model")
            or raw.get("generationtime_ms")
            or "open_meteo_historical_forecast"
        )
        if source.strip().casefold() in {"open_meteo_archive", "archive", "actual"}:
            return None, "forecast_response_looks_like_actual_archive", None, cache_key
        value = extract_open_meteo_daily_value(raw, variable, target_date)
        if value is None:
            return None, "missing_forecast_mean", None, cache_key
        return value, source, model, cache_key

    def get_actual_value(
        self,
        *,
        mapping: LocationMapping,
        metric: str,
        target_date: str,
    ) -> tuple[float | None, str | None, str | None]:
        variable = metric_to_open_meteo_daily(metric)
        params = _open_meteo_params(mapping, variable, target_date)
        cache_key = weather_dataset_cache_key(
            provider="open_meteo",
            location=mapping.location_name,
            metric=metric,
            target_date=target_date,
            forecast_issued_at=None,
            horizon_hours=None,
            request_type="actual",
        )
        raw = self._fetch_cached("actual", cache_key, params)
        value = extract_open_meteo_daily_value(raw, variable, target_date)
        source = str(raw.get("source") or raw.get("actual_source") or "open_meteo_archive")
        return value, source, cache_key

    def _fetch_cached(
        self,
        request_type: Literal["forecast", "actual"],
        cache_key: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not self.refresh_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            raw = cached.get("raw", cached)
            if not isinstance(raw, dict):
                raise ValueError(f"cached {request_type} response must be a JSON object")
            return raw
        if request_type in self.disabled_request_types:
            raise httpx.ConnectError(f"previous Open-Meteo {request_type} request failed")
        try:
            raw = self.fetcher(request_type, params)
        except httpx.HTTPError:
            self.disabled_request_types.add(request_type)
            self.request_errors[request_type] += 1
            raise
        payload = {"request_type": request_type, "params": params, "raw": raw}
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return raw

    def _http_fetch(self, request_type: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self.forecast_base_url if request_type == "forecast" else self.actual_base_url
        if self.print_request_url:
            print(f"{request_type} request URL: {_request_url(url, params)}")
        kwargs: dict[str, Any] = {
            "params": params,
            "timeout": self.timeout_seconds,
            "trust_env": self.trust_env,
        }
        if self.proxy:
            kwargs["proxy"] = self.proxy
        response = httpx.get(url, **kwargs)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Open-Meteo {request_type} response must be a JSON object")
        return data


def build_weather_dataset(
    *,
    locations_file: str | Path,
    output_path: str | Path,
    provider: OpenMeteoHistoricalDatasetProvider,
    start_date: str,
    end_date: str,
    metrics: list[str],
    forecast_issue_hours: list[int],
    horizons: list[int],
) -> WeatherDatasetBuildSummary:
    """Build and write forecast-vs-actual rows."""

    result = collect_weather_dataset_samples(
        locations_file=locations_file,
        provider=provider,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        forecast_issue_hours=forecast_issue_hours,
        horizons=horizons,
    )
    write_weather_dataset_csv(result.samples, output_path)
    return result.summary(
        output_path=output_path,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
    )


def build_provider_semantics_audit(
    *,
    summary: WeatherDatasetBuildSummary,
    cache_dir: str | Path,
    start_date: str,
    end_date: str,
    provider: str = "open_meteo",
    proxy: str | None = None,
    trust_env: bool = True,
) -> ProviderSemanticsAudit:
    cache_path = Path(cache_dir)
    forecast_payload = _first_cached_payload(cache_path, "forecast")
    actual_payload = _first_cached_payload(cache_path, "actual")
    forecast_meta = _response_metadata(forecast_payload)
    actual_meta = _response_metadata(actual_payload)
    forecast_supported = bool(forecast_meta.get("daily_variable_names"))
    actual_supported = bool(actual_meta.get("daily_variable_names"))
    warnings: list[str] = []
    missing_fields = _unique(
        list(forecast_meta.get("missing_fields", []))
        + [f"actual:{field}" for field in actual_meta.get("missing_fields", [])]
    )
    if provider == "open_meteo":
        warnings.append("historical_forecast_semantics_uncertain")
    if summary.samples_generated == 0 and _network_failed(summary.skipped_reasons):
        smoke_status: Literal["SUCCESS", "NETWORK_FAILED", "API_UNSUPPORTED", "PARTIAL"] = (
            "NETWORK_FAILED"
        )
        error_classification = "CONNECTION_REFUSED"
    elif not forecast_supported or not actual_supported:
        smoke_status = "API_UNSUPPORTED"
        error_classification = "API_UNSUPPORTED"
    elif summary.samples_generated > 0 and summary.skipped_count == 0:
        smoke_status = "SUCCESS"
        error_classification = "NONE"
    else:
        smoke_status = "PARTIAL"
        error_classification = "UNKNOWN_NETWORK_ERROR"
    return ProviderSemanticsAudit(
        provider=provider,
        smoke_status=smoke_status,
        actual_date_range_used=f"{start_date}..{end_date}",
        forecast_request_supported=forecast_supported,
        actual_request_supported=actual_supported,
        forecast_mean_field=_first_variable(forecast_meta),
        actual_value_field=_first_variable(actual_meta),
        forecast_issued_at_construction_method=(
            "CLI issue date/hour converted from location timezone to UTC; "
            "currently not sent as an Open-Meteo forecast run selector"
        ),
        target_datetime_construction_method="forecast_issued_at + horizon_hours",
        timezone_handling="locations.csv timezone is used when available; UTC fallback if zone data missing",
        historical_forecast_request_params=_payload_params(forecast_payload),
        historical_forecast_response_key_fields=forecast_meta,
        actual_archive_request_params=_payload_params(actual_payload),
        actual_archive_response_key_fields=actual_meta,
        daily_variable_names=_unique(
            list(forecast_meta.get("daily_variable_names", []))
            + list(actual_meta.get("daily_variable_names", []))
        ),
        returned_dates=_unique(
            list(forecast_meta.get("returned_dates", []))
            + list(actual_meta.get("returned_dates", []))
        ),
        missing_fields=missing_fields,
        known_limitations=[
            "This builder separates historical forecast and archive actual endpoints.",
            "Open-Meteo historical forecast daily endpoint is not yet proven to be a point-in-time model run selected by forecast_issued_at.",
            "For strict forecast-as-of replay, a provider with explicit model run initialization time should be preferred.",
        ],
        warnings=_unique(warnings),
        error_classification=error_classification,
        proxy_used=mask_proxy_url(proxy) if proxy else None,
        trust_env=trust_env,
    )


def write_provider_semantics_audit(
    audit: ProviderSemanticsAudit,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def debug_open_meteo_provider(
    *,
    location: str,
    latitude: float,
    longitude: float,
    target_date: str,
    forecast_issued_at: str,
    metric: str,
    horizon: float,
    cache_dir: str | Path = "data/weather/debug_cache",
    refresh_cache: bool = False,
    timeout_seconds: float = 30,
    print_request_url: bool = False,
    proxy: str | None = None,
    trust_env: bool = True,
    fetcher: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> ProviderDebugReport:
    """Run a minimal real-provider debug for one forecast and one actual request."""

    try:
        metric = normalize_metric(metric)
        variable = metric_to_open_meteo_daily(metric)
        issued_at = _aware_datetime(_parse_datetime(forecast_issued_at))
        target_dt = issued_at + timedelta(hours=horizon)
        api_target_date = target_date[:10]
    except (ValueError, TypeError) as exc:
        return ProviderDebugReport(
            status="PARAMETER_ERROR",
            provider="open_meteo",
            request_type="forecast",
            forecast_issued_at_construction_method="parsed from CLI --forecast-issued-at",
            target_datetime_construction_method="forecast_issued_at + horizon_hours",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            warnings=["parameter_error"],
        )
    mapping = LocationMapping(
        location_name=location,
        latitude=latitude,
        longitude=longitude,
        timezone="UTC",
    )
    provider = OpenMeteoHistoricalDatasetProvider(
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        fetcher=fetcher,
        timeout_seconds=timeout_seconds,
        print_request_url=print_request_url,
        proxy=proxy,
        trust_env=trust_env,
    )
    forecast_params = _open_meteo_params(mapping, variable, api_target_date)
    actual_params = _open_meteo_params(mapping, variable, api_target_date)
    forecast_url = _request_url(provider.forecast_base_url, forecast_params)
    actual_url = _request_url(provider.actual_base_url, actual_params)
    warnings = ["historical_forecast_semantics_uncertain"]
    if target_dt.date().isoformat() != api_target_date:
        warnings.append("target_date_differs_from_forecast_issued_at_plus_horizon")
    forecast_raw: dict[str, Any] | None = None
    actual_raw: dict[str, Any] | None = None
    try:
        forecast_key = weather_dataset_cache_key(
            provider="open_meteo",
            location=location,
            metric=metric,
            target_date=api_target_date,
            forecast_issued_at=issued_at,
            horizon_hours=horizon,
            request_type="forecast",
        )
        forecast_raw = provider._fetch_cached("forecast", forecast_key, forecast_params)
        forecast_meta = _response_metadata({"raw": forecast_raw})
        forecast_value = extract_open_meteo_daily_value(forecast_raw, variable, api_target_date)
        forecast_source = str(
            forecast_raw.get("source") or forecast_raw.get("forecast_source") or "open_meteo"
        )
        if forecast_source.strip().casefold() in {"open_meteo_archive", "archive", "actual"}:
            warnings.append("forecast_response_looks_like_actual_archive")
            forecast_value = None
        if forecast_source.strip().casefold() in {"mock", "sample"}:
            warnings.append("mock_or_sample_response_not_real")
        actual_key = weather_dataset_cache_key(
            provider="open_meteo",
            location=location,
            metric=metric,
            target_date=api_target_date,
            forecast_issued_at=None,
            horizon_hours=None,
            request_type="actual",
        )
        actual_raw = provider._fetch_cached("actual", actual_key, actual_params)
        actual_meta = _response_metadata({"raw": actual_raw})
        actual_value = extract_open_meteo_daily_value(actual_raw, variable, api_target_date)
        actual_source = str(actual_raw.get("source") or actual_raw.get("actual_source") or "open_meteo_archive")
    except Exception as exc:
        status = _debug_status_for_exception(exc)
        error_classification = classify_network_error(exc)
        raw = forecast_raw or actual_raw
        return ProviderDebugReport(
            status=status,
            provider="open_meteo",
            request_type="forecast",
            endpoint_url=forecast_url,
            actual_endpoint_url=actual_url,
            request_params=forecast_params,
            actual_request_params=actual_params,
            response_top_level_keys=sorted(str(key) for key in raw) if raw else [],
            returned_dates=_returned_dates(raw),
            forecast_issued_at_construction_method="parsed from CLI --forecast-issued-at",
            target_datetime_construction_method="forecast_issued_at + horizon_hours",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            response_body_excerpt=_response_excerpt(raw),
            warnings=_unique(warnings),
            error_classification=error_classification,
            proxy_used=mask_proxy_url(proxy) if proxy else None,
            trust_env=trust_env,
        )
    status: ProviderDebugStatus = "SUCCESS"
    if forecast_value is None or actual_value is None:
        status = "API_UNSUPPORTED"
    return ProviderDebugReport(
        status=status,
        provider="open_meteo",
        request_type="forecast",
        endpoint_url=forecast_url,
        actual_endpoint_url=actual_url,
        request_params=forecast_params,
        actual_request_params=actual_params,
        response_top_level_keys=forecast_meta.get("top_level_keys", []),
        actual_response_top_level_keys=actual_meta.get("top_level_keys", []),
        returned_dates=forecast_meta.get("returned_dates", []),
        actual_returned_dates=actual_meta.get("returned_dates", []),
        forecast_mean_extracted=forecast_value,
        actual_value_extracted=actual_value,
        forecast_source=forecast_source,
        actual_source=actual_source,
        forecast_issued_at_construction_method="parsed from CLI --forecast-issued-at",
        target_datetime_construction_method="forecast_issued_at + horizon_hours",
        response_body_excerpt=_response_excerpt(forecast_raw),
        warnings=_unique(warnings),
        error_classification="NONE",
        proxy_used=mask_proxy_url(proxy) if proxy else None,
        trust_env=trust_env,
    )


def write_provider_debug_report(report: ProviderDebugReport, output_path: str | Path) -> None:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def provider_debug_report_to_markdown(report: ProviderDebugReport, output_path: str) -> str:
    return "\n".join(
        [
            "# Provider Debug Report",
            "",
            f"- output: `{output_path}`",
            f"- status: `{report.status}`",
            f"- provider: `{report.provider}`",
            f"- forecast_endpoint: `{report.endpoint_url or 'n/a'}`",
            f"- actual_endpoint: `{report.actual_endpoint_url or 'n/a'}`",
            f"- forecast_mean_extracted: `{report.forecast_mean_extracted if report.forecast_mean_extracted is not None else 'n/a'}`",
            f"- actual_value_extracted: `{report.actual_value_extracted if report.actual_value_extracted is not None else 'n/a'}`",
            f"- exception: `{report.exception_type or 'n/a'}` `{report.exception_message or ''}`",
            f"- warnings: `{', '.join(report.warnings) if report.warnings else 'none'}`",
        ]
    )


def collect_weather_dataset_samples(
    *,
    locations_file: str | Path,
    provider: OpenMeteoHistoricalDatasetProvider,
    start_date: str,
    end_date: str,
    metrics: list[str],
    forecast_issue_hours: list[int],
    horizons: list[int],
) -> DatasetBuildResult:
    mappings, location_skips, total_locations = _load_dataset_locations(locations_file)
    result = DatasetBuildResult(locations_processed=total_locations)
    result.skipped_reasons.update(location_skips)
    for mapping in mappings:
        if mapping.latitude is None or mapping.longitude is None:
            result.skipped_reasons["location_missing_coordinates"] += 1
            continue
        for issue_date in _date_range(start_date, end_date):
            for issue_hour in forecast_issue_hours:
                forecast_issued_at = _local_datetime(issue_date, issue_hour, mapping.timezone)
                for horizon in horizons:
                    target_datetime = forecast_issued_at + timedelta(hours=horizon)
                    target_date = target_datetime.date().isoformat()
                    for metric in metrics:
                        sample, skip_reason = build_one_sample(
                            provider=provider,
                            mapping=mapping,
                            metric=metric,
                            target_datetime=target_datetime,
                            forecast_issued_at=forecast_issued_at,
                            horizon_hours=float(horizon),
                        )
                        if skip_reason:
                            result.skipped_reasons[skip_reason] += 1
                        elif sample is not None:
                            result.samples.append(sample)
    return result


def build_one_sample(
    *,
    provider: OpenMeteoHistoricalDatasetProvider,
    mapping: LocationMapping,
    metric: str,
    target_datetime: datetime,
    forecast_issued_at: datetime,
    horizon_hours: float | None = None,
) -> tuple[WeatherForecastActualSample | None, str | None]:
    if forecast_issued_at >= target_datetime:
        return None, "forecast_issued_at_not_before_target_datetime"
    if mapping.latitude is None or mapping.longitude is None:
        return None, "location_missing_coordinates"
    metric = normalize_metric(metric)
    target_datetime = _aware_datetime(target_datetime)
    forecast_issued_at = _aware_datetime(forecast_issued_at)
    target_date = target_datetime.date().isoformat()
    horizon = (
        horizon_hours
        if horizon_hours is not None
        else (target_datetime - forecast_issued_at).total_seconds() / 3600
    )
    try:
        forecast_mean, forecast_source, forecast_model, _ = provider.get_forecast_mean(
            mapping=mapping,
            metric=metric,
            target_date=target_date,
            forecast_issued_at=forecast_issued_at,
            horizon_hours=horizon,
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return None, _skip_reason_from_exception(exc, "forecast")
    if forecast_mean is None:
        return None, forecast_source or "missing_forecast_mean"
    try:
        actual_value, actual_source, _ = provider.get_actual_value(
            mapping=mapping,
            metric=metric,
            target_date=target_date,
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return None, _skip_reason_from_exception(exc, "actual")
    if actual_value is None:
        return None, "missing_actual_value"
    if not _valid_temperature(actual_value):
        return None, "actual_value_obviously_invalid"
    if not _valid_temperature(forecast_mean):
        return None, "forecast_mean_obviously_invalid"
    notes = ["target_datetime_from_forecast_issue_plus_horizon"]
    return (
        WeatherForecastActualSample(
            location=mapping.location_name,
            station_id=mapping.station_id,
            source_location_name=mapping.source_location_name,
            latitude=mapping.latitude,
            longitude=mapping.longitude,
            metric=metric,  # type: ignore[arg-type]
            target_date=target_date,
            target_datetime=target_datetime.isoformat(),
            forecast_issued_at=forecast_issued_at.isoformat(),
            forecast_mean=float(forecast_mean),
            actual_value=float(actual_value),
            unit="C",
            forecast_source=forecast_source or "open_meteo",
            forecast_model=forecast_model or "open_meteo_historical_forecast",
            actual_source=actual_source or "open_meteo_archive",
            horizon_hours=horizon,
            timezone=mapping.timezone,
            notes=";".join(notes),
        ),
        None,
    )


def write_weather_dataset_csv(
    samples: list[WeatherForecastActualSample],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            data = sample.model_dump()
            writer.writerow({field: data.get(field) for field in DATASET_CSV_FIELDS})
    return len(samples)


def write_manual_forecast_actual_template(
    *,
    output_path: str | Path,
    locations_file: str | Path = "data/weather/locations.csv",
    rows_per_location: int = 2,
) -> int:
    mappings, _, _ = _load_dataset_locations(locations_file)
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANUAL_TEMPLATE_FIELDS)
        writer.writeheader()
        for mapping in mappings:
            for _ in range(rows_per_location):
                writer.writerow(
                    {
                        "location": mapping.location_name,
                        "station_id": mapping.station_id,
                        "source_location_name": mapping.source_location_name,
                        "latitude": mapping.latitude,
                        "longitude": mapping.longitude,
                        "metric": "high_temperature",
                        "target_date": "",
                        "target_datetime": "",
                        "forecast_issued_at": "",
                        "forecast_mean": "",
                        "actual_value": "",
                        "unit": "C",
                        "forecast_source": "",
                        "forecast_model": "",
                        "actual_source": "",
                        "timezone": mapping.timezone,
                        "notes": (
                            "manual template, not real data until filled and verified; "
                            "suggested forecast_source=manual_verified_forecast; "
                            "suggested actual_source=wunderground/open_meteo_archive/NOAA"
                        ),
                        "verification_url_forecast": "",
                        "verification_url_actual": "",
                    }
                )
                count += 1
    return count


def validate_manual_forecast_actual_csv(input_path: str | Path) -> ManualCsvValidationSummary:
    path = Path(input_path)
    errors: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    total = 0
    valid = 0
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            row_errors = _manual_row_errors(row)
            row_warnings = _manual_row_warnings(row)
            errors.update(row_errors)
            warnings.update(row_warnings)
            if not row_errors:
                valid += 1
    return ManualCsvValidationSummary(
        input_path=str(input_path),
        total_rows=total,
        valid_rows=valid,
        invalid_rows=total - valid,
        errors=dict(sorted(errors.items())),
        warnings=dict(sorted(warnings.items())),
        can_use_for_calibration=valid > 0,
    )


def manual_validation_to_markdown(summary: ManualCsvValidationSummary) -> str:
    lines = [
        "# Manual Weather CSV Validation",
        "",
        f"- input: `{summary.input_path}`",
        f"- total_rows: `{summary.total_rows}`",
        f"- valid_rows: `{summary.valid_rows}`",
        f"- invalid_rows: `{summary.invalid_rows}`",
        f"- can_use_for_calibration: `{summary.can_use_for_calibration}`",
    ]
    if summary.errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in summary.errors.items())
    if summary.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in summary.warnings.items())
    return "\n".join(lines)


def write_data_source_options_markdown(
    *,
    output_path: str | Path,
    open_meteo_status: str,
    failure_reason: str,
) -> None:
    text = f"""# Weather Data Source Options

## Open-Meteo Current Failure

- status: `{open_meteo_status}`
- observed_failure_reason: `{failure_reason or 'n/a'}`
- current stance: do not treat mock/sample cache as real data.

## Options

| Source | Historical forecast | Actuals | Station-level | API key | Calibration fit | Strict backtest fit | Main risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Open-Meteo Historical Forecast | Yes, but as-of/run semantics must be audited | Indirect/model archive | Grid/city, not true station | No | Potentially useful after semantics validation | Needs explicit as-of confidence | Forecast run time may not match requested `forecast_issued_at` |
| Open-Meteo Historical Weather only | No | Yes/model archive | Grid/city | No | No, actuals only | No for forecast calibration | Actual archive cannot be used as forecast |
| NOAA / NCEI actual observations | No forecast in this path | Yes | Often station-level | Sometimes/no depending endpoint | Actual side only | Useful for resolution/actual validation | Station mapping and access complexity |
| Meteostat | No forecast baseline | Yes | Station-level possible | Usually no/free tiers vary | Actual side only | Useful for actual validation | Coverage and station continuity |
| Visual Crossing historical forecast | Potentially yes | Yes | Location/grid, depends plan | Usually yes | Possible | Possible if API exposes forecast issue/run timing | API key/cost and semantics need audit |
| Self-collected live forecast snapshots | Yes from collection start onward | Needs paired actual source | Whatever provider supplies | Depends provider | Best for forward calibration | Strong for strict replay after enough samples | Requires waiting to accumulate data |

## Recommendation

- If Open-Meteo failed because of network/timeout/proxy, first fix network or run in a stable environment.
- If Open-Meteo does not expose the required historical forecast as-of semantics, use live forecast snapshot accumulation or a provider with explicit historical forecast run times.
- If a source only provides actual observations, use it for resolution/actual_value only; never use it as forecast_mean.

## Minimal Next Patch

1. Keep `debug-provider` as the first smoke gate.
2. Add a provider only after a single-market debug proves forecast and actual fields are semantically separated.
3. For strict backtest, require forecast snapshots with `forecast_issued_at <= as_of_time` and verified source metadata.
"""
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def weather_dataset_summary_to_markdown(summary: WeatherDatasetBuildSummary) -> str:
    lines = [
        "# Weather Dataset Build",
        "",
        "- Dataset type: `forecast_actual_history_for_calibration`",
        "- Forecast values are not allowed to come from actual archive data.",
        f"- output: `{summary.output_path}`",
        f"- date_range: `{summary.date_range}`",
        f"- metrics: `{', '.join(summary.metrics)}`",
        f"- locations_processed: `{summary.locations_processed}`",
        f"- samples_generated: `{summary.samples_generated}`",
        f"- skipped_count: `{summary.skipped_count}`",
    ]
    if summary.skipped_reasons:
        lines.extend(["", "## Skipped Reasons", ""])
        for reason, count in summary.skipped_reasons.items():
            lines.append(f"- `{reason}`: `{count}`")
    return "\n".join(lines)


def provider_semantics_audit_to_markdown(audit: ProviderSemanticsAudit, output_path: str) -> str:
    return "\n".join(
        [
            "# Provider Semantics Audit",
            "",
            f"- audit_output: `{output_path}`",
            f"- provider: `{audit.provider}`",
            f"- smoke_status: `{audit.smoke_status}`",
            f"- actual_date_range_used: `{audit.actual_date_range_used}`",
            f"- forecast_request_supported: `{audit.forecast_request_supported}`",
            f"- actual_request_supported: `{audit.actual_request_supported}`",
            f"- forecast_mean_field: `{audit.forecast_mean_field or 'n/a'}`",
            f"- actual_value_field: `{audit.actual_value_field or 'n/a'}`",
            f"- warnings: `{', '.join(audit.warnings) if audit.warnings else 'none'}`",
            f"- error_classification: `{audit.error_classification or 'n/a'}`",
            f"- proxy_used: `{audit.proxy_used or 'n/a'}`",
            f"- trust_env: `{audit.trust_env}`",
        ]
    )


def weather_dataset_cache_key(
    *,
    provider: str,
    location: str,
    metric: str,
    target_date: str,
    forecast_issued_at: datetime | str | None,
    horizon_hours: float | int | None,
    request_type: str,
) -> str:
    issued = "none"
    if forecast_issued_at is not None:
        issued = _aware_datetime(_parse_datetime(forecast_issued_at)).isoformat()
    horizon = "none" if horizon_hours is None else f"h{float(horizon_hours):g}"
    readable = "__".join(
        [
            _slug(provider),
            _slug(request_type),
            _slug(location),
            _slug(metric),
            target_date[:10],
            _slug(issued),
            horizon,
        ]
    )
    digest = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:8]
    return f"{readable}__{digest}"


def metric_to_open_meteo_daily(metric: str) -> str:
    normalized = normalize_metric(metric)
    if normalized == "high_temperature":
        return "temperature_2m_max"
    if normalized == "low_temperature":
        return "temperature_2m_min"
    if normalized == "average_temperature":
        return "temperature_2m_mean"
    raise ValueError(f"unsupported weather metric: {metric}")


def extract_open_meteo_daily_value(raw: dict[str, Any], variable: str, target_date: str) -> float | None:
    daily = raw.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("Open-Meteo response missing daily data")
    dates = daily.get("time")
    values = daily.get(variable)
    if not isinstance(dates, list) or not isinstance(values, list):
        raise ValueError(f"Open-Meteo response missing daily {variable}")
    for index, value_date in enumerate(dates):
        if value_date == target_date:
            value = values[index]
            if value is None or value == "":
                return None
            return float(value)
    return None


def parse_csv_list(value: str, *, cast: Callable[[str], Any] = str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def normalize_metric(metric: str) -> str:
    normalized = metric.strip().casefold()
    if normalized in {"high_temperature", "temperature_2m_max", "high"}:
        return "high_temperature"
    if normalized in {"low_temperature", "temperature_2m_min", "low"}:
        return "low_temperature"
    if normalized in {"average_temperature", "temperature_2m_mean", "average"}:
        return "average_temperature"
    raise ValueError(f"unsupported weather metric: {metric}")


def _open_meteo_params(mapping: LocationMapping, variable: str, target_date: str) -> dict[str, Any]:
    return {
        "latitude": mapping.latitude,
        "longitude": mapping.longitude,
        "daily": variable,
        "temperature_unit": "celsius",
        "timezone": mapping.timezone or "UTC",
        "start_date": target_date,
        "end_date": target_date,
    }


def _load_dataset_locations(
    locations_file: str | Path,
) -> tuple[list[LocationMapping], Counter[str], int]:
    path = Path(locations_file)
    if not path.exists():
        raise FileNotFoundError(f"locations file not found: {path}")
    mappings: list[LocationMapping] = []
    skipped: Counter[str] = Counter()
    total = 0
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            name = _optional_string(row.get("location_name"))
            latitude = _optional_float(row.get("latitude"))
            longitude = _optional_float(row.get("longitude"))
            if name is None or latitude is None or longitude is None:
                skipped["location_missing_coordinates"] += 1
                continue
            mappings.append(
                LocationMapping(
                    location_name=name,
                    latitude=latitude,
                    longitude=longitude,
                    station_id=_optional_string(row.get("station_id")),
                    source_location_name=_optional_string(row.get("source_location_name")),
                    timezone=_optional_string(row.get("timezone")),
                    notes=_optional_string(row.get("notes")),
                    default_forecast_std=_optional_float(row.get("default_forecast_std")),
                    std_source=_optional_string(row.get("std_source")),
                )
            )
    return mappings, skipped, total


def _date_range(start_date: str, end_date: str) -> list[date]:
    start = date.fromisoformat(start_date[:10])
    end = date.fromisoformat(end_date[:10])
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(days + 1)]


def _local_datetime(day: date, hour: int, timezone: str | None) -> datetime:
    if hour < 0 or hour > 23:
        raise ValueError("forecast issue hours must be between 0 and 23")
    try:
        zone = ZoneInfo(timezone or "UTC")
    except ZoneInfoNotFoundError:
        zone = UTC
    return datetime.combine(day, time(hour), tzinfo=zone).astimezone(UTC)


def _parse_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _valid_temperature(value: float) -> bool:
    return math.isfinite(value) and -90 <= value <= 70


def _skip_reason_from_exception(exc: Exception, prefix: str) -> str:
    text = str(exc).casefold()
    if "missing daily" in text or "missing target date" in text:
        return f"missing_{prefix}_value"
    if "actual archive" in text:
        return "forecast_response_looks_like_actual_archive"
    return f"{prefix}_provider_error"


def _manual_row_errors(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for field_name in ("forecast_mean", "actual_value", "forecast_issued_at"):
        if not _optional_string(row.get(field_name)):
            errors.append(f"missing_{field_name}")
    forecast_source = _normalized(row.get("forecast_source"))
    if forecast_source in {"sample", "mock"}:
        errors.append("forecast_source_sample_or_mock")
    for field_name in ("forecast_mean", "actual_value"):
        value = _optional_string(row.get(field_name))
        if value is not None:
            try:
                float(value)
            except ValueError:
                errors.append(f"invalid_{field_name}")
    return errors


def _manual_row_warnings(row: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    if not _optional_string(row.get("actual_source")):
        warnings.append("missing_actual_source")
    if not _optional_string(row.get("verification_url_forecast")):
        warnings.append("missing_verification_url_forecast")
    if not _optional_string(row.get("verification_url_actual")):
        warnings.append("missing_verification_url_actual")
    return warnings


def _debug_status_for_exception(exc: Exception) -> ProviderDebugStatus:
    classification = classify_network_error(exc)
    if classification == "CONNECTION_REFUSED":
        return "CONNECTION_REFUSED"
    if classification == "TIMEOUT":
        return "TIMEOUT"
    if classification == "DNS_FAILED":
        return "DNS_FAILED"
    if classification == "TLS_FAILED":
        return "TLS_FAILED"
    if classification == "HTTP_ERROR":
        return "HTTP_ERROR"
    if isinstance(exc, ValueError):
        return "API_UNSUPPORTED"
    if classification == "UNKNOWN_NETWORK_ERROR":
        return "NETWORK_FAILED"
    return "PARSE_ERROR"


def _request_url(url: str, params: dict[str, Any]) -> str:
    return f"{url}?{urlencode(params)}"


def _returned_dates(raw: dict[str, Any] | None) -> list[str]:
    if not raw:
        return []
    daily = raw.get("daily")
    if not isinstance(daily, dict):
        return []
    dates = daily.get("time")
    return [str(item) for item in dates] if isinstance(dates, list) else []


def _response_excerpt(raw: dict[str, Any] | None) -> str | None:
    if raw is None:
        return None
    return json.dumps(raw, ensure_ascii=False)[:1000]


def _first_cached_payload(cache_dir: Path, request_type: str) -> dict[str, Any] | None:
    pattern = f"*__{_slug(request_type)}__*.json"
    for path in sorted(cache_dir.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("request_type") == request_type:
            return payload
    return None


def _payload_params(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    params = payload.get("params")
    return params if isinstance(params, dict) else None


def _response_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"missing_fields": ["raw_response"]}
    raw = payload.get("raw")
    if not isinstance(raw, dict):
        return {"missing_fields": ["raw_response"]}
    daily = raw.get("daily")
    missing: list[str] = []
    daily_keys: list[str] = []
    returned_dates: list[str] = []
    variables: list[str] = []
    if not isinstance(daily, dict):
        missing.append("daily")
    else:
        daily_keys = sorted(str(key) for key in daily)
        dates = daily.get("time")
        if not isinstance(dates, list):
            missing.append("daily.time")
        else:
            returned_dates = [str(item) for item in dates]
        variables = [key for key in daily_keys if key != "time"]
        if not variables:
            missing.append("daily.temperature_variable")
    return {
        "top_level_keys": sorted(str(key) for key in raw),
        "daily_keys": daily_keys,
        "daily_variable_names": variables,
        "returned_dates": returned_dates,
        "timezone": raw.get("timezone"),
        "utc_offset_seconds": raw.get("utc_offset_seconds"),
        "missing_fields": missing,
    }


def _first_variable(meta: dict[str, Any]) -> str | None:
    values = meta.get("daily_variable_names")
    if isinstance(values, list) and values:
        return str(values[0])
    return None


def _network_failed(skipped_reasons: dict[str, int]) -> bool:
    if not skipped_reasons:
        return False
    provider_errors = {
        "forecast_provider_error",
        "actual_provider_error",
    }
    return all(reason in provider_errors for reason in skipped_reasons)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _optional_string(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalized(value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip().casefold()


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "none"
