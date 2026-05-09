"""Pluggable weather forecast data providers for weather alpha research."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, Field, field_validator

TemperatureUnit = Literal["C", "F"]


class WeatherForecast(BaseModel):
    """Forecast snapshot used by weather threshold models."""

    date: str
    location: str
    metric: str
    forecast_mean: float
    forecast_std: float
    actual_value: float | None = None
    unit: TemperatureUnit
    forecast_issued_at: str | None = None
    station_id: str | None = None
    source_location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    forecast_source: str | None = None
    forecast_model: str | None = None
    std_method: str | None = None
    actual_source: str | None = None
    notes: str | None = None
    timezone: str | None = None
    cache_key: str | None = None
    raw_data_reference: str | None = None
    location_mapping_found: bool = True
    provider_warnings: list[str] = Field(default_factory=list)

    @field_validator("unit", mode="before")
    @classmethod
    def normalize_unit(cls, value: object) -> str:
        text = str(value).strip().upper().replace("°", "")
        if text in {"C", "CELSIUS"}:
            return "C"
        if text in {"F", "FAHRENHEIT"}:
            return "F"
        raise ValueError("unit must be C or F")

    @field_validator("forecast_issued_at")
    @classmethod
    def normalize_forecast_issued_at(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return str(value).strip()


class WeatherDataProvider(ABC):
    """Abstract weather data provider interface."""

    @abstractmethod
    def get_forecast(
        self,
        location: str,
        target_date: str,
        metric: str,
        *,
        as_of_time: str | datetime | None = None,
    ) -> WeatherForecast | None:
        """Return the forecast available for a location/date/metric."""

    @abstractmethod
    def get_actual(self, location: str, target_date: str, metric: str) -> float | None:
        """Return actual observed value for later backtests."""


class CsvWeatherDataProvider(WeatherDataProvider):
    """CSV-backed provider for offline, replayable tests and research."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self._forecasts = self._load_csv(self.csv_path)

    def get_forecast(
        self,
        location: str,
        target_date: str,
        metric: str,
        *,
        as_of_time: str | datetime | None = None,
    ) -> WeatherForecast | None:
        key = _key(location, target_date, metric)
        forecasts = self._forecasts.get(key, [])
        if not forecasts:
            return None
        if as_of_time is not None:
            cutoff = _parse_datetime(as_of_time)
            candidates = [
                (issued_at, forecast)
                for forecast in forecasts
                if (issued_at := _try_parse_datetime(forecast.forecast_issued_at)) is not None
                and issued_at <= cutoff
            ]
            if not candidates:
                if len(forecasts) == 1 and _try_parse_datetime(
                    forecasts[0].forecast_issued_at
                ) is None:
                    warnings = list(forecasts[0].provider_warnings)
                    warnings.append("missing_forecast_issued_at_single_snapshot_used_for_replay")
                    return forecasts[0].model_copy(update={"provider_warnings": _unique(warnings)})
                return None
            return max(candidates, key=lambda item: item[0])[1]
        if len(forecasts) == 1:
            return forecasts[0]

        dated_forecasts = [
            (issued_at, forecast)
            for forecast in forecasts
            if (issued_at := _try_parse_datetime(forecast.forecast_issued_at)) is not None
        ]
        if not dated_forecasts:
            return None
        selected = max(dated_forecasts, key=lambda item: item[0])[1]
        warnings = list(selected.provider_warnings)
        warnings.append("as_of_time_missing_multiple_forecasts")
        if len(dated_forecasts) != len(forecasts):
            warnings.append("undated_forecast_rows_ignored")
        return selected.model_copy(update={"provider_warnings": _unique(warnings)})

    def get_actual(self, location: str, target_date: str, metric: str) -> float | None:
        forecast = self.get_forecast(location, target_date, metric)
        return None if forecast is None else forecast.actual_value

    def _load_csv(self, csv_path: Path) -> dict[tuple[str, str, str], list[WeatherForecast]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"weather data CSV not found: {csv_path}")
        forecasts: dict[tuple[str, str, str], list[WeatherForecast]] = {}
        with csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                forecast = WeatherForecast(
                    date=_required(row, "date"),
                    location=_required(row, "location"),
                    metric=_required(row, "metric"),
                    forecast_mean=float(_required(row, "forecast_mean")),
                    forecast_std=float(_required(row, "forecast_std")),
                    actual_value=_optional_float(row.get("actual_value")),
                    unit=_required(row, "unit"),
                    forecast_issued_at=_optional_string(row.get("forecast_issued_at")),
                    station_id=_optional_string(row.get("station_id")),
                    source_location_name=_optional_string(row.get("source_location_name")),
                    latitude=_optional_float(row.get("latitude")),
                    longitude=_optional_float(row.get("longitude")),
                    forecast_source=_optional_string(
                        row.get("forecast_source") or row.get("source")
                    ),
                    forecast_model=_optional_string(row.get("forecast_model")),
                    std_method=_optional_string(row.get("std_method")),
                    actual_source=_optional_string(row.get("actual_source")),
                    notes=_optional_string(row.get("notes")),
                    timezone=_optional_string(row.get("timezone")),
                    cache_key=_optional_string(row.get("cache_key")),
                    raw_data_reference=_optional_string(row.get("raw_data_reference")),
                )
                forecasts.setdefault(_key(forecast.location, forecast.date, forecast.metric), []).append(
                    forecast
                )
        return forecasts


class StubWeatherDataProvider(WeatherDataProvider):
    """In-memory provider for tests."""

    def __init__(self, forecasts: list[WeatherForecast] | None = None) -> None:
        self._forecasts = {
            _key(forecast.location, forecast.date, forecast.metric): forecast
            for forecast in (forecasts or [])
        }

    def get_forecast(
        self,
        location: str,
        target_date: str,
        metric: str,
        *,
        as_of_time: str | datetime | None = None,
    ) -> WeatherForecast | None:
        return self._forecasts.get(_key(location, target_date, metric))

    def get_actual(self, location: str, target_date: str, metric: str) -> float | None:
        forecast = self.get_forecast(location, target_date, metric)
        return None if forecast is None else forecast.actual_value


def _required(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"weather CSV missing required field: {field}")
    return value.strip()


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def _optional_string(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _key(location: str, target_date: str, metric: str) -> tuple[str, str, str]:
    return (_normalize_location(location), target_date[:10], metric.strip().casefold())


def _normalize_location(location: str) -> str:
    value = unicodedata.normalize("NFKD", location)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.casefold().split())


class LocationMapping(BaseModel):
    location_name: str
    latitude: float
    longitude: float
    station_id: str | None = None
    source_location_name: str | None = None
    timezone: str | None = None
    notes: str | None = None
    default_forecast_std: float | None = None
    std_source: str | None = None


class LocationResolver:
    """CSV-backed resolver for weather market locations."""

    def __init__(self, locations_file: str | Path) -> None:
        self.locations_file = Path(locations_file)
        self._locations = self._load_locations(self.locations_file)

    def resolve(self, location_name: str) -> LocationMapping | None:
        return self._locations.get(_normalize_location(location_name))

    def _load_locations(self, path: Path) -> dict[str, LocationMapping]:
        if not path.exists():
            return {}
        locations: dict[str, LocationMapping] = {}
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                mapping = LocationMapping(
                    location_name=_required(row, "location_name"),
                    latitude=float(_required(row, "latitude")),
                    longitude=float(_required(row, "longitude")),
                    station_id=_optional_string(row.get("station_id")),
                    source_location_name=_optional_string(row.get("source_location_name")),
                    timezone=_optional_string(row.get("timezone")),
                    notes=_optional_string(row.get("notes")),
                    default_forecast_std=_optional_float(row.get("default_forecast_std")),
                    std_source=_optional_string(row.get("std_source")),
                )
                locations[_normalize_location(mapping.location_name)] = mapping
        return locations


class OpenMeteoForecastProvider(WeatherDataProvider):
    """Open-Meteo current forecast provider with cache and explicit std handling."""

    base_url = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        *,
        location_resolver: LocationResolver,
        fallback_forecast_std: float | None = None,
        cache_dir: str | Path = "data/weather/cache",
        refresh_cache: bool = False,
        fetcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        now: Callable[[], datetime] | None = None,
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self.location_resolver = location_resolver
        self.fallback_forecast_std = fallback_forecast_std
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_cache = refresh_cache
        self.fetcher = fetcher or self._http_fetch
        self.now = now or (lambda: datetime.now(UTC))
        self.proxy = proxy
        self.trust_env = trust_env

    def get_forecast(
        self,
        location: str,
        target_date: str,
        metric: str,
        *,
        as_of_time: str | datetime | None = None,
    ) -> WeatherForecast | None:
        mapping = self.location_resolver.resolve(location)
        if mapping is None:
            return None
        issued_at = _aware_datetime(self.now())
        target = _parse_date(target_date)
        if target < issued_at.date():
            raise ValueError("OpenMeteo current forecast cannot be used for past target_date")
        parsed_as_of_time = _parse_datetime(as_of_time) if as_of_time is not None else issued_at
        cache_key = open_meteo_cache_key(
            location=location,
            target_date=target.isoformat(),
            metric=metric,
            as_of_time=parsed_as_of_time,
        )
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists() and not self.refresh_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return WeatherForecast(**cached["forecast"])

        daily_variable = _metric_to_open_meteo_daily(metric)
        params = {
            "latitude": mapping.latitude,
            "longitude": mapping.longitude,
            "daily": daily_variable,
            "temperature_unit": "celsius",
            "timezone": mapping.timezone or "UTC",
            "start_date": target.isoformat(),
            "end_date": target.isoformat(),
        }
        raw = self.fetcher(params)
        forecast_mean = _extract_open_meteo_daily_value(raw, daily_variable, target.isoformat())
        forecast_std, std_method = self._forecast_std(mapping)
        forecast = WeatherForecast(
            date=target.isoformat(),
            location=location,
            metric=metric,
            forecast_mean=forecast_mean,
            forecast_std=forecast_std,
            unit="C",
            forecast_issued_at=issued_at.isoformat().replace("+00:00", "Z"),
            station_id=mapping.station_id,
            source_location_name=mapping.source_location_name or mapping.location_name,
            latitude=mapping.latitude,
            longitude=mapping.longitude,
            forecast_source="open_meteo",
            forecast_model="open_meteo_current_forecast",
            std_method=std_method,
            timezone=mapping.timezone,
            notes=mapping.notes,
            cache_key=cache_key,
            raw_data_reference=str(cache_path),
        )
        cache_payload = {"forecast": forecast.model_dump(mode="json"), "raw": raw}
        cache_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return forecast

    def get_actual(self, location: str, target_date: str, metric: str) -> float | None:
        return None

    def _forecast_std(self, mapping: LocationMapping) -> tuple[float, str]:
        if mapping.default_forecast_std is not None:
            return mapping.default_forecast_std, "configured_std"
        if self.fallback_forecast_std is not None:
            return self.fallback_forecast_std, "fallback_error_std"
        raise ValueError("forecast_std unavailable; provide configured_std or --fallback-forecast-std")

    def _http_fetch(self, params: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "params": params,
            "timeout": 20.0,
            "trust_env": self.trust_env,
        }
        if self.proxy:
            kwargs["proxy"] = self.proxy
        response = httpx.get(self.base_url, **kwargs)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("OpenMeteo response must be a JSON object")
        return data


def open_meteo_cache_key(
    *,
    location: str,
    target_date: str,
    metric: str,
    as_of_time: str | datetime,
) -> str:
    parsed_as_of_time = _parse_datetime(as_of_time)
    text = "|".join(
        [
            "open_meteo",
            _normalize_location(location),
            target_date[:10],
            metric.strip().casefold(),
            parsed_as_of_time.isoformat(),
        ]
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _metric_to_open_meteo_daily(metric: str) -> str:
    normalized = metric.strip().casefold()
    if normalized == "high_temperature":
        return "temperature_2m_max"
    if normalized == "low_temperature":
        return "temperature_2m_min"
    if normalized == "average_temperature":
        return "temperature_2m_mean"
    raise ValueError(f"unsupported weather metric for OpenMeteo: {metric}")


def _extract_open_meteo_daily_value(raw: dict[str, Any], variable: str, target_date: str) -> float:
    daily = raw.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("OpenMeteo response missing daily data")
    dates = daily.get("time")
    values = daily.get(variable)
    if not isinstance(dates, list) or not isinstance(values, list):
        raise ValueError(f"OpenMeteo response missing daily {variable}")
    for index, value_date in enumerate(dates):
        if value_date == target_date:
            return float(values[index])
    raise ValueError(f"OpenMeteo response missing target date {target_date}")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value[:10])


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return _aware_datetime(value)
    return _aware_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _try_parse_datetime(value: str | None) -> datetime | None:
    if value is None or not str(value).strip():
        return None
    try:
        return _parse_datetime(str(value))
    except ValueError:
        return None


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
