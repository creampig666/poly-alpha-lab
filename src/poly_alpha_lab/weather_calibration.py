"""Historical temperature forecast error calibration."""

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

CalibrationQuality = Literal["HIGH", "MEDIUM", "LOW", "INSUFFICIENT"]

SUPPORTED_GROUP_FIELDS = {
    "metric",
    "horizon_bucket",
    "location",
    "station_id",
    "forecast_source",
    "forecast_model",
    "month",
}

CALIBRATION_CSV_FIELDS = [
    "group_key",
    "metric",
    "horizon_bucket",
    "location",
    "station_id",
    "forecast_source",
    "forecast_model",
    "month",
    "n",
    "bias",
    "mean_error",
    "std_error",
    "mae",
    "rmse",
    "q05",
    "q25",
    "q50",
    "q75",
    "q95",
    "min_error",
    "max_error",
    "tail_abs_1",
    "tail_abs_2",
    "tail_abs_3",
    "calibration_quality",
    "min_samples_required",
    "bias_shrinkage_applied",
    "bias_raw",
    "bias_shrunk",
    "std_error_raw",
    "std_error_used",
    "quality_warnings",
]


class WeatherForecastErrorSample(BaseModel):
    location: str
    station_id: str | None = None
    metric: str
    target_date: str
    forecast_issued_at: str
    forecast_mean: float
    actual_value: float
    unit: str
    forecast_source: str | None = None
    forecast_model: str | None = None
    horizon_hours: float
    error: float
    horizon_bucket: str
    month: str
    warnings: list[str] = Field(default_factory=list)


class WeatherCalibrationSummary(BaseModel):
    group_key: str
    metric: str | None = None
    horizon_bucket: str | None = None
    location: str | None = None
    station_id: str | None = None
    forecast_source: str | None = None
    forecast_model: str | None = None
    month: str | None = None
    n: int
    bias: float
    mean_error: float
    std_error: float
    mae: float
    rmse: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    min_error: float
    max_error: float
    tail_abs_1: float
    tail_abs_2: float
    tail_abs_3: float
    calibration_quality: CalibrationQuality = "INSUFFICIENT"
    min_samples_required: int = 0
    bias_shrinkage_applied: bool = False
    bias_raw: float | None = None
    bias_shrunk: float | None = None
    std_error_raw: float | None = None
    std_error_used: float | None = None
    quality_warnings: list[str] = Field(default_factory=list)


def load_forecast_error_samples(input_path: str | Path) -> list[WeatherForecastErrorSample]:
    """Load forecast-vs-actual rows and compute forecast errors."""

    path = Path(input_path)
    samples: list[WeatherForecastErrorSample] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            samples.append(_sample_from_row(row))
    return samples


def fit_weather_calibration(
    input_path: str | Path,
    *,
    group_by: list[str] | None = None,
    min_samples: int = 20,
    bias_shrinkage_k: float = 30,
) -> list[WeatherCalibrationSummary]:
    """Fit calibration summaries from historical forecast errors."""

    group_by = normalize_group_by(group_by or ["metric", "horizon_bucket"])
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if bias_shrinkage_k < 0:
        raise ValueError("bias_shrinkage_k must be >= 0")
    groups: dict[str, list[WeatherForecastErrorSample]] = {}
    for sample in load_forecast_error_samples(input_path):
        key = calibration_group_key(sample, group_by)
        groups.setdefault(key, []).append(sample)
    summaries = [
        _summary_for_group(
            group_key,
            samples,
            group_by,
            min_samples_required=min_samples,
            bias_shrinkage_k=bias_shrinkage_k,
        )
        for group_key, samples in sorted(groups.items())
    ]
    return summaries


def write_calibration_json(
    summaries: list[WeatherCalibrationSummary],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([summary.model_dump() for summary in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(summaries)


def write_calibration_csv(
    summaries: list[WeatherCalibrationSummary],
    output_path: str | Path,
) -> int:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CALIBRATION_CSV_FIELDS)
        writer.writeheader()
        for summary in summaries:
            data = summary.model_dump()
            writer.writerow({field: data.get(field) for field in CALIBRATION_CSV_FIELDS})
    return len(summaries)


def load_calibration_summaries(path: str | Path) -> dict[str, WeatherCalibrationSummary]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("calibration JSON must contain a list of summaries")
    summaries = [WeatherCalibrationSummary.model_validate(_upgrade_summary_payload(item)) for item in data]
    return {summary.group_key: summary for summary in summaries}


def normalize_group_by(group_by: list[str] | str) -> list[str]:
    if isinstance(group_by, str):
        values = [item.strip() for item in group_by.split(",") if item.strip()]
    else:
        values = [item.strip() for item in group_by if item.strip()]
    invalid = [value for value in values if value not in SUPPORTED_GROUP_FIELDS]
    if invalid:
        raise ValueError(f"unsupported calibration group fields: {', '.join(invalid)}")
    return values


def calibration_group_key(sample: Any, group_by: list[str] | str) -> str:
    fields = normalize_group_by(group_by)
    parts = []
    for field_name in fields:
        value = getattr(sample, field_name, None)
        if value is None and isinstance(sample, dict):
            value = sample.get(field_name)
        parts.append(f"{field_name}={value or 'unknown'}")
    return "|".join(parts)


def horizon_bucket(horizon_hours: float) -> str:
    if horizon_hours < 12:
        return "0_12h"
    if horizon_hours < 24:
        return "12_24h"
    if horizon_hours < 48:
        return "24_48h"
    if horizon_hours < 72:
        return "48_72h"
    return "72h_plus"


def current_forecast_calibration_key(
    *,
    metric: str,
    target_date: str,
    forecast_issued_at: str | None,
    group_by: list[str] | str,
    location: str | None = None,
    station_id: str | None = None,
    forecast_source: str | None = None,
    forecast_model: str | None = None,
) -> tuple[str | None, str | None]:
    """Build a calibration group key for a current forecast."""

    if forecast_issued_at is None:
        return None, "missing_forecast_issued_at_for_calibration"
    try:
        issued_at = _parse_datetime(forecast_issued_at)
        target_dt, _ = _target_datetime(target_date=target_date, target_datetime=None)
    except ValueError:
        return None, "invalid_calibration_time"
    hours = (target_dt - issued_at).total_seconds() / 3600
    sample_like = {
        "metric": metric,
        "horizon_bucket": horizon_bucket(hours),
        "location": location,
        "station_id": station_id,
        "forecast_source": forecast_source,
        "forecast_model": forecast_model,
        "month": target_dt.date().isoformat()[:7],
    }
    return calibration_group_key(sample_like, group_by), None


def _sample_from_row(row: dict[str, str]) -> WeatherForecastErrorSample:
    target_dt, warnings = _target_datetime(
        target_date=_required(row, "target_date"),
        target_datetime=_optional_string(row.get("target_datetime")),
    )
    issued_at = _parse_datetime(_required(row, "forecast_issued_at"))
    horizon_hours = (target_dt - issued_at).total_seconds() / 3600
    forecast_mean = float(_required(row, "forecast_mean"))
    actual_value = float(_required(row, "actual_value"))
    error = actual_value - forecast_mean
    return WeatherForecastErrorSample(
        location=_required(row, "location"),
        station_id=_optional_string(row.get("station_id")),
        metric=_required(row, "metric"),
        target_date=_required(row, "target_date")[:10],
        forecast_issued_at=_required(row, "forecast_issued_at"),
        forecast_mean=forecast_mean,
        actual_value=actual_value,
        unit=_required(row, "unit"),
        forecast_source=_optional_string(row.get("forecast_source")),
        forecast_model=_optional_string(row.get("forecast_model")),
        horizon_hours=horizon_hours,
        error=error,
        horizon_bucket=horizon_bucket(horizon_hours),
        month=target_dt.date().isoformat()[:7],
        warnings=warnings,
    )


def _summary_for_group(
    group_key: str,
    samples: list[WeatherForecastErrorSample],
    group_by: list[str],
    *,
    min_samples_required: int,
    bias_shrinkage_k: float,
) -> WeatherCalibrationSummary:
    errors = [sample.error for sample in samples]
    n = len(errors)
    mean_error = sum(errors) / n
    std_error = _sample_std(errors)
    bias_shrinkage_applied = bias_shrinkage_k > 0
    bias_shrunk = (n / (n + bias_shrinkage_k)) * mean_error if bias_shrinkage_applied else mean_error
    quality = calibration_quality_for_n(n, min_samples_required)
    sorted_errors = sorted(errors)
    context: dict[str, str | None] = {}
    for field_name in group_by:
        values = {str(getattr(sample, field_name) or "unknown") for sample in samples}
        context[field_name] = values.pop() if len(values) == 1 else "mixed"
    return WeatherCalibrationSummary(
        group_key=group_key,
        metric=context.get("metric"),
        horizon_bucket=context.get("horizon_bucket"),
        location=context.get("location"),
        station_id=context.get("station_id"),
        forecast_source=context.get("forecast_source"),
        forecast_model=context.get("forecast_model"),
        month=context.get("month"),
        n=n,
        bias=bias_shrunk,
        mean_error=mean_error,
        std_error=std_error,
        mae=sum(abs(error) for error in errors) / n,
        rmse=math.sqrt(sum(error * error for error in errors) / n),
        q05=_quantile(sorted_errors, 0.05),
        q25=_quantile(sorted_errors, 0.25),
        q50=_quantile(sorted_errors, 0.50),
        q75=_quantile(sorted_errors, 0.75),
        q95=_quantile(sorted_errors, 0.95),
        min_error=min(errors),
        max_error=max(errors),
        tail_abs_1=sum(1 for error in errors if abs(error) >= 1) / n,
        tail_abs_2=sum(1 for error in errors if abs(error) >= 2) / n,
        tail_abs_3=sum(1 for error in errors if abs(error) >= 3) / n,
        calibration_quality=quality,
        min_samples_required=min_samples_required,
        bias_shrinkage_applied=bias_shrinkage_applied,
        bias_raw=mean_error,
        bias_shrunk=bias_shrunk,
        std_error_raw=std_error,
        std_error_used=std_error,
        quality_warnings=calibration_quality_warnings(quality),
    )


def calibration_quality_for_n(n: int, min_samples: int) -> CalibrationQuality:
    if n < min_samples:
        return "INSUFFICIENT"
    if n < 2 * min_samples:
        return "LOW"
    if n < 5 * min_samples:
        return "MEDIUM"
    return "HIGH"


def calibration_quality_warnings(quality: CalibrationQuality) -> list[str]:
    if quality == "INSUFFICIENT":
        return ["insufficient_calibration_samples"]
    if quality == "LOW":
        return ["low_quality_calibration"]
    return []


def _upgrade_summary_payload(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    upgraded = dict(item)
    n = int(upgraded.get("n") or 0)
    min_samples_required = int(upgraded.get("min_samples_required") or max(n, 1))
    bias_raw = float(upgraded.get("bias_raw", upgraded.get("mean_error", upgraded.get("bias", 0.0))))
    bias_shrunk = float(upgraded.get("bias_shrunk", upgraded.get("bias", bias_raw)))
    std_error_raw = float(upgraded.get("std_error_raw", upgraded.get("std_error", 0.0)))
    std_error_used = float(upgraded.get("std_error_used", upgraded.get("std_error", std_error_raw)))
    quality = upgraded.get("calibration_quality") or calibration_quality_for_n(n, min_samples_required)
    upgraded.setdefault("calibration_quality", quality)
    upgraded.setdefault("min_samples_required", min_samples_required)
    upgraded.setdefault("bias_shrinkage_applied", bias_shrunk != bias_raw)
    upgraded.setdefault("bias_raw", bias_raw)
    upgraded.setdefault("bias_shrunk", bias_shrunk)
    upgraded.setdefault("std_error_raw", std_error_raw)
    upgraded.setdefault("std_error_used", std_error_used)
    upgraded.setdefault("quality_warnings", calibration_quality_warnings(quality))
    upgraded.setdefault("bias", bias_shrunk)
    upgraded.setdefault("std_error", std_error_used)
    return upgraded


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = q * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _target_datetime(*, target_date: str, target_datetime: str | None) -> tuple[datetime, list[str]]:
    if target_datetime:
        return _parse_datetime(target_datetime), []
    parsed_date = date.fromisoformat(target_date[:10])
    return datetime.combine(parsed_date, time(23, 59), tzinfo=UTC), [
        "target_datetime_approximated_from_date"
    ]


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _required(row: dict[str, str], field: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"forecast actual history missing required field: {field}")
    return value.strip()


def _optional_string(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()
