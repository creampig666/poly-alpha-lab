"""Temperature threshold probability model."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.weather_data import WeatherForecast

Comparator = Literal[
    "above",
    "below",
    "at_or_above",
    "at_or_below",
    "equals",
    "exact_bucket",
]
BucketMode = Literal["rounded", "floor"]
TemperatureUnit = Literal["C", "F"]
WeatherModel = Literal["normal", "student_t", "normal_mixture"]


class WeatherProbabilityError(ValueError):
    """Raised when a weather probability cannot be estimated safely."""


class WeatherProbabilityResult(BaseModel):
    model_p_yes: float = Field(ge=0, le=1)
    model_name: str = "normal_temperature_threshold_v1"
    assumptions: list[str]
    warnings: list[str] = Field(default_factory=list)
    bucket_mode: BucketMode | None = None
    bucket_lower_bound: float | None = None
    bucket_upper_bound: float | None = None
    bucket_assumption: str | None = None
    weather_model: WeatherModel = "normal"
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    distribution_assumption: str = "normal forecast error"
    tail_model_warning: str | None = None


def estimate_temperature_threshold_probability(
    *,
    forecast: WeatherForecast,
    threshold: float,
    comparator: Comparator,
    threshold_unit: TemperatureUnit,
    bucket_mode: BucketMode = "rounded",
    weather_model: WeatherModel = "normal",
    student_t_df: float = 5,
    mixture_tail_weight: float = 0.10,
    mixture_tail_scale: float = 2.5,
) -> WeatherProbabilityResult:
    """Estimate YES probability using a configurable forecast error distribution."""

    if forecast.forecast_std <= 0:
        raise WeatherProbabilityError("forecast_std must be greater than 0")
    weather_model = _normalize_weather_model(weather_model)
    model_parameters = _validate_model_parameters(
        weather_model=weather_model,
        student_t_df=student_t_df,
        mixture_tail_weight=mixture_tail_weight,
        mixture_tail_scale=mixture_tail_scale,
    )
    forecast_unit = _normalize_unit(forecast.unit)
    threshold_unit = _normalize_unit(threshold_unit)
    warnings: list[str] = []
    forecast_mean = forecast.forecast_mean
    forecast_std = forecast.forecast_std
    if forecast_unit != threshold_unit:
        forecast_mean = convert_temperature(forecast_mean, forecast_unit, threshold_unit)
        forecast_std = convert_temperature_std(forecast_std, forecast_unit, threshold_unit)
        warnings.append("converted_forecast_unit")

    cdf = _distribution_cdf(
        weather_model=weather_model,
        mean=forecast_mean,
        std=forecast_std,
        student_t_df=student_t_df,
        mixture_tail_weight=mixture_tail_weight,
        mixture_tail_scale=mixture_tail_scale,
    )
    bucket_lower_bound: float | None = None
    bucket_upper_bound: float | None = None
    bucket_assumption: str | None = None
    result_bucket_mode: BucketMode | None = None
    if comparator in {"below", "at_or_below"}:
        probability = cdf(threshold)
        comparison = "T <= threshold"
    elif comparator in {"above", "at_or_above"}:
        probability = 1 - cdf(threshold)
        comparison = "T >= threshold"
    elif comparator in {"equals", "exact_bucket"}:
        if bucket_mode not in {"rounded", "floor"}:
            raise WeatherProbabilityError("bucket_mode must be rounded or floor")
        result_bucket_mode = bucket_mode
        if bucket_mode == "rounded":
            bucket_lower_bound = threshold - 0.5
            bucket_upper_bound = threshold + 0.5
            bucket_assumption = "rounded integer bucket [K - 0.5, K + 0.5)"
        else:
            bucket_lower_bound = threshold
            bucket_upper_bound = threshold + 1
            bucket_assumption = "floor integer bucket [K, K + 1)"
        probability = cdf(bucket_upper_bound) - cdf(bucket_lower_bound)
        comparison = (
            "Exact temperature market is modeled as integer bucket, "
            "not literal real-valued equality."
        )
    else:
        raise WeatherProbabilityError(f"unsupported comparator: {comparator}")

    probability = min(1.0, max(0.0, probability))
    distribution_assumption = _distribution_assumption(weather_model)
    tail_model_warning = (
        "tail model changes center and tail bucket probabilities; compare against normal baseline"
        if weather_model != "normal"
        else None
    )
    return WeatherProbabilityResult(
        model_p_yes=probability,
        model_name=f"{weather_model}_temperature_threshold_v1",
        assumptions=[
            f"temperature follows {distribution_assumption}",
            "strict and inclusive threshold comparators are approximated identically",
            comparison,
            f"bucket_mode={bucket_mode}" if result_bucket_mode else "bucket_mode=n/a",
            f"bucket_interpretation={bucket_assumption}" if bucket_assumption else "bucket_interpretation=n/a",
            "actual_value is not used for prediction",
        ],
        warnings=warnings,
        bucket_mode=result_bucket_mode,
        bucket_lower_bound=bucket_lower_bound,
        bucket_upper_bound=bucket_upper_bound,
        bucket_assumption=bucket_assumption,
        weather_model=weather_model,
        model_parameters=model_parameters,
        distribution_assumption=distribution_assumption,
        tail_model_warning=tail_model_warning,
    )


def convert_temperature(value: float, from_unit: TemperatureUnit, to_unit: TemperatureUnit) -> float:
    """Convert a temperature value between Celsius and Fahrenheit."""

    from_unit = _normalize_unit(from_unit)
    to_unit = _normalize_unit(to_unit)
    if from_unit == to_unit:
        return value
    if from_unit == "C" and to_unit == "F":
        return value * 9 / 5 + 32
    return (value - 32) * 5 / 9


def convert_temperature_std(value: float, from_unit: TemperatureUnit, to_unit: TemperatureUnit) -> float:
    """Convert temperature standard deviation between units."""

    from_unit = _normalize_unit(from_unit)
    to_unit = _normalize_unit(to_unit)
    if from_unit == to_unit:
        return value
    if from_unit == "C" and to_unit == "F":
        return value * 9 / 5
    return value * 5 / 9


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _distribution_cdf(
    *,
    weather_model: WeatherModel,
    mean: float,
    std: float,
    student_t_df: float,
    mixture_tail_weight: float,
    mixture_tail_scale: float,
):
    if weather_model == "normal":
        return lambda value: _normal_cdf((value - mean) / std)
    if weather_model == "student_t":
        scale = std * math.sqrt((student_t_df - 2) / student_t_df)
        return lambda value: _student_t_cdf((value - mean) / scale, student_t_df)
    if weather_model == "normal_mixture":
        return lambda value: (
            (1 - mixture_tail_weight) * _normal_cdf((value - mean) / std)
            + mixture_tail_weight * _normal_cdf((value - mean) / (mixture_tail_scale * std))
        )
    raise WeatherProbabilityError(f"unsupported weather_model: {weather_model}")


def _student_t_cdf(value: float, df: float) -> float:
    try:
        from scipy.stats import t as scipy_t  # type: ignore

        return float(scipy_t.cdf(value, df))
    except ImportError:
        return _student_t_cdf_fallback(value, df)


def _student_t_cdf_fallback(value: float, df: float) -> float:
    if value == 0:
        return 0.5
    sign = 1 if value > 0 else -1
    limit = abs(value)
    steps = max(256, int(limit * 256))
    if steps % 2:
        steps += 1
    step = limit / steps
    total = _student_t_pdf(0, df) + _student_t_pdf(limit, df)
    for index in range(1, steps):
        multiplier = 4 if index % 2 else 2
        total += multiplier * _student_t_pdf(index * step, df)
    area = total * step / 3
    probability = 0.5 + sign * area
    return min(1.0, max(0.0, probability))


def _student_t_pdf(value: float, df: float) -> float:
    coefficient = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return coefficient * (1 + value * value / df) ** (-(df + 1) / 2)


def _normalize_weather_model(value: str) -> WeatherModel:
    normalized = value.strip().casefold()
    if normalized not in {"normal", "student_t", "normal_mixture"}:
        raise WeatherProbabilityError("weather_model must be normal, student_t, or normal_mixture")
    return normalized  # type: ignore[return-value]


def _validate_model_parameters(
    *,
    weather_model: WeatherModel,
    student_t_df: float,
    mixture_tail_weight: float,
    mixture_tail_scale: float,
) -> dict[str, Any]:
    if weather_model == "student_t":
        if student_t_df <= 2:
            raise WeatherProbabilityError("student_t_df must be greater than 2")
        return {"student_t_df": student_t_df}
    if weather_model == "normal_mixture":
        if not 0 <= mixture_tail_weight <= 1:
            raise WeatherProbabilityError("mixture_tail_weight must be between 0 and 1")
        if mixture_tail_scale < 1:
            raise WeatherProbabilityError("mixture_tail_scale must be greater than or equal to 1")
        return {
            "mixture_tail_weight": mixture_tail_weight,
            "mixture_tail_scale": mixture_tail_scale,
        }
    return {}


def _distribution_assumption(weather_model: WeatherModel) -> str:
    if weather_model == "student_t":
        return "student_t forecast error with variance matched to forecast_std"
    if weather_model == "normal_mixture":
        return "normal mixture forecast error with tail component"
    return "normal forecast error"


def _normalize_unit(unit: str) -> TemperatureUnit:
    text = unit.strip().upper().replace("°", "")
    if text in {"C", "CELSIUS"}:
        return "C"
    if text in {"F", "FAHRENHEIT"}:
        return "F"
    raise WeatherProbabilityError("temperature unit must be C or F")
