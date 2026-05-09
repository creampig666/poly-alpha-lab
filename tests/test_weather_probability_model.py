import pytest

from poly_alpha_lab.weather_data import WeatherForecast
from poly_alpha_lab.weather_probability_model import (
    WeatherProbabilityError,
    estimate_temperature_threshold_probability,
)


def forecast(**overrides):
    data = {
        "date": "2026-05-09",
        "location": "Sao Paulo",
        "metric": "high_temperature",
        "forecast_mean": 24.0,
        "forecast_std": 2.0,
        "actual_value": 999.0,
        "unit": "C",
        "forecast_issued_at": "2026-05-08T00:00:00Z",
    }
    data.update(overrides)
    return WeatherForecast(**data)


def test_normal_model_temperature_at_or_below_probability() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=2),
        threshold=24,
        comparator="at_or_below",
        threshold_unit="C",
    )

    assert result.model_p_yes == pytest.approx(0.5)


def test_normal_model_temperature_at_or_above_probability() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=20, forecast_std=2),
        threshold=22,
        comparator="at_or_above",
        threshold_unit="C",
    )

    assert result.model_p_yes == pytest.approx(0.158655, rel=1e-4)


def test_forecast_std_must_be_positive() -> None:
    with pytest.raises(WeatherProbabilityError):
        estimate_temperature_threshold_probability(
            forecast=forecast(forecast_std=0),
            threshold=24,
            comparator="at_or_below",
            threshold_unit="C",
        )


def test_temperature_unit_conversion() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=20, forecast_std=1, unit="C"),
        threshold=68,
        comparator="at_or_above",
        threshold_unit="F",
    )

    assert result.model_p_yes == pytest.approx(0.5)
    assert "converted_forecast_unit" in result.warnings


def test_exact_bucket_rounded_probability() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        bucket_mode="rounded",
    )

    assert result.model_p_yes == pytest.approx(0.3829, rel=1e-3)
    assert result.bucket_lower_bound == pytest.approx(23.5)
    assert result.bucket_upper_bound == pytest.approx(24.5)
    assert result.bucket_mode == "rounded"
    assert any("not literal real-valued equality" in item for item in result.assumptions)


def test_exact_bucket_floor_probability() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        bucket_mode="floor",
    )

    assert result.model_p_yes == pytest.approx(0.3413, rel=1e-3)
    assert result.bucket_lower_bound == pytest.approx(24)
    assert result.bucket_upper_bound == pytest.approx(25)
    assert result.bucket_mode == "floor"


def test_exact_bucket_probability_is_not_real_valued_equality_zero() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
    )

    assert result.model_p_yes > 0


def test_actual_value_is_not_used_for_model_probability() -> None:
    base = forecast(actual_value=10)
    changed_actual = forecast(actual_value=100)

    first = estimate_temperature_threshold_probability(
        forecast=base,
        threshold=24,
        comparator="at_or_below",
        threshold_unit="C",
    )
    second = estimate_temperature_threshold_probability(
        forecast=changed_actual,
        threshold=24,
        comparator="at_or_below",
        threshold_unit="C",
    )

    assert first.model_p_yes == second.model_p_yes


def test_student_t_df_must_be_greater_than_two() -> None:
    with pytest.raises(WeatherProbabilityError, match="student_t_df"):
        estimate_temperature_threshold_probability(
            forecast=forecast(),
            threshold=24,
            comparator="exact_bucket",
            threshold_unit="C",
            weather_model="student_t",
            student_t_df=2,
        )


def test_student_t_exact_bucket_probability_is_nonzero() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="student_t",
        student_t_df=5,
    )

    assert result.model_p_yes > 0
    assert result.weather_model == "student_t"
    assert result.model_parameters["student_t_df"] == 5
    assert "student_t forecast error" in result.distribution_assumption


def test_student_t_center_bucket_differs_from_normal() -> None:
    normal = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="normal",
    )
    student_t = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="student_t",
        student_t_df=5,
    )

    assert student_t.model_p_yes != pytest.approx(normal.model_p_yes)


def test_student_t_far_bucket_probability_higher_than_normal() -> None:
    normal = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=30,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="normal",
    )
    student_t = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=30,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="student_t",
        student_t_df=5,
    )

    assert student_t.model_p_yes > normal.model_p_yes


def test_normal_mixture_exact_bucket_probability_is_nonzero() -> None:
    result = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=24,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="normal_mixture",
        mixture_tail_weight=0.1,
        mixture_tail_scale=2.5,
    )

    assert result.model_p_yes > 0
    assert result.weather_model == "normal_mixture"
    assert result.model_parameters["mixture_tail_weight"] == pytest.approx(0.1)


def test_normal_mixture_far_bucket_probability_higher_than_normal() -> None:
    normal = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=30,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="normal",
    )
    mixture = estimate_temperature_threshold_probability(
        forecast=forecast(forecast_mean=24, forecast_std=1),
        threshold=30,
        comparator="exact_bucket",
        threshold_unit="C",
        weather_model="normal_mixture",
        mixture_tail_weight=0.1,
        mixture_tail_scale=2.5,
    )

    assert mixture.model_p_yes > normal.model_p_yes


def test_normal_mixture_parameters_are_validated() -> None:
    with pytest.raises(WeatherProbabilityError, match="mixture_tail_weight"):
        estimate_temperature_threshold_probability(
            forecast=forecast(),
            threshold=24,
            comparator="exact_bucket",
            threshold_unit="C",
            weather_model="normal_mixture",
            mixture_tail_weight=1.5,
        )
    with pytest.raises(WeatherProbabilityError, match="mixture_tail_scale"):
        estimate_temperature_threshold_probability(
            forecast=forecast(),
            threshold=24,
            comparator="exact_bucket",
            threshold_unit="C",
            weather_model="normal_mixture",
            mixture_tail_scale=0.5,
        )
