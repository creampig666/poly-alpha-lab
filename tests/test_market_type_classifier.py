from datetime import date

from poly_alpha_lab.market_type_classifier import MarketType, classify_market_text


def test_classifier_identifies_sao_paulo_high_at_or_below_celsius() -> None:
    result = classify_market_text(
        "Will the highest temperature in Sao Paulo be 24\u00b0C or below on May 9?",
        end_date="2026-05-09T23:59:00+00:00",
    )

    assert result.market_type == MarketType.weather_temperature_threshold
    assert result.location_name == "Sao Paulo"
    assert result.metric == "high_temperature"
    assert result.comparator == "at_or_below"
    assert result.threshold_value == 24
    assert result.unit == "C"
    assert result.target_date == "2026-05-09"


def test_classifier_identifies_paris_high_above_celsius() -> None:
    result = classify_market_text(
        "Will the highest temperature in Paris be above 25\u00b0C on May 10?",
        reference_date=date(2026, 5, 8),
    )

    assert result.market_type == MarketType.weather_temperature_threshold
    assert result.location_name == "Paris"
    assert result.metric == "high_temperature"
    assert result.comparator == "above"
    assert result.threshold_value == 25
    assert result.unit == "C"
    assert result.target_date == "2026-05-10"


def test_classifier_identifies_chicago_low_below_fahrenheit() -> None:
    result = classify_market_text(
        "Will the low temperature in Chicago be below 40\u00b0F on May 12?",
        reference_date=date(2026, 5, 8),
    )

    assert result.market_type == MarketType.weather_temperature_threshold
    assert result.location_name == "Chicago"
    assert result.metric == "low_temperature"
    assert result.comparator == "below"
    assert result.threshold_value == 40
    assert result.unit == "F"
    assert result.target_date == "2026-05-12"


def test_classifier_returns_unknown_for_non_weather_market() -> None:
    result = classify_market_text("Will CPI be above 3% in April?")

    assert result.market_type == MarketType.unknown
    assert "not_weather_temperature_threshold" in result.warnings


def test_classifier_identifies_exact_bucket_sao_paulo_high() -> None:
    result = classify_market_text(
        "Will the highest temperature in Sao Paulo be 24\u00b0C on May 9?",
        end_date="2026-05-09T23:59:00+00:00",
    )

    assert result.market_type == MarketType.weather_temperature_exact_bucket
    assert result.location_name == "Sao Paulo"
    assert result.metric == "high_temperature"
    assert result.comparator == "exact_bucket"
    assert result.threshold_value == 24
    assert result.unit == "C"
    assert result.target_date == "2026-05-09"


def test_classifier_identifies_exact_bucket_chicago_low() -> None:
    result = classify_market_text(
        "Will the lowest temperature in Chicago be 40\u00b0F on May 12?",
        reference_date=date(2026, 5, 8),
    )

    assert result.market_type == MarketType.weather_temperature_exact_bucket
    assert result.location_name == "Chicago"
    assert result.metric == "low_temperature"
    assert result.comparator == "exact_bucket"
    assert result.unit == "F"


def test_classifier_does_not_treat_equity_price_as_weather_exact_bucket() -> None:
    result = classify_market_text("Will Apple hit $264 in May?")

    assert result.market_type != MarketType.weather_temperature_exact_bucket
    assert result.market_type != MarketType.weather_temperature_threshold


def test_classifier_identifies_or_higher_temperature_threshold() -> None:
    result = classify_market_text(
        "Will the highest temperature in New York City be 74\u00b0F or higher on May 8?",
        reference_date=date(2026, 5, 8),
    )

    assert result.market_type == MarketType.weather_temperature_threshold
    assert result.location_name == "New York City"
    assert result.comparator == "at_or_above"
