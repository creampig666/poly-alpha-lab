from datetime import UTC, datetime

import pytest

from poly_alpha_lab.weather_data import (
    CsvWeatherDataProvider,
    LocationResolver,
    OpenMeteoForecastProvider,
    WeatherForecast,
    open_meteo_cache_key,
)


def write_locations(path):
    path.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes,default_forecast_std,std_source\n"
        "Milan,45.6306,8.7281,LIMC,Malpensa Intl Airport Station,Europe/Rome,test,0.8,station history\n"
        "Sao Paulo,-23.5505,-46.6333,,Sao Paulo city centroid,America/Sao_Paulo,test,1.2,city history\n",
        encoding="utf-8",
    )


def open_meteo_response(value=24.0):
    return {
        "daily": {
            "time": ["2026-05-08"],
            "temperature_2m_max": [value],
        }
    }


def fixed_now():
    return datetime(2026, 5, 7, 12, 0, tzinfo=UTC)


def test_locations_csv_resolves_milan_and_sao_paulo(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    write_locations(locations)
    resolver = LocationResolver(locations)

    milan = resolver.resolve("Milan")
    sao_paulo = resolver.resolve("Sao Paulo")

    assert milan is not None
    assert milan.station_id == "LIMC"
    assert sao_paulo is not None
    assert sao_paulo.latitude == pytest.approx(-23.5505)


def test_location_resolver_prefers_station_id(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    write_locations(locations)
    resolver = LocationResolver(locations)

    resolved = resolver.resolve("Sao Paulo", station_id="LIMC")

    assert resolved is not None
    assert resolved.location_name == "Milan"
    assert resolved.station_id == "LIMC"


def test_open_meteo_cache_key_is_stable() -> None:
    first = open_meteo_cache_key(
        location="Milan",
        target_date="2026-05-08",
        metric="high_temperature",
        as_of_time="2026-05-08T01:00:00Z",
    )
    second = open_meteo_cache_key(
        location=" Milan ",
        target_date="2026-05-08T12:00:00Z",
        metric="HIGH_TEMPERATURE",
        as_of_time=datetime(2026, 5, 8, 1, tzinfo=UTC),
    )

    assert first == second


def test_open_meteo_provider_mock_response(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    cache_dir = tmp_path / "cache"
    write_locations(locations)
    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        cache_dir=cache_dir,
        fetcher=lambda params: open_meteo_response(24.5),
        now=fixed_now,
    )

    forecast = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-07T12:00:00Z",
    )

    assert isinstance(forecast, WeatherForecast)
    assert forecast.forecast_mean == pytest.approx(24.5)
    assert forecast.forecast_source == "open_meteo"
    assert forecast.forecast_model == "open_meteo_current_forecast"
    assert forecast.std_method == "configured_std"
    assert forecast.cache_key is not None
    assert forecast.raw_data_reference is not None


def test_open_meteo_station_missing_falls_back_to_city_centroid_warning(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    cache_dir = tmp_path / "cache"
    write_locations(locations)
    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        cache_dir=cache_dir,
        fetcher=lambda params: open_meteo_response(24.5),
        now=fixed_now,
    )

    forecast = provider.get_forecast(
        "Sao Paulo",
        "2026-05-08",
        "high_temperature",
        station_id="SBGR",
    )

    assert forecast is not None
    assert "station_forecast_fallback_to_city_centroid" in forecast.provider_warnings
    assert forecast.station_id is None


def test_open_meteo_provider_uses_cache_without_refetch(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    cache_dir = tmp_path / "cache"
    write_locations(locations)
    calls = {"count": 0}

    def fetcher(params):
        calls["count"] += 1
        return open_meteo_response(24.5)

    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        cache_dir=cache_dir,
        fetcher=fetcher,
        now=fixed_now,
    )
    first = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-07T12:00:00Z",
    )
    second = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-07T12:00:00Z",
    )

    assert calls["count"] == 1
    assert first is not None and second is not None
    assert first.cache_key == second.cache_key


def test_current_forecast_cannot_target_past_date(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    write_locations(locations)
    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        cache_dir=tmp_path / "no-std-cache",
        fetcher=lambda params: open_meteo_response(24.5),
        now=fixed_now,
    )

    with pytest.raises(ValueError, match="past target_date"):
        provider.get_forecast("Milan", "2026-05-06", "high_temperature")


def test_fallback_std_used_when_config_missing(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    locations.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes\n"
        "Milan,45.6306,8.7281,LIMC,Malpensa Intl Airport Station,Europe/Rome,test\n",
        encoding="utf-8",
    )
    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        fallback_forecast_std=1.4,
        fetcher=lambda params: open_meteo_response(24.5),
        now=fixed_now,
    )

    forecast = provider.get_forecast("Milan", "2026-05-08", "high_temperature")

    assert forecast is not None
    assert forecast.forecast_std == pytest.approx(1.4)
    assert forecast.std_method == "fallback_error_std"


def test_no_silent_default_std(tmp_path) -> None:
    locations = tmp_path / "locations.csv"
    locations.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes\n"
        "Milan,45.6306,8.7281,LIMC,Malpensa Intl Airport Station,Europe/Rome,test\n",
        encoding="utf-8",
    )
    provider = OpenMeteoForecastProvider(
        location_resolver=LocationResolver(locations),
        cache_dir=tmp_path / "no-std-cache",
        fetcher=lambda params: open_meteo_response(24.5),
        now=fixed_now,
    )

    with pytest.raises(ValueError, match="forecast_std unavailable"):
        provider.get_forecast("Milan", "2026-05-08", "high_temperature")


def test_csv_multi_snapshot_selects_latest_prior_as_of_time(tmp_path) -> None:
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-08,Milan,high_temperature,21,1,,C,2026-05-07T00:00:00Z\n"
        "2026-05-08,Milan,high_temperature,22,1,,C,2026-05-07T06:00:00Z\n"
        "2026-05-08,Milan,high_temperature,23,1,,C,2026-05-07T10:00:00Z\n",
        encoding="utf-8",
    )
    provider = CsvWeatherDataProvider(csv_path)

    selected = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-07T07:00:00Z",
    )

    assert selected is not None
    assert selected.forecast_mean == pytest.approx(22)


def test_csv_as_of_time_before_all_forecasts_returns_none(tmp_path) -> None:
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-08,Milan,high_temperature,21,1,,C,2026-05-07T00:00:00Z\n"
        "2026-05-08,Milan,high_temperature,22,1,,C,2026-05-07T06:00:00Z\n",
        encoding="utf-8",
    )
    provider = CsvWeatherDataProvider(csv_path)

    selected = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-06T23:00:00Z",
    )

    assert selected is None


def test_csv_missing_forecast_issued_at_multi_rows_not_silently_selected(tmp_path) -> None:
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-08,Milan,high_temperature,21,1,,C,\n"
        "2026-05-08,Milan,high_temperature,22,1,,C,\n",
        encoding="utf-8",
    )
    provider = CsvWeatherDataProvider(csv_path)

    assert provider.get_forecast("Milan", "2026-05-08", "high_temperature") is None
    assert (
        provider.get_forecast(
            "Milan",
            "2026-05-08",
            "high_temperature",
            as_of_time="2026-05-08T01:00:00Z",
        )
        is None
    )


def test_csv_single_missing_forecast_issued_at_can_be_used_with_replay_warning(tmp_path) -> None:
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-08,Milan,high_temperature,21,1,,C,\n",
        encoding="utf-8",
    )
    provider = CsvWeatherDataProvider(csv_path)

    selected = provider.get_forecast(
        "Milan",
        "2026-05-08",
        "high_temperature",
        as_of_time="2026-05-08T01:00:00Z",
    )

    assert selected is not None
    assert "missing_forecast_issued_at_single_snapshot_used_for_replay" in selected.provider_warnings


def test_csv_multiple_snapshots_without_as_of_time_marks_manual_review_warning(tmp_path) -> None:
    csv_path = tmp_path / "weather.csv"
    csv_path.write_text(
        "date,location,metric,forecast_mean,forecast_std,actual_value,unit,forecast_issued_at\n"
        "2026-05-08,Milan,high_temperature,21,1,,C,2026-05-07T00:00:00Z\n"
        "2026-05-08,Milan,high_temperature,22,1,,C,2026-05-07T06:00:00Z\n",
        encoding="utf-8",
    )
    provider = CsvWeatherDataProvider(csv_path)

    selected = provider.get_forecast("Milan", "2026-05-08", "high_temperature")

    assert selected is not None
    assert selected.forecast_mean == pytest.approx(22)
    assert "as_of_time_missing_multiple_forecasts" in selected.provider_warnings
