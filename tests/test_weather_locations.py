import csv

from poly_alpha_lab.weather_locations import (
    OpenMeteoGeocodingClient,
    enrich_pending_locations,
    promote_location_suggestions,
)


def write_pending(path):
    path.write_text(
        "source,detected_location_name,detected_station_id,detected_station_name,country_hint,example_market_id,example_slug,example_question,resolution_source,suggested_latitude,suggested_longitude,suggested_action,status\n"
        "daily_capture_missing_forecast,Chengdu,,,,m1,chengdu,Will the highest temperature in Chengdu be 25C?,,,,,pending_manual_coordinates\n",
        encoding="utf-8",
    )


def write_locations(path):
    path.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes,default_forecast_std,std_source\n"
        "Milan,45.6306,8.7281,LIMC,Malpensa Intl Airport Station,Europe/Rome,test,0.8,manual\n",
        encoding="utf-8",
    )


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def geocoding_response(*names):
    return {
        "results": [
            {
                "name": name,
                "latitude": 30.0 + index,
                "longitude": 104.0 + index,
                "country": "China",
                "timezone": "Asia/Shanghai",
            }
            for index, name in enumerate(names)
        ]
    }


def test_enrich_pending_reads_csv_and_writes_suggestions(tmp_path):
    pending = tmp_path / "pending.csv"
    existing = tmp_path / "locations.csv"
    output = tmp_path / "suggestions.csv"
    write_pending(pending)
    write_locations(existing)
    client = OpenMeteoGeocodingClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda params: geocoding_response("Chengdu"),
    )

    summary = enrich_pending_locations(
        pending_path=pending,
        existing_path=existing,
        output_path=output,
        client=client,
    )
    rows = read_rows(output)

    assert summary["pending_total"] == 1
    assert rows[0]["detected_location_name"] == "Chengdu"
    assert rows[0]["status"] == "suggested"
    assert rows[0]["suggested_latitude"]


def test_enrich_pending_marks_multiple_candidates_with_rank(tmp_path):
    pending = tmp_path / "pending.csv"
    existing = tmp_path / "locations.csv"
    output = tmp_path / "suggestions.csv"
    write_pending(pending)
    write_locations(existing)
    client = OpenMeteoGeocodingClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda params: geocoding_response("Chengdu", "Chengdu Shi"),
    )

    enrich_pending_locations(
        pending_path=pending,
        existing_path=existing,
        output_path=output,
        client=client,
    )
    rows = read_rows(output)

    assert [row["rank"] for row in rows] == ["1", "2"]
    assert rows[1]["match_type"] == "ambiguous"
    assert rows[1]["status"] == "needs_manual_review"


def test_enrich_pending_marks_already_existing_location(tmp_path):
    pending = tmp_path / "pending.csv"
    existing = tmp_path / "locations.csv"
    output = tmp_path / "suggestions.csv"
    write_pending(pending)
    existing.write_text(
        "location_name,latitude,longitude,station_id,source_location_name,timezone,notes\n"
        "Chengdu,30.5728,104.0668,,Chengdu city centroid,Asia/Shanghai,test\n",
        encoding="utf-8",
    )

    summary = enrich_pending_locations(
        pending_path=pending,
        existing_path=existing,
        output_path=output,
        client=OpenMeteoGeocodingClient(cache_dir=tmp_path / "cache", fetcher=lambda params: geocoding_response("Chengdu")),
    )
    rows = read_rows(output)

    assert summary["status_counts"]["already_exists"] == 1
    assert rows[0]["status"] == "already_exists"


def test_promote_suggestions_only_high_confidence(tmp_path):
    suggestions = tmp_path / "suggestions.csv"
    locations = tmp_path / "locations.csv"
    output = tmp_path / "locations_updated.csv"
    write_locations(locations)
    suggestions.write_text(
        ",".join(
            [
                "detected_location_name",
                "detected_station_id",
                "detected_station_name",
                "source",
                "example_market_id",
                "example_question",
                "query_used",
                "suggested_location_name",
                "suggested_station_id",
                "suggested_latitude",
                "suggested_longitude",
                "suggested_country",
                "suggested_timezone",
                "provider",
                "confidence",
                "rank",
                "match_type",
                "status",
                "warning",
                "resolution_source",
            ]
        )
        + "\n"
        "Chengdu,,,src,m1,q,Chengdu,Chengdu,,30.572800,104.066800,China,Asia/Shanghai,open-meteo-geocoding,0.960,1,exact_name,suggested,,\n"
        "Lowtown,,,src,m2,q,Lowtown,Lowtown,,1,2,Nowhere,UTC,open-meteo-geocoding,0.600,1,ambiguous,suggested,,\n",
        encoding="utf-8",
    )

    summary = promote_location_suggestions(
        suggestions_path=suggestions,
        locations_path=locations,
        output_path=output,
        min_confidence=0.85,
        dry_run=True,
    )
    rows = read_rows(output)

    assert summary["promoted_count"] == 1
    assert summary["skipped_low_confidence"] == 1
    assert any(row["location_name"] == "Chengdu" for row in rows)
    assert "Chengdu" not in locations.read_text(encoding="utf-8")


def test_promote_preserves_station_and_city_centroid_fields(tmp_path):
    suggestions = tmp_path / "suggestions.csv"
    locations = tmp_path / "locations.csv"
    output = tmp_path / "locations_updated.csv"
    write_locations(locations)
    suggestions.write_text(
        ",".join(
            [
                "detected_location_name",
                "detected_station_id",
                "detected_station_name",
                "source",
                "example_market_id",
                "example_question",
                "query_used",
                "suggested_location_name",
                "suggested_station_id",
                "suggested_latitude",
                "suggested_longitude",
                "suggested_country",
                "suggested_timezone",
                "provider",
                "confidence",
                "rank",
                "match_type",
                "status",
                "warning",
                "resolution_source",
            ]
        )
        + "\n"
        "Milan,LIMC,Malpensa Intl Airport Station,src,m1,q,Malpensa,Milan,LIMC,45.630600,8.728100,Italy,Europe/Rome,open-meteo-geocoding,0.960,1,station_name,suggested,,\n",
        encoding="utf-8",
    )

    promote_location_suggestions(
        suggestions_path=suggestions,
        locations_path=locations,
        output_path=output,
        min_confidence=0.85,
        dry_run=True,
    )
    rows = read_rows(output)

    milan_rows = [row for row in rows if row["station_id"] == "LIMC"]
    assert len(milan_rows) == 1
    assert "Malpensa" in milan_rows[0]["source_location_name"]
