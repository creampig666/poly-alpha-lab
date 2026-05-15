"""Weather location enrichment and promotion helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


SUGGESTION_FIELDS = [
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

LOCATIONS_FIELDS = [
    "location_name",
    "latitude",
    "longitude",
    "station_id",
    "station_name",
    "source_location_name",
    "country",
    "timezone",
    "provider",
    "notes",
    "default_forecast_std",
    "std_source",
]


@dataclass(frozen=True)
class GeocodingCandidate:
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    timezone: str | None = None
    admin1: str | None = None


class OpenMeteoGeocodingClient:
    """Small read-only Open-Meteo geocoding client with file cache."""

    base_url = "https://geocoding-api.open-meteo.com/v1/search"

    def __init__(
        self,
        *,
        cache_dir: str | Path = "data/weather/geocoding_cache",
        proxy: str | None = None,
        trust_env: bool = True,
        fetcher: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.proxy = proxy
        self.trust_env = trust_env
        self.fetcher = fetcher or self._http_fetch

    def search(
        self,
        query: str,
        *,
        country_hint: str | None = None,
        limit: int = 10,
    ) -> list[GeocodingCandidate]:
        query = " ".join(str(query or "").split())
        if not query:
            return []
        params: dict[str, Any] = {
            "name": query,
            "count": limit,
            "language": "en",
            "format": "json",
        }
        if country_hint:
            params["country_code"] = country_hint
        cache_path = self.cache_dir / f"{geocoding_cache_key(query, country_hint)}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = self.fetcher(params)
            cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return _parse_geocoding_response(raw)

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
            raise ValueError("Open-Meteo geocoding response must be a JSON object")
        return data


def geocoding_cache_key(query: str, country_hint: str | None) -> str:
    text = "|".join(["open_meteo_geocoding", _normalize(query), _normalize(country_hint or "")])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def enrich_pending_locations(
    *,
    pending_path: str | Path,
    existing_path: str | Path,
    output_path: str | Path,
    provider: str = "open-meteo-geocoding",
    proxy: str | None = None,
    trust_env: bool = True,
    min_confidence: float = 0.70,
    limit: int | None = None,
    client: OpenMeteoGeocodingClient | None = None,
) -> dict[str, Any]:
    """Read locations_pending.csv and write geocoding suggestions."""

    if provider != "open-meteo-geocoding":
        raise ValueError("Only open-meteo-geocoding is supported")
    pending_rows = _read_csv_rows(pending_path)
    if limit is not None:
        pending_rows = pending_rows[: max(0, limit)]
    existing = _existing_index(existing_path)
    client = client or OpenMeteoGeocodingClient(proxy=proxy, trust_env=trust_env)
    suggestions: list[dict[str, Any]] = []
    for row in pending_rows:
        detected_location = _clean(row.get("detected_location_name"))
        detected_station_id = _clean(row.get("detected_station_id"))
        detected_station_name = _clean(row.get("detected_station_name"))
        country_hint = _clean(row.get("country_hint"))
        exists = _already_exists(
            existing,
            location_name=detected_location,
            station_id=detected_station_id,
        )
        if exists:
            suggestions.append(
                _base_suggestion(row)
                | {
                    "query_used": detected_station_name or detected_location,
                    "provider": provider,
                    "confidence": "1.000",
                    "rank": "1",
                    "match_type": "exact_name" if detected_location else "station_name",
                    "status": "already_exists",
                    "warning": "",
                }
            )
            continue

        query = detected_station_name or detected_location
        if not query:
            suggestions.append(
                _base_suggestion(row)
                | {
                    "query_used": "",
                    "provider": provider,
                    "confidence": "0.000",
                    "rank": "",
                    "match_type": "no_match",
                    "status": "no_match",
                    "warning": "missing_query",
                }
            )
            continue

        try:
            candidates = client.search(query, country_hint=country_hint, limit=10)
        except Exception as exc:
            suggestions.append(
                _base_suggestion(row)
                | {
                    "query_used": query,
                    "provider": provider,
                    "confidence": "0.000",
                    "rank": "",
                    "match_type": "no_match",
                    "status": "no_match",
                    "warning": f"geocoding_error:{type(exc).__name__}:{exc}",
                }
            )
            continue

        if not candidates:
            suggestions.append(
                _base_suggestion(row)
                | {
                    "query_used": query,
                    "provider": provider,
                    "confidence": "0.000",
                    "rank": "",
                    "match_type": "no_match",
                    "status": "no_match",
                    "warning": "no_geocoding_results",
                }
            )
            continue

        for rank, candidate in enumerate(candidates, start=1):
            confidence, match_type = _confidence_and_match_type(
                query=query,
                detected_location=detected_location,
                detected_station_name=detected_station_name,
                candidate=candidate,
                rank=rank,
            )
            status = "suggested" if rank == 1 and confidence >= min_confidence else "needs_manual_review"
            warning = "" if status == "suggested" else "ambiguous_or_low_confidence"
            suggestions.append(
                _base_suggestion(row)
                | {
                    "query_used": query,
                    "suggested_location_name": candidate.name,
                    "suggested_station_id": detected_station_id,
                    "suggested_latitude": f"{candidate.latitude:.6f}",
                    "suggested_longitude": f"{candidate.longitude:.6f}",
                    "suggested_country": candidate.country or "",
                    "suggested_timezone": candidate.timezone or "",
                    "provider": provider,
                    "confidence": f"{confidence:.3f}",
                    "rank": str(rank),
                    "match_type": match_type,
                    "status": status,
                    "warning": warning,
                }
            )

    _write_csv(output_path, SUGGESTION_FIELDS, suggestions)
    status_counts = _count_by(suggestions, "status")
    return {
        "pending_total": len(pending_rows),
        "suggestions_total": len(suggestions),
        "status_counts": status_counts,
        "high_confidence_suggestions": sum(
            1
            for row in suggestions
            if row.get("status") == "suggested" and _float(row.get("confidence")) >= min_confidence
        ),
        "output_path": str(output_path),
    }


def promote_location_suggestions(
    *,
    suggestions_path: str | Path,
    locations_path: str | Path,
    output_path: str | Path,
    min_confidence: float = 0.85,
    only_status: str = "suggested",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote high-confidence suggestions into a new locations CSV."""

    suggestions = _read_csv_rows(suggestions_path)
    existing_rows = _read_csv_rows(locations_path)
    normalized_existing = {_normalize(row.get("location_name")) for row in existing_rows}
    station_existing = {
        (_normalize(row.get("location_name")), _normalize(row.get("station_id")))
        for row in existing_rows
    }
    output_rows = [_normalize_location_row(row) for row in existing_rows]
    promoted = 0
    skipped_low = 0
    skipped_ambiguous = 0
    already_exists = 0
    for suggestion in suggestions:
        status = _clean(suggestion.get("status"))
        confidence = _float(suggestion.get("confidence"))
        match_type = _clean(suggestion.get("match_type"))
        if status == "already_exists":
            already_exists += 1
            continue
        if status != only_status:
            if match_type == "ambiguous":
                skipped_ambiguous += 1
            continue
        if confidence < min_confidence:
            skipped_low += 1
            continue
        if match_type == "ambiguous":
            skipped_ambiguous += 1
            continue
        row = _suggestion_to_location_row(suggestion)
        location_key = _normalize(row["location_name"])
        station_key = (location_key, _normalize(row.get("station_id")))
        if location_key in normalized_existing or station_key in station_existing:
            already_exists += 1
            continue
        output_rows.append(row)
        normalized_existing.add(location_key)
        station_existing.add(station_key)
        promoted += 1
    _write_csv(output_path, LOCATIONS_FIELDS, output_rows)
    return {
        "suggestions_total": len(suggestions),
        "promoted_count": promoted,
        "skipped_low_confidence": skipped_low,
        "skipped_ambiguous": skipped_ambiguous,
        "already_exists": already_exists,
        "output_path": str(output_path),
        "dry_run": dry_run,
    }


def enrichment_summary_to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Weather Location Enrichment Summary",
        "",
        f"- pending_total: `{summary.get('pending_total', 0)}`",
        f"- suggestions_total: `{summary.get('suggestions_total', 0)}`",
        f"- high_confidence_suggestions: `{summary.get('high_confidence_suggestions', 0)}`",
        f"- output_path: `{summary.get('output_path')}`",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in (summary.get("status_counts") or {}).items():
        lines.append(f"- `{status}`: `{count}`")
    return "\n".join(lines) + "\n"


def promotion_summary_to_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Weather Location Promotion Summary",
            "",
            f"- suggestions_total: `{summary.get('suggestions_total', 0)}`",
            f"- promoted_count: `{summary.get('promoted_count', 0)}`",
            f"- skipped_low_confidence: `{summary.get('skipped_low_confidence', 0)}`",
            f"- skipped_ambiguous: `{summary.get('skipped_ambiguous', 0)}`",
            f"- already_exists: `{summary.get('already_exists', 0)}`",
            f"- output_path: `{summary.get('output_path')}`",
            f"- dry_run: `{summary.get('dry_run')}`",
            "",
            "Formal locations.csv was not overwritten.",
        ]
    ) + "\n"


def _parse_geocoding_response(raw: dict[str, Any]) -> list[GeocodingCandidate]:
    results = raw.get("results")
    if not isinstance(results, list):
        return []
    candidates: list[GeocodingCandidate] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("latitude") is None or item.get("longitude") is None or not item.get("name"):
            continue
        candidates.append(
            GeocodingCandidate(
                name=str(item["name"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                country=_clean(item.get("country")),
                timezone=_clean(item.get("timezone")),
                admin1=_clean(item.get("admin1")),
            )
        )
    return candidates


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_csv(path: str | Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _existing_index(path: str | Path) -> dict[str, set[str]]:
    rows = _read_csv_rows(path)
    return {
        "locations": {_normalize(row.get("location_name")) for row in rows},
        "stations": {_normalize(row.get("station_id")) for row in rows if _clean(row.get("station_id"))},
    }


def _already_exists(
    existing: dict[str, set[str]],
    *,
    location_name: str,
    station_id: str,
) -> bool:
    if station_id and _normalize(station_id) in existing["stations"]:
        return True
    return bool(location_name and _normalize(location_name) in existing["locations"])


def _base_suggestion(row: dict[str, str]) -> dict[str, Any]:
    return {
        "detected_location_name": _clean(row.get("detected_location_name")),
        "detected_station_id": _clean(row.get("detected_station_id")),
        "detected_station_name": _clean(row.get("detected_station_name")),
        "source": _clean(row.get("source")),
        "example_market_id": _clean(row.get("example_market_id")),
        "example_question": _clean(row.get("example_question")),
        "suggested_location_name": "",
        "suggested_station_id": "",
        "suggested_latitude": "",
        "suggested_longitude": "",
        "suggested_country": "",
        "suggested_timezone": "",
        "resolution_source": _clean(row.get("resolution_source")),
    }


def _confidence_and_match_type(
    *,
    query: str,
    detected_location: str,
    detected_station_name: str,
    candidate: GeocodingCandidate,
    rank: int,
) -> tuple[float, str]:
    query_norm = _normalize(query)
    candidate_norm = _normalize(candidate.name)
    location_norm = _normalize(detected_location)
    station_norm = _normalize(detected_station_name)
    if rank > 1:
        return max(0.40, 0.68 - (rank - 1) * 0.06), "ambiguous"
    if query_norm and query_norm == candidate_norm:
        return max(0.88, 0.96 - (rank - 1) * 0.05), "exact_name"
    if location_norm and location_norm == candidate_norm:
        return max(0.86, 0.94 - (rank - 1) * 0.05), "normalized_name"
    if station_norm and station_norm == candidate_norm:
        return max(0.82, 0.90 - (rank - 1) * 0.05), "station_name"
    if query_norm and (query_norm in candidate_norm or candidate_norm in query_norm):
        return max(0.70, 0.82 - (rank - 1) * 0.05), "city_fallback"
    return max(0.40, 0.68 - (rank - 1) * 0.06), "ambiguous"


def _suggestion_to_location_row(suggestion: dict[str, str]) -> dict[str, Any]:
    station_id = _clean(suggestion.get("suggested_station_id")) or _clean(suggestion.get("detected_station_id"))
    location_name = _clean(suggestion.get("suggested_location_name")) or _clean(suggestion.get("detected_location_name"))
    station_name = _clean(suggestion.get("detected_station_name"))
    source_location_name = station_name or f"{location_name} city centroid"
    notes = "geocoding suggestion; manually review before strict use"
    return {
        "location_name": location_name,
        "latitude": _clean(suggestion.get("suggested_latitude")),
        "longitude": _clean(suggestion.get("suggested_longitude")),
        "station_id": station_id,
        "station_name": station_name,
        "source_location_name": source_location_name,
        "country": _clean(suggestion.get("suggested_country")),
        "timezone": _clean(suggestion.get("suggested_timezone")),
        "provider": _clean(suggestion.get("provider")) or "open-meteo-geocoding",
        "notes": notes,
        "default_forecast_std": "",
        "std_source": "",
    }


def _normalize_location_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "location_name": _clean(row.get("location_name")),
        "latitude": _clean(row.get("latitude")),
        "longitude": _clean(row.get("longitude")),
        "station_id": _clean(row.get("station_id")),
        "station_name": _clean(row.get("station_name")),
        "source_location_name": _clean(row.get("source_location_name")),
        "country": _clean(row.get("country")),
        "timezone": _clean(row.get("timezone")),
        "provider": _clean(row.get("provider")),
        "notes": _clean(row.get("notes")),
        "default_forecast_std": _clean(row.get("default_forecast_std")),
        "std_source": _clean(row.get("std_source")),
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clean(value: Any) -> str:
    return "" if value is None else " ".join(str(value).strip().split())


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _clean(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())
