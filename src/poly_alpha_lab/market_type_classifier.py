"""Rule-based market type classification for alpha modules."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MarketType(str, Enum):
    weather_temperature_threshold = "weather_temperature_threshold"
    weather_temperature_exact_bucket = "weather_temperature_exact_bucket"
    weather_precipitation_threshold = "weather_precipitation_threshold"
    crypto_price_threshold = "crypto_price_threshold"
    equity_price_threshold = "equity_price_threshold"
    sports_match = "sports_match"
    politics = "politics"
    longshot = "longshot"
    unknown = "unknown"


TemperatureMetric = Literal[
    "high_temperature",
    "low_temperature",
    "average_temperature",
]
ThresholdComparator = Literal[
    "above",
    "below",
    "at_or_above",
    "at_or_below",
    "exact_bucket",
]
TemperatureUnit = Literal["C", "F"]

MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


class MarketClassification(BaseModel):
    """Structured classification parsed from a market question."""

    market_type: MarketType = MarketType.unknown
    location_name: str | None = None
    metric: TemperatureMetric | None = None
    comparator: ThresholdComparator | None = None
    threshold_value: float | None = None
    unit: TemperatureUnit | None = None
    target_date: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)


def classify_strategy_candidate(candidate: dict[str, Any]) -> MarketClassification:
    """Classify one strategy candidate dictionary from strategy JSON."""

    question = str(candidate.get("question") or "")
    end_date = _candidate_end_date(candidate)
    return classify_market_text(
        question,
        slug=candidate.get("slug"),
        category=candidate.get("category"),
        end_date=end_date,
    )


def classify_market_text(
    question: str,
    *,
    slug: str | None = None,
    category: str | None = None,
    end_date: str | date | datetime | None = None,
    reference_date: date | None = None,
) -> MarketClassification:
    """Classify supported market types.

    v0.7.1 intentionally supports weather temperature threshold and exact
    integer bucket markets with enough structure to calculate a replayable
    model probability.
    """

    text = _normalize_text(" ".join(part for part in (question, slug or "") if part))
    warnings: list[str] = []

    if not _looks_like_temperature_market(text, category):
        return MarketClassification(
            market_type=_non_weather_type_hint(text, category),
            confidence=0.0,
            warnings=["not_weather_temperature_threshold"],
        )

    metric = _extract_metric(text)
    comparator = _extract_comparator(text)
    threshold_value, unit = _extract_threshold(text)
    target_date = _extract_target_date(
        text,
        end_date=end_date,
        reference_date=reference_date,
    )
    location = _extract_location(question)

    missing: list[str] = []
    if metric is None:
        missing.append("metric")
    if comparator is None:
        missing.append("comparator")
    if threshold_value is None:
        missing.append("threshold_value")
    if unit is None:
        missing.append("unit")
    if target_date is None:
        missing.append("target_date")
    if location is None:
        missing.append("location_name")

    if missing:
        warnings.extend(f"missing_{field}" for field in missing)
        return MarketClassification(
            market_type=MarketType.unknown,
            location_name=location,
            metric=metric,
            comparator=comparator,
            threshold_value=threshold_value,
            unit=unit,
            target_date=target_date,
            confidence=0.35,
            warnings=warnings,
        )

    market_type = (
        MarketType.weather_temperature_exact_bucket
        if comparator == "exact_bucket"
        else MarketType.weather_temperature_threshold
    )

    return MarketClassification(
        market_type=market_type,
        location_name=location,
        metric=metric,
        comparator=comparator,
        threshold_value=threshold_value,
        unit=unit,
        target_date=target_date,
        confidence=0.95,
        warnings=warnings,
    )


def _looks_like_temperature_market(text: str, category: str | None) -> bool:
    has_weather_category = (category or "").casefold() == "weather"
    has_temperature = re.search(r"\b(temp|temperature|celsius|fahrenheit)\b|°\s*[cf]\b", text) is not None
    has_unit = re.search(r"(?:°\s*)?[cf]\b|celsius|fahrenheit", text) is not None
    return (has_weather_category or has_temperature) and has_unit


def _non_weather_type_hint(text: str, category: str | None) -> MarketType:
    category_text = (category or "").casefold()
    if "weather" in category_text and re.search(r"\b(rain|rainfall|hurricane|snow)\b", text):
        return MarketType.weather_precipitation_threshold
    if re.search(r"\b(bitcoin|btc|ethereum|eth|crypto)\b", text):
        return MarketType.crypto_price_threshold
    if re.search(r"\b(stock|earnings|market cap|ipo)\b", text):
        return MarketType.equity_price_threshold
    if "politic" in category_text or re.search(r"\b(election|president|senate|house)\b", text):
        return MarketType.politics
    if re.search(r"\b(winner|champion|outright|longshot)\b", text):
        return MarketType.longshot
    return MarketType.unknown


def _extract_metric(text: str) -> TemperatureMetric | None:
    if re.search(r"\b(highest|max(?:imum)?|high)\s+temperature\b", text):
        return "high_temperature"
    if re.search(r"\b(lowest|min(?:imum)?|low)\s+temperature\b", text):
        return "low_temperature"
    if re.search(r"\baverage\s+temperature\b", text):
        return "average_temperature"
    if re.search(r"\btemperature\b", text):
        return "high_temperature"
    return None


def _extract_comparator(text: str) -> ThresholdComparator | None:
    if re.search(r"\b(or below|or lower|at most|no more than|equal to or below)\b", text):
        return "at_or_below"
    if re.search(r"\b(or above|or higher|at least|no less than|equal to or above)\b", text):
        return "at_or_above"
    if re.search(r"\b(above|over|exceed|exceeds|greater than|more than)\b", text):
        return "above"
    if re.search(r"\b(below|under|less than)\b", text):
        return "below"
    if re.search(
        r"\bbe\s+-?\d+(?:\.\d+)?\s*(?:°\s*)?(?:celsius|fahrenheit|c|f)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return "exact_bucket"
    return None


def _extract_threshold(text: str) -> tuple[float | None, TemperatureUnit | None]:
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(celsius|fahrenheit|c|f)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    value = float(match.group(1))
    unit_text = match.group(2).casefold()
    unit: TemperatureUnit = "C" if unit_text in {"c", "celsius"} else "F"
    return value, unit


def _extract_target_date(
    text: str,
    *,
    end_date: str | date | datetime | None,
    reference_date: date | None,
) -> str | None:
    match = re.search(
        r"\b(?:on|by|before|after)?\s*"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
        r"dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    month = MONTHS[match.group(1).casefold()]
    day = int(match.group(2))
    explicit_year = match.group(3)
    year = int(explicit_year) if explicit_year else _infer_year(end_date, reference_date)
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_location(question: str) -> str | None:
    patterns = [
        r"\bin\s+(.+?)\s+be\b",
        r"\bin\s+(.+?)\s+(?:on|by|before|after)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            location = _clean_location(match.group(1))
            return location or None
    return None


def _clean_location(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"^(?:the|city of)\s+", "", value.strip(), flags=re.IGNORECASE)
    value = value.strip(" ?.,;:\"'")
    aliases = {
        "nyc": "New York City",
        "new york": "New York City",
        "são paulo": "Sao Paulo",
    }
    return aliases.get(value.casefold(), value)


def _candidate_end_date(candidate: dict[str, Any]) -> str | None:
    for key in ("end_date", "endDate"):
        value = candidate.get(key)
        if value:
            return str(value)
    for draft_key in ("journal_draft_payload_yes", "journal_draft_payload_no"):
        draft = candidate.get(draft_key)
        if isinstance(draft, dict) and draft.get("end_date"):
            return str(draft["end_date"])
    return None


def _infer_year(end_date: str | date | datetime | None, reference_date: date | None) -> int:
    parsed = _parse_date_like(end_date)
    if parsed is not None:
        return parsed.year
    if reference_date is not None:
        return reference_date.year
    return datetime.now(UTC).date().year


def _parse_date_like(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()
