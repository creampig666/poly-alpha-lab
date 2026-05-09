"""Category extraction and normalization for Gamma market metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poly_alpha_lab.models import Market


@dataclass(frozen=True)
class CategoryInfo:
    raw_category: str
    normalized_category: str
    raw_category_source: str
    used_keyword_fallback: bool = False

    @property
    def is_unknown(self) -> bool:
        return self.normalized_category == "unknown"


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sports": (
        "sports",
        "sport",
        "nba",
        "nfl",
        "mlb",
        "nhl",
        "soccer",
        "football",
        "tennis",
        "f1",
        "formula 1",
        "ufc",
        "mma",
        "championship",
        "wimbledon",
    ),
    "politics": (
        "politics",
        "election",
        "elections",
        "president",
        "presidential",
        "congress",
        "senate",
        "house",
        "governor",
        "mayor",
    ),
    "economics": (
        "economics",
        "economy",
        "inflation",
        "cpi",
        "fed",
        "fomc",
        "rate",
        "rates",
        "jobs",
        "unemployment",
        "gdp",
    ),
    "finance": (
        "finance",
        "stocks",
        "stock",
        "ipo",
        "market cap",
        "earnings",
        "treasury",
        "bond",
    ),
    "crypto": (
        "crypto",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "xrp",
    ),
    "tech": (
        "tech",
        "ai",
        "artificial intelligence",
        "openai",
        "google",
        "apple",
        "microsoft",
        "nvidia",
        "discord",
    ),
    "weather": (
        "weather",
        "temperature",
        "rainfall",
        "hurricane",
        "storm",
        "snow",
    ),
    "geopolitics": (
        "geopolitics",
        "war",
        "military",
        "israel",
        "iran",
        "russia",
        "ukraine",
        "china",
        "taiwan",
    ),
    "culture": (
        "culture",
        "entertainment",
        "oscars",
        "grammy",
        "nobel",
        "movie",
        "music",
        "celebrity",
    ),
}

RAW_CATEGORY_SOURCES = (
    "market.category",
    'market.raw["category"]',
    'market.raw["event"]["category"]',
    'market.raw["event"]["categories"]',
    'market.raw["events"][0]["category"]',
    'market.raw["events"][0]["categories"]',
    'market.raw["tags"]',
    'market.raw["event"]["tags"]',
    'market.raw["events"][0]["tags"]',
    'market.raw["series"]',
    'market.raw["event"]["series"]',
    'market.raw["events"][0]["series"]',
)


def extract_category_info(market: Market) -> CategoryInfo:
    """Extract the best available category from Gamma metadata."""

    for source in RAW_CATEGORY_SOURCES:
        value = _value_for_source(market, source)
        for raw in _flatten_labels(value):
            normalized = normalize_category_text(raw)
            if normalized != "unknown":
                return CategoryInfo(
                    raw_category=raw,
                    normalized_category=normalized,
                    raw_category_source=source,
                )

    keyword_text = " ".join(_keyword_text_parts(market))
    normalized = normalize_category_text(keyword_text)
    if normalized != "unknown":
        return CategoryInfo(
            raw_category=keyword_text[:120] or "unknown",
            normalized_category=normalized,
            raw_category_source="keyword_fallback",
            used_keyword_fallback=True,
        )

    return CategoryInfo(
        raw_category="unknown",
        normalized_category="unknown",
        raw_category_source="none",
    )


def normalize_category_text(value: str | None) -> str:
    if not value:
        return "unknown"
    text = _normalize_text(value)
    if text in {"n/a", "na", "none", "null", "unknown", ""}:
        return "unknown"
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if _keyword_matches(text, keyword):
                return category
    return "unknown"


def is_known_category(category: str | None) -> bool:
    return normalize_category_text(category) != "unknown"


def _keyword_matches(text: str, keyword: str) -> bool:
    keyword = _normalize_text(keyword)
    if len(keyword) <= 3:
        return keyword in text.split()
    return keyword in text


def _normalize_text(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )


def _value_for_source(market: Market, source: str) -> Any:
    if source == "market.category":
        return market.category
    raw = market.raw
    event = raw.get("event")
    events = raw.get("events")
    first_event = events[0] if isinstance(events, list) and events else None
    mapping = {
        'market.raw["category"]': raw.get("category"),
        'market.raw["event"]["category"]': _dict_get(event, "category"),
        'market.raw["event"]["categories"]': _dict_get(event, "categories"),
        'market.raw["events"][0]["category"]': _dict_get(first_event, "category"),
        'market.raw["events"][0]["categories"]': _dict_get(first_event, "categories"),
        'market.raw["tags"]': raw.get("tags"),
        'market.raw["event"]["tags"]': _dict_get(event, "tags"),
        'market.raw["events"][0]["tags"]': _dict_get(first_event, "tags"),
        'market.raw["series"]': raw.get("series"),
        'market.raw["event"]["series"]': _dict_get(event, "series"),
        'market.raw["events"][0]["series"]': _dict_get(first_event, "series"),
    }
    return mapping.get(source)


def _dict_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _flatten_labels(value: Any) -> list[str]:
    labels: list[str] = []
    if value is None:
        return labels
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, dict):
        for key in ("category", "label", "name", "title", "slug", "ticker"):
            nested = value.get(key)
            if nested:
                labels.extend(_flatten_labels(nested))
        return labels
    if isinstance(value, list):
        for item in value:
            labels.extend(_flatten_labels(item))
    return labels


def _keyword_text_parts(market: Market) -> list[str]:
    parts = [
        market.question,
        market.slug,
        market.raw.get("title"),
        market.raw.get("description"),
        market.raw.get("groupItemTitle"),
    ]
    for event_key in ("event",):
        event = market.raw.get(event_key)
        if isinstance(event, dict):
            parts.extend([event.get("title"), event.get("slug"), event.get("description")])
    events = market.raw.get("events")
    if isinstance(events, list):
        for event in events[:2]:
            if isinstance(event, dict):
                parts.extend([event.get("title"), event.get("slug"), event.get("description")])
    parts.extend(_flatten_labels(market.raw.get("tags")))
    parts.extend(_flatten_labels(_dict_get(market.raw.get("event"), "tags")))
    if isinstance(events, list) and events:
        parts.extend(_flatten_labels(_dict_get(events[0], "tags")))
    return [str(part) for part in parts if part]

