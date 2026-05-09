"""Rule-based resolution criteria risk analysis."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.category_normalization import extract_category_info
from poly_alpha_lab.models import Market, MarketStructureError

RiskLabel = Literal["LOW", "MEDIUM", "HIGH"]

SOURCE_KEYWORDS = [
    "official",
    "announced by",
    "according to",
    "as reported by",
    "source",
    "website",
    "api",
    "government",
    "court",
    "sec",
    "fomc",
    "cpi",
    "bls",
    "noaa",
    "nhc",
    "wikipedia",
    "imdb",
    "box office mojo",
    "coinmarketcap",
    "binance",
    "coinbase",
]
OFFICIAL_SOURCE_KEYWORDS = [
    "official",
    "government",
    "court",
    "sec",
    "fomc",
    "cpi",
    "bls",
    "noaa",
    "nhc",
]
MEDIA_SOURCE_KEYWORDS = ["credible reports", "major media", "as reported by"]
DEADLINE_KEYWORDS = [
    "by",
    "before",
    "after",
    "on or before",
    "as of",
    "at",
    "between",
    "through",
    "end of",
    "close of",
    "11:59 pm",
    "utc",
    "et",
    "est",
    "edt",
    "local time",
]
TIME_ZONE_KEYWORDS = ["UTC", "ET", "EST", "EDT", "PST", "PDT", "CST", "CDT", "local time"]
SUBJECTIVE_KEYWORDS = [
    "substantially",
    "significant",
    "widely recognized",
    "credible reports",
    "major media",
    "likely",
    "expected",
    "apparent",
    "unclear",
    "disputed",
    "temporary",
    "permanent",
]
DATA_FINALITY_KEYWORDS = ["certified", "final", "preliminary", "revised"]
CONDITIONAL_KEYWORDS = [
    "if and only if",
    "unless",
    "except",
    "at least",
    "more than",
    "greater than",
    "equal to or above",
    "below",
    "not including",
    "including",
    "regardless of",
]
ANNOUNCEMENT_KEYWORDS = ["announced", "signed", "passed", "approved", "certified"]
IMPLEMENTATION_KEYWORDS = [
    "implemented",
    "takes effect",
    "enters into force",
    "officially becomes",
    "final result",
]
YES_PATTERNS = [
    "resolve to yes",
    "resolve to “yes”",
    "resolve to \"yes\"",
    "market resolves to yes",
    "market resolves to “yes”",
    "will resolve yes if",
    "will resolve to yes if",
    "will resolve to “yes” if",
    "this market will resolve to yes if",
    "this market will resolve to “yes” if",
]
NO_PATTERNS = [
    "resolve to no",
    "resolve to “no”",
    "resolve to \"no\"",
    "resolves to no",
    "resolves to “no”",
    "will resolve no if",
    "will resolve to no if",
    "will resolve to “no” if",
    "otherwise",
]
NUMERIC_VALUE_RE = re.compile(r"([$]\d|\b\d+(?:\.\d+)?\s*%)", re.I)


class ResolutionAnalysis(BaseModel):
    """Rule-based audit of market resolution criteria."""

    market_id: str
    slug: str | None = None
    question: str | None = None
    category: str
    has_resolution_text: bool
    resolution_text_source: str | None = None
    resolution_text_excerpt: str | None = None
    what_counts_as_yes: str | None = None
    what_counts_as_no: str | None = None
    resolution_source: str | None = None
    deadline: str | None = None
    time_zone: str | None = None
    ambiguity_risk: RiskLabel
    dispute_risk: RiskLabel
    risk_score: float = Field(ge=0, le=100)
    critical_phrases: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    research_notes: list[str] = Field(default_factory=list)


def analyze_resolution(market: Market) -> ResolutionAnalysis:
    """Analyze resolution criteria clarity without producing a trade signal."""

    category = extract_category_info(market).normalized_category
    text, source, has_resolution_text = _extract_resolution_text(market)
    sentences = _sentences(text)
    lower_text = text.lower()
    warnings: list[str] = []
    missing: list[str] = []
    notes: list[str] = []
    critical = _critical_phrases(lower_text)

    if not has_resolution_text:
        warnings.append("question_only_no_resolution_text")
        missing.append("resolution_text")

    what_yes = _first_sentence_with(sentences, YES_PATTERNS)
    what_no = _first_sentence_with(sentences, NO_PATTERNS)
    resolution_source = _first_sentence_with(sentences, SOURCE_KEYWORDS)
    deadline = _deadline_sentence(sentences)
    time_zone = _time_zone(text)

    if resolution_source is None:
        missing.append("resolution_source")
    if deadline is None:
        missing.append("deadline")
    if deadline is not None and time_zone is None:
        missing.append("time_zone")
    if what_yes is None:
        missing.append("what_counts_as_yes")
    if what_no is None:
        missing.append("what_counts_as_no")

    if any(_contains_phrase(lower_text, phrase) for phrase in ["preliminary", "revised"]):
        warnings.append("preliminary_revised_data_risk")
    if _has_announcement_vs_implementation(lower_text):
        warnings.append("announcement_vs_implementation_ambiguity")
    if _multiple_source_mentions(lower_text):
        warnings.append("multiple_possible_sources")
    if any(_contains_phrase(lower_text, phrase) for phrase in MEDIA_SOURCE_KEYWORDS):
        warnings.append("media_report_source")
    if any(_contains_phrase(lower_text, phrase) for phrase in SUBJECTIVE_KEYWORDS):
        notes.append("review subjective or vague resolution language")
    if _has_numeric_threshold(lower_text):
        notes.append("objective numeric threshold detected")

    risk_score = _risk_score(
        market=market,
        has_resolution_text=has_resolution_text,
        resolution_source=resolution_source,
        deadline=deadline,
        time_zone=time_zone,
        what_yes=what_yes,
        what_no=what_no,
        lower_text=lower_text,
        warnings=warnings,
    )
    ambiguity = _risk_label(risk_score)
    dispute = _risk_label(risk_score)
    if _has_dispute_language(lower_text) and dispute == "LOW":
        dispute = "MEDIUM"

    return ResolutionAnalysis(
        market_id=market.id,
        slug=market.slug,
        question=market.question,
        category=category,
        has_resolution_text=has_resolution_text,
        resolution_text_source=source,
        resolution_text_excerpt=_excerpt(text),
        what_counts_as_yes=what_yes,
        what_counts_as_no=what_no,
        resolution_source=resolution_source,
        deadline=deadline,
        time_zone=time_zone,
        ambiguity_risk=ambiguity,
        dispute_risk=dispute,
        risk_score=risk_score,
        critical_phrases=critical,
        missing_fields=_unique(missing),
        warnings=_unique(warnings),
        research_notes=_unique(notes),
    )


def _extract_resolution_text(market: Market) -> tuple[str, str | None, bool]:
    raw = market.raw or {}
    candidates: list[tuple[str, Any]] = [
        ('market.raw["resolutionCriteria"]', raw.get("resolutionCriteria")),
        ('market.raw["resolution_criteria"]', raw.get("resolution_criteria")),
        ('market.raw["rules"]', raw.get("rules")),
        ('market.raw["description"]', raw.get("description")),
        ('market.raw["event"]["description"]', _nested(raw, "event", "description")),
        ('market.raw["events"][0]["description"]', _events_first(raw, "description")),
        ("market.question", market.question),
    ]
    for source, value in candidates:
        text = _string_value(value)
        if text:
            return text, source, source != "market.question"
    return "", None, False


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _events_first(data: dict[str, Any], key: str) -> Any:
    events = data.get("events")
    if not isinstance(events, list) or not events or not isinstance(events[0], dict):
        return None
    return events[0].get(key)


def _string_value(value: Any) -> str | None:
    if isinstance(value, str):
        text = " ".join(value.split())
        return text or None
    if isinstance(value, dict):
        for key in ("text", "description", "rules"):
            if key in value:
                return _string_value(value[key])
    return None


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _first_sentence_with(sentences: list[str], keywords: list[str]) -> str | None:
    for sentence in sentences:
        lower = sentence.lower()
        if any(_contains_phrase(lower, keyword) for keyword in keywords):
            return sentence
    return None


def _deadline_sentence(sentences: list[str]) -> str | None:
    for sentence in sentences:
        lower = sentence.lower()
        if any(
            _contains_phrase(lower, phrase)
            for phrase in [
                "on or before",
                "as of",
                "between",
                "through",
                "end of",
                "close of",
                "11:59 pm",
                "utc",
                "local time",
            ]
        ):
            return sentence
        if any(_contains_phrase(lower, phrase.lower()) for phrase in TIME_ZONE_KEYWORDS):
            return sentence
        if re.search(r"\b(before|after|at)\b", lower) and _has_date_or_time_cue(lower):
            return sentence
        if _contains_phrase(lower, "by") and _has_date_or_time_cue(lower):
            if not re.search(r"\b(published|reported|announced)\s+by\b", lower):
                return sentence
    return None


def _has_date_or_time_cue(lower_text: str) -> bool:
    months = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    return (
        any(month in lower_text for month in months)
        or re.search(r"\b\d{1,2}:\d{2}\b", lower_text) is not None
        or re.search(r"\b20\d{2}\b", lower_text) is not None
        or any(_contains_phrase(lower_text, phrase.lower()) for phrase in TIME_ZONE_KEYWORDS)
    )


def _time_zone(text: str) -> str | None:
    for keyword in TIME_ZONE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.I):
            return keyword
    return None


def _critical_phrases(lower_text: str) -> list[str]:
    phrases = (
        SOURCE_KEYWORDS
        + DEADLINE_KEYWORDS
        + SUBJECTIVE_KEYWORDS
        + DATA_FINALITY_KEYWORDS
        + CONDITIONAL_KEYWORDS
        + ANNOUNCEMENT_KEYWORDS
        + IMPLEMENTATION_KEYWORDS
    )
    return _unique([phrase for phrase in phrases if _contains_phrase(lower_text, phrase)])


def _has_announcement_vs_implementation(lower_text: str) -> bool:
    return any(_contains_phrase(lower_text, phrase) for phrase in ANNOUNCEMENT_KEYWORDS) and any(
        _contains_phrase(lower_text, phrase) for phrase in IMPLEMENTATION_KEYWORDS
    )


def _multiple_source_mentions(lower_text: str) -> bool:
    source_hits = [phrase for phrase in SOURCE_KEYWORDS if _contains_phrase(lower_text, phrase)]
    named_hits = [phrase for phrase in source_hits if phrase not in {"official", "source", "website"}]
    return len(named_hits) >= 2


def _has_dispute_language(lower_text: str) -> bool:
    dispute_terms = ["credible reports", "major media", "disputed", "unclear", "apparent"]
    return any(_contains_phrase(lower_text, term) for term in dispute_terms)


def _contains_phrase(lower_text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in lower_text
    return re.search(rf"\b{re.escape(phrase)}\b", lower_text) is not None


def _risk_score(
    *,
    market: Market,
    has_resolution_text: bool,
    resolution_source: str | None,
    deadline: str | None,
    time_zone: str | None,
    what_yes: str | None,
    what_no: str | None,
    lower_text: str,
    warnings: list[str],
) -> float:
    score = 35.0
    if not has_resolution_text:
        score += 30
    if resolution_source is None:
        score += 15
    if deadline is None:
        score += 15
    if deadline is not None and time_zone is None:
        score += 8
    subjective_hits = [
        phrase for phrase in SUBJECTIVE_KEYWORDS if _contains_phrase(lower_text, phrase)
    ]
    score += min(20, len(subjective_hits) * 5)
    if any(_contains_phrase(lower_text, phrase) for phrase in MEDIA_SOURCE_KEYWORDS):
        score += 10
    if any(_contains_phrase(lower_text, phrase) for phrase in ["preliminary", "revised", "final"]):
        score += 10
    if "announcement_vs_implementation_ambiguity" in warnings:
        score += 10
    if "multiple_possible_sources" in warnings:
        score += 8
    if what_yes is None:
        score += 8
    if what_no is None:
        score += 8

    if _has_numeric_threshold(lower_text):
        score -= 10
    if any(_contains_phrase(lower_text, keyword) for keyword in OFFICIAL_SOURCE_KEYWORDS):
        score -= 10
    if deadline is not None:
        score -= 8
    if time_zone is not None:
        score -= 5
    if what_yes is not None and what_no is not None:
        score -= 10
    if _standard_binary_market(market):
        score -= 5
    return round(min(100.0, max(0.0, score)), 2)


def _standard_binary_market(market: Market) -> bool:
    if len(market.outcomes) != 2:
        return False
    try:
        market.yes_outcome_index
        market.no_outcome_index
    except MarketStructureError:
        return False
    return True


def _has_numeric_threshold(lower_text: str) -> bool:
    threshold_phrases = [
        "at least",
        "more than",
        "greater than",
        "equal to or above",
        "below",
        "not including",
        "including",
    ]
    return NUMERIC_VALUE_RE.search(lower_text) is not None or any(
        _contains_phrase(lower_text, phrase) for phrase in threshold_phrases
    )


def _risk_label(score: float) -> RiskLabel:
    if score < 30:
        return "LOW"
    if score < 60:
        return "MEDIUM"
    return "HIGH"


def _excerpt(text: str, length: int = 280) -> str | None:
    if not text:
        return None
    if len(text) <= length:
        return text
    return text[: length - 3].rstrip() + "..."


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
