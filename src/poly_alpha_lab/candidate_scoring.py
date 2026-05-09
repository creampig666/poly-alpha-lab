"""Candidate market scoring for human research triage."""

from __future__ import annotations

from datetime import UTC, datetime
from math import log10
from typing import Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.category_normalization import extract_category_info
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import Market, MarketStructureError

CandidateGrade = Literal["A", "B", "C", "SKIP"]


class CandidateScore(BaseModel):
    """Rule-based candidate score for a standard binary market."""

    liquidity_score: float = Field(ge=0, le=20)
    spread_score: float = Field(ge=0, le=20)
    depth_score: float = Field(ge=0, le=20)
    volume_score: float = Field(ge=0, le=15)
    time_to_resolution_score: float = Field(ge=0, le=15)
    fee_score: float = Field(ge=0, le=5)
    structure_score: float = Field(ge=0, le=5)
    total_score: float = Field(ge=0, le=100)
    candidate_grade: CandidateGrade
    reasons: list[str] = Field(default_factory=list)
    normalized_category: str = "unknown"
    raw_category: str = "unknown"
    raw_category_source: str = "none"


def score_candidate_market(
    market: Market,
    liquidity: BinaryMarketLiquidity | None,
    *,
    size: float,
    now: datetime | None = None,
) -> CandidateScore:
    """Score a market for manual research, not for trading."""

    now = now or datetime.now(UTC)
    reasons: list[str] = []
    category_info = extract_category_info(market)
    if category_info.is_unknown:
        reasons.append("category_unknown")
    if category_info.used_keyword_fallback:
        reasons.append("category_keyword_fallback")

    hard_skip_reason = _hard_skip_reason(market, liquidity, now)
    if hard_skip_reason is not None:
        if hard_skip_reason not in reasons:
            reasons.append(hard_skip_reason)
        return CandidateScore(
            liquidity_score=0,
            spread_score=0,
            depth_score=0,
            volume_score=0,
            time_to_resolution_score=0,
            fee_score=0,
            structure_score=0,
            total_score=0,
            candidate_grade="SKIP",
            reasons=reasons,
            normalized_category=category_info.normalized_category,
            raw_category=category_info.raw_category,
            raw_category_source=category_info.raw_category_source,
        )

    liquidity_score = _liquidity_score(market.liquidity)
    volume_score = _volume_score(market.volume)
    spread_score, spread_cap_no_a = _spread_score(liquidity, reasons)
    depth_score = _depth_score(liquidity, size, reasons)
    time_score = _time_to_resolution_score(market, now, reasons)
    fee_score = _fee_score(market, category_info.normalized_category, reasons)
    structure_score = 5

    total = round(
        liquidity_score
        + spread_score
        + depth_score
        + volume_score
        + time_score
        + fee_score
        + structure_score,
        2,
    )
    grade = _grade(total)
    if grade == "A" and (spread_cap_no_a or "insufficient_near_touch_depth" in reasons):
        grade = "B"
    if grade == "A" and "insufficient_ask_depth" in reasons:
        grade = "B"

    return CandidateScore(
        liquidity_score=liquidity_score,
        spread_score=spread_score,
        depth_score=depth_score,
        volume_score=volume_score,
        time_to_resolution_score=time_score,
        fee_score=fee_score,
        structure_score=structure_score,
        total_score=total,
        candidate_grade=grade,
        reasons=reasons,
        normalized_category=category_info.normalized_category,
        raw_category=category_info.raw_category,
        raw_category_source=category_info.raw_category_source,
    )


def _hard_skip_reason(
    market: Market,
    liquidity: BinaryMarketLiquidity | None,
    now: datetime,
) -> str | None:
    if not market.active:
        return "inactive"
    if market.closed or market.archived:
        return "closed_or_archived"
    if market.raw.get("acceptingOrders") is False:
        return "accepting_orders_false"
    if not market.enable_order_book:
        return "orderbook_disabled"
    if len(market.outcomes) != 2:
        return "non_standard_binary_market"
    try:
        market.yes_outcome_index
        market.no_outcome_index
        market.yes_token_id
        market.no_token_id
    except MarketStructureError:
        return "non_standard_binary_market"
    if market.end_date is None:
        return "missing_end_date"
    end_date = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=UTC)
    if end_date <= now:
        return "end_date_past"
    if liquidity is None or liquidity.yes_book is None or liquidity.no_book is None:
        return "missing_orderbook"
    return None


def _liquidity_score(liquidity: float | None) -> float:
    return _log_score(liquidity, max_score=20, low=100, high=100_000)


def _volume_score(volume: float | None) -> float:
    return _log_score(volume, max_score=15, low=100, high=1_000_000)


def _log_score(value: float | None, *, max_score: float, low: float, high: float) -> float:
    if value is None or value <= 0:
        return 0
    if value >= high:
        return max_score
    if value <= low:
        return round(max_score * 0.25, 2)
    fraction = (log10(value) - log10(low)) / (log10(high) - log10(low))
    return round(max_score * (0.25 + 0.75 * fraction), 2)


def _spread_score(liquidity: BinaryMarketLiquidity | None, reasons: list[str]) -> tuple[float, bool]:
    if liquidity is None or liquidity.yes_spread is None or liquidity.no_spread is None:
        reasons.append("missing_spread")
        return 0, True
    cap_no_a = liquidity.yes_spread > 0.08 or liquidity.no_spread > 0.08
    avg_spread = (liquidity.yes_spread + liquidity.no_spread) / 2
    if avg_spread <= 0.01:
        return 20, cap_no_a
    if avg_spread <= 0.03:
        return 16, cap_no_a
    if avg_spread <= 0.05:
        return 12, cap_no_a
    if avg_spread <= 0.10:
        if cap_no_a:
            reasons.append("wide_spread")
        return 8, cap_no_a
    if avg_spread <= 0.20:
        reasons.append("wide_spread")
        return 4, True
    reasons.append("very_wide_spread")
    return 0, True


def _depth_score(liquidity: BinaryMarketLiquidity | None, size: float, reasons: list[str]) -> float:
    if liquidity is None:
        reasons.append("missing_orderbook")
        return 0
    yes_total_depth = _total_ask_depth(liquidity.yes_book)
    no_total_depth = _total_ask_depth(liquidity.no_book)
    if min(yes_total_depth, no_total_depth) < size:
        reasons.append("insufficient_ask_depth")
    yes_ratio = _near_touch_depth_ratio(
        liquidity.yes_book,
        best_ask=liquidity.yes_best_ask,
        size=size,
    )
    no_ratio = _near_touch_depth_ratio(
        liquidity.no_book,
        best_ask=liquidity.no_best_ask,
        size=size,
    )
    ratio = min(yes_ratio, no_ratio)
    if ratio < 1:
        reasons.append("insufficient_near_touch_depth")
        return round(max(0.0, ratio) * 5, 2)
    if ratio < 2:
        return round(5 + (ratio - 1) * 5, 2)
    if ratio < 5:
        return round(10 + ((ratio - 2) / 3) * 5, 2)
    if ratio < 20:
        return round(15 + ((ratio - 5) / 15) * 5, 2)
    return 20


def _total_ask_depth(book: object) -> float:
    asks = getattr(book, "asks", None)
    if asks is None:
        return 0
    return sum(level.size for level in asks)


def _near_touch_depth_ratio(book: object, *, best_ask: float | None, size: float) -> float:
    if book is None or best_ask is None or size <= 0:
        return 0
    requested_notional = max(size * best_ask, 1e-9)
    max_price = min(1.0, best_ask * 1.03)
    asks = getattr(book, "asks", None) or []
    near_touch_notional = sum(level.size * level.price for level in asks if level.price <= max_price)
    return near_touch_notional / requested_notional


def _time_to_resolution_score(market: Market, now: datetime, reasons: list[str]) -> float:
    if market.end_date is None:
        reasons.append("missing_end_date")
        return 0
    end_date = market.end_date if market.end_date.tzinfo else market.end_date.replace(tzinfo=UTC)
    days = (end_date - now).total_seconds() / 86_400
    if days <= 0:
        reasons.append("end_date_past")
        return 0
    if days <= 30:
        return 15
    if days <= 90:
        return 12
    if days <= 180:
        return 9
    if days <= 365:
        reasons.append("long_time_to_resolution")
        return 6
    reasons.append("very_long_time_to_resolution")
    return 3


def _fee_score(market: Market, normalized_category: str, reasons: list[str]) -> float:
    unknown = normalized_category == "unknown"
    if market.fees_enabled is False:
        if unknown and "category_unknown" not in reasons:
            reasons.append("category_unknown")
        return 5
    if market.fees_enabled is None:
        reasons.append("fees_enabled_unknown")
        if unknown:
            return 2
        return 3
    if unknown:
        return 4
    return 5


def _grade(total_score: float) -> CandidateGrade:
    if total_score >= 80:
        return "A"
    if total_score >= 60:
        return "B"
    if total_score >= 40:
        return "C"
    return "SKIP"


def candidate_grade_label(grade: CandidateGrade) -> str:
    return {
        "A": "worth_research",
        "B": "watch",
        "C": "low_priority",
        "SKIP": "skip",
    }[grade]
