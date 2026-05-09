"""Breakeven paper strategy candidate runner."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from poly_alpha_lab.candidate_scoring import CandidateScore, score_candidate_market
from poly_alpha_lab.fees import taker_fee_per_share
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import Market
from poly_alpha_lab.resolution_analyzer import ResolutionAnalysis, analyze_resolution

Grade = Literal["A", "B", "C", "SKIP"]
Risk = Literal["LOW", "MEDIUM", "HIGH"]

GRADE_RANK = {"SKIP": 0, "C": 1, "B": 2, "A": 3}
RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
MANUAL_RATIONALE = (
    "strategy runner breakeven candidate; requires manual probability judgment; "
    "not a real trading signal"
)


class StrategyCandidate(BaseModel):
    """Breakeven paper strategy candidate. This is not a trading signal."""

    market_id: str
    slug: str | None = None
    question: str | None = None
    category: str
    candidate_score: float
    candidate_grade: Grade
    resolution_risk_score: float
    ambiguity_risk: Risk
    dispute_risk: Risk
    yes_executable_avg_buy_price: float
    no_executable_avg_buy_price: float
    yes_fee_per_share: float
    no_fee_per_share: float
    yes_breakeven_probability: float
    no_breakeven_probability: float
    yes_required_probability: float
    no_required_yes_probability_upper_bound: float
    yes_spread: float | None = None
    no_spread: float | None = None
    yes_depth_score: float = Field(ge=0, le=20)
    no_depth_score: float = Field(ge=0, le=20)
    strategy_score: float
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    journal_draft_payload_yes: dict[str, Any] = Field(default_factory=dict)
    journal_draft_payload_no: dict[str, Any] = Field(default_factory=dict)


def build_strategy_candidate(
    market: Market,
    liquidity: BinaryMarketLiquidity,
    *,
    size: float,
    now: datetime | None = None,
) -> StrategyCandidate:
    """Build one breakeven candidate from executable avg buy prices."""

    now = _aware_now(now)
    candidate_score = score_candidate_market(market, liquidity, size=size, now=now)
    resolution = analyze_resolution(market)
    yes_price = liquidity.yes_avg_buy_price(size)
    no_price = liquidity.no_avg_buy_price(size)
    yes_fee = taker_fee_per_share(
        yes_price,
        category=candidate_score.normalized_category,
        fees_enabled=market.fees_enabled,
    )
    no_fee = taker_fee_per_share(
        no_price,
        category=candidate_score.normalized_category,
        fees_enabled=market.fees_enabled,
    )
    yes_breakeven = yes_price + yes_fee
    no_breakeven = no_price + no_fee
    no_yes_upper = 1 - no_breakeven
    yes_depth_score = _side_depth_score(
        liquidity.yes_ask_depth_3pct,
        price=liquidity.yes_best_ask,
        size=size,
    )
    no_depth_score = _side_depth_score(
        liquidity.no_ask_depth_3pct,
        price=liquidity.no_best_ask,
        size=size,
    )

    reasons = list(candidate_score.reasons)
    warnings = list(resolution.warnings)
    if yes_breakeven < no_yes_upper:
        warnings.append("market_data_warning:overlapping_breakeven_region")
    if _is_long_dated(candidate_score):
        warnings.append("long_time_to_resolution_penalty")
    if _is_sports_outright_longshot(market, candidate_score, yes_breakeven):
        warnings.append("sports_outright_longshot_penalty")
    warnings.extend(_time_risk_warnings(market, now))
    if _is_geopolitics_military_event(market, candidate_score):
        warnings.append("geopolitics_military_event_risk")
    dispute_risk = _risk_floor(
        resolution.dispute_risk,
        "MEDIUM" if "geopolitics_military_event_risk" in warnings else None,
    )

    strategy_score = _strategy_score(
        candidate_score=candidate_score,
        resolution=resolution,
        liquidity=liquidity,
        yes_depth_score=yes_depth_score,
        no_depth_score=no_depth_score,
        yes_breakeven=yes_breakeven,
        no_yes_upper=no_yes_upper,
        warnings=warnings,
    )

    common_payload = {
        "market_id": market.id,
        "slug": market.slug,
        "question": market.question or market.slug or market.id,
        "category": candidate_score.normalized_category,
        "end_date": market.end_date.isoformat() if market.end_date else None,
        "candidate_score": candidate_score.total_score,
        "candidate_grade": candidate_score.candidate_grade,
        "probability_source": "manual",
        "fair_yes_probability": None,
        "requires_user_fair_yes_probability": True,
        "entry_size": size,
        "rationale": MANUAL_RATIONALE,
        "status": "OPEN",
        "resolution_risk_score": resolution.risk_score,
        "ambiguity_risk": resolution.ambiguity_risk,
        "dispute_risk": dispute_risk,
    }
    yes_payload = {
        **_without_none(common_payload),
        "fair_yes_probability": None,
        "requires_user_fair_yes_probability": True,
        "side": "YES",
        "entry_price": yes_price,
        "fee_per_share": yes_fee,
    }
    no_payload = {
        **_without_none(common_payload),
        "fair_yes_probability": None,
        "requires_user_fair_yes_probability": True,
        "side": "NO",
        "entry_price": no_price,
        "fee_per_share": no_fee,
    }

    return StrategyCandidate(
        market_id=market.id,
        slug=market.slug,
        question=market.question,
        category=candidate_score.normalized_category,
        candidate_score=candidate_score.total_score,
        candidate_grade=candidate_score.candidate_grade,
        resolution_risk_score=resolution.risk_score,
        ambiguity_risk=resolution.ambiguity_risk,
        dispute_risk=dispute_risk,
        yes_executable_avg_buy_price=yes_price,
        no_executable_avg_buy_price=no_price,
        yes_fee_per_share=yes_fee,
        no_fee_per_share=no_fee,
        yes_breakeven_probability=yes_breakeven,
        no_breakeven_probability=no_breakeven,
        yes_required_probability=yes_breakeven,
        no_required_yes_probability_upper_bound=no_yes_upper,
        yes_spread=liquidity.yes_spread,
        no_spread=liquidity.no_spread,
        yes_depth_score=yes_depth_score,
        no_depth_score=no_depth_score,
        strategy_score=strategy_score,
        reasons=_unique(reasons),
        warnings=_unique(warnings),
        journal_draft_payload_yes=yes_payload,
        journal_draft_payload_no=no_payload,
    )


def scan_strategy_candidates(
    markets: list[Market],
    liquidities: dict[str, BinaryMarketLiquidity],
    *,
    size: float,
    min_grade: str = "B",
    max_resolution_risk: str = "HIGH",
    category: str | None = None,
    include_long_dated: bool = True,
    include_longshots: bool = False,
    now: datetime | None = None,
) -> list[StrategyCandidate]:
    """Return sorted paper strategy candidates using only breakeven probabilities."""

    now = _aware_now(now)
    min_rank = GRADE_RANK[min_grade.upper()]
    max_risk_rank = RISK_RANK[max_resolution_risk.upper()]
    candidates: list[StrategyCandidate] = []
    for market in markets:
        liquidity = liquidities.get(market.id)
        if liquidity is None:
            continue
        try:
            candidate = build_strategy_candidate(market, liquidity, size=size, now=now)
        except ValueError:
            continue
        if candidate.candidate_grade == "SKIP":
            continue
        if GRADE_RANK[candidate.candidate_grade] < min_rank:
            continue
        candidate_risk_rank = max(
            RISK_RANK[candidate.ambiguity_risk],
            RISK_RANK[candidate.dispute_risk],
        )
        if candidate_risk_rank > max_risk_rank:
            continue
        if category and candidate.category != category:
            continue
        if not include_long_dated and "long_time_to_resolution_penalty" in candidate.warnings:
            continue
        if not include_longshots and "sports_outright_longshot_penalty" in candidate.warnings:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: item.strategy_score, reverse=True)
    return candidates


def write_strategy_candidates_json(candidates: list[StrategyCandidate], output_path: str | Path) -> int:
    """Write candidates to JSON and return the number of records."""

    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    data = [candidate.model_dump() for candidate in candidates]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(data)


def strategy_candidates_report(candidates: list[StrategyCandidate]) -> str:
    """Render breakeven strategy candidates as Markdown."""

    lines = [
        "# Breakeven Paper Strategy Candidates",
        "",
        f"- Candidates: `{len(candidates)}`",
        "- Purpose: `manual_probability_review_not_trade_signal`",
        "- The runner never places orders and never writes journal entries automatically.",
        "- No directional action label is emitted; compare your own fair probability to breakeven.",
        "",
    ]
    for candidate in candidates:
        lines.extend(_candidate_section(candidate))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _candidate_section(candidate: StrategyCandidate) -> list[str]:
    region_text = _breakeven_region_text(candidate)
    return [
        f"## {candidate.question or candidate.slug or candidate.market_id}",
        "",
        f"- market_id: `{candidate.market_id}`",
        f"- slug: `{candidate.slug or 'n/a'}`",
        f"- category: `{candidate.category}`",
        f"- candidate_grade / score: `{candidate.candidate_grade}` / `{candidate.candidate_score:.2f}`",
        f"- strategy_score: `{candidate.strategy_score:.2f}`",
        f"- resolution risk: `{candidate.ambiguity_risk}` ambiguity, `{candidate.dispute_risk}` dispute, score `{candidate.resolution_risk_score:.2f}`",
        f"- YES requires p_yes > `{candidate.yes_required_probability:.2%}`",
        f"- NO requires p_yes < `{candidate.no_required_yes_probability_upper_bound:.2%}`",
        f"- {region_text}",
        "",
        "| Side | Executable avg buy | Fee/share | Breakeven | Spread | Depth score |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| YES | {candidate.yes_executable_avg_buy_price:.4f} | {candidate.yes_fee_per_share:.6f} | {candidate.yes_breakeven_probability:.2%} | {_fmt_float(candidate.yes_spread)} | {candidate.yes_depth_score:.2f} |",
        f"| NO | {candidate.no_executable_avg_buy_price:.4f} | {candidate.no_fee_per_share:.6f} | {candidate.no_breakeven_probability:.2%} | {_fmt_float(candidate.no_spread)} | {candidate.no_depth_score:.2f} |",
        "",
        f"- reasons: `{_fmt_list(candidate.reasons)}`",
        f"- warnings: `{_fmt_list(candidate.warnings)}`",
        "",
        "### Journal Draft Payloads",
        "",
        "```json",
        json.dumps(
            {
                "YES": candidate.journal_draft_payload_yes,
                "NO": candidate.journal_draft_payload_no,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "```",
    ]


def _strategy_score(
    *,
    candidate_score: CandidateScore,
    resolution: ResolutionAnalysis,
    liquidity: BinaryMarketLiquidity,
    yes_depth_score: float,
    no_depth_score: float,
    yes_breakeven: float,
    no_yes_upper: float,
    warnings: list[str],
) -> float:
    risk_penalty = resolution.risk_score * 0.35
    avg_spread = ((liquidity.yes_spread or 0.0) + (liquidity.no_spread or 0.0)) / 2
    spread_penalty = avg_spread * 120
    depth_penalty = max(0.0, 20 - min(yes_depth_score, no_depth_score)) * 0.8
    breakeven_attractiveness = max(0.0, 0.5 - yes_breakeven) * 8
    breakeven_attractiveness += max(0.0, no_yes_upper - 0.5) * 8
    penalty = 0.0
    if "long_time_to_resolution_penalty" in warnings:
        penalty += 10
    if "near_expiry_penalty" in warnings:
        penalty += 15
    if "same_day_expiry_penalty" in warnings:
        penalty += 20
    if "geopolitics_military_event_risk" in warnings:
        penalty += 15
    if "sports_outright_longshot_penalty" in warnings:
        penalty += 25
    if "market_data_warning:overlapping_breakeven_region" in warnings:
        penalty += 8
    score = (
        candidate_score.total_score
        - risk_penalty
        - spread_penalty
        - depth_penalty
        + breakeven_attractiveness
        - penalty
    )
    return round(max(0.0, score), 2)


def _side_depth_score(depth: float | None, *, price: float | None, size: float) -> float:
    if depth is None or price is None or size <= 0:
        return 0.0
    requested_notional = max(price * size, 1e-9)
    ratio = depth * price / requested_notional
    if ratio < 1:
        return round(max(0.0, ratio) * 5, 2)
    if ratio < 2:
        return round(5 + (ratio - 1) * 5, 2)
    if ratio < 5:
        return round(10 + ((ratio - 2) / 3) * 5, 2)
    if ratio < 20:
        return round(15 + ((ratio - 5) / 15) * 5, 2)
    return 20.0


def _is_long_dated(score: CandidateScore) -> bool:
    return any(reason in score.reasons for reason in ("long_time_to_resolution", "very_long_time_to_resolution"))


def _is_sports_outright_longshot(
    market: Market,
    score: CandidateScore,
    yes_breakeven: float,
) -> bool:
    if yes_breakeven >= 0.15:
        return False
    text = f"{market.question or ''} {market.slug or ''}".lower()
    has_outright_language = (
        re.search(r"\b(win|winner|champion|championship|outright|title)\b", text) is not None
    )
    has_sports_context = (
        score.normalized_category == "sports"
        or re.search(
            r"\b(wimbledon|french open|us open|australian open|nba|nfl|mlb|nhl|f1|drivers|mvp|cup|league)\b",
            text,
        )
        is not None
    )
    return has_outright_language and has_sports_context


def _time_risk_warnings(market: Market, now: datetime) -> list[str]:
    if market.end_date is None:
        return []
    end_date = _aware_datetime(market.end_date)
    warnings: list[str] = []
    if end_date <= now:
        return ["expired_market"]
    if (end_date - now).total_seconds() < 86_400:
        warnings.append("near_expiry_penalty")
    if end_date.date() == now.date():
        warnings.append("same_day_expiry_penalty")
    return warnings


def _is_geopolitics_military_event(market: Market, score: CandidateScore) -> bool:
    if score.normalized_category not in {"geopolitics", "politics"}:
        return False
    text = f"{market.question or ''} {market.slug or ''}".lower()
    terms = [
        "military",
        "clash",
        "war",
        "attack",
        "invasion",
        "strike",
        "conflict",
        "troops",
        "missile",
        "combat",
    ]
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _risk_floor(value: Risk, floor: Risk | None) -> Risk:
    if floor is None:
        return value
    return floor if RISK_RANK[value] < RISK_RANK[floor] else value


def _breakeven_region_text(candidate: StrategyCandidate) -> str:
    if candidate.yes_required_probability < candidate.no_required_yes_probability_upper_bound:
        return (
            "Warning: overlapping breakeven region. Both sides appear positive in part "
            "of the interval; treat as market data anomaly, not a signal."
        )
    return (
        "No-trade zone: if your fair p_yes is between NO upper bound and YES threshold, "
        "neither side clears breakeven."
    )


def _aware_now(now: datetime | None) -> datetime:
    return _aware_datetime(now or datetime.now(UTC))


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _without_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
