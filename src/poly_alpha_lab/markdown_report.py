"""Markdown report generation for market analysis."""

from __future__ import annotations

import json
from typing import Any

from poly_alpha_lab.candidate_scoring import (
    CandidateScore,
    candidate_grade_label,
    score_candidate_market,
)
from poly_alpha_lab.category_normalization import extract_category_info
from poly_alpha_lab.config import settings
from poly_alpha_lab.ev import calculate_liquidity_ev
from poly_alpha_lab.fees import fee_assumption
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import EVResult, Market, MarketStructureError, OrderBook
from poly_alpha_lab.resolution_analyzer import ResolutionAnalysis, analyze_resolution


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _fmt_prob(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _safe_market_attr(market: Market, attr: str) -> tuple[object | None, str | None]:
    try:
        return getattr(market, attr), None
    except MarketStructureError as exc:
        return None, str(exc)


def suggested_action(
    *,
    market: Market,
    liquidity: BinaryMarketLiquidity | None,
    yes_ev: EVResult | None = None,
    no_ev: EVResult | None = None,
    min_net_edge: float = settings.min_net_edge,
) -> str:
    """Return BUY_YES, BUY_NO, WATCH, or SKIP."""

    try:
        market.yes_token_id
        market.no_token_id
    except MarketStructureError:
        return "SKIP"

    if liquidity is None or liquidity.yes_book is None or liquidity.no_book is None:
        return "SKIP"
    if liquidity.yes_best_ask is None or liquidity.no_best_ask is None:
        return "SKIP"
    if yes_ev is None or no_ev is None:
        return "SKIP"
    yes_buy = yes_ev.net_edge >= min_net_edge
    no_buy = no_ev.net_edge >= min_net_edge
    if yes_buy and no_buy:
        return "SKIP"
    if yes_buy:
        return "BUY_YES"
    if no_buy:
        return "BUY_NO"
    if yes_ev.net_edge > 0 or no_ev.net_edge > 0:
        return "WATCH"
    return "WATCH"


def market_report(
    market: Market,
    *,
    fair_yes_probability: float,
    size: float,
    yes_book: OrderBook | None = None,
    no_book: OrderBook | None = None,
    liquidity: BinaryMarketLiquidity | None = None,
    min_net_edge: float = settings.min_net_edge,
) -> str:
    """Render one market analysis report in Markdown."""

    if liquidity is None and (yes_book is not None or no_book is not None):
        liquidity = BinaryMarketLiquidity(yes_book=yes_book, no_book=no_book)

    indicative_yes, yes_error = _safe_market_attr(market, "yes_price")
    indicative_no, no_error = _safe_market_attr(market, "no_price")
    yes_token_id, yes_token_error = _safe_market_attr(market, "yes_token_id")
    no_token_id, no_token_error = _safe_market_attr(market, "no_token_id")
    category_info = extract_category_info(market)

    skip_reasons = [
        reason
        for reason in (yes_error, no_error, yes_token_error, no_token_error)
        if reason is not None
    ]
    if liquidity is None:
        skip_reasons.append("missing CLOB orderbook")
    else:
        if liquidity.yes_book is None:
            skip_reasons.append("missing YES CLOB orderbook")
        if liquidity.no_book is None:
            skip_reasons.append("missing NO CLOB orderbook")
        if liquidity.yes_book is not None and yes_token_id and liquidity.yes_book.token_id != yes_token_id:
            skip_reasons.append("token_id_mismatch")
        if liquidity.no_book is not None and no_token_id and liquidity.no_book.token_id != no_token_id:
            skip_reasons.append("token_id_mismatch")

    yes_ev: EVResult | None = None
    no_ev: EVResult | None = None
    ev_error: str | None = None
    if not skip_reasons and liquidity is not None:
        try:
            yes_ev, no_ev = calculate_liquidity_ev(
                liquidity=liquidity,
                fair_yes_probability=fair_yes_probability,
                size=size,
                category=category_info.normalized_category,
                fees_enabled=market.fees_enabled,
            )
        except ValueError as exc:
            ev_error = str(exc)
            skip_reasons.append(ev_error)

    if yes_ev is not None and no_ev is not None:
        if yes_ev.net_edge >= min_net_edge and no_ev.net_edge >= min_net_edge:
            skip_reasons.append("conflicting_positive_edges")

    action = suggested_action(
        market=market,
        liquidity=liquidity,
        yes_ev=yes_ev,
        no_ev=no_ev,
        min_net_edge=min_net_edge,
    )

    lines = [
        f"## {market.question or market.slug or market.id}",
        "",
        f"- ID: `{market.id}`",
        f"- Slug: `{market.slug or 'n/a'}`",
        f"- Category: `{category_info.normalized_category}`",
        f"- Active / Closed / Order Book: `{market.active}` / `{market.closed}` / "
        f"`{market.enable_order_book}`",
        f"- Liquidity: `{_fmt_money(market.liquidity)}`",
        f"- Volume: `{_fmt_money(market.volume)}`",
        f"- End date: `{market.end_date.isoformat() if market.end_date else 'n/a'}`",
        f"- fee_assumption=`{fee_assumption(market.fees_enabled, category_info.normalized_category)}`",
        f"- Min net edge: `{min_net_edge:.2%}`",
        f"- Suggested action: `{action}`",
    ]
    if skip_reasons:
        lines.append(f"- Skip reason: `{' ; '.join(skip_reasons)}`")

    lines.extend(_resolution_analysis_section(analyze_resolution(market)))

    lines.extend(
        [
            "",
            "### Prices And Liquidity",
            "",
            "| Field | Value |",
            "| --- | ---: |",
            f"| Indicative YES price from Gamma | {_fmt_prob(indicative_yes)} |",
            f"| Indicative NO price from Gamma | {_fmt_prob(indicative_no)} |",
            f"| YES token id | `{yes_token_id or 'n/a'}` |",
            f"| NO token id | `{no_token_id or 'n/a'}` |",
            f"| Executable YES bid | {_fmt_prob(liquidity.yes_best_bid if liquidity else None)} |",
            f"| Display-only YES best ask | {_fmt_prob(liquidity.yes_best_ask if liquidity else None)} |",
            f"| Executable YES avg buy price ({size:g}) | "
            f"{_fmt_prob(_avg_or_none(liquidity, 'YES', size))} |",
            f"| YES spread | {_fmt_prob(liquidity.yes_spread if liquidity else None)} |",
            f"| YES ask depth within 3% | "
            f"{_fmt_money(liquidity.yes_ask_depth_3pct if liquidity else None)} |",
            f"| Executable NO bid | {_fmt_prob(liquidity.no_best_bid if liquidity else None)} |",
            f"| Display-only NO best ask | {_fmt_prob(liquidity.no_best_ask if liquidity else None)} |",
            f"| Executable NO avg buy price ({size:g}) | "
            f"{_fmt_prob(_avg_or_none(liquidity, 'NO', size))} |",
            f"| NO spread | {_fmt_prob(liquidity.no_spread if liquidity else None)} |",
            f"| NO ask depth within 3% | "
            f"{_fmt_money(liquidity.no_ask_depth_3pct if liquidity else None)} |",
        ]
    )

    lines.extend(
        [
            "",
            "### EV",
            "",
            "| Side | Fair probability | Executable price | Price source | Gross edge | Cost | Fee | Net edge | Expected profit |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if yes_ev is None or no_ev is None:
        lines.append("| YES | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
        lines.append("| NO | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    else:
        for result in (yes_ev, no_ev):
            lines.append(
                f"| {result.side} | {_fmt_pct(result.fair_probability)} | "
                f"{_fmt_prob(result.price)} | `{result.price_source}` | "
                f"{_fmt_pct(result.gross_edge)} | {_fmt_money(result.cost)} | "
                f"{_fmt_money(result.estimated_fee)} | {_fmt_pct(result.net_edge)} | "
                f"{_fmt_money(result.expected_profit)} |"
            )

    lines.extend(
        _journal_draft_section(
            market=market,
            action=action,
            fair_yes_probability=fair_yes_probability,
            size=size,
            liquidity=liquidity,
            yes_ev=yes_ev,
            no_ev=no_ev,
            skip_reasons=skip_reasons,
        )
    )

    return "\n".join(lines)


def _avg_or_none(
    liquidity: BinaryMarketLiquidity | None,
    side: str,
    size: float,
) -> float | None:
    if liquidity is None:
        return None
    try:
        if side == "YES":
            return liquidity.yes_avg_buy_price(size)
        return liquidity.no_avg_buy_price(size)
    except ValueError:
        return None


def _journal_draft_section(
    *,
    market: Market,
    action: str,
    fair_yes_probability: float,
    size: float,
    liquidity: BinaryMarketLiquidity | None,
    yes_ev: EVResult | None,
    no_ev: EVResult | None,
    skip_reasons: list[str],
) -> list[str]:
    score = score_candidate_market(market, liquidity, size=size)
    payload = _journal_draft_payload(
        market=market,
        score=score,
        action=action,
        fair_yes_probability=fair_yes_probability,
        size=size,
        yes_ev=yes_ev,
        no_ev=no_ev,
        skip_reasons=skip_reasons,
    )
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    return [
        "",
        "### Journal Command Draft",
        "",
        "This draft does not write to the journal automatically. Save the JSON payload as `journal_draft.json`, review it, then run:",
        "",
        "```powershell",
        "poly-alpha-lab journal add --from-json-file journal_draft.json",
        "```",
        "",
        "```json",
        payload_json,
        "```",
    ]


def _journal_draft_payload(
    *,
    market: Market,
    score: CandidateScore,
    action: str,
    fair_yes_probability: float,
    size: float,
    yes_ev: EVResult | None,
    no_ev: EVResult | None,
    skip_reasons: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "market_id": market.id,
        "slug": market.slug,
        "question": market.question or market.slug or market.id,
        "category": score.normalized_category,
        "end_date": market.end_date.isoformat() if market.end_date else None,
        "candidate_score": score.total_score,
        "candidate_grade": score.candidate_grade,
        "fair_yes_probability": fair_yes_probability,
        "probability_source": "manual",
        "entry_size": size,
    }

    selected: EVResult | None = None
    if action == "BUY_YES":
        payload["side"] = "YES"
        selected = yes_ev
        payload["rationale"] = (
            "manual paper trade; conditional EV from user-provided fair_yes_probability"
        )
        payload["status"] = "OPEN"
    elif action == "BUY_NO":
        payload["side"] = "NO"
        selected = no_ev
        payload["rationale"] = (
            "manual paper trade; conditional EV from user-provided fair_yes_probability"
        )
        payload["status"] = "OPEN"
    elif action == "WATCH":
        payload["side"] = "NONE"
        payload["rationale"] = "watch only, conditional EV did not pass threshold"
        payload["status"] = "SKIPPED"
    else:
        payload["side"] = "NONE"
        reason = " ; ".join(skip_reasons) if skip_reasons else "suggested_action_skip"
        payload["rationale"] = f"skip: {reason}"
        payload["status"] = "SKIPPED"

    if selected is not None:
        payload["entry_price"] = selected.price
        payload["fee_per_share"] = selected.estimated_fee / selected.size
        payload["expected_value_per_share"] = selected.net_edge
        payload["expected_profit"] = selected.expected_profit

    return {key: value for key, value in payload.items() if value is not None}


def _resolution_analysis_section(analysis: ResolutionAnalysis) -> list[str]:
    return [
        "",
        "### Resolution Analysis",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| resolution_text_source | `{analysis.resolution_text_source or 'n/a'}` |",
        f"| has_resolution_text | `{analysis.has_resolution_text}` |",
        f"| ambiguity_risk | `{analysis.ambiguity_risk}` |",
        f"| dispute_risk | `{analysis.dispute_risk}` |",
        f"| risk_score | `{analysis.risk_score:.2f}` |",
        f"| resolution_source | `{analysis.resolution_source or 'n/a'}` |",
        f"| deadline | `{analysis.deadline or 'n/a'}` |",
        f"| time_zone | `{analysis.time_zone or 'n/a'}` |",
        f"| critical_phrases | `{_fmt_list(analysis.critical_phrases)}` |",
        f"| missing_fields | `{_fmt_list(analysis.missing_fields)}` |",
        f"| warnings | `{_fmt_list(analysis.warnings)}` |",
    ]


def resolution_analysis_report(market: Market) -> str:
    """Render a single-market resolution criteria analysis."""

    analysis = analyze_resolution(market)
    lines = [
        f"# Resolution Analysis: {market.question or market.slug or market.id}",
        "",
        f"- market_id: `{analysis.market_id}`",
        f"- slug: `{analysis.slug or 'n/a'}`",
        f"- category: `{analysis.category}`",
        "",
        "## Extracted Criteria",
        "",
        f"- resolution_text_source: `{analysis.resolution_text_source or 'n/a'}`",
        f"- has_resolution_text: `{analysis.has_resolution_text}`",
        f"- resolution_text_excerpt: `{analysis.resolution_text_excerpt or 'n/a'}`",
        f"- what_counts_as_yes: `{analysis.what_counts_as_yes or 'n/a'}`",
        f"- what_counts_as_no: `{analysis.what_counts_as_no or 'n/a'}`",
        f"- resolution_source: `{analysis.resolution_source or 'n/a'}`",
        f"- deadline: `{analysis.deadline or 'n/a'}`",
        f"- time_zone: `{analysis.time_zone or 'n/a'}`",
        "",
        "## Risk Review",
        "",
        f"- ambiguity_risk: `{analysis.ambiguity_risk}`",
        f"- dispute_risk: `{analysis.dispute_risk}`",
        f"- risk_score: `{analysis.risk_score:.2f}`",
        f"- critical_phrases: `{_fmt_list(analysis.critical_phrases)}`",
        f"- missing_fields: `{_fmt_list(analysis.missing_fields)}`",
        f"- warnings: `{_fmt_list(analysis.warnings)}`",
        f"- research_notes: `{_fmt_list(analysis.research_notes)}`",
        "",
        "This is a rule-based resolution criteria review only. It is not a trading signal.",
    ]
    return "\n".join(lines)


def markets_report(
    markets: list[Market],
    *,
    fair_yes_probability: float,
    size: float,
    liquidities: dict[str, BinaryMarketLiquidity] | None = None,
    min_net_edge: float = settings.min_net_edge,
) -> str:
    """Render a multi-market Markdown report."""

    sections = [
        "# Polymarket Research Report",
        "",
        f"- Markets: `{len(markets)}`",
        f"- Position size: `{size:g}` shares",
        f"- Fair YES probability: `{fair_yes_probability:.2%}`",
        f"- Minimum net edge for BUY: `{min_net_edge:.2%}`",
        "- EV is conditional on user-provided fair_yes_probability.",
        "- Journal: use `poly-alpha-lab journal add ...` to manually record a paper trade; scan never writes journal entries automatically.",
        "",
    ]
    liquidities = liquidities or {}
    for market in markets:
        sections.append(
            market_report(
                market,
                fair_yes_probability=fair_yes_probability,
                size=size,
                liquidity=liquidities.get(market.id),
                min_net_edge=min_net_edge,
            )
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def candidate_market_report(
    market: Market,
    *,
    size: float,
    liquidity: BinaryMarketLiquidity | None = None,
    score: CandidateScore | None = None,
) -> str:
    """Render one candidate-scanner market section."""

    score = score or score_candidate_market(market, liquidity, size=size)
    indicative_yes, _ = _safe_market_attr(market, "yes_price")
    indicative_no, _ = _safe_market_attr(market, "no_price")
    resolution = analyze_resolution(market)

    return "\n".join(
        [
            f"## {market.question or market.slug or market.id}",
            "",
            f"- market_id: `{market.id}`",
            f"- slug: `{market.slug or 'n/a'}`",
            f"- category: `{score.normalized_category}`",
            f"- raw_category_source: `{score.raw_category_source}`",
            f"- end_date: `{market.end_date.isoformat() if market.end_date else 'n/a'}`",
            f"- indicative_yes_price: `{_fmt_prob(indicative_yes)}`",
            f"- indicative_no_price: `{_fmt_prob(indicative_no)}`",
            f"- display_yes_best_ask: `{_fmt_prob(liquidity.yes_best_ask if liquidity else None)}`",
            f"- display_no_best_ask: `{_fmt_prob(liquidity.no_best_ask if liquidity else None)}`",
            f"- yes_spread: `{_fmt_prob(liquidity.yes_spread if liquidity else None)}`",
            f"- no_spread: `{_fmt_prob(liquidity.no_spread if liquidity else None)}`",
            f"- yes_ask_depth_3pct: `{_fmt_money(liquidity.yes_ask_depth_3pct if liquidity else None)}`",
            f"- no_ask_depth_3pct: `{_fmt_money(liquidity.no_ask_depth_3pct if liquidity else None)}`",
            f"- fee_assumption: `{fee_assumption(market.fees_enabled, score.normalized_category)}`",
            f"- candidate_score: `{score.total_score:.2f}`",
            f"- candidate_grade: `{score.candidate_grade}` `{candidate_grade_label(score.candidate_grade)}`",
            f"- reasons: `{', '.join(score.reasons) if score.reasons else 'none'}`",
            f"- resolution_risk_score: `{resolution.risk_score:.2f}`",
            f"- ambiguity_risk: `{resolution.ambiguity_risk}`",
            f"- dispute_risk: `{resolution.dispute_risk}`",
            f"- resolution_warnings: `{_fmt_list(resolution.warnings)}`",
            f"- resolution_missing_fields: `{_fmt_list(resolution.missing_fields)}`",
        ]
    )


def candidates_report(
    markets: list[Market],
    *,
    size: float,
    liquidities: dict[str, BinaryMarketLiquidity] | None = None,
) -> str:
    """Render the candidate scanner report."""

    liquidities = liquidities or {}
    scored = [
        (market, score_candidate_market(market, liquidities.get(market.id), size=size))
        for market in markets
    ]
    scored.sort(key=lambda item: item[1].total_score, reverse=True)

    counts = {grade: 0 for grade in ("A", "B", "C", "SKIP")}
    for _, score in scored:
        counts[score.candidate_grade] += 1

    sections = [
        "# Polymarket Candidate Market Scanner",
        "",
        f"- Markets: `{len(scored)}`",
        f"- Position size for depth checks: `{size:g}` shares",
        "- Mode: `candidates`",
        "- Purpose: `human_research_triage_not_trade_signal`",
        "- Journal: use `poly-alpha-lab journal add ...` to manually record a market; scan never writes journal entries automatically.",
        f"- Grades: `A={counts['A']}` `B={counts['B']}` `C={counts['C']}` `SKIP={counts['SKIP']}`",
        "",
    ]
    for market, score in scored:
        sections.append(
            candidate_market_report(
                market,
                size=size,
                liquidity=liquidities.get(market.id),
                score=score,
            )
        )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"
