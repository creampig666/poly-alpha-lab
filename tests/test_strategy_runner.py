import json
from datetime import UTC, datetime, timedelta

import pytest

import poly_alpha_lab.main as cli
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import Market, OrderBook, OrderLevel
from poly_alpha_lab.strategy_runner import (
    MANUAL_RATIONALE,
    build_strategy_candidate,
    scan_strategy_candidates,
    strategy_candidates_report,
    write_strategy_candidates_json,
)


NOW = datetime(2026, 5, 8, tzinfo=UTC)
RULES = (
    "This market will resolve to Yes if the official BLS CPI report is above 3% "
    "by 8:30 AM ET on May 12, 2026. Otherwise, this market resolves to No."
)


def market(**overrides: object) -> Market:
    data = {
        "id": "m1",
        "question": "Will CPI be above 3%?",
        "slug": "will-cpi-be-above-3",
        "category": "Economics",
        "active": True,
        "closed": False,
        "archived": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "liquidity": 50_000,
        "volume": 200_000,
        "endDate": (NOW + timedelta(days=15)).isoformat(),
        "feesEnabled": True,
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.4, 0.6],
        "clobTokenIds": ["yes-token", "no-token"],
        "resolutionCriteria": RULES,
    }
    data.update(overrides)
    return Market.model_validate(data)


def liquidity(
    *,
    yes_token_id: str = "yes-token",
    no_token_id: str = "no-token",
    yes_bid: float = 0.39,
    yes_ask: float = 0.40,
    no_bid: float = 0.59,
    no_ask: float = 0.60,
    yes_depth: float = 1_000,
    no_depth: float = 1_000,
) -> BinaryMarketLiquidity:
    return BinaryMarketLiquidity(
        yes_book=OrderBook(
            token_id=yes_token_id,
            bids=[OrderLevel(price=yes_bid, size=1_000)],
            asks=[OrderLevel(price=yes_ask, size=yes_depth)],
        ),
        no_book=OrderBook(
            token_id=no_token_id,
            bids=[OrderLevel(price=no_bid, size=1_000)],
            asks=[OrderLevel(price=no_ask, size=no_depth)],
        ),
    )


def test_yes_breakeven_is_price_plus_fee() -> None:
    candidate = build_strategy_candidate(market(), liquidity(), size=10)

    assert candidate.yes_breakeven_probability == pytest.approx(
        candidate.yes_executable_avg_buy_price + candidate.yes_fee_per_share
    )


def test_no_upper_bound_is_one_minus_no_price_plus_fee() -> None:
    candidate = build_strategy_candidate(market(), liquidity(), size=10)

    assert candidate.no_required_yes_probability_upper_bound == pytest.approx(
        1 - (candidate.no_executable_avg_buy_price + candidate.no_fee_per_share)
    )


def test_strategy_report_emits_no_buy_or_sell_action() -> None:
    candidate = build_strategy_candidate(market(), liquidity(), size=10)
    report = strategy_candidates_report([candidate])

    assert "BUY_YES" not in report
    assert "BUY_NO" not in report
    assert "SELL" not in report


def test_strategy_scan_excludes_skip_candidates() -> None:
    skip = market(id="skip", endDate=(NOW - timedelta(days=1)).isoformat())
    ok = market(id="ok")
    result = scan_strategy_candidates(
        [skip, ok],
        {"skip": liquidity(), "ok": liquidity()},
        size=10,
    )

    assert [item.market_id for item in result] == ["ok"]


def test_strategy_scan_respects_min_grade() -> None:
    grade_b = market(id="b", liquidity=1_000, volume=1_000)
    grade_a = market(id="a")
    result = scan_strategy_candidates(
        [grade_b, grade_a],
        {"b": liquidity(), "a": liquidity()},
        size=10,
        min_grade="A",
    )

    assert [item.market_id for item in result] == ["a"]


def test_strategy_scan_respects_max_resolution_risk() -> None:
    high_risk = market(id="high", resolutionCriteria=None, question="Will this happen?")
    low_risk = market(id="low")
    result = scan_strategy_candidates(
        [high_risk, low_risk],
        {"high": liquidity(), "low": liquidity()},
        size=10,
        max_resolution_risk="LOW",
    )

    assert [item.market_id for item in result] == ["low"]


def test_long_time_to_resolution_creates_warning_and_penalty() -> None:
    short = build_strategy_candidate(market(id="short"), liquidity(), size=10, now=NOW)
    long = build_strategy_candidate(
        market(id="long", endDate=(NOW + timedelta(days=500)).isoformat()),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert "long_time_to_resolution_penalty" in long.warnings
    assert long.strategy_score < short.strategy_score


def test_near_expiry_market_gets_warning_and_lower_score() -> None:
    normal = build_strategy_candidate(market(id="normal"), liquidity(), size=10, now=NOW)
    near = build_strategy_candidate(
        market(id="near", endDate=(NOW + timedelta(hours=12)).isoformat()),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert "near_expiry_penalty" in near.warnings
    assert near.strategy_score < normal.strategy_score


def test_same_day_expiry_market_gets_warning() -> None:
    candidate = build_strategy_candidate(
        market(id="same-day", endDate=(NOW + timedelta(hours=3)).isoformat()),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert "same_day_expiry_penalty" in candidate.warnings


def test_expired_market_excluded_from_strategy_candidates() -> None:
    expired = market(id="expired", endDate=(NOW - timedelta(hours=1)).isoformat())
    result = scan_strategy_candidates(
        [expired],
        {"expired": liquidity()},
        size=10,
        now=NOW,
    )

    assert result == []


def test_geopolitics_military_event_gets_warning_and_lower_score() -> None:
    economics = build_strategy_candidate(market(id="econ"), liquidity(), size=10, now=NOW)
    military = build_strategy_candidate(
        market(
            id="military",
            question="US x Cuba military clash in 2026?",
            slug="us-x-cuba-military-clash-in-2026",
            category="Geopolitics",
        ),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert "geopolitics_military_event_risk" in military.warnings
    assert military.dispute_risk in {"MEDIUM", "HIGH"}
    assert military.strategy_score < economics.strategy_score


def test_sports_outright_longshot_excluded_by_default() -> None:
    sports = market(
        id="sports",
        question="Will Amanda Anisimova win 2026 Wimbledon?",
        slug="amanda-anisimova-2026-wimbledon-winner",
        category="Sports",
        endDate=(NOW + timedelta(days=400)).isoformat(),
        outcomePrices=[0.02, 0.98],
    )
    result = scan_strategy_candidates(
        [sports],
        {"sports": liquidity(yes_bid=0.01, yes_ask=0.02, no_bid=0.97, no_ask=0.98)},
        size=10,
        now=NOW,
    )

    assert result == []


def test_unknown_category_tennis_outright_longshot_excluded_by_default() -> None:
    sports = market(
        id="unknown-tennis",
        question="Will Victoria Azarenka win the 2026 Women’s French Open?",
        slug="will-victoria-azarenka-win-the-2026-womens-french-open",
        category="Unmapped",
        endDate=(NOW + timedelta(days=45)).isoformat(),
        outcomePrices=[0.003, 0.997],
    )
    result = scan_strategy_candidates(
        [sports],
        {"unknown-tennis": liquidity(yes_bid=0.002, yes_ask=0.003, no_bid=0.996, no_ask=0.998)},
        size=10,
        now=NOW,
    )

    assert result == []


def test_sports_outright_longshot_included_only_when_requested() -> None:
    sports = market(
        id="sports",
        question="Will Amanda Anisimova win 2026 Wimbledon?",
        slug="amanda-anisimova-2026-wimbledon-winner",
        category="Sports",
        endDate=(NOW + timedelta(days=400)).isoformat(),
        outcomePrices=[0.02, 0.98],
    )
    result = scan_strategy_candidates(
        [sports],
        {"sports": liquidity(yes_bid=0.01, yes_ask=0.02, no_bid=0.97, no_ask=0.98)},
        size=10,
        include_longshots=True,
        now=NOW,
    )

    assert len(result) == 1
    assert "sports_outright_longshot_penalty" in result[0].warnings


def test_sports_outright_low_price_does_not_outrank_clear_short_dated_economics() -> None:
    economics = build_strategy_candidate(market(id="econ"), liquidity(), size=10, now=NOW)
    sports = build_strategy_candidate(
        market(
            id="sports",
            question="Will Amanda Anisimova win 2026 Wimbledon?",
            slug="amanda-anisimova-2026-wimbledon-winner",
            category="Sports",
            endDate=(NOW + timedelta(days=400)).isoformat(),
            outcomePrices=[0.02, 0.98],
        ),
        liquidity(yes_bid=0.01, yes_ask=0.02, no_bid=0.97, no_ask=0.98),
        size=10,
        now=NOW,
    )

    assert "sports_outright_longshot_penalty" in sports.warnings
    assert economics.strategy_score > sports.strategy_score


def test_overlapping_breakeven_report_uses_anomaly_wording() -> None:
    candidate = build_strategy_candidate(
        market(id="overlap"),
        liquidity(yes_bid=0.19, yes_ask=0.20, no_bid=0.19, no_ask=0.20),
        size=10,
        now=NOW,
    )
    report = strategy_candidates_report([candidate])

    assert "overlapping breakeven region" in report
    assert "market data anomaly" in report
    assert "No-trade zone" not in report


def test_normal_breakeven_gap_report_uses_no_trade_zone_wording() -> None:
    candidate = build_strategy_candidate(market(id="normal"), liquidity(), size=10, now=NOW)
    report = strategy_candidates_report([candidate])

    assert "No-trade zone" in report
    assert "market data anomaly" not in report


def test_output_json_works(tmp_path) -> None:
    candidate = build_strategy_candidate(market(), liquidity(), size=10)
    output = tmp_path / "strategy_candidates.json"

    row_count = write_strategy_candidates_json([candidate], output)

    assert row_count == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data[0]["market_id"] == "m1"


def test_journal_draft_payloads_require_manual_probability_judgment() -> None:
    candidate = build_strategy_candidate(market(), liquidity(), size=10, now=NOW)

    assert candidate.journal_draft_payload_yes["side"] == "YES"
    assert candidate.journal_draft_payload_no["side"] == "NO"
    assert candidate.journal_draft_payload_yes["requires_user_fair_yes_probability"] is True
    assert candidate.journal_draft_payload_no["requires_user_fair_yes_probability"] is True
    assert candidate.journal_draft_payload_yes["fair_yes_probability"] is None
    assert candidate.journal_draft_payload_no["fair_yes_probability"] is None
    assert candidate.journal_draft_payload_yes["rationale"] == MANUAL_RATIONALE
    assert "manual probability judgment" in candidate.journal_draft_payload_no["rationale"]


def test_journal_add_from_json_rejects_yes_without_fair_yes(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli.sqlite"
    monkeypatch.setattr(cli.settings, "journal_db_path", str(db_path))
    candidate = build_strategy_candidate(market(), liquidity(), size=10, now=NOW)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(candidate.journal_draft_payload_yes), encoding="utf-8")

    with pytest.raises(SystemExit):
        cli.run(["journal", "add", "--from-json-file", str(draft)])

    assert make_journal_entries(db_path) == []


def test_journal_add_from_json_accepts_cli_fair_yes_override(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "cli.sqlite"
    monkeypatch.setattr(cli.settings, "journal_db_path", str(db_path))
    candidate = build_strategy_candidate(market(), liquidity(), size=10, now=NOW)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(candidate.journal_draft_payload_yes), encoding="utf-8")

    assert cli.run(["journal", "add", "--from-json-file", str(draft), "--fair-yes", "0.6"]) == 0
    entries = make_journal_entries(db_path)

    assert len(entries) == 1
    assert entries[0].fair_yes_probability == pytest.approx(0.6)


def make_journal_entries(db_path):
    from poly_alpha_lab.journal import ResearchJournal

    return ResearchJournal(db_path).list_entries(limit=10)
