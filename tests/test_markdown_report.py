import json
import re

from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.markdown_report import market_report
from poly_alpha_lab.models import Market, OrderBook, OrderLevel


def binary_market(**overrides: object) -> Market:
    data = {
        "id": "m1",
        "question": "Will test pass?",
        "category": "Politics",
        "feesEnabled": False,
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.40","0.60"]',
        "clobTokenIds": '["yes-token","no-token"]',
    }
    data.update(overrides)
    return Market.model_validate(data)


def liquidity(
    *,
    yes_token_id: str = "yes-token",
    no_token_id: str = "no-token",
    yes_ask: float = 0.53,
    no_ask: float = 0.60,
    yes_size: float = 100,
    no_size: float = 100,
) -> BinaryMarketLiquidity:
    return BinaryMarketLiquidity(
        yes_book=OrderBook(
            token_id=yes_token_id,
            bids=[OrderLevel(price=max(0.0, yes_ask - 0.03), size=100)],
            asks=[OrderLevel(price=yes_ask, size=yes_size)],
        ),
        no_book=OrderBook(
            token_id=no_token_id,
            bids=[OrderLevel(price=max(0.0, no_ask - 0.03), size=100)],
            asks=[OrderLevel(price=no_ask, size=no_size)],
        ),
    )


def journal_payload(report: str) -> dict:
    match = re.search(r"```json\n(.*?)\n```", report, flags=re.S)
    assert match is not None
    return json.loads(match.group(1))


def test_missing_orderbook_sets_suggested_action_skip() -> None:
    market = binary_market()

    report = market_report(market, fair_yes_probability=0.55, size=10)
    payload = journal_payload(report)

    assert "Suggested action: `SKIP`" in report
    assert "missing CLOB orderbook" in report
    assert "Journal Command Draft" in report
    assert payload["side"] == "NONE"
    assert "missing CLOB orderbook" in payload["rationale"]


def test_buy_yes_report_includes_journal_draft_side_yes() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.90,
        size=10,
        liquidity=liquidity(yes_ask=0.20, no_ask=0.85),
        min_net_edge=0.03,
    )
    payload = journal_payload(report)

    assert "Suggested action: `BUY_YES`" in report
    assert payload["side"] == "YES"
    assert payload["entry_price"] == 0.20
    assert payload["expected_profit"] > 0


def test_buy_no_report_includes_journal_draft_side_no() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.10,
        size=10,
        liquidity=liquidity(yes_ask=0.85, no_ask=0.20),
        min_net_edge=0.03,
    )
    payload = journal_payload(report)

    assert "Suggested action: `BUY_NO`" in report
    assert payload["side"] == "NO"
    assert payload["entry_price"] == 0.20
    assert payload["expected_profit"] > 0


def test_watch_report_includes_research_only_journal_draft() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.55,
        size=10,
        liquidity=liquidity(yes_ask=0.53, no_ask=0.70),
        min_net_edge=0.03,
    )
    payload = journal_payload(report)

    assert "Suggested action: `WATCH`" in report
    assert payload["side"] == "NONE"
    assert "watch only" in payload["rationale"]
    assert "entry_price" not in payload


def test_insufficient_ask_depth_does_not_output_buy() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.95,
        size=10,
        liquidity=liquidity(yes_ask=0.20, no_ask=0.80, yes_size=5, no_size=100),
    )
    payload = journal_payload(report)

    assert "Suggested action: `SKIP`" in report
    assert "insufficient_ask_depth:YES" in report
    assert payload["side"] == "NONE"
    assert "insufficient_ask_depth:YES" in payload["rationale"]
    assert "BUY_YES" not in report
    assert "BUY_NO" not in report


def test_token_id_mismatch_outputs_skip() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.90,
        size=10,
        liquidity=liquidity(yes_token_id="wrong-yes-token", yes_ask=0.20),
    )

    assert "Suggested action: `SKIP`" in report
    assert "token_id_mismatch" in report


def test_positive_edge_below_min_net_edge_outputs_watch() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.55,
        size=10,
        liquidity=liquidity(yes_ask=0.53, no_ask=0.70),
        min_net_edge=0.03,
    )

    assert "Suggested action: `WATCH`" in report
    assert "BUY_YES" not in report


def test_conflicting_positive_edges_output_skip() -> None:
    report = market_report(
        binary_market(),
        fair_yes_probability=0.50,
        size=10,
        liquidity=liquidity(yes_ask=0.20, no_ask=0.20),
        min_net_edge=0.03,
    )

    assert "Suggested action: `SKIP`" in report
    assert "conflicting_positive_edges" in report


def test_category_unknown_report_shows_fee_assumption() -> None:
    report = market_report(
        binary_market(category="Unmapped", question="Will this happen?", feesEnabled=True),
        fair_yes_probability=0.55,
        size=10,
        liquidity=liquidity(),
    )

    assert "fee_assumption=`category_unknown_using_default`" in report


def test_fees_enabled_unknown_with_economics_uses_category_rate_assumption() -> None:
    report = market_report(
        binary_market(category="Economics", feesEnabled=None),
        fair_yes_probability=0.55,
        size=10,
        liquidity=liquidity(),
    )

    assert "fee_assumption=`fees_enabled_unknown_using_category_rate`" in report


def test_fees_enabled_false_report_shows_zero_fee() -> None:
    report = market_report(
        binary_market(feesEnabled=False),
        fair_yes_probability=0.60,
        size=10,
        liquidity=liquidity(yes_ask=0.50, no_ask=0.70),
    )

    assert "fee_assumption=`fees_enabled_false_fee_zero`" in report
    assert "| YES | 60.00% | 0.5000 | `executable_avg_buy` | 10.00% | $5.00 | $0.00 |" in report


def test_journal_draft_json_handles_special_characters() -> None:
    question = 'Will "quoted", comma, and question marks? survive'
    report = market_report(
        binary_market(question=question),
        fair_yes_probability=0.90,
        size=10,
        liquidity=liquidity(yes_ask=0.20, no_ask=0.85),
        min_net_edge=0.03,
    )
    payload = journal_payload(report)

    assert payload["question"] == question
