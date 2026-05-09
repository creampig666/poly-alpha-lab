from datetime import UTC, datetime, timedelta

from poly_alpha_lab.category_normalization import extract_category_info
from poly_alpha_lab.candidate_scoring import score_candidate_market
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.markdown_report import candidates_report
from poly_alpha_lab.models import Market, OrderBook, OrderLevel


NOW = datetime(2026, 5, 7, tzinfo=UTC)


def market(**overrides: object) -> Market:
    data = {
        "id": "m1",
        "question": "Will this be worth researching?",
        "slug": "worth-researching",
        "category": "Politics",
        "active": True,
        "closed": False,
        "archived": False,
        "enableOrderBook": True,
        "acceptingOrders": True,
        "liquidity": 10_000,
        "volume": 50_000,
        "endDate": (NOW + timedelta(days=30)).isoformat(),
        "feesEnabled": True,
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.45, 0.55],
        "clobTokenIds": ["yes-token", "no-token"],
    }
    data.update(overrides)
    return Market.model_validate(data)


def liquidity(
    *,
    yes_bid: float = 0.49,
    yes_ask: float = 0.50,
    no_bid: float = 0.49,
    no_ask: float = 0.50,
    yes_depth: float = 100,
    no_depth: float = 100,
) -> BinaryMarketLiquidity:
    return BinaryMarketLiquidity(
        yes_book=OrderBook(
            token_id="yes-token",
            bids=[OrderLevel(price=yes_bid, size=100)],
            asks=[OrderLevel(price=yes_ask, size=yes_depth)],
        ),
        no_book=OrderBook(
            token_id="no-token",
            bids=[OrderLevel(price=no_bid, size=100)],
            asks=[OrderLevel(price=no_ask, size=no_depth)],
        ),
    )


def test_tight_spread_scores_higher_than_wide_spread() -> None:
    tight = score_candidate_market(market(), liquidity(), size=10, now=NOW)
    wide = score_candidate_market(
        market(),
        liquidity(yes_bid=0.30, yes_ask=0.70, no_bid=0.30, no_ask=0.70),
        size=10,
        now=NOW,
    )

    assert tight.spread_score > wide.spread_score
    assert tight.total_score > wide.total_score


def test_sufficient_depth_scores_higher_than_insufficient_depth() -> None:
    sufficient = score_candidate_market(market(), liquidity(yes_depth=100, no_depth=100), size=10, now=NOW)
    insufficient = score_candidate_market(market(), liquidity(yes_depth=5, no_depth=100), size=10, now=NOW)

    assert sufficient.depth_score > insufficient.depth_score
    assert "insufficient_ask_depth" in insufficient.reasons


def test_past_end_date_is_skip() -> None:
    score = score_candidate_market(
        market(endDate=(NOW - timedelta(days=1)).isoformat()),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert score.candidate_grade == "SKIP"
    assert "end_date_past" in score.reasons


def test_missing_end_date_is_skip() -> None:
    score = score_candidate_market(market(endDate=None), liquidity(), size=10, now=NOW)

    assert score.candidate_grade == "SKIP"
    assert "missing_end_date" in score.reasons


def test_fees_enabled_unknown_adds_reason() -> None:
    score = score_candidate_market(market(feesEnabled=None), liquidity(), size=10, now=NOW)

    assert "fees_enabled_unknown" in score.reasons


def test_category_unknown_adds_reason() -> None:
    score = score_candidate_market(
        market(category="Unmapped", question="Will this happen?", slug="will-this-happen"),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert score.normalized_category == "unknown"
    assert "category_unknown" in score.reasons


def test_candidates_report_does_not_need_fair_yes_or_emit_buy_actions() -> None:
    report = candidates_report([market()], size=10, liquidities={"m1": liquidity()})

    assert "fair_yes" not in report
    assert "BUY_YES" not in report
    assert "BUY_NO" not in report


def test_non_standard_binary_market_is_skip() -> None:
    score = score_candidate_market(
        market(
            outcomes=["Yes", "No", "Maybe"],
            outcomePrices=[0.4, 0.5, 0.1],
            clobTokenIds=["yes-token", "no-token", "maybe-token"],
        ),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert score.candidate_grade == "SKIP"
    assert "non_standard_binary_market" in score.reasons


def test_category_from_market_category() -> None:
    info = extract_category_info(market(category="NBA"))

    assert info.normalized_category == "sports"
    assert info.raw_category_source == "market.category"


def test_category_from_events_category() -> None:
    info = extract_category_info(
        market(category=None, events=[{"category": "Inflation"}])
    )

    assert info.normalized_category == "economics"
    assert info.raw_category_source == 'market.raw["events"][0]["category"]'


def test_category_from_tags() -> None:
    info = extract_category_info(market(category=None, tags=[{"label": "Weather"}]))

    assert info.normalized_category == "weather"
    assert info.raw_category_source == 'market.raw["tags"]'


def test_category_keyword_fallback_fed_rate_hike() -> None:
    score = score_candidate_market(
        market(category=None, question="Fed rate hike in 2026?", slug="fed-rate-hike-in-2026"),
        liquidity(),
        size=10,
        now=NOW,
    )

    assert score.normalized_category == "economics"
    assert "category_keyword_fallback" in score.reasons


def test_category_keyword_fallback_bitcoin() -> None:
    info = extract_category_info(
        market(category=None, question="Will Bitcoin dip to $50,000?", slug="bitcoin-dip")
    )

    assert info.normalized_category == "crypto"
    assert info.used_keyword_fallback


def test_category_keyword_fallback_f1() -> None:
    info = extract_category_info(
        market(category=None, question="Will Valtteri Bottas be the 2026 F1 Champion?")
    )

    assert info.normalized_category == "sports"


def test_category_keyword_fallback_discord_ipo_market_cap() -> None:
    info = extract_category_info(
        market(category=None, question="Will Discord IPO market cap be above $30B?")
    )

    assert info.normalized_category in {"tech", "finance"}
    assert info.normalized_category != "unknown"


def test_size_increase_reduces_depth_score_on_same_orderbook() -> None:
    book = liquidity(yes_ask=0.50, no_ask=0.50, yes_depth=50, no_depth=50)

    score10 = score_candidate_market(market(), book, size=10, now=NOW)
    score100 = score_candidate_market(market(), book, size=100, now=NOW)

    assert score100.depth_score < score10.depth_score


def test_low_near_touch_depth_cannot_be_grade_a() -> None:
    book = liquidity(yes_ask=0.50, no_ask=0.50, yes_depth=10, no_depth=1_000)

    score = score_candidate_market(
        market(liquidity=1_000_000, volume=1_000_000),
        book,
        size=100,
        now=NOW,
    )

    assert score.candidate_grade != "A"
    assert "insufficient_near_touch_depth" in score.reasons


def test_wide_spread_cannot_be_grade_a() -> None:
    book = liquidity(yes_bid=0.40, yes_ask=0.50, no_bid=0.40, no_ask=0.50, yes_depth=10_000, no_depth=10_000)

    score = score_candidate_market(
        market(liquidity=1_000_000, volume=1_000_000),
        book,
        size=10,
        now=NOW,
    )

    assert score.candidate_grade != "A"
    assert "wide_spread" in score.reasons
