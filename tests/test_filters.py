from poly_alpha_lab.filters import filter_markets, is_research_candidate
from poly_alpha_lab.models import Market


def market(**overrides: object) -> Market:
    data = {
        "id": "1",
        "question": "Will test pass?",
        "active": True,
        "closed": False,
        "archived": False,
        "enableOrderBook": True,
        "liquidity": "1500",
        "outcomes": '["Yes","No"]',
        "outcomePrices": '["0.40","0.60"]',
        "clobTokenIds": '["yes-token","no-token"]',
    }
    data.update(overrides)
    return Market.model_validate(data)


def test_research_candidate_requires_active_open_orderbook_and_liquidity() -> None:
    assert is_research_candidate(market(), min_liquidity=1000)
    assert not is_research_candidate(market(active=False), min_liquidity=1000)
    assert not is_research_candidate(market(closed=True), min_liquidity=1000)
    assert not is_research_candidate(market(enableOrderBook=False), min_liquidity=1000)
    assert not is_research_candidate(market(liquidity="999"), min_liquidity=1000)


def test_filter_markets() -> None:
    markets = [market(id="1"), market(id="2", liquidity="10")]

    assert [item.id for item in filter_markets(markets, min_liquidity=100)] == ["1"]


def test_non_binary_market_is_not_research_candidate() -> None:
    assert not is_research_candidate(
        market(
            outcomes='["Yes","No","Maybe"]',
            outcomePrices='["0.30","0.60","0.10"]',
            clobTokenIds='["yes-token","no-token","maybe-token"]',
        )
    )


def test_accepting_orders_false_is_not_research_candidate() -> None:
    assert not is_research_candidate(market(acceptingOrders=False))
