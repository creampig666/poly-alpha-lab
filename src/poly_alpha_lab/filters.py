"""Market filters for research universe construction."""

from __future__ import annotations

from poly_alpha_lab.models import Market, MarketStructureError


def is_research_candidate(market: Market, min_liquidity: float = 0) -> bool:
    """Return whether a market passes the default read-only research filters."""

    if not market.active:
        return False
    if market.closed or market.archived:
        return False
    if not market.enable_order_book:
        return False
    if market.liquidity is None or market.liquidity < min_liquidity:
        return False
    if market.raw.get("acceptingOrders") is False:
        return False
    if len(market.outcomes) != 2:
        return False
    try:
        market.yes_outcome_index
        market.no_outcome_index
        market.yes_token_id
        market.no_token_id
    except MarketStructureError:
        return False
    return True


def filter_markets(markets: list[Market], min_liquidity: float = 0) -> list[Market]:
    """Filter active, open, order-book-enabled markets with enough liquidity."""

    return [market for market in markets if is_research_candidate(market, min_liquidity)]
