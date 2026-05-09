"""Expected-value calculations for binary Polymarket outcomes."""

from __future__ import annotations

from poly_alpha_lab.fees import estimate_taker_fee
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import EVResult


def calculate_buy_ev(
    *,
    side: str,
    fair_yes_probability: float,
    price: float,
    price_source: str = "executable_best_ask",
    size: float,
    category: str | None = None,
    fees_enabled: bool | None = None,
) -> EVResult:
    """Calculate EV for buying YES or NO shares at an executable CLOB price.

    ``gross_edge`` and ``net_edge`` are expressed per share. ``expected_profit`` is
    expected dollars for the requested share size after estimated taker fees.
    """

    side_upper = side.upper()
    if side_upper not in {"YES", "NO"}:
        raise ValueError("side must be YES or NO")
    if not 0 <= fair_yes_probability <= 1:
        raise ValueError("fair_yes_probability must be between 0 and 1")
    if not 0 <= price <= 1:
        raise ValueError("price must be between 0 and 1")
    if size <= 0:
        raise ValueError("size must be positive")
    if price_source not in {"executable_best_ask", "executable_avg_buy"}:
        raise ValueError("price_source must be executable_best_ask or executable_avg_buy")

    fair_probability = fair_yes_probability if side_upper == "YES" else 1 - fair_yes_probability
    gross_edge = fair_probability - price
    cost = price * size
    estimated_fee = estimate_taker_fee(
        price=price,
        size=size,
        category=category,
        fees_enabled=fees_enabled,
    )
    net_edge = gross_edge - (estimated_fee / size)
    expected_profit = net_edge * size

    return EVResult(
        side=side_upper,
        fair_probability=fair_probability,
        price=price,
        price_source=price_source,
        size=size,
        gross_edge=gross_edge,
        cost=cost,
        estimated_fee=estimated_fee,
        net_edge=net_edge,
        expected_profit=expected_profit,
    )


def calculate_yes_no_ev(
    *,
    fair_yes_probability: float,
    yes_executable_price: float,
    no_executable_price: float,
    price_source: str = "executable_best_ask",
    size: float,
    category: str | None = None,
    fees_enabled: bool | None = None,
) -> tuple[EVResult, EVResult]:
    """Calculate EV results for buying YES and NO from executable CLOB prices."""

    yes = calculate_buy_ev(
        side="YES",
        fair_yes_probability=fair_yes_probability,
        price=yes_executable_price,
        price_source=price_source,
        size=size,
        category=category,
        fees_enabled=fees_enabled,
    )
    no = calculate_buy_ev(
        side="NO",
        fair_yes_probability=fair_yes_probability,
        price=no_executable_price,
        price_source=price_source,
        size=size,
        category=category,
        fees_enabled=fees_enabled,
    )
    return yes, no


def _executable_price_for_size(
    avg_buy_price: float,
) -> tuple[float, str]:
    return avg_buy_price, "executable_avg_buy"


def calculate_liquidity_ev(
    *,
    liquidity: BinaryMarketLiquidity,
    fair_yes_probability: float,
    size: float,
    category: str | None = None,
    fees_enabled: bool | None = None,
) -> tuple[EVResult, EVResult]:
    """Calculate EV from CLOB order books, preferring full-size average buy prices."""

    try:
        yes_avg = liquidity.yes_avg_buy_price(size)
    except ValueError as exc:
        raise ValueError(f"insufficient_ask_depth:YES:{exc}") from exc
    try:
        no_avg = liquidity.no_avg_buy_price(size)
    except ValueError as exc:
        raise ValueError(f"insufficient_ask_depth:NO:{exc}") from exc

    yes_price, yes_source = _executable_price_for_size(yes_avg)
    no_price, no_source = _executable_price_for_size(no_avg)

    yes = calculate_buy_ev(
        side="YES",
        fair_yes_probability=fair_yes_probability,
        price=yes_price,
        price_source=yes_source,
        size=size,
        category=category,
        fees_enabled=fees_enabled,
    )
    no = calculate_buy_ev(
        side="NO",
        fair_yes_probability=fair_yes_probability,
        price=no_price,
        price_source=no_source,
        size=size,
        category=category,
        fees_enabled=fees_enabled,
    )
    return yes, no
