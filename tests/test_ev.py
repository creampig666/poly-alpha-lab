import pytest

from poly_alpha_lab.ev import calculate_buy_ev, calculate_liquidity_ev
from poly_alpha_lab.fees import estimate_maker_fee, estimate_taker_fee, taker_fee_per_share
from poly_alpha_lab.liquidity import BinaryMarketLiquidity
from poly_alpha_lab.models import OrderBook, OrderLevel


def make_liquidity() -> BinaryMarketLiquidity:
    return BinaryMarketLiquidity(
        yes_book=OrderBook(
            token_id="yes-token",
            bids=[OrderLevel(price=0.50, size=10)],
            asks=[OrderLevel(price=0.55, size=100)],
        ),
        no_book=OrderBook(
            token_id="no-token",
            bids=[OrderLevel(price=0.38, size=10)],
            asks=[OrderLevel(price=0.42, size=100)],
        ),
    )


def test_calculate_yes_ev_after_fee() -> None:
    result = calculate_buy_ev(
        side="YES",
        fair_yes_probability=0.60,
        price=0.55,
        price_source="executable_best_ask",
        size=100,
        category="Politics",
    )

    assert result.side == "YES"
    assert result.fair_probability == pytest.approx(0.60)
    assert result.gross_edge == pytest.approx(0.05)
    assert result.cost == pytest.approx(55.0)
    assert result.estimated_fee == pytest.approx(0.495)
    assert result.net_edge == pytest.approx(0.04505)
    assert result.expected_profit == pytest.approx(4.505)


def test_calculate_no_ev_uses_complement_probability() -> None:
    result = calculate_buy_ev(
        side="NO",
        fair_yes_probability=0.60,
        price=0.42,
        size=100,
    )

    assert result.fair_probability == pytest.approx(0.40)
    assert result.gross_edge == pytest.approx(-0.02)


def test_yes_ev_uses_yes_ask_not_gamma_indicative_price() -> None:
    yes, no = calculate_liquidity_ev(
        liquidity=make_liquidity(),
        fair_yes_probability=0.60,
        size=10,
    )

    assert yes.price == pytest.approx(0.55)
    assert yes.price_source == "executable_avg_buy"
    assert no.price == pytest.approx(0.42)


def test_no_ev_uses_no_ask_not_yes_complement_or_gamma_price() -> None:
    _, no = calculate_liquidity_ev(
        liquidity=make_liquidity(),
        fair_yes_probability=0.60,
        size=10,
    )

    assert no.price == pytest.approx(0.42)
    assert no.fair_probability == pytest.approx(0.40)


def test_avg_buy_price_missing_does_not_fallback_to_best_ask() -> None:
    liquidity = BinaryMarketLiquidity(
        yes_book=OrderBook(
            token_id="yes-token",
            bids=[OrderLevel(price=0.10, size=10)],
            asks=[OrderLevel(price=0.20, size=5)],
        ),
        no_book=OrderBook(
            token_id="no-token",
            bids=[OrderLevel(price=0.70, size=10)],
            asks=[OrderLevel(price=0.80, size=100)],
        ),
    )

    with pytest.raises(ValueError, match="insufficient_ask_depth:YES"):
        calculate_liquidity_ev(liquidity=liquidity, fair_yes_probability=0.90, size=10)


def test_fee_per_share_formula_and_fee_flags() -> None:
    assert taker_fee_per_share(0.55, category="Politics") == pytest.approx(0.02 * 0.55 * 0.45)
    assert estimate_taker_fee(0.55, 100, category="Politics") == pytest.approx(0.495)
    assert estimate_taker_fee(0.55, 100, fees_enabled=False) == pytest.approx(0)
    assert estimate_maker_fee() == pytest.approx(0)


def test_fees_enabled_false_keeps_ev_fee_zero() -> None:
    result = calculate_buy_ev(
        side="YES",
        fair_yes_probability=0.60,
        price=0.55,
        size=100,
        fees_enabled=False,
    )

    assert result.estimated_fee == pytest.approx(0)
    assert result.net_edge == pytest.approx(0.05)


def test_calculate_ev_validates_inputs() -> None:
    with pytest.raises(ValueError, match="side"):
        calculate_buy_ev(side="MAYBE", fair_yes_probability=0.5, price=0.5, size=1)

    with pytest.raises(ValueError, match="price"):
        calculate_buy_ev(side="YES", fair_yes_probability=0.5, price=1.5, size=1)
