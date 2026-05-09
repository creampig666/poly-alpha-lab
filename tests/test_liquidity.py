import pytest

from poly_alpha_lab.liquidity import BinaryMarketLiquidity, average_buy_price, best_ask, best_bid, spread
from poly_alpha_lab.models import OrderBook, OrderLevel


def yes_book() -> OrderBook:
    return OrderBook(
        token_id="yes-token",
        bids=[
            OrderLevel(price=0.48, size=100),
            OrderLevel(price=0.47, size=200),
        ],
        asks=[
            OrderLevel(price=0.52, size=50),
            OrderLevel(price=0.53, size=150),
            OrderLevel(price=0.56, size=100),
        ],
    )


def no_book() -> OrderBook:
    return OrderBook(
        token_id="no-token",
        bids=[
            OrderLevel(price=0.43, size=100),
            OrderLevel(price=0.42, size=200),
        ],
        asks=[
            OrderLevel(price=0.46, size=80),
            OrderLevel(price=0.47, size=120),
        ],
    )


def test_single_token_top_of_book_metrics() -> None:
    book = yes_book()

    assert best_bid(book) == pytest.approx(0.48)
    assert best_ask(book) == pytest.approx(0.52)
    assert spread(book) == pytest.approx(0.04)


def test_average_buy_price_consumes_token_asks() -> None:
    avg = average_buy_price(yes_book(), 100)

    assert avg == pytest.approx(((50 * 0.52) + (50 * 0.53)) / 100)


def test_average_buy_price_rejects_insufficient_depth() -> None:
    with pytest.raises(ValueError, match="insufficient ask depth"):
        average_buy_price(yes_book(), 500)


def test_binary_market_liquidity_separates_yes_and_no_books() -> None:
    liquidity = BinaryMarketLiquidity(yes_book=yes_book(), no_book=no_book())

    assert liquidity.yes_best_bid == pytest.approx(0.48)
    assert liquidity.yes_best_ask == pytest.approx(0.52)
    assert liquidity.no_best_bid == pytest.approx(0.43)
    assert liquidity.no_best_ask == pytest.approx(0.46)
    assert liquidity.yes_spread == pytest.approx(0.04)
    assert liquidity.no_spread == pytest.approx(0.03)
    assert liquidity.yes_avg_buy_price(100) == pytest.approx(0.525)
    assert liquidity.no_avg_buy_price(100) == pytest.approx(((80 * 0.46) + (20 * 0.47)) / 100)
    assert liquidity.yes_ask_depth_3pct == pytest.approx(200)
    assert liquidity.no_ask_depth_3pct == pytest.approx(200)

