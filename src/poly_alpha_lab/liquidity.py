"""Order-book and liquidity calculations."""

from __future__ import annotations

from pydantic import BaseModel

from poly_alpha_lab.models import OrderBook, OrderLevel


def best_bid(book: OrderBook | None) -> float | None:
    """Return the highest bid in one token order book."""

    if book is None or not book.bids:
        return None
    return book.bids[0].price


def best_ask(book: OrderBook | None) -> float | None:
    """Return the lowest ask in one token order book."""

    if book is None or not book.asks:
        return None
    return book.asks[0].price


def spread(book: OrderBook | None) -> float | None:
    """Return ask minus bid in one token order book."""

    bid = best_bid(book)
    ask = best_ask(book)
    if bid is None or ask is None:
        return None
    return ask - bid


def average_buy_price(book: OrderBook, size: float) -> float:
    """Compute size-weighted average taker price for buying one token.

    Raises:
        ValueError: If size is non-positive or asks cannot fill the requested size.
    """

    if size <= 0:
        raise ValueError("size must be positive")

    remaining = size
    notional = 0.0
    for level in book.asks:
        fill_size = min(remaining, level.size)
        notional += fill_size * level.price
        remaining -= fill_size
        if remaining <= 1e-12:
            return notional / size

    available = size - remaining
    raise ValueError(f"insufficient ask depth for {size:g}; available {available:g}")


def ask_depth_within(book: OrderBook | None, pct: float = 0.03) -> float | None:
    """Return ask size available within ``pct`` above the best ask."""

    if book is None:
        return None
    ask = best_ask(book)
    if ask is None:
        return None
    max_price = min(1.0, ask * (1 + pct))
    return sum(level.size for level in book.asks if level.price <= max_price)


class BinaryMarketLiquidity(BaseModel):
    """Executable liquidity for a binary market using separate YES and NO token books."""

    yes_book: OrderBook | None = None
    no_book: OrderBook | None = None

    @property
    def yes_best_bid(self) -> float | None:
        return best_bid(self.yes_book)

    @property
    def yes_best_ask(self) -> float | None:
        return best_ask(self.yes_book)

    @property
    def no_best_bid(self) -> float | None:
        return best_bid(self.no_book)

    @property
    def no_best_ask(self) -> float | None:
        return best_ask(self.no_book)

    @property
    def yes_spread(self) -> float | None:
        return spread(self.yes_book)

    @property
    def no_spread(self) -> float | None:
        return spread(self.no_book)

    def yes_avg_buy_price(self, size: float) -> float:
        if self.yes_book is None:
            raise ValueError("missing YES orderbook")
        return average_buy_price(self.yes_book, size)

    def no_avg_buy_price(self, size: float) -> float:
        if self.no_book is None:
            raise ValueError("missing NO orderbook")
        return average_buy_price(self.no_book, size)

    @property
    def yes_ask_depth_3pct(self) -> float | None:
        return ask_depth_within(self.yes_book, pct=0.03)

    @property
    def no_ask_depth_3pct(self) -> float | None:
        return ask_depth_within(self.no_book, pct=0.03)


def average_execution_price(book: OrderBook, side: str, size: float) -> float:
    """Backward-compatible wrapper; use BinaryMarketLiquidity in new code."""

    if side.upper() not in {"YES", "NO"}:
        raise ValueError("side must be YES or NO")
    return average_buy_price(book, size)


def available_depth(book: OrderBook, side: str, max_price: float | None = None) -> float:
    """Backward-compatible ask-depth helper for one token book."""

    if side.upper() not in {"YES", "NO"}:
        raise ValueError("side must be YES or NO")
    levels: list[OrderLevel] = book.asks
    if max_price is None:
        return sum(level.size for level in levels)
    return sum(level.size for level in levels if level.price <= max_price)
