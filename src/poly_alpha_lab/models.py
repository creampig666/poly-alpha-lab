"""Pydantic data models for Polymarket market research."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _list_from_jsonish(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        return parsed if isinstance(parsed, list) else []
    return []


class MarketStructureError(ValueError):
    """Raised when a Gamma market cannot be mapped to an explicit YES/NO structure."""


class OrderLevel(BaseModel):
    """One price level in a binary outcome order book."""

    model_config = ConfigDict(extra="ignore")

    price: float = Field(ge=0, le=1)
    size: float = Field(gt=0)


class OrderBook(BaseModel):
    """CLOB order book for one token, either the YES token or the NO token."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    token_id: str | None = Field(default=None, alias="asset_id")
    market_id: str | None = None
    bids: list[OrderLevel] = Field(default_factory=list)
    asks: list[OrderLevel] = Field(default_factory=list)

    @field_validator("token_id", "market_id", mode="before")
    @classmethod
    def coerce_optional_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @field_validator("bids")
    @classmethod
    def sort_bids_desc(cls, levels: list[OrderLevel]) -> list[OrderLevel]:
        return sorted(levels, key=lambda level: level.price, reverse=True)

    @field_validator("asks")
    @classmethod
    def sort_asks_asc(cls, levels: list[OrderLevel]) -> list[OrderLevel]:
        return sorted(levels, key=lambda level: level.price)


class Market(BaseModel):
    """Normalized subset of Gamma market data used by the research modules."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    question: str | None = None
    slug: str | None = None
    condition_id: str | None = Field(default=None, alias="conditionId")
    category: str | None = None
    end_date: datetime | None = Field(default=None, alias="endDate")
    active: bool = False
    closed: bool = False
    archived: bool = False
    enable_order_book: bool = Field(default=False, alias="enableOrderBook")
    liquidity: float | None = None
    volume: float | None = None
    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float | None] = Field(default_factory=list, alias="outcomePrices")
    clob_token_ids: list[str] = Field(default_factory=list, alias="clobTokenIds")
    fees_enabled: bool | None = Field(default=None, alias="feesEnabled")
    best_bid: float | None = Field(default=None, alias="bestBid")
    best_ask: float | None = Field(default=None, alias="bestAsk")
    last_trade_price: float | None = Field(default=None, alias="lastTradePrice")
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("id", mode="before")
    @classmethod
    def coerce_id(cls, value: Any) -> str:
        return str(value)

    @field_validator("active", "closed", "archived", "enable_order_book", mode="before")
    @classmethod
    def parse_bool(cls, value: Any) -> bool:
        return _bool_or_false(value)

    @field_validator("fees_enabled", mode="before")
    @classmethod
    def parse_optional_bool(cls, value: Any) -> bool | None:
        if value is None or value == "":
            return None
        return _bool_or_false(value)

    @field_validator(
        "liquidity", "volume", "best_bid", "best_ask", "last_trade_price", mode="before"
    )
    @classmethod
    def parse_float(cls, value: Any) -> float | None:
        return _float_or_none(value)

    @field_validator("outcomes", "clob_token_ids", mode="before")
    @classmethod
    def parse_list(cls, value: Any) -> list[str]:
        return [str(item) for item in _list_from_jsonish(value)]

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def parse_price_list(cls, value: Any) -> list[float | None]:
        return [_float_or_none(item) for item in _list_from_jsonish(value)]

    @model_validator(mode="before")
    @classmethod
    def capture_raw(cls, data: Any) -> Any:
        if isinstance(data, dict) and "raw" not in data:
            data = dict(data)
            data["raw"] = dict(data)
        return data

    @classmethod
    def model_validate(cls, obj: Any, *args: Any, **kwargs: Any) -> "Market":
        market = super().model_validate(obj, *args, **kwargs)
        market._validate_outcome_vector_lengths()
        return market

    def _validate_outcome_vector_lengths(self) -> None:
        if self.outcome_prices and len(self.outcome_prices) != len(self.outcomes):
            raise MarketStructureError(
                f"Market {self.id} has outcomePrices length {len(self.outcome_prices)} "
                f"but outcomes length {len(self.outcomes)}"
            )
        if self.clob_token_ids and len(self.clob_token_ids) != len(self.outcomes):
            raise MarketStructureError(
                f"Market {self.id} has clobTokenIds length {len(self.clob_token_ids)} "
                f"but outcomes length {len(self.outcomes)}"
            )

    def _outcome_index(self, label: str) -> int:
        normalized = label.strip().casefold()
        for index, outcome in enumerate(self.outcomes):
            if outcome.strip().casefold() == normalized:
                return index
        raise MarketStructureError(
            f"Market {self.id} is missing {label!r} in outcomes={self.outcomes!r}"
        )

    def _price_at(self, index: int, label: str) -> float:
        if index >= len(self.outcome_prices):
            raise MarketStructureError(
                f"Market {self.id} is missing {label} price at outcome index {index}; "
                f"outcomePrices={self.outcome_prices!r}"
            )
        price = self.outcome_prices[index]
        if price is None:
            raise MarketStructureError(
                f"Market {self.id} has null {label} price at outcome index {index}"
            )
        return price

    def _token_at(self, index: int, label: str) -> str:
        if index >= len(self.clob_token_ids):
            raise MarketStructureError(
                f"Market {self.id} is missing {label} CLOB token id at outcome index {index}; "
                f"clobTokenIds={self.clob_token_ids!r}"
            )
        token_id = self.clob_token_ids[index]
        if not token_id:
            raise MarketStructureError(
                f"Market {self.id} has empty {label} CLOB token id at outcome index {index}"
            )
        return token_id

    @property
    def yes_outcome_index(self) -> int:
        return self._outcome_index("Yes")

    @property
    def no_outcome_index(self) -> int:
        return self._outcome_index("No")

    @property
    def yes_price(self) -> float:
        return self._price_at(self.yes_outcome_index, "YES")

    @property
    def no_price(self) -> float:
        return self._price_at(self.no_outcome_index, "NO")

    @property
    def yes_token_id(self) -> str:
        return self._token_at(self.yes_outcome_index, "YES")

    @property
    def no_token_id(self) -> str:
        return self._token_at(self.no_outcome_index, "NO")


class EVResult(BaseModel):
    """Expected-value estimate for buying one side of a binary market."""

    side: Literal["YES", "NO"]
    fair_probability: float = Field(ge=0, le=1)
    price: float = Field(ge=0, le=1)
    price_source: Literal["executable_best_ask", "executable_avg_buy"]
    size: float = Field(gt=0)
    gross_edge: float
    cost: float
    estimated_fee: float = Field(ge=0)
    net_edge: float
    expected_profit: float
