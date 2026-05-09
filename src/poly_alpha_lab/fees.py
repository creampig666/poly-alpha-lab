"""Fee estimates for read-only research calculations."""

from __future__ import annotations

from poly_alpha_lab.category_normalization import normalize_category_text

FEE_RATE_BY_CATEGORY = {
    "crypto": 0.02,
    "sports": 0.02,
    "politics": 0.02,
    "economics": 0.02,
    "finance": 0.02,
    "tech": 0.02,
    "weather": 0.02,
    "geopolitics": 0.02,
    "culture": 0.02,
    "default": 0.02,
}


def fee_rate_for_category(category: str | None) -> float:
    """Return a conservative category fee rate assumption."""

    if not category:
        return FEE_RATE_BY_CATEGORY["default"]
    normalized = normalize_category_text(category)
    return FEE_RATE_BY_CATEGORY.get(normalized, FEE_RATE_BY_CATEGORY["default"])


def is_mapped_category(category: str | None) -> bool:
    if not category:
        return False
    normalized = normalize_category_text(category)
    return normalized in FEE_RATE_BY_CATEGORY and normalized != "default"


def taker_fee_per_share(price: float, category: str | None = None, fees_enabled: bool | None = None) -> float:
    """Estimate taker fee per share: rate * price * (1 - price)."""

    if not 0 <= price <= 1:
        raise ValueError("price must be between 0 and 1")
    if fees_enabled is False:
        return 0.0
    return fee_rate_for_category(category) * price * (1 - price)


def estimate_taker_fee(
    price: float,
    size: float,
    category: str | None = None,
    fees_enabled: bool | None = None,
) -> float:
    """Estimate taker fee for binary share purchases.

    Polymarket fee schedules may change. This function keeps the fee assumption explicit
    and conservative for research: rate * price * (1 - price) * size.
    """

    if not 0 <= price <= 1:
        raise ValueError("price must be between 0 and 1")
    if size <= 0:
        raise ValueError("size must be positive")

    return taker_fee_per_share(price, category=category, fees_enabled=fees_enabled) * size


def estimate_maker_fee() -> float:
    """Polymarket maker fee assumption for this read-only research model."""

    return 0.0


def fee_assumption(fees_enabled: bool | None, category: str | None = None) -> str:
    """Human-readable fee assumption for reports."""

    if fees_enabled is False:
        return "fees_enabled_false_fee_zero"
    normalized = normalize_category_text(category)
    if normalized == "unknown":
        return "category_unknown_using_default"
    if fees_enabled is None:
        return "fees_enabled_unknown_using_category_rate"
    return "mapped_category_fee_rate"
