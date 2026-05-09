"""Read-only Polymarket Gamma API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from poly_alpha_lab.config import settings
from poly_alpha_lab.models import Market


class GammaClient:
    """Small synchronous client for the public Gamma API.

    The client intentionally exposes only read methods. Gamma API market discovery is
    public and requires no authentication.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        user_agent: str | None = None,
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self.base_url = str(base_url or settings.gamma_base_url).rstrip("/")
        self.timeout = timeout or settings.http_timeout_seconds
        self.user_agent = user_agent or settings.user_agent
        self.proxy = proxy
        self.trust_env = trust_env

    def _request(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "headers": headers,
            "trust_env": self.trust_env,
        }
        if self.proxy:
            kwargs["proxy"] = self.proxy
        with httpx.Client(**kwargs) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def list_markets(
        self,
        *,
        active: bool | None = True,
        closed: bool | None = False,
        limit: int = 100,
        offset: int | None = None,
        order: str | None = None,
        ascending: bool | None = None,
        tag_id: int | None = None,
        slug: str | None = None,
    ) -> list[Market]:
        """Fetch markets from ``GET /markets`` and normalize them into ``Market`` models."""

        params: dict[str, Any] = {"limit": limit}
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if offset is not None:
            params["offset"] = offset
        if order:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()
        if tag_id is not None:
            params["tag_id"] = tag_id
        if slug:
            params["slug"] = slug

        data = self._request("/markets", params=params)
        if not isinstance(data, list):
            raise ValueError("Gamma /markets returned a non-list response")
        return [Market.model_validate(item) for item in data]

    def get_market(self, market_id: str | int) -> Market:
        """Fetch one market from ``GET /markets/{id}``."""

        data = self._request(f"/markets/{market_id}")
        if not isinstance(data, dict):
            raise ValueError("Gamma /markets/{id} returned a non-object response")
        return Market.model_validate(data)

    def get_market_by_slug(self, slug: str) -> Market:
        """Fetch the first market returned by ``GET /markets?slug=...``."""

        markets = self.list_markets(active=None, closed=None, limit=1, slug=slug)
        if not markets:
            raise LookupError(f"No market found for slug: {slug}")
        return markets[0]
