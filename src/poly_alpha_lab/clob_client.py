"""Read-only Polymarket CLOB API client."""

from __future__ import annotations

from typing import Any

import httpx

from poly_alpha_lab.config import settings
from poly_alpha_lab.models import OrderBook


class ClobClient:
    """Small synchronous client for public CLOB order-book reads only."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        user_agent: str | None = None,
        proxy: str | None = None,
        trust_env: bool = True,
    ) -> None:
        self.base_url = str(base_url or settings.clob_base_url).rstrip("/")
        self.timeout = timeout or settings.http_timeout_seconds
        self.user_agent = user_agent or settings.user_agent
        self.proxy = proxy
        self.trust_env = trust_env

    def _request(self, path: str, params: dict[str, Any] | None = None) -> Any:
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

    def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch ``GET /book?token_id=...`` for a YES or NO CLOB token."""

        if not token_id:
            raise ValueError("token_id is required")
        data = self._request("/book", params={"token_id": token_id})
        if not isinstance(data, dict):
            raise ValueError("CLOB /book returned a non-object response")
        book = OrderBook.model_validate(data)
        if book.token_id is None:
            book.token_id = token_id
        return book
