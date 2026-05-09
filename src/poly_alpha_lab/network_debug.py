"""Network diagnostics for weather data providers."""

from __future__ import annotations

import json
import os
import socket
import ssl
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

NetworkErrorClass = Literal[
    "CONNECTION_REFUSED",
    "DNS_FAILED",
    "TIMEOUT",
    "TLS_FAILED",
    "HTTP_ERROR",
    "API_PARAMETER_ERROR",
    "PARSE_ERROR",
    "UNKNOWN_NETWORK_ERROR",
    "NONE",
]


class NetworkDebugReport(BaseModel):
    url: str
    host: str
    dns_status: str
    dns_addresses: list[str] = Field(default_factory=list)
    dns_error: str | None = None
    tcp_status: str
    tcp_error: str | None = None
    http_status: str
    http_status_code: int | None = None
    http_error_type: str | None = None
    http_error_message: str | None = None
    api_status: str
    api_status_code: int | None = None
    api_error_type: str | None = None
    api_error_message: str | None = None
    api_response_body_excerpt: str | None = None
    error_classification: NetworkErrorClass
    proxy_used: str | None = None
    trust_env: bool
    env_http_proxy_present: bool
    env_https_proxy_present: bool
    env_all_proxy_present: bool
    env_http_proxy: str | None = None
    env_https_proxy: str | None = None
    env_all_proxy: str | None = None
    recommendation: str


def run_network_debug(
    *,
    url: str,
    timeout_seconds: float = 30,
    proxy: str | None = None,
    trust_env: bool = True,
    print_env_proxy: bool = False,
    getaddrinfo: Callable[..., Any] = socket.getaddrinfo,
    create_connection: Callable[..., Any] = socket.create_connection,
    http_get: Callable[..., httpx.Response] | None = None,
) -> NetworkDebugReport:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    env = _proxy_env()
    dns_status = "SKIPPED"
    dns_addresses: list[str] = []
    dns_error: str | None = None
    tcp_status = "SKIPPED"
    tcp_error: str | None = None
    http_status = "SKIPPED"
    http_status_code: int | None = None
    http_error_type: str | None = None
    http_error_message: str | None = None
    api_status = "SKIPPED"
    api_status_code: int | None = None
    api_error_type: str | None = None
    api_error_message: str | None = None
    api_body_excerpt: str | None = None
    classification: NetworkErrorClass = "NONE"

    try:
        infos = getaddrinfo(host, port)
        dns_addresses = sorted({str(info[4][0]) for info in infos})
        dns_status = "SUCCESS"
    except OSError as exc:
        dns_status = "FAILED"
        dns_error = str(exc)
        classification = "DNS_FAILED"

    if dns_status == "SUCCESS":
        try:
            conn = create_connection((host, port), timeout=timeout_seconds)
            close = getattr(conn, "close", None)
            if callable(close):
                close()
            tcp_status = "SUCCESS"
        except OSError as exc:
            tcp_status = "FAILED"
            tcp_error = str(exc)
            classification = classify_network_error(exc)

    if dns_status == "SUCCESS" and (tcp_status == "SUCCESS" or proxy or trust_env):
        try:
            response = _http_get(
                url,
                timeout_seconds=timeout_seconds,
                proxy=proxy,
                trust_env=trust_env,
                http_get=http_get,
            )
            http_status_code = response.status_code
            http_status = "SUCCESS" if response.status_code < 500 else "HTTP_ERROR"
            if response.status_code >= 400:
                classification = "HTTP_ERROR"
        except Exception as exc:
            http_status = "FAILED"
            http_error_type = type(exc).__name__
            http_error_message = str(exc)
            classification = classify_network_error(exc)

    api_url = _with_minimal_open_meteo_query(url)
    if http_status == "SUCCESS":
        try:
            response = _http_get(
                api_url,
                timeout_seconds=timeout_seconds,
                proxy=proxy,
                trust_env=trust_env,
                http_get=http_get,
            )
            api_status_code = response.status_code
            api_body_excerpt = response.text[:1000]
            if response.status_code >= 400:
                api_status = "HTTP_ERROR"
                classification = "API_PARAMETER_ERROR" if response.status_code == 400 else "HTTP_ERROR"
            else:
                api_status = "SUCCESS"
        except Exception as exc:
            api_status = "FAILED"
            api_error_type = type(exc).__name__
            api_error_message = str(exc)
            classification = classify_network_error(exc)

    recommendation = _recommendation(
        classification=classification,
        env_proxy_present=env["HTTP_PROXY"] or env["HTTPS_PROXY"] or env["ALL_PROXY"],
        proxy=proxy,
        trust_env=trust_env,
    )
    return NetworkDebugReport(
        url=url,
        host=host,
        dns_status=dns_status,
        dns_addresses=dns_addresses,
        dns_error=dns_error,
        tcp_status=tcp_status,
        tcp_error=tcp_error,
        http_status=http_status,
        http_status_code=http_status_code,
        http_error_type=http_error_type,
        http_error_message=http_error_message,
        api_status=api_status,
        api_status_code=api_status_code,
        api_error_type=api_error_type,
        api_error_message=api_error_message,
        api_response_body_excerpt=api_body_excerpt,
        error_classification=classification,
        proxy_used=mask_proxy_url(proxy) if proxy else None,
        trust_env=trust_env,
        env_http_proxy_present=env["HTTP_PROXY"] is not None,
        env_https_proxy_present=env["HTTPS_PROXY"] is not None,
        env_all_proxy_present=env["ALL_PROXY"] is not None,
        env_http_proxy=mask_proxy_url(env["HTTP_PROXY"]) if print_env_proxy else None,
        env_https_proxy=mask_proxy_url(env["HTTPS_PROXY"]) if print_env_proxy else None,
        env_all_proxy=mask_proxy_url(env["ALL_PROXY"]) if print_env_proxy else None,
        recommendation=recommendation,
    )


def write_network_debug_report(report: NetworkDebugReport, output_path: str | Path) -> None:
    path = Path(output_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def network_debug_report_to_markdown(report: NetworkDebugReport) -> str:
    return "\n".join(
        [
            "# Network Debug Report",
            "",
            f"- url: `{report.url}`",
            f"- host: `{report.host}`",
            f"- dns_status: `{report.dns_status}`",
            f"- dns_addresses: `{', '.join(report.dns_addresses) if report.dns_addresses else 'n/a'}`",
            f"- tcp_status: `{report.tcp_status}`",
            f"- http_status: `{report.http_status}`",
            f"- http_status_code: `{report.http_status_code if report.http_status_code is not None else 'n/a'}`",
            f"- api_status: `{report.api_status}`",
            f"- api_status_code: `{report.api_status_code if report.api_status_code is not None else 'n/a'}`",
            f"- error_classification: `{report.error_classification}`",
            f"- trust_env: `{report.trust_env}`",
            f"- proxy_used: `{report.proxy_used or 'n/a'}`",
            f"- env HTTP_PROXY present: `{report.env_http_proxy_present}`",
            f"- env HTTPS_PROXY present: `{report.env_https_proxy_present}`",
            f"- env ALL_PROXY present: `{report.env_all_proxy_present}`",
            f"- recommendation: `{report.recommendation}`",
        ]
    )


def classify_network_error(exc: BaseException) -> NetworkErrorClass:
    message = str(exc).casefold()
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timed out" in message:
        return "TIMEOUT"
    if isinstance(exc, ssl.SSLError) or "ssl" in message or "tls" in message:
        return "TLS_FAILED"
    if isinstance(exc, socket.gaierror) or "getaddrinfo" in message or "name or service" in message:
        return "DNS_FAILED"
    if "10061" in message or "connection refused" in message or "actively refused" in message:
        return "CONNECTION_REFUSED"
    if isinstance(exc, httpx.HTTPStatusError):
        return "HTTP_ERROR"
    if isinstance(exc, httpx.ConnectError):
        return "UNKNOWN_NETWORK_ERROR"
    if isinstance(exc, ValueError):
        return "PARSE_ERROR"
    return "UNKNOWN_NETWORK_ERROR"


def mask_proxy_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlparse(value)
    if parsed.username is None and parsed.password is None:
        return value
    host = parsed.hostname or ""
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    masked_user = parsed.username or ""
    auth = f"{masked_user}:***@" if masked_user else "***@"
    return urlunparse(parsed._replace(netloc=f"{auth}{netloc}"))


def _http_get(
    url: str,
    *,
    timeout_seconds: float,
    proxy: str | None,
    trust_env: bool,
    http_get: Callable[..., httpx.Response] | None,
) -> httpx.Response:
    kwargs: dict[str, Any] = {
        "timeout": timeout_seconds,
        "follow_redirects": True,
        "trust_env": trust_env,
    }
    if proxy:
        kwargs["proxy"] = proxy
    if http_get is not None:
        return http_get(url, **kwargs)
    return httpx.get(url, **kwargs)


def _with_minimal_open_meteo_query(url: str) -> str:
    parsed = urlparse(url)
    params = dict(parse_qsl(parsed.query))
    params.update(
        {
            "latitude": "45.4642",
            "longitude": "9.19",
            "daily": "temperature_2m_max",
            "start_date": "2026-04-01",
            "end_date": "2026-04-01",
            "timezone": "UTC",
        }
    )
    return urlunparse(parsed._replace(query=urlencode(params)))


def _proxy_env() -> dict[str, str | None]:
    return {
        "HTTP_PROXY": os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
        "ALL_PROXY": os.environ.get("ALL_PROXY") or os.environ.get("all_proxy"),
    }


def _recommendation(
    *,
    classification: NetworkErrorClass,
    env_proxy_present: str | None,
    proxy: str | None,
    trust_env: bool,
) -> str:
    if classification == "NONE":
        return "Network path is reachable; retry Route A."
    if classification == "CONNECTION_REFUSED":
        if env_proxy_present and trust_env:
            return "Environment proxy is present but connection is refused; verify proxy availability or pass --proxy explicitly."
        if proxy:
            return "Explicit proxy was used but connection is refused; verify proxy host/port and credentials."
        return "TCP was refused; configure a working proxy or try another network before retrying Route A."
    if classification == "DNS_FAILED":
        return "DNS resolution failed; check DNS/proxy settings."
    if classification == "TIMEOUT":
        return "Connection timed out; try a proxy, longer timeout, or another network."
    if classification == "API_PARAMETER_ERROR":
        return "Network works but API parameters are rejected; inspect response body and provider params."
    return "Network/provider path failed; inspect report details before retrying Route A."
