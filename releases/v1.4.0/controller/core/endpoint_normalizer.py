"""Endpoint normalization utilities for route registry and probes."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


_STRIPPABLE_SUFFIXES = (
    "/responses",
    "/chat/completions",
    "/completions",
    "/models",
    "/v1/messages",
    "/v1/models",
    "/v1",
)


def normalize_provider_base_url(url: str) -> str:
    """Normalize endpoint URL to stable base form without API-path tail."""
    if not url:
        return ""

    parsed = urlparse(url.strip())
    path = parsed.path or ""

    # Trim trailing slash first for stable suffix checks.
    path = path.rstrip("/")

    changed = True
    while changed and path:
        changed = False
        for suffix in _STRIPPABLE_SUFFIXES:
            if path.endswith(suffix):
                path = path[: -len(suffix)].rstrip("/")
                changed = True

    # Preserve explicit version prefix when endpoint is exactly /v1
    if path == "":
        path = ""

    normalized = parsed._replace(path=path or "", query="", fragment="")
    result = urlunparse(normalized).rstrip("/")
    return result


def build_url(base_url: str, endpoint: str) -> str:
    base = normalize_provider_base_url(base_url)
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base}{endpoint}"
