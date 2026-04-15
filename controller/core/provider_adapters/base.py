"""Base contract for provider adapters."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ..provider_contracts import (
    AdapterError,
    ApiDialect,
    CanonicalMessage,
    CanonicalRequest,
    RouteTarget,
    ValidationKind,
)


class BaseProviderAdapter(ABC):
    supported_dialects: List[ApiDialect] = []

    def can_handle(self, dialect: ApiDialect) -> bool:
        return dialect in self.supported_dialects

    @abstractmethod
    async def execute(
        self,
        request: CanonicalRequest,
        route: RouteTarget,
        messages: List[CanonicalMessage],
        timeout_seconds: float,
    ):
        raise NotImplementedError

    def build_auth_headers(self, credential_env: Optional[str], anthropic: bool = False) -> Dict[str, str]:
        if not credential_env:
            return {}

        token = os.getenv(credential_env)
        if not token:
            return {}

        if anthropic:
            return {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }


def classify_http_error(status_code: int, message: str) -> AdapterError:
    lower_msg = (message or "").lower()
    if status_code in {401, 403}:
        return AdapterError(message="Unauthorized", kind=ValidationKind.CREDENTIAL, http_status=status_code)
    if status_code in {404, 405}:
        return AdapterError(message="Endpoint unavailable", kind=ValidationKind.ENDPOINT, http_status=status_code)
    if "model" in lower_msg and ("not found" in lower_msg or "invalid" in lower_msg):
        return AdapterError(message="Invalid model", kind=ValidationKind.MODEL, http_status=status_code)
    return AdapterError(message=message or f"HTTP {status_code}", kind=ValidationKind.UNKNOWN, http_status=status_code)
