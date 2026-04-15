"""Token estimation service with safety margin."""

from __future__ import annotations

from typing import Iterable

from .provider_contracts import CanonicalMessage


class TokenEstimator:
    def __init__(self, chars_per_token: int = 4, safety_margin: float = 1.15):
        self.chars_per_token = max(1, chars_per_token)
        self.safety_margin = max(1.0, safety_margin)

    def estimate_message_tokens(self, content: str) -> int:
        return int((len(content) / self.chars_per_token) * self.safety_margin) + 1

    def estimate_messages(self, messages: Iterable[CanonicalMessage]) -> int:
        total = 0
        for msg in messages:
            total += self.estimate_message_tokens(msg.content)
        return total
