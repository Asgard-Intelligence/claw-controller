"""Response normalization from execution result to public API payload."""

from __future__ import annotations

import time
from typing import Any, Dict

from .provider_contracts import ExecutionResult, RoutingDecision


class ResponseNormalizer:
    def to_chat_completions_response(
        self,
        request_id: str,
        session_id: str,
        requested_model: str,
        decision: RoutingDecision,
        execution_result: ExecutionResult,
    ) -> Dict[str, Any]:
        usage = {
            "prompt_tokens": execution_result.usage.prompt_tokens,
            "completion_tokens": execution_result.usage.completion_tokens,
            "total_tokens": execution_result.usage.total_tokens,
        }

        return {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "session_id": session_id,
            "routing": {
                "provider": decision.provider_id,
                "model": decision.model_id,
                "api_dialect": decision.api_dialect.value,
                "route_key": decision.route_key,
                "reason": decision.reason_str,
                "context_transfer": decision.context_transfer_required,
                "fallback_chain": decision.fallback_chain,
            },
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": execution_result.content,
                    },
                    "finish_reason": execution_result.finish_reason,
                }
            ],
            "usage": usage,
        }
