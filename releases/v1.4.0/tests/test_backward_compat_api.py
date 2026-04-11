from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from controller.api import routes_v130
from controller.core.provider_contracts import (
    ApiDialect,
    ExecutionResult,
    ExecutionUsage,
    RoutingDecision,
    TransferMode,
    TransferPolicy,
)
from controller.core.request_normalizer import RequestNormalizer
from controller.core.response_normalizer import ResponseNormalizer
from controller.core.session_manager import FileSessionStore, IdentityManager, SessionManager
from controller.core.token_estimator import TokenEstimator


class FakeHybridRouter:
    async def select_route(self, session, requirements):
        return RoutingDecision(
            provider_id="openai",
            model_id="gpt-4",
            api_dialect=ApiDialect.OPENAI_CHAT_COMPLETIONS,
            route_key="openai:gpt-4:openai_chat_completions",
            reason_code="test",
            reason_str="test route",
            context_transfer_required=session.current_route is not None,
            transfer_policy=TransferPolicy(mode=TransferMode.REPLAY_FULL),
            estimated_tokens=requirements.estimated_prompt_tokens,
            alternatives=[],
            fallback_chain=[],
            metadata={},
        )

    def build_route_chain(self, decision):
        return [decision.route_key]


class FakeExecutionEngine:
    async def execute(self, canonical_request, decision, route_chain):
        return (
            ExecutionResult(
                content="compat-ok",
                provider_id=decision.provider_id,
                model_id=decision.model_id,
                api_dialect=decision.api_dialect,
                usage=ExecutionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                finish_reason="stop",
            ),
            SimpleNamespace(used_route_key=decision.route_key, attempts=[]),
        )


def _build_client(tmp_path):
    routes_v130.session_manager = SessionManager(
        store=FileSessionStore(base_path=str(tmp_path / "sessions")),
        identity_manager=IdentityManager(identity_dir=str(tmp_path / "identity")),
    )
    routes_v130.hybrid_router = FakeHybridRouter()
    routes_v130.execution_engine = FakeExecutionEngine()
    routes_v130.request_normalizer = RequestNormalizer(TokenEstimator())
    routes_v130.response_normalizer = ResponseNormalizer()
    routes_v130.settings_obj = SimpleNamespace(ROUTING_MODE="balanced")

    app = FastAPI()
    app.include_router(routes_v130.router)
    return TestClient(app)


def test_chat_completions_keeps_legacy_headers_contract(tmp_path):
    identity_dir = tmp_path / "identity" / "agent2"
    identity_dir.mkdir(parents=True)
    (identity_dir / "IDENTITY.md").write_text("Your name is CompatAgent.")

    client = _build_client(tmp_path)

    response = client.post(
        "/v1/chat/completions",
        headers={"x-agent-id": "agent2", "x-pin-provider": "openai"},
        json={"model": "controller", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"]
    assert payload["routing"]["provider"] == "openai"
    assert payload["choices"][0]["message"]["content"] == "compat-ok"

    session_id = payload["session_id"]

    response2 = client.post(
        "/v1/chat/completions",
        headers={"x-session-id": session_id},
        json={"model": "controller", "messages": [{"role": "user", "content": "again"}]},
    )
    assert response2.status_code == 200
    payload2 = response2.json()
    assert payload2["session_id"] == session_id
