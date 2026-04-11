import json
from datetime import datetime

import pytest

from controller.core.session_manager import FileSessionStore, IdentityManager, SessionManager


@pytest.mark.asyncio
async def test_session_roundtrip_is_lossless(tmp_path):
    identity_dir = tmp_path / "identity"
    agent_dir = identity_dir / "agent1"
    agent_dir.mkdir(parents=True)
    (agent_dir / "IDENTITY.md").write_text("Your name is AgentOne.")

    store = FileSessionStore(base_path=str(tmp_path / "sessions"))
    manager = SessionManager(store=store, identity_manager=IdentityManager(identity_dir=str(identity_dir)))

    session = await manager.create_session(agent_id="agent1", pinned_provider="openai")
    await manager.add_message(session.session_id, "user", "hello", tokens=2)
    await manager.add_message(session.session_id, "assistant", "hi", provider="openai", model="gpt-4", tokens=2)
    await manager.record_routing_event(
        session.session_id,
        from_provider=None,
        to_provider="openai",
        reason="init",
        context_preserved=True,
        token_count=4,
        from_route=None,
        to_route="openai:gpt-4:openai_chat_completions",
        api_dialect="openai_chat_completions",
    )

    loaded = await manager.get_session(session.session_id)
    assert loaded is not None
    assert loaded.identity.system_prompt
    assert loaded.message_count == 2
    assert len(loaded.messages) == 2
    assert len(loaded.routing_history) == 1
    assert loaded.metadata["agent_id"] == "agent1"


@pytest.mark.asyncio
async def test_legacy_session_migration(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    session_id = "legacy-1"

    legacy_payload = {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "identity": {
            "name": "LegacyAgent",
            "hash": "abc123",
            "version": "1.0",
        },
        "messages": [],
        "message_count": 0,
        "total_tokens": 0,
        "current_provider": "openai",
        "pinned_provider": "openai",
        "routing_history": 5,
    }

    (sessions_dir / f"{session_id}.json").write_text(json.dumps(legacy_payload))

    manager = SessionManager(
        store=FileSessionStore(base_path=str(sessions_dir)),
        identity_manager=IdentityManager(identity_dir=str(tmp_path / "identity")),
    )
    session = await manager.get_session(session_id)

    assert session is not None
    assert session.session_version == "1.4"
    assert session.identity.system_prompt
    assert session.metadata.get("legacy_routing_history_count") == 5
