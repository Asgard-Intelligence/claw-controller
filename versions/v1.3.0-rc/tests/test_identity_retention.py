"""
Identity Retention Tests for Controller v1.3.0
Tests that agent remembers its name and user's name across messages
"""

import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.core.session_manager import (
    SessionManager,
    FileSessionStore,
    IdentityManager,
)
from controller.core.safe_router import (
    SafeRoutingEngine,
    ProviderRegistry,
    RoutingPolicy,
    ModelCapabilities,
    ContextPreservationEngine,
)


async def test_identity_loaded_on_session_creation():
    """Test that identity is loaded when session is created"""
    print("\n" + "="*60)
    print("TEST 1: Identity loaded on session creation")
    print("="*60)
    
    # Setup
    store = FileSessionStore(base_path="./test_sessions")
    identity_manager = IdentityManager(identity_dir="./test_identity")
    session_manager = SessionManager(store=store, identity_manager=identity_manager)
    
    # Create test identity file
    os.makedirs("./test_identity/agent2", exist_ok=True)
    with open("./test_identity/agent2/IDENTITY.md", "w") as f:
        f.write("""# Identity for agent2

Your name is Толик.
You are a helpful AI assistant.
Always remember your name is Толик.
""")
    
    # Create session
    session = await session_manager.create_session(agent_id="agent2")
    
    # Verify identity loaded
    assert session.identity.name == "Толик", f"Expected 'Толик', got '{session.identity.name}'"
    assert "Толик" in session.identity.system_prompt, "Name should be in system prompt"
    
    print(f"✅ Session created: {session.session_id}")
    print(f"✅ Identity name: {session.identity.name}")
    print(f"✅ Identity hash: {session.identity.hash[:16]}...")
    
    return session.session_id


async def test_identity_preserved_across_messages(session_id: str):
    """Test that identity is preserved when adding messages"""
    print("\n" + "="*60)
    print("TEST 2: Identity preserved across messages")
    print("="*60)
    
    # Setup
    store = FileSessionStore(base_path="./test_sessions")
    identity_manager = IdentityManager(identity_dir="./test_identity")
    session_manager = SessionManager(store=store, identity_manager=identity_manager)
    
    # Get session
    session = await session_manager.get_session(session_id)
    assert session is not None, "Session not found"
    
    original_name = session.identity.name
    original_hash = session.identity.hash
    
    # Add messages
    await session_manager.add_message(
        session_id=session_id,
        role="user",
        content="Меня зовут Александр",
        tokens=5
    )
    
    await session_manager.add_message(
        session_id=session_id,
        role="assistant",
        content="Привет, Александр! Я Толик.",
        tokens=7
    )
    
    # Get updated session
    session = await session_manager.get_session(session_id)
    
    # Verify identity unchanged
    assert session.identity.name == original_name, "Identity name changed!"
    assert session.identity.hash == original_hash, "Identity hash changed!"
    
    print(f"✅ Identity name still: {session.identity.name}")
    print(f"✅ Identity hash unchanged: {session.identity.hash[:16]}...")
    print(f"✅ Message count: {session.message_count}")
    

async def test_context_preservation_on_routing():
    """Test that context is preserved when routing between providers"""
    print("\n" + "="*60)
    print("TEST 3: Context preservation on routing")
    print("="*60)
    
    # Setup
    store = FileSessionStore(base_path="./test_sessions")
    identity_manager = IdentityManager(identity_dir="./test_identity")
    session_manager = SessionManager(store=store, identity_manager=identity_manager)
    
    provider_registry = ProviderRegistry()
    provider_registry.register(ModelCapabilities(
        provider="openai",
        model="gpt-4",
        context_window=8192
    ))
    provider_registry.set_health("openai", "healthy")
    
    routing_policy = RoutingPolicy(registry=provider_registry)
    routing_engine = SafeRoutingEngine(
        registry=provider_registry,
        policy=routing_policy,
        default_provider="openai"
    )
    context_engine = ContextPreservationEngine(registry=provider_registry)
    
    # Create session with messages
    session = await session_manager.create_session(agent_id="agent2")
    
    # Add messages
    for i in range(5):
        await session_manager.add_message(
            session_id=session.session_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i}",
            tokens=3
        )
    
    # Prepare context for routing
    context_result = await context_engine.prepare_context_for_routing(
        session,
        target_provider="openai"
    )
    
    messages = context_result["messages"]
    
    # Verify context preserved
    assert len(messages) > 0, "No messages in context"
    assert messages[0]["role"] == "system", "First message should be system"
    assert "Толик" in messages[0]["content"], "Identity name should be in system prompt"
    assert context_result["identity_preserved"], "Identity should be preserved"
    
    print(f"✅ Context prepared: {context_result['original_count']} -> {context_result['final_count']} messages")
    print(f"✅ Identity preserved: {context_result['identity_preserved']}")
    print(f"✅ System prompt contains identity: {'Толик' in messages[0]['content']}")


async def test_routing_decision_with_pinned_provider():
    """Test that pinned provider is respected"""
    print("\n" + "="*60)
    print("TEST 4: Routing decision with pinned provider")
    print("="*60)
    
    # Setup
    store = FileSessionStore(base_path="./test_sessions")
    identity_manager = IdentityManager(identity_dir="./test_identity")
    session_manager = SessionManager(store=store, identity_manager=identity_manager)
    
    provider_registry = ProviderRegistry()
    provider_registry.register(ModelCapabilities(
        provider="openai",
        model="gpt-4",
        context_window=8192
    ))
    provider_registry.register(ModelCapabilities(
        provider="anthropic",
        model="claude-3",
        context_window=200000
    ))
    provider_registry.set_health("openai", "healthy")
    provider_registry.set_health("anthropic", "healthy")
    
    routing_policy = RoutingPolicy(registry=provider_registry)
    routing_engine = SafeRoutingEngine(
        registry=provider_registry,
        policy=routing_policy,
        default_provider="openai"
    )
    
    # Create session with pinned provider
    session = await session_manager.create_session(
        agent_id="agent2",
        pinned_provider="anthropic"
    )
    
    # Get routing decision
    class MockRequest:
        messages = [{"role": "user", "content": "Hello"}]
    
    decision = await routing_engine.select_provider(session, MockRequest())
    
    # Verify pinned provider used
    assert decision.provider == "anthropic", f"Expected 'anthropic', got '{decision.provider}'"
    assert decision.context_transfer_required == False, "Should not require context transfer for pinned"
    
    print(f"✅ Pinned provider respected: {decision.provider}")
    print(f"✅ Context transfer required: {decision.context_transfer_required}")
    print(f"✅ Routing reason: {decision.reason_str}")


async def test_context_window_check():
    """Test context window checking"""
    print("\n" + "="*60)
    print("TEST 5: Context window check")
    print("="*60)
    
    # Setup
    provider_registry = ProviderRegistry()
    provider_registry.register(ModelCapabilities(
        provider="gemma",
        model="gemma4",
        context_window=8192
    ))
    provider_registry.set_health("gemma", "healthy")
    
    routing_policy = RoutingPolicy(registry=provider_registry)
    
    # Create mock session with large context
    class MockSession:
        total_tokens = 7500  # 91.5% of 8192
    
    # Check context window
    fits, msg = routing_policy.check_context_window(MockSession(), "gemma")
    
    # Should warn but allow (above 90% threshold)
    assert fits == True, "Should fit but warn"
    assert "near limit" in msg.lower(), f"Expected warning, got: {msg}"
    
    print(f"✅ Context window check: {fits}")
    print(f"✅ Warning message: {msg}")
    
    # Test exceeding limit
    MockSession.total_tokens = 9000  # Exceeds 8192
    fits, msg = routing_policy.check_context_window(MockSession(), "gemma")
    
    assert fits == False, "Should not fit"
    assert "exceeds" in msg.lower(), f"Expected exceed message, got: {msg}"
    
    print(f"✅ Exceed check: {fits}")
    print(f"✅ Exceed message: {msg}")


async def test_identity_critical_routing_blocked():
    """Test that routing is blocked for identity-critical requests"""
    print("\n" + "="*60)
    print("TEST 6: Identity-critical routing blocked")
    print("="*60)
    
    # Setup
    provider_registry = ProviderRegistry()
    routing_policy = RoutingPolicy(registry=provider_registry)
    
    # Create mock session without pinned provider
    class MockSession:
        pinned_provider = None
    
    # Test identity-critical request
    class MockRequest:
        messages = [{"role": "user", "content": "Тебя зовут Толик"}]
    
    allowed, reason = routing_policy.should_allow_routing(MockSession(), MockRequest())
    
    # Should be blocked (identity-critical)
    assert allowed == False, "Should block identity-critical routing"
    assert "identity-critical" in reason.lower(), f"Expected identity-critical reason, got: {reason}"
    
    print(f"✅ Routing allowed: {allowed}")
    print(f"✅ Block reason: {reason}")


async def run_all_tests():
    """Run all identity retention tests"""
    print("\n" + "="*60)
    print("CONTROLLER v1.3.0 - IDENTITY RETENTION TESTS")
    print("="*60)
    
    try:
        # Run tests
        session_id = await test_identity_loaded_on_session_creation()
        await test_identity_preserved_across_messages(session_id)
        await test_context_preservation_on_routing()
        await test_routing_decision_with_pinned_provider()
        await test_context_window_check()
        await test_identity_critical_routing_blocked()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
    finally:
        # Cleanup
        import shutil
        if os.path.exists("./test_sessions"):
            shutil.rmtree("./test_sessions")
        if os.path.exists("./test_identity"):
            shutil.rmtree("./test_identity")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
