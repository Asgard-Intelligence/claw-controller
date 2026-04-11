#!/usr/bin/env python3
"""
Quick test for Controller v1.3.0
Tests basic functionality without external dependencies
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


async def quick_test():
    """Quick functionality test"""
    print("\n" + "="*60)
    print("CONTROLLER v1.3.0 - QUICK TEST")
    print("="*60)
    
    # Setup
    store = FileSessionStore(base_path="./test_sessions_quick")
    identity_manager = IdentityManager(identity_dir="./test_identity_quick")
    session_manager = SessionManager(store=store, identity_manager=identity_manager)
    
    # Create test identity
    os.makedirs("./test_identity_quick/test_agent", exist_ok=True)
    with open("./test_identity_quick/test_agent/IDENTITY.md", "w") as f:
        f.write("Your name is TestBot. Always remember your name is TestBot.")
    
    print("\n[1] Creating session...")
    session = await session_manager.create_session(agent_id="test_agent")
    print(f"    ✅ Session ID: {session.session_id}")
    print(f"    ✅ Identity: {session.identity.name}")
    
    print("\n[2] Adding messages...")
    await session_manager.add_message(
        session_id=session.session_id,
        role="user",
        content="Hello!",
        tokens=2
    )
    await session_manager.add_message(
        session_id=session.session_id,
        role="assistant",
        content="Hi! I'm TestBot.",
        tokens=5
    )
    print(f"    ✅ Messages added")
    
    print("\n[3] Retrieving session...")
    retrieved = await session_manager.get_session(session.session_id)
    print(f"    ✅ Session retrieved")
    print(f"    ✅ Message count: {retrieved.message_count}")
    print(f"    ✅ Identity still: {retrieved.identity.name}")
    
    print("\n[4] Verifying identity preservation...")
    assert retrieved.identity.name == "TestBot", "Identity changed!"
    assert retrieved.message_count == 2, "Message count wrong!"
    print("    ✅ Identity preserved correctly")
    
    print("\n" + "="*60)
    print("✅ QUICK TEST PASSED")
    print("="*60)
    print("\nController v1.3.0 is working correctly!")
    print("You can now run the full test suite with:")
    print("  python tests/test_identity_retention.py")
    
    # Cleanup
    import shutil
    if os.path.exists("./test_sessions_quick"):
        shutil.rmtree("./test_sessions_quick")
    if os.path.exists("./test_identity_quick"):
        shutil.rmtree("./test_identity_quick")


if __name__ == "__main__":
    try:
        asyncio.run(quick_test())
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
