"""
Integration tests for Controller 1.0.0 RC.

Run with: python -m pytest tests/test_integration.py -v
"""

import pytest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["CONTROLLER_API_KEY"] = "test-api-key-for-integration-tests-12345"
os.environ["LOG_LEVEL"] = "ERROR"  # Reduce noise during tests

from controller.main import app
from controller.core.config import get_settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {"Authorization": "Bearer test-api-key-for-integration-tests-12345"}


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_unauthenticated(self, client):
        """Health endpoint should not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self, client):
        """Root endpoint should return basic info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Controller"
        assert "version" in data


class TestModelsEndpoint:
    """Test models listing endpoint."""
    
    def test_models_requires_auth(self, client):
        """Models endpoint should require authentication."""
        response = client.get("/v1/models")
        assert response.status_code == 403  # FastAPI HTTPBearer auto_error=False returns 403
    
    def test_models_with_auth(self, client, auth_headers):
        """Models endpoint should return list with auth."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        
        # Should include controller model
        model_ids = [m["id"] for m in data["data"]]
        assert "controller" in model_ids


class TestChatCompletions:
    """Test chat completions endpoint."""
    
    def test_chat_requires_auth(self, client):
        """Chat endpoint should require authentication."""
        response = client.post("/v1/chat/completions", json={
            "model": "controller",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 403
    
    def test_chat_empty_messages(self, client, auth_headers):
        """Chat endpoint should reject empty messages."""
        response = client.post("/v1/chat/completions", 
            headers=auth_headers,
            json={"model": "controller", "messages": []}
        )
        assert response.status_code == 422  # Validation error
    
    def test_chat_mock_mode(self, client, auth_headers):
        """Chat endpoint should work in mock mode without providers."""
        response = client.post("/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        # Should work even without providers (mock mode)
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "controller_metadata" in data


class TestControllerStatus:
    """Test controller status endpoint."""
    
    def test_status_requires_auth(self, client):
        """Status endpoint should require authentication."""
        response = client.get("/v1/controller/status")
        assert response.status_code == 403
    
    def test_status_with_auth(self, client, auth_headers):
        """Status endpoint should return detailed status."""
        response = client.get("/v1/controller/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "providers" in data
        assert "total_requests" in data
        assert "config" in data


class TestExplainEndpoint:
    """Test explain endpoint."""
    
    def test_explain_requires_auth(self, client):
        """Explain endpoint should require authentication."""
        response = client.post("/v1/controller/explain", json={
            "model": "controller",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        assert response.status_code == 403
    
    def test_explain_with_auth(self, client, auth_headers):
        """Explain endpoint should return routing explanation."""
        response = client.post("/v1/controller/explain",
            headers=auth_headers,
            json={
                "model": "controller",
                "messages": [{"role": "user", "content": "Write a Python function"}]
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "route_decision" in data
        assert "confidence_breakdown" in data
        assert "safety_analysis" in data
        assert "intent_classification" in data
        
        # Check route decision structure
        route = data["route_decision"]
        assert "selected_provider" in route
        assert "confidence" in route
        assert "safety" in route
        assert "intent" in route


class TestIntelligenceComponents:
    """Test intelligence components directly."""
    
    def test_intent_classifier(self):
        """Test intent classification."""
        from controller.core.intelligence import IntentClassifier
        from controller.models.schemas import ChatCompletionRequest, Message, Role
        
        classifier = IntentClassifier()
        
        # Code intent
        request = ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="Write a Python function to sort a list")]
        )
        intent = classifier.classify(request)
        assert intent.value in ["code", "technical", "unknown"]
        
        # Creative intent
        request = ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="Write a story about a dragon")]
        )
        intent = classifier.classify(request)
        assert intent.value in ["creative", "unknown"]
    
    def test_safety_checker(self):
        """Test safety checking."""
        from controller.core.intelligence import SafetyChecker
        from controller.models.schemas import ChatCompletionRequest, Message, Role
        
        checker = SafetyChecker()
        
        # Safe request
        request = ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="Hello, how are you?")]
        )
        safety = checker.check(request)
        assert safety.overall_risk.value in ["low", "medium", "high", "critical"]
        assert safety.risk_score >= 0.0 and safety.risk_score <= 1.0
        
        # Request with PII
        request = ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="My email is test@example.com")]
        )
        safety = checker.check(request)
        assert len(safety.pii_detected) > 0
        assert safety.pii_detected[0].type == "email"
    
    def test_confidence_calculator(self):
        """Test confidence calculation."""
        from controller.core.intelligence import ConfidenceCalculator
        from controller.models.schemas import (
            ChatCompletionRequest, Message, Role, SafetyScore, RiskLevel
        )
        
        calc = ConfidenceCalculator()
        
        request = ChatCompletionRequest(
            messages=[Message(role=Role.USER, content="Hello")]
        )
        safety = SafetyScore(
            overall_risk=RiskLevel.LOW,
            risk_score=0.1,
            pii_detected=[],
            jailbreak_attempts=[],
            content_violations=[],
            checks=[],
            recommendations=[]
        )
        
        from controller.models.schemas import IntentType, ProviderHealth
        confidence = calc.calculate(
            request=request,
            provider_health={},
            safety_score=safety,
            intent=IntentType.CONVERSATIONAL
        )
        
        assert confidence.overall >= 0.0 and confidence.overall <= 1.0
        assert len(confidence.factors) > 0
        assert confidence.tier_recommendation in ["premium", "standard", "basic"]


class TestConfiguration:
    """Test configuration loading."""
    
    def test_settings_loading(self):
        """Test that settings load correctly."""
        settings = get_settings()
        
        assert settings.CONTROLLER_API_KEY == "test-api-key-for-integration-tests-12345"
        assert settings.LOG_LEVEL == "ERROR"
        assert settings.CONTROLLER_PORT == 8080
    
    def test_provider_config(self):
        """Test provider configuration retrieval."""
        settings = get_settings()
        
        openai_config = settings.get_provider_config("openai")
        assert "base_url" in openai_config
        assert "default_model" in openai_config
        
        anthropic_config = settings.get_provider_config("anthropic")
        assert "base_url" in anthropic_config
        assert "default_model" in anthropic_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
