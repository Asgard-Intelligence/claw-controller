"""
Intelligence Mode - Core decision-making engine.

Implements:
- Confidence calculation (multi-factor)
- Safety checking (PII, jailbreak detection)
- Intent classification
- Route selection
- Fallback management
"""

import re
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ..models.schemas import (
    ConfidenceScore, ConfidenceFactor, SafetyScore, SafetyCheck,
    PIIFinding, RouteDecision, IntentType, RiskLevel,
    ChatCompletionRequest, ProviderHealth
)

logger = logging.getLogger(__name__)


# PII detection patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "api_key": re.compile(r'\b(?:api[_-]?key|apikey)[\s]*[:=][\s]*["\']?[a-zA-Z0-9]{16,}["\']?\b', re.IGNORECASE),
}

# Jailbreak detection patterns
JAILBREAK_PATTERNS = [
    re.compile(r'ignore\s+(?:previous|prior|above)\s+instructions?', re.IGNORECASE),
    re.compile(r'disregard\s+(?:your\s+)?(?:programming|training|guidelines?)', re.IGNORECASE),
    re.compile(r'you\s+are\s+now\s+(?:in\s+)?(?:DAN|developer|admin)\s+mode', re.IGNORECASE),
    re.compile(r'(?:pretend|act\s+as\s+if)\s+you\s+(?:are|have\s+no)', re.IGNORECASE),
    re.compile(r'(?:bypass|circumvent|override)\s+(?:safety|restriction|limitation)', re.IGNORECASE),
    re.compile(r'(?:system\s+prompt|instruction\s+override)', re.IGNORECASE),
]

# Intent classification keywords
INTENT_KEYWORDS = {
    IntentType.CODE: ["code", "function", "script", "program", "debug", "error", "compile", "import", "class", "def "],
    IntentType.CREATIVE: ["write", "story", "poem", "creative", "imagine", "fiction", "character", "plot"],
    IntentType.ANALYTICAL: ["analyze", "compare", "evaluate", "assess", "review", "examine", "study"],
    IntentType.FACTUAL: ["what is", "who is", "when", "where", "how many", "define", "explain"],
    IntentType.TECHNICAL: ["api", "database", "server", "config", "deploy", "architecture", "framework"],
}


class CircuitBreaker:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record a successful call."""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time and (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        return True  # half-open


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for key."""
        now = time.time()
        window_start = now - 60.0
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [t for t in self.requests[key] if t > window_start]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= self.requests_per_minute:
            return False
        
        self.requests[key].append(now)
        return True


class ConfidenceCalculator:
    """Multi-factor confidence calculator."""
    
    def __init__(self):
        self.request_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def calculate(
        self,
        request: ChatCompletionRequest,
        provider_health: Dict[str, ProviderHealth],
        safety_score: SafetyScore,
        intent: IntentType
    ) -> ConfidenceScore:
        """Calculate multi-factor confidence score."""
        
        factors = []
        
        # Factor 1: Request complexity
        complexity_score = self._assess_complexity(request)
        factors.append(ConfidenceFactor(
            name="complexity",
            score=complexity_score,
            weight=0.2,
            description="Request complexity assessment"
        ))
        
        # Factor 2: Provider health
        health_score = self._assess_provider_health(provider_health)
        factors.append(ConfidenceFactor(
            name="provider_health",
            score=health_score,
            weight=0.25,
            description="Provider health status"
        ))
        
        # Factor 3: Safety score
        safety_confidence = 1.0 - safety_score.risk_score
        factors.append(ConfidenceFactor(
            name="safety",
            score=safety_confidence,
            weight=0.25,
            description="Safety/privacy confidence"
        ))
        
        # Factor 4: Historical success
        history_score = self._assess_historical_success(request)
        factors.append(ConfidenceFactor(
            name="historical_success",
            score=history_score,
            weight=0.15,
            description="Historical request success rate"
        ))
        
        # Factor 5: Intent clarity
        intent_score = 0.9 if intent != IntentType.UNKNOWN else 0.5
        factors.append(ConfidenceFactor(
            name="intent_clarity",
            score=intent_score,
            weight=0.15,
            description="Intent classification confidence"
        ))
        
        # Calculate weighted overall score
        overall = sum(f.score * f.weight for f in factors)
        
        # Determine tier recommendation
        if overall >= 0.8:
            tier = "premium"
        elif overall >= 0.5:
            tier = "standard"
        else:
            tier = "basic"
        
        reasoning = f"Overall confidence {overall:.2f} based on {len(factors)} factors. Recommended tier: {tier}."
        if safety_score.overall_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            reasoning += " Elevated safety risk detected."
        
        return ConfidenceScore(
            overall=round(overall, 3),
            factors=factors,
            tier_recommendation=tier,
            reasoning=reasoning
        )
    
    def _assess_complexity(self, request: ChatCompletionRequest) -> float:
        """Assess request complexity (0-1, higher is more complex)."""
        total_content = " ".join(m.content for m in request.messages)
        
        # Length factor
        length_score = min(len(total_content) / 2000, 1.0)
        
        # Message count factor
        message_score = min(len(request.messages) / 10, 1.0)
        
        # Code indicators
        code_indicators = sum(1 for indicator in ["```", "def ", "class ", "import ", "function"] 
                             if indicator in total_content)
        code_score = min(code_indicators / 3, 1.0)
        
        # Combine factors
        complexity = (length_score * 0.4 + message_score * 0.3 + code_score * 0.3)
        
        # Invert: simple requests = higher confidence
        return 1.0 - complexity
    
    def _assess_provider_health(self, provider_health: Dict[str, ProviderHealth]) -> float:
        """Assess overall provider health."""
        if not provider_health:
            return 0.5
        
        scores = []
        for health in provider_health.values():
            if health.status == "healthy":
                scores.append(1.0)
            elif health.status == "degraded":
                scores.append(0.6)
            elif health.status == "unhealthy":
                scores.append(0.2)
            else:
                scores.append(0.5)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _assess_historical_success(self, request: ChatCompletionRequest) -> float:
        """Assess historical success rate for similar requests."""
        # Simple hash of request pattern
        content_preview = " ".join(m.content[:50] for m in request.messages[:2])
        request_hash = hashlib.md5(content_preview.encode()).hexdigest()[:8]
        
        history = self.request_history.get(request_hash, [])
        if not history:
            return 0.7  # Default for unknown patterns
        
        successes = sum(1 for h in history if h.get("success", False))
        return successes / len(history)
    
    def record_result(self, request: ChatCompletionRequest, success: bool):
        """Record request result for history."""
        content_preview = " ".join(m.content[:50] for m in request.messages[:2])
        request_hash = hashlib.md5(content_preview.encode()).hexdigest()[:8]
        
        if request_hash not in self.request_history:
            self.request_history[request_hash] = []
        
        self.request_history[request_hash].append({
            "timestamp": time.time(),
            "success": success
        })
        
        # Keep only last 100 entries per pattern
        self.request_history[request_hash] = self.request_history[request_hash][-100:]


class SafetyChecker:
    """Safety and privacy checker."""
    
    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_jailbreak_detection: bool = True,
        risk_threshold: float = 0.7
    ):
        self.enable_pii_detection = enable_pii_detection
        self.enable_jailbreak_detection = enable_jailbreak_detection
        self.risk_threshold = risk_threshold
    
    def check(self, request: ChatCompletionRequest) -> SafetyScore:
        """Perform comprehensive safety check."""
        
        all_content = " ".join(m.content for m in request.messages)
        
        # PII detection
        pii_findings = []
        if self.enable_pii_detection:
            pii_findings = self._detect_pii(all_content)
        
        # Jailbreak detection
        jailbreak_attempts = []
        if self.enable_jailbreak_detection:
            jailbreak_attempts = self._detect_jailbreak(all_content)
        
        # Content violations
        violations = self._check_content_violations(all_content)
        
        # Individual checks
        checks = [
            SafetyCheck(
                check_name="pii_scan",
                passed=len(pii_findings) == 0,
                score=1.0 if len(pii_findings) == 0 else 0.3,
                details=f"Found {len(pii_findings)} PII instances" if pii_findings else None
            ),
            SafetyCheck(
                check_name="jailbreak_scan",
                passed=len(jailbreak_attempts) == 0,
                score=1.0 if len(jailbreak_attempts) == 0 else 0.1,
                details=f"Found {len(jailbreak_attempts)} jailbreak patterns" if jailbreak_attempts else None
            ),
            SafetyCheck(
                check_name="content_policy",
                passed=len(violations) == 0,
                score=1.0 if len(violations) == 0 else 0.2,
                details=f"Found {len(violations)} violations" if violations else None
            ),
        ]
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(pii_findings, jailbreak_attempts, violations, checks)
        
        # Determine risk level
        if risk_score >= 0.9:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= self.risk_threshold:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate recommendations
        recommendations = self._generate_recommendations(pii_findings, jailbreak_attempts, violations)
        
        # Mask content if PII found
        masked_content = None
        if pii_findings:
            masked_content = self._mask_pii(all_content, pii_findings)
        
        return SafetyScore(
            overall_risk=risk_level,
            risk_score=round(risk_score, 3),
            pii_detected=pii_findings,
            jailbreak_attempts=jailbreak_attempts,
            content_violations=violations,
            checks=checks,
            masked_content=masked_content,
            recommendations=recommendations
        )
    
    def _detect_pii(self, content: str) -> List[PIIFinding]:
        """Detect PII in content."""
        findings = []
        
        for pii_type, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(content):
                findings.append(PIIFinding(
                    type=pii_type,
                    position=match.start(),
                    length=len(match.group()),
                    confidence=0.9,
                    masked="[REDACTED]"
                ))
        
        return findings
    
    def _detect_jailbreak(self, content: str) -> List[str]:
        """Detect jailbreak attempts."""
        attempts = []
        
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(content):
                attempts.append(pattern.pattern[:50] + "...")
        
        return attempts
    
    def _check_content_violations(self, content: str) -> List[str]:
        """Check for content policy violations."""
        violations = []
        
        # Simple checks (in production, use more sophisticated methods)
        violent_terms = ["kill", "murder", "attack", "bomb", "weapon"]
        for term in violent_terms:
            if re.search(rf'\b{term}\b', content, re.IGNORECASE):
                violations.append(f"potentially_violent_content: {term}")
        
        return violations[:3]  # Limit to top 3
    
    def _calculate_risk_score(
        self,
        pii: List[PIIFinding],
        jailbreaks: List[str],
        violations: List[str],
        checks: List[SafetyCheck]
    ) -> float:
        """Calculate overall risk score."""
        score = 0.0
        
        # PII contributes up to 0.3
        score += min(len(pii) * 0.1, 0.3)
        
        # Jailbreak contributes up to 0.5
        score += min(len(jailbreaks) * 0.25, 0.5)
        
        # Violations contribute up to 0.2
        score += min(len(violations) * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _generate_recommendations(
        self,
        pii: List[PIIFinding],
        jailbreaks: List[str],
        violations: List[str]
    ) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if pii:
            recommendations.append("PII detected: Consider masking sensitive information")
        if jailbreaks:
            recommendations.append("Potential jailbreak attempt: Review request carefully")
        if violations:
            recommendations.append("Content policy concerns: Verify request compliance")
        
        if not recommendations:
            recommendations.append("No safety concerns detected")
        
        return recommendations
    
    def _mask_pii(self, content: str, findings: List[PIIFinding]) -> str:
        """Mask PII in content."""
        masked = content
        # Sort by position in reverse to avoid offset issues
        for finding in sorted(findings, key=lambda f: f.position, reverse=True):
            start = finding.position
            end = finding.position + finding.length
            masked = masked[:start] + "[REDACTED]" + masked[end:]
        return masked


class IntentClassifier:
    """Simple intent classifier."""
    
    def classify(self, request: ChatCompletionRequest) -> IntentType:
        """Classify user intent from request."""
        all_content = " ".join(m.content.lower() for m in request.messages)
        
        scores = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in all_content)
            scores[intent] = score
        
        # Return intent with highest score
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                return best_intent
        
        return IntentType.UNKNOWN


class RouteSelector:
    """Provider and tier selector based on confidence and safety."""
    
    PROVIDER_TIERS = {
        "premium": ["openai", "anthropic"],
        "standard": ["openai", "anthropic"],
        "basic": ["openai"],
    }
    
    MODELS_BY_TIER = {
        "premium": {"openai": "gpt-4o", "anthropic": "claude-3-opus-20240229"},
        "standard": {"openai": "gpt-4o-mini", "anthropic": "claude-3-sonnet-20240229"},
        "basic": {"openai": "gpt-4o-mini", "anthropic": "claude-3-haiku-20240307"},
    }
    
    def select(
        self,
        request: ChatCompletionRequest,
        confidence: ConfidenceScore,
        safety: SafetyScore,
        intent: IntentType,
        provider_health: Dict[str, ProviderHealth],
        available_providers: List[str]
    ) -> RouteDecision:
        """Select optimal provider and tier."""
        
        # Determine tier from confidence
        tier = confidence.tier_recommendation
        
        # Adjust for safety concerns
        if safety.overall_risk == RiskLevel.CRITICAL:
            # Block request
            return RouteDecision(
                selected_provider="blocked",
                selected_model="none",
                selected_tier="blocked",
                confidence=confidence,
                safety=safety,
                intent=intent,
                reasoning="Request blocked due to critical safety risk",
                fallback_chain=[]
            )
        
        # Force provider if specified
        if request.force_provider:
            provider = request.force_provider
            model = self.MODELS_BY_TIER.get(tier, {}).get(provider, "gpt-4o-mini")
        else:
            # Select best available provider
            tier_providers = self.PROVIDER_TIERS.get(tier, ["openai"])
            provider = None
            
            for p in tier_providers:
                if p in available_providers:
                    health = provider_health.get(p)
                    if health and health.status == "healthy":
                        provider = p
                        break
                    elif not health or health.status != "unhealthy":
                        provider = p
            
            if not provider:
                provider = available_providers[0] if available_providers else "openai"
            
            model = self.MODELS_BY_TIER.get(tier, {}).get(provider, "gpt-4o-mini")
        
        # Build fallback chain
        fallback_chain = [p for p in available_providers if p != provider]
        
        # Estimate cost and latency
        estimated_cost = self._estimate_cost(tier, request)
        estimated_latency = self._estimate_latency(tier, provider_health.get(provider))
        
        reasoning = f"Selected {provider}/{model} based on {tier} tier recommendation. "
        reasoning += f"Confidence: {confidence.overall:.2f}. Intent: {intent.value}."
        
        if safety.overall_risk != RiskLevel.LOW:
            reasoning += f" Safety risk: {safety.overall_risk.value}."
        
        return RouteDecision(
            selected_provider=provider,
            selected_model=model,
            selected_tier=tier,
            confidence=confidence,
            safety=safety,
            intent=intent,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
            fallback_chain=fallback_chain
        )
    
    def _estimate_cost(self, tier: str, request: ChatCompletionRequest) -> float:
        """Estimate request cost."""
        total_chars = sum(len(m.content) for m in request.messages)
        
        # Rough estimation
        tier_multipliers = {"premium": 0.03, "standard": 0.01, "basic": 0.005}
        multiplier = tier_multipliers.get(tier, 0.01)
        
        return round((total_chars / 1000) * multiplier, 4)
    
    def _estimate_latency(self, tier: str, health: Optional[ProviderHealth]) -> int:
        """Estimate request latency in ms."""
        base_latencies = {"premium": 2000, "standard": 1000, "basic": 500}
        base = base_latencies.get(tier, 1000)
        
        if health and health.latency_ms:
            return int((base + health.latency_ms) / 2)
        
        return base


class FallbackManager:
    """Manages fallback chain execution."""
    
    def __init__(self, providers: Dict[str, Any]):
        self.providers = providers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            name: CircuitBreaker() for name in providers.keys()
        }
    
    async def execute_with_fallback(
        self,
        route_decision: RouteDecision,
        request: ChatCompletionRequest,
        execute_fn
    ) -> Tuple[Any, str]:
        """Execute with fallback chain."""
        
        providers_to_try = [route_decision.selected_provider] + route_decision.fallback_chain
        
        last_error = None
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
            
            cb = self.circuit_breakers.get(provider_name)
            if cb and not cb.can_execute():
                logger.warning(f"Circuit breaker open for {provider_name}")
                continue
            
            try:
                provider = self.providers[provider_name]
                result = await execute_fn(provider, request)
                
                if cb:
                    cb.record_success()
                
                return result, provider_name
            
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                last_error = e
                if cb:
                    cb.record_failure()
        
        # All providers failed
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    def get_provider_health(self) -> Dict[str, ProviderHealth]:
        """Get health status for all providers."""
        health = {}
        for name, cb in self.circuit_breakers.items():
            if cb.state == "open":
                status = "unhealthy"
            elif cb.state == "half-open":
                status = "degraded"
            else:
                status = "healthy"
            
            health[name] = ProviderHealth(
                provider=name,
                status=status,
                error_rate=cb.failures / max(cb.failure_threshold, 1),
                consecutive_failures=cb.failures
            )
        
        return health
