"""Intelligent model router — selects the optimal model for each request.

The router analyzes the incoming request to estimate complexity,
then selects the best model given the user's budget and preferences.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from koda_router.catalog import ALL_MODELS, list_models
from koda_router.models import (
    LatencyTier,
    ModelDefinition,
    ModelTier,
    RoutingConfig,
    RoutingDecision,
    TaskComplexity,
)

logger = logging.getLogger("koda_router.router")

# Keywords that suggest higher complexity
_COMPLEX_KEYWORDS = {
    "analyze", "compare", "research", "plan", "design", "architect",
    "debug", "refactor", "optimize", "review", "implement", "build",
    "integrate", "migrate", "investigate", "explain", "summarize",
    "write", "create", "generate", "strategy", "evaluate", "assess",
    "troubleshoot", "diagnose", "recommend", "configure", "transform",
}
_CRITICAL_KEYWORDS = {
    "delete", "remove", "transfer", "payment", "financial", "security",
    "password", "credential", "production", "deploy", "destroy", "wipe",
    "purge", "uninstall", "revoke", "terminate", "cancel", "refund",
}
_SIMPLE_KEYWORDS = {
    "hello", "hi", "thanks", "bye", "weather", "time",
    "help", "status", "list", "show", "hey", "good morning",
    "good night", "ok", "yes", "no", "sure", "got it",
    # Casual conversation — these should stay local, not go to cloud
    "think", "feel", "like", "love", "favorite", "enjoy",
    "read", "reading", "book", "story", "poem", "chapter",
    "tell", "talk", "chat", "about", "remember", "thought",
    "cool", "nice", "awesome", "interesting", "beautiful",
    "haha", "lol", "wow", "oh", "hm", "hmm", "right",
    "morning", "evening", "night", "today", "yesterday",
}
# Multi-step indicators (suggest complex even without complex keywords)
_MULTI_STEP_PATTERNS = [
    r"\b(?:first|then|after that|next|finally|also|and then)\b",
    r"\b(?:step \d|steps?:)\b",
    r"\d+\.\s",  # Numbered lists
    r"\b(?:both|all|each|every|multiple)\b.*\b(?:and|or)\b",
]
# Question complexity indicators
_DEEP_QUESTION_PATTERNS = [
    r"\bwhy\b.*\b(?:does|did|is|are|would|should)\b",
    r"\bhow\b.*\b(?:does|would|should|can|could)\b.*\b(?:work|handle|manage)\b",
    r"\bwhat.*\bdifference\b",
    r"\bcompare\b|\bpros?\s+(?:and|&)\s+cons?\b",
]

# Tier → model tier mapping
_COMPLEXITY_TO_TIER = {
    TaskComplexity.SIMPLE: ModelTier.FAST,
    TaskComplexity.MODERATE: ModelTier.BALANCED,
    TaskComplexity.COMPLEX: ModelTier.POWERFUL,
    TaskComplexity.CRITICAL: ModelTier.POWERFUL,
}


class RoutingError(ValueError):
    """No model satisfies the configured capability, provider or budget policy."""


class ModelRouter:
    """Selects the best model for a given request."""

    def __init__(self, config: Optional[RoutingConfig] = None) -> None:
        self._config = config or RoutingConfig()
        self._daily_spend = 0.0
        self._daily_spend_provider: Callable[[], float] | None = None

    def set_daily_spend_provider(self, provider: Callable[[], float]) -> None:
        """Use authoritative daily usage (e.g. RoutingStorage.get_daily_cost).

        The host must persist usage to that source; record_usage only updates
        the built-in in-memory counter used when no provider is installed.
        """
        self._daily_spend_provider = provider

    @property
    def config(self) -> RoutingConfig:
        return self._config

    @config.setter
    def config(self, value: RoutingConfig) -> None:
        self._config = value

    def route(
        self,
        message: str,
        has_tools: bool = False,
        has_images: bool = False,
        latency: LatencyTier = LatencyTier.REALTIME,
        force_model: Optional[str] = None,
    ) -> RoutingDecision:
        """Select the best model for a request.

        Args:
            message: The user's message text
            has_tools: Whether tool use is expected
            has_images: Whether images are in the request
            latency: Latency requirements
            force_model: Override to force a specific model ID

        Returns:
            RoutingDecision with the selected model and reasoning
        """
        complexity = self.estimate_complexity(message, has_tools)
        provider = self._config.preferred_provider
        compatible = [m for m in ALL_MODELS.values() if m.enabled
                      and (not provider or m.provider == provider)
                      and (not has_tools or m.supports_tools)
                      and (not has_images or m.supports_vision)]
        if not compatible:
            raise RoutingError("No compatible model satisfies the provider and capability requirements")
        budget = min(self._config.per_request_budget, self.budget_remaining)
        affordable = [m for m in compatible if self._estimate_cost(m, message) <= budget]
        if force_model:
            candidates = [m for m in affordable if m.id == force_model]
            if not candidates:
                raise RoutingError("Forced model is unknown, unavailable, incompatible or over budget")
            model = candidates[0]
            reason = f"Forced model: {model.display_name}"
        else:
            if not affordable:
                raise RoutingError("No compatible model fits the remaining daily and per-request budget")
            if self._config.preferred_tier:
                target_tier = self._config.preferred_tier
            elif complexity == TaskComplexity.SIMPLE:
                target_tier = ModelTier.FAST if self._config.auto_downgrade else ModelTier.BALANCED
            elif complexity in (TaskComplexity.COMPLEX, TaskComplexity.CRITICAL):
                target_tier = ModelTier.POWERFUL if self._config.auto_upgrade else ModelTier.BALANCED
            else:
                target_tier = ModelTier.BALANCED
            candidates = [m for m in affordable if m.tier == target_tier]
            if not candidates:
                candidates = [m for m in affordable if m.tier == ModelTier.BALANCED]
            model = self._select_best(candidates or affordable)
            reason = self._build_reason(model, complexity, target_tier)
            if provider:
                reason = f"Forced provider: {provider}; " + reason
        fallbacks = []
        if self._config.fallback_enabled:
            remaining = [m for m in affordable if m.id != model.id and m.tier == model.tier]
            while remaining and len(fallbacks) < 3:
                best = self._select_best(remaining)
                fallbacks.append(best.id)
                remaining = [m for m in remaining if m.id != best.id]
        return RoutingDecision(model=model, reason=reason, complexity=complexity,
                               estimated_cost=round(self._estimate_cost(model, message), 6),
                               fallback_models=fallbacks)

    def estimate_complexity(self, message: str, has_tools: bool = False) -> TaskComplexity:
        """Estimate task complexity using a scoring system.

        Scores multiple signals and combines them rather than using
        simple keyword matching alone. This produces better routing
        for ambiguous or multi-faceted requests.
        """
        lower = message.lower().strip()
        words = set(re.findall(r"\w+", lower))
        # Start with a neutral score (0 = simple, 1 = moderate, 2 = complex, 3 = critical)
        score = 0.0

        # --- Critical detection (highest priority) ---
        critical_hits = len(words & _CRITICAL_KEYWORDS)
        if critical_hits >= 1:
            score += 2.5 + (critical_hits - 1) * 0.5

        # --- Complexity keyword scoring ---
        complex_hits = len(words & _COMPLEX_KEYWORDS)
        score += complex_hits * 0.6

        # --- Multi-step pattern detection ---
        multi_step_hits = sum(
            1 for p in _MULTI_STEP_PATTERNS if re.search(p, lower)
        )
        score += multi_step_hits * 0.5

        # --- Deep question detection ---
        deep_q_hits = sum(
            1 for p in _DEEP_QUESTION_PATTERNS if re.search(p, lower)
        )
        score += deep_q_hits * 0.7

        # --- Tool usage boost ---
        if has_tools:
            score += 0.5

        # --- Length signals ---
        msg_len = len(message)
        if msg_len > 800:
            score += 1.0
        elif msg_len > 400:
            score += 0.5
        elif msg_len > 200:
            score += 0.2

        # --- Sentence count (more sentences = more complex) ---
        sentences = len(re.split(r"[.!?]+", message.strip()))
        if sentences >= 5:
            score += 0.5
        elif sentences >= 3:
            score += 0.2

        # --- Simple / conversational detection (pull score down) ---
        simple_hits = len(words & _SIMPLE_KEYWORDS)
        if simple_hits > 0 and msg_len < 50:
            score -= 1.0
        elif simple_hits >= 2 and msg_len < 200:
            # Multiple casual keywords in a medium message → likely chatting
            score -= 0.6
        if msg_len < 15 and not has_tools:
            score -= 0.5
        # Strongly conversational (high ratio of simple to complex keywords)
        if simple_hits >= 3 and complex_hits == 0 and critical_hits == 0:
            score -= 0.5

        # --- Code detection (code = complex) ---
        if re.search(r"```|def |class |function |import |<[a-z]+>", lower):
            score += 1.0

        # --- Map score to complexity ---
        if score >= 2.5:
            return TaskComplexity.CRITICAL
        if score >= 1.5:
            return TaskComplexity.COMPLEX
        if score >= 0.5:
            return TaskComplexity.MODERATE
        return TaskComplexity.SIMPLE

    def record_usage(self, cost: float) -> None:
        """Record a cost against the daily budget."""
        import math
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Usage cost must be finite and nonnegative")
        self._daily_spend += cost

    def reset_daily_spend(self) -> None:
        """Reset the built-in counter; an external daily source remains authoritative."""
        self._daily_spend = 0.0

    @property
    def daily_spend(self) -> float:
        import math
        spend = self._daily_spend_provider() if self._daily_spend_provider else self._daily_spend
        if not math.isfinite(spend) or spend < 0:
            raise RoutingError("Daily usage source returned an invalid spend")
        return spend

    @property
    def budget_remaining(self) -> float:
        return max(0, self._config.daily_budget - self.daily_spend)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_candidates(
        self,
        tier: ModelTier,
        has_tools: bool,
        has_images: bool,
    ) -> list[ModelDefinition]:
        """Get candidate models matching requirements."""
        return [
            m for m in ALL_MODELS.values()
            if m.enabled
            and m.tier == tier
            and (not has_tools or m.supports_tools)
            and (not has_images or m.supports_vision)
        ]

    def _select_best(self, candidates: list[ModelDefinition]) -> ModelDefinition:
        """Select the best model from candidates using fallback order."""
        fallback = self._config.fallback_order

        # Sort by: provider preference order, then lower cost
        def sort_key(m: ModelDefinition) -> tuple:
            provider_rank = (
                fallback.index(m.provider) if m.provider in fallback else 99
            )
            return (provider_rank, m.input_cost_per_m + m.output_cost_per_m)

        candidates.sort(key=sort_key)
        return candidates[0]

    @staticmethod
    def _estimate_cost(model: ModelDefinition, message: str) -> float:
        """Rough cost estimate based on message length."""
        # Rough token estimate: ~4 chars per token
        input_tokens = len(message) / 4
        output_tokens = input_tokens * 2  # Assume 2:1 output/input ratio
        cost = (
            (input_tokens / 1_000_000) * model.input_cost_per_m
            + (output_tokens / 1_000_000) * model.output_cost_per_m
        )
        return cost

    @staticmethod
    def _build_reason(
        model: ModelDefinition,
        complexity: TaskComplexity,
        target_tier: ModelTier,
    ) -> str:
        parts = [
            f"Complexity: {complexity.value}",
            f"Tier: {target_tier.value}",
            f"Selected: {model.display_name}",
        ]
        return " → ".join(parts)
