"""Data models for multi-model routing."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskComplexity(str, Enum):
    """Estimated task complexity — affects model selection."""

    SIMPLE = "simple"        # Greetings, single-step questions, formatting
    MODERATE = "moderate"    # Multi-step tasks, summarization, analysis
    COMPLEX = "complex"      # Multi-tool orchestration, reasoning, code gen
    CRITICAL = "critical"    # High-stakes actions, financial, security-sensitive


class LatencyTier(str, Enum):
    """Latency requirements."""

    REALTIME = "realtime"    # Chat: <2s response time
    STANDARD = "standard"    # Normal: <10s acceptable
    BACKGROUND = "background"  # Workflows, batch: no latency constraint


class ModelTier(str, Enum):
    """Model capability tiers mapped to cost."""

    FAST = "fast"            # Cheapest, fastest (Haiku, GPT-4o-mini, small Ollama)
    BALANCED = "balanced"    # Good quality/cost trade-off (Sonnet, GPT-4o)
    POWERFUL = "powerful"    # Best quality (Opus, o1, large Ollama)


class ModelDefinition(BaseModel):
    """A configured model that the router can select."""

    id: str = Field(..., description="Unique model ID (e.g. 'anthropic:claude-sonnet')")
    provider: str = Field(..., description="Provider name: anthropic, openai, local")
    model: str = Field(..., description="Model identifier for the API")
    tier: ModelTier = ModelTier.BALANCED
    display_name: str = Field(default="")

    # Cost per 1M tokens (for budget tracking)
    input_cost_per_m: float = Field(default=0.0, description="$ per 1M input tokens")
    output_cost_per_m: float = Field(default=0.0, description="$ per 1M output tokens")

    # Capabilities
    max_context: int = Field(default=128000, description="Max context window in tokens")
    supports_tools: bool = Field(default=True)
    supports_vision: bool = Field(default=False)

    # Availability
    enabled: bool = Field(default=True)
    requires_api_key: bool = Field(default=True)

    @property
    def cost_per_1k_input(self) -> float:
        return self.input_cost_per_m / 1000

    @property
    def cost_per_1k_output(self) -> float:
        return self.output_cost_per_m / 1000


class RoutingDecision(BaseModel):
    """The result of a routing decision."""

    model: ModelDefinition
    reason: str = Field(default="", description="Why this model was chosen")
    complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_cost: float = Field(default=0.0, description="Estimated cost in $")
    fallback_models: list[str] = Field(default_factory=list)


class UsageRecord(BaseModel):
    """Record of a single LLM API call for cost tracking."""

    id: str = Field(default="")
    model_id: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    complexity: str = ""
    conversation_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class RoutingConfig(BaseModel):
    """User-configurable routing preferences."""

    # Budget
    daily_budget: float = Field(default=5.0, description="Max daily spend in $")
    per_request_budget: float = Field(default=0.50, description="Max cost per request in $")

    # Model preferences
    preferred_provider: Optional[str] = Field(
        default=None, description="Force a specific provider (None = auto)"
    )
    preferred_tier: Optional[ModelTier] = Field(
        default=None, description="Force a specific tier (None = auto)"
    )

    # Routing strategy
    auto_downgrade: bool = Field(
        default=True,
        description="Automatically use cheaper model for simple tasks",
    )
    auto_upgrade: bool = Field(
        default=True,
        description="Automatically use better model for complex tasks",
    )

    # Fallback behavior
    fallback_enabled: bool = Field(default=True)
    fallback_order: list[str] = Field(
        default_factory=lambda: ["anthropic", "openai", "local"],
    )
