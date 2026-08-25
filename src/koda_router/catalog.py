"""Built-in model catalog with pricing and capabilities.

Pricing as of early 2026. Updated periodically.
"""

from __future__ import annotations

from koda_router.models import ModelDefinition, ModelTier

# ------------------------------------------------------------------
# Anthropic Models
# ------------------------------------------------------------------

ANTHROPIC_HAIKU = ModelDefinition(
    id="anthropic:claude-haiku",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    tier=ModelTier.FAST,
    display_name="Claude Haiku 4.5",
    input_cost_per_m=0.80,
    output_cost_per_m=4.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

ANTHROPIC_SONNET = ModelDefinition(
    id="anthropic:claude-sonnet",
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    tier=ModelTier.BALANCED,
    display_name="Claude Sonnet 4.5",
    input_cost_per_m=3.0,
    output_cost_per_m=15.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

ANTHROPIC_OPUS_45 = ModelDefinition(
    id="anthropic:claude-opus-4.5",
    provider="anthropic",
    model="claude-opus-4-5-20251101",
    tier=ModelTier.POWERFUL,
    display_name="Claude Opus 4.5",
    input_cost_per_m=15.0,
    output_cost_per_m=75.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

ANTHROPIC_OPUS_46 = ModelDefinition(
    id="anthropic:claude-opus-4.6",
    provider="anthropic",
    model="claude-opus-4-6-20260101",
    tier=ModelTier.POWERFUL,
    display_name="Claude Opus 4.6 (Recommended)",
    input_cost_per_m=15.0,
    output_cost_per_m=75.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

ANTHROPIC_SONNET_4 = ModelDefinition(
    id="anthropic:claude-sonnet-4",
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    tier=ModelTier.BALANCED,
    display_name="Claude Sonnet 4",
    input_cost_per_m=3.0,
    output_cost_per_m=15.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

# ------------------------------------------------------------------
# OpenAI Models
# ------------------------------------------------------------------

OPENAI_GPT4O_MINI = ModelDefinition(
    id="openai:gpt-4o-mini",
    provider="openai",
    model="gpt-4o-mini",
    tier=ModelTier.FAST,
    display_name="GPT-4o Mini",
    input_cost_per_m=0.15,
    output_cost_per_m=0.60,
    max_context=128000,
    supports_tools=True,
    supports_vision=True,
)

OPENAI_GPT4O = ModelDefinition(
    id="openai:gpt-4o",
    provider="openai",
    model="gpt-4o",
    tier=ModelTier.BALANCED,
    display_name="GPT-4o",
    input_cost_per_m=2.50,
    output_cost_per_m=10.0,
    max_context=128000,
    supports_tools=True,
    supports_vision=True,
)

OPENAI_O1 = ModelDefinition(
    id="openai:o1",
    provider="openai",
    model="o1",
    tier=ModelTier.POWERFUL,
    display_name="o1",
    input_cost_per_m=15.0,
    output_cost_per_m=60.0,
    max_context=200000,
    supports_tools=True,
    supports_vision=True,
)

# ------------------------------------------------------------------
# Local (Ollama) Models
# ------------------------------------------------------------------

OLLAMA_LLAMA = ModelDefinition(
    id="local:llama3",
    provider="local",
    model="llama3",
    tier=ModelTier.FAST,
    display_name="Llama 3 (Local)",
    input_cost_per_m=0.0,
    output_cost_per_m=0.0,
    max_context=8192,
    supports_tools=False,
    supports_vision=False,
    requires_api_key=False,
)

OLLAMA_MIXTRAL = ModelDefinition(
    id="local:mixtral",
    provider="local",
    model="mixtral",
    tier=ModelTier.BALANCED,
    display_name="Mixtral (Local)",
    input_cost_per_m=0.0,
    output_cost_per_m=0.0,
    max_context=32768,
    supports_tools=False,
    supports_vision=False,
    requires_api_key=False,
)

# ------------------------------------------------------------------
# All models in a dict for easy lookup
# ------------------------------------------------------------------

ALL_MODELS: dict[str, ModelDefinition] = {
    m.id: m
    for m in [
        ANTHROPIC_HAIKU,
        ANTHROPIC_SONNET,
        ANTHROPIC_SONNET_4,
        ANTHROPIC_OPUS_45,
        ANTHROPIC_OPUS_46,
        OPENAI_GPT4O_MINI,
        OPENAI_GPT4O,
        OPENAI_O1,
        OLLAMA_LLAMA,
        OLLAMA_MIXTRAL,
    ]
}

# Default model — most capable by default
DEFAULT_MODEL_ID = "anthropic:claude-opus-4.6"


def get_model(model_id: str) -> ModelDefinition | None:
    return ALL_MODELS.get(model_id)


def list_models(provider: str | None = None, tier: str | None = None) -> list[ModelDefinition]:
    models = list(ALL_MODELS.values())
    if provider:
        models = [m for m in models if m.provider == provider]
    if tier:
        models = [m for m in models if m.tier.value == tier]
    return models


def list_providers() -> list[str]:
    return sorted({m.provider for m in ALL_MODELS.values()})
