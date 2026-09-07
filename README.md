# koda-router

Multi-model LLM router: scores task complexity and picks the right model by tier, cost budget, latency need, and provider availability, with fallback chains.

## What & why

Sending every request to your most capable (and most expensive) model wastes money; sending everything to the cheapest one wastes quality. `koda-router` is the decision layer in between: it analyzes an incoming message, estimates how hard the task is, and returns a `RoutingDecision` naming the model to use, why it was chosen, an estimated cost, and a fallback chain.

It is a pure decision engine — it does not call any LLM provider itself. You feed the decision into your own Anthropic/OpenAI/Ollama client code.

Extracted from [Koda AI](https://github.com/HeyKodaAI) ("the AI agent that never lies"), where it routed every agent request across cloud and local models. Part of the Koda project by Mike Wandzilak / Wandzilak Web Design.

## Features

- **Complexity scoring** — combines multiple signals rather than bare keyword matching: complex/critical/casual keyword sets, multi-step patterns ("first... then...", numbered lists), deep-question patterns ("why does...", "compare..."), message length, sentence count, code detection, and tool usage. Maps to four levels: `simple`, `moderate`, `complex`, `critical`.
- **Tier-based selection** — models are grouped into `fast`, `balanced`, and `powerful` tiers; complexity maps to a target tier, with configurable auto-upgrade and auto-downgrade.
- **Cost budgets** — per-request and daily budget limits; requests that would exceed either limit use a cheaper compatible model or raise `RoutingError`.
- **Fallback chains** — every decision includes up to three same-tier fallback models, ordered by a configurable provider preference (`["anthropic", "openai", "local"]` by default).
- **Overrides** — force a specific model ID or pin all routing to one provider.
- **Capability filtering** — models that don't support tools or vision are excluded when the request needs them.
- **Built-in catalog** — Anthropic (Haiku/Sonnet/Opus), OpenAI (GPT-4o mini/GPT-4o/o1), and local Ollama models (Llama 3, Mixtral), each with pricing, context window, and capability metadata.
- **Usage tracking** — SQLite-backed storage for routing config and per-call usage records, with cost summaries by provider and model.
- **FastAPI routes** — a ready-made `APIRouter` exposing the router and usage data over HTTP.

## Review fixes (0.1.1)

See [CHANGELOG.md](CHANGELOG.md) for fixes, compatibility changes and upgrade guidance.

## Install

Not yet on PyPI. Requires Python 3.11+.

```bash
pip install git+https://github.com/HeyKodaAI/koda-router.git
```

## Quickstart

```python
from koda_router.models import RoutingConfig
from koda_router.router import ModelRouter

router = ModelRouter(RoutingConfig(daily_budget=5.0, per_request_budget=0.50))

messages = [
    "hey, good morning",
    "Refactor the auth module, then add tests, and finally update the docs",
    "Delete the production database backups",
]

for msg in messages:
    decision = router.route(msg)
    print(f"{msg!r}")
    print(f"  model:      {decision.model.id} ({decision.model.display_name})")
    print(f"  complexity: {decision.complexity.value}")
    print(f"  est. cost:  ${decision.estimated_cost}")
    print(f"  fallbacks:  {decision.fallback_models}")
```

Output:

```text
'hey, good morning'
  model:      anthropic:claude-haiku (Claude Haiku 4.5)
  complexity: simple
  est. cost:  $3.7e-05
  fallbacks:  ['openai:gpt-4o-mini', 'local:llama3']
'Refactor the auth module, then add tests, and finally update the docs'
  model:      anthropic:claude-sonnet (Claude Sonnet 4.5)
  complexity: moderate
  est. cost:  $0.000569
  fallbacks:  ['anthropic:claude-sonnet-4', 'openai:gpt-4o', 'local:mixtral']
'Delete the production database backups'
  model:      anthropic:claude-opus-4.5 (Claude Opus 4.5)
  complexity: critical
  est. cost:  $0.001568
  fallbacks:  ['anthropic:claude-opus-4.6', 'openai:o1']
```

Note: the top-level `koda_router` package holds only documentation; import from the submodules as shown above.

## API overview

### `koda_router.router`

- `ModelRouter(config: RoutingConfig | None)` — the core class.
  - `route(message, has_tools=False, has_images=False, latency=LatencyTier.REALTIME, force_model=None) -> RoutingDecision`
  - `estimate_complexity(message, has_tools=False) -> TaskComplexity`
  - `record_usage(cost)` / `reset_daily_spend()` / `daily_spend` / `budget_remaining`

### `koda_router.models`

- `RoutingConfig` — budgets, preferred provider/tier, auto up/downgrade, fallback order.
- `RoutingDecision` — selected `model`, `reason`, `complexity`, `estimated_cost`, `fallback_models`.
- `ModelDefinition` — id, provider, tier, per-1M-token pricing, context window, tool/vision support.
- `TaskComplexity`, `LatencyTier`, `ModelTier` — enums.
- `UsageRecord` — a single tracked LLM call (tokens, cost, latency).

### `koda_router.catalog`

- `ALL_MODELS` — dict of all built-in `ModelDefinition`s keyed by ID.
- `get_model(model_id)`, `list_models(provider=None, tier=None)`, `list_providers()`, `DEFAULT_MODEL_ID`.

Pricing in the catalog is a snapshot (early 2026); verify against current provider pricing before relying on cost estimates.

### `koda_router.storage`

- `RoutingStorage(db_path="data/routing.db")` — SQLite (WAL mode) persistence: `get_config`/`save_config`, `record_usage`, `get_usage`, `get_cost_summary`, `get_daily_cost`.

### `koda_router.routes`

FastAPI `APIRouter` mounted at `/api/v1/routing`. Call `init_routing_routes(storage, model_router)` once at startup, then include `routes.router` in your app.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/routing/models` | List models (filter by provider/tier) |
| GET | `/api/v1/routing/providers` | List providers |
| GET / PUT | `/api/v1/routing/config` | Read / update routing config |
| POST | `/api/v1/routing/estimate` | Dry-run a routing decision for a message |
| GET | `/api/v1/routing/usage` | Usage records |
| GET | `/api/v1/routing/usage/summary` | Aggregated cost summary |
| GET | `/api/v1/routing/budget` | Daily budget status |

## Testing

```bash
pip install -e . pytest pytest-asyncio
pytest
```

43 tests.

## License

MIT.
