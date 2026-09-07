# Changelog

## 0.1.1 - 2026-09-07

- Forced providers, including local-only, fail with `RoutingError` when no compatible model exists. Forced models cannot override provider, capability or budget constraints. (Review 10)
- Daily remaining budget and per-request limits apply to every path. A free compatible model can still be selected at zero remaining budget; otherwise routing raises `RoutingError`. (Review 11)
- Fallbacks obey capabilities, provider pin, budget, enabled state, preference order and `fallback_enabled`. (Review 12)
- HTTP estimates return 409 when no route satisfies policy. The route adapter reads daily spending from `RoutingStorage`; invalid configuration updates return 422.

### Upgrade

Catch `koda_router.router.RoutingError` and report the constraint instead of invoking a different provider. For standalone use, call `record_usage(actual_cost)` and reset its in-memory counter daily, or supply `set_daily_spend_provider(callable)` backed by durable usage. With the supplied HTTP adapter, record actual usage in `RoutingStorage`; its daily query is authoritative. If a provider is installed, `record_usage()` updates only the unused in-memory counter.

Cost checks are estimates from the existing catalog and message-length heuristic, not an atomic spending reservation or a guarantee of final token charges. The host must record actual costs and coordinate concurrent requests. This change does not update model IDs, pricing or latency estimation.

### Validation

43 tests pass, including `tests/test_review_regressions.py`; main README quickstart checked. Tests use synthetic data and isolated databases. No live provider calls or service actions were used.
