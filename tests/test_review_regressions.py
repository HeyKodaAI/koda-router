import pytest
from koda_router.router import ModelRouter, RoutingError
from koda_router.models import RoutingConfig
from koda_router.catalog import ALL_MODELS


def test_local_pin_never_falls_back_to_cloud():
    router = ModelRouter(RoutingConfig(preferred_provider="local"))
    with pytest.raises(RoutingError):
        router.route("describe this image", has_images=True)
    assert router.route("hi").model.provider == "local"
    with pytest.raises(RoutingError):
        router.route("hi", force_model="anthropic:claude-haiku")


@pytest.mark.parametrize("options", [{}, {"preferred_provider": "anthropic"}])
def test_daily_budget_blocks_paid_models(options):
    router = ModelRouter(RoutingConfig(daily_budget=0.01, **options))
    router.record_usage(0.02)
    if options:
        with pytest.raises(RoutingError):
            router.route("hi")
    else:
        assert router.route("hi").estimated_cost == 0
        with pytest.raises(RoutingError):
            router.route("hi", has_images=True)
    with pytest.raises(RoutingError):
        router.route("hi", force_model="anthropic:claude-haiku")


def test_provider_and_force_model_paths_respect_per_request_budget():
    router = ModelRouter(RoutingConfig(per_request_budget=0, preferred_provider="openai"))
    with pytest.raises(RoutingError):
        router.route("hi")
    with pytest.raises(RoutingError):
        router.route("hi", force_model="openai:gpt-4o")


def test_fallbacks_obey_capabilities_order_and_enabled_flag():
    router = ModelRouter(RoutingConfig(fallback_order=["openai", "local", "anthropic"]))
    decision = router.route("hi", has_tools=True, has_images=True)
    assert decision.model.provider == "openai"
    assert all(ALL_MODELS[id].supports_tools and ALL_MODELS[id].supports_vision for id in decision.fallback_models)
    router.config.fallback_enabled = False
    assert router.route("hi").fallback_models == []


@pytest.mark.parametrize("model", ["unknown:model", "local:llama3"])
def test_invalid_forced_model_does_not_bypass_policy(model):
    with pytest.raises(RoutingError):
        ModelRouter().route("image", has_images=True, force_model=model)


def test_all_disabled_models_fail_closed(monkeypatch):
    for model in ALL_MODELS.values():
        monkeypatch.setattr(model, "enabled", False)
    with pytest.raises(RoutingError):
        ModelRouter().route("hi")


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_invalid_usage_cannot_restore_budget(cost):
    with pytest.raises(ValueError):
        ModelRouter().record_usage(cost)


def test_remaining_budget_applies_to_fallbacks():
    router = ModelRouter(RoutingConfig(daily_budget=0.00001))
    decision = router.route("hi", has_images=True)
    assert all(router._estimate_cost(ALL_MODELS[id], "hi") <= router.budget_remaining
               for id in decision.fallback_models)


@pytest.mark.asyncio
async def test_http_estimate_uses_persisted_budget_and_returns_conflict():
    from koda_router import routes
    from koda_router.storage import RoutingStorage
    from koda_router.models import UsageRecord
    from fastapi import HTTPException
    storage = RoutingStorage(":memory:")
    storage.record_usage(UsageRecord(model_id="test", model="test", provider="test", cost=1))
    router = ModelRouter(RoutingConfig(daily_budget=0.1))
    routes.init_routing_routes(storage, router)
    with pytest.raises(HTTPException) as exc:
        await routes.estimate_model(routes.EstimateRequest(message="image", has_images=True))
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_config_route_rejects_invalid_budget():
    from koda_router import routes
    from koda_router.storage import RoutingStorage
    from fastapi import HTTPException
    routes.init_routing_routes(RoutingStorage(":memory:"), ModelRouter())
    with pytest.raises(HTTPException) as exc:
        await routes.update_config(routes.ConfigUpdateRequest(daily_budget=-1))
    assert exc.value.status_code == 422


def test_reset_does_not_disconnect_authoritative_spending():
    router = ModelRouter(RoutingConfig(daily_budget=1))
    router.set_daily_spend_provider(lambda: 2.0)
    router.reset_daily_spend()
    assert router.daily_spend == 2.0
    with pytest.raises(RoutingError):
        router.route("hi", has_images=True)
