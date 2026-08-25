"""Tests for multi-model routing — catalog, router, storage."""

import os
import tempfile

import pytest

from koda_router.catalog import ALL_MODELS, get_model, list_models, list_providers
from koda_router.models import (
    LatencyTier,
    ModelTier,
    RoutingConfig,
    TaskComplexity,
    UsageRecord,
)
from koda_router.router import ModelRouter
from koda_router.storage import RoutingStorage


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def storage(tmp_db):
    return RoutingStorage(db_path=tmp_db)


@pytest.fixture
def router_default():
    return ModelRouter()


# ==================================================================
# Catalog Tests
# ==================================================================


class TestCatalog:
    def test_models_loaded(self):
        assert len(ALL_MODELS) >= 8

    def test_get_known_model(self):
        m = get_model("anthropic:claude-sonnet")
        assert m is not None
        assert m.provider == "anthropic"

    def test_get_unknown_model(self):
        assert get_model("nope:nope") is None

    def test_list_by_provider(self):
        anthropic = list_models(provider="anthropic")
        assert len(anthropic) >= 3
        assert all(m.provider == "anthropic" for m in anthropic)

    def test_list_by_tier(self):
        fast = list_models(tier="fast")
        assert len(fast) >= 2
        assert all(m.tier == ModelTier.FAST for m in fast)

    def test_list_providers(self):
        providers = list_providers()
        assert "anthropic" in providers
        assert "openai" in providers
        assert "local" in providers

    def test_model_pricing(self):
        m = get_model("anthropic:claude-haiku")
        assert m is not None
        assert m.input_cost_per_m > 0
        assert m.cost_per_1k_input > 0

    def test_local_models_free(self):
        local = list_models(provider="local")
        assert all(m.input_cost_per_m == 0 for m in local)


# ==================================================================
# Router Tests
# ==================================================================


class TestRouter:
    def test_simple_message_routes_to_fast(self, router_default):
        decision = router_default.route("hello")
        assert decision.complexity == TaskComplexity.SIMPLE
        assert decision.model.tier == ModelTier.FAST

    def test_complex_message_routes_to_powerful(self, router_default):
        decision = router_default.route(
            "Analyze and compare the top 5 approaches to implement the new feature, "
            "then design an architecture plan with integration tests"
        )
        assert decision.complexity in (TaskComplexity.COMPLEX, TaskComplexity.CRITICAL)

    def test_critical_keywords(self, router_default):
        decision = router_default.route("delete all production credentials")
        assert decision.complexity == TaskComplexity.CRITICAL

    def test_moderate_message(self, router_default):
        decision = router_default.route("Summarize my emails from today")
        assert decision.complexity == TaskComplexity.MODERATE

    def test_force_model(self, router_default):
        decision = router_default.route("hello", force_model="openai:gpt-4o")
        assert decision.model.id == "openai:gpt-4o"
        assert "Forced" in decision.reason

    def test_forced_provider(self):
        config = RoutingConfig(preferred_provider="openai")
        router = ModelRouter(config)
        decision = router.route("hello")
        assert decision.model.provider == "openai"

    def test_forced_tier(self):
        config = RoutingConfig(preferred_tier=ModelTier.POWERFUL)
        router = ModelRouter(config)
        decision = router.route("hello")
        assert decision.model.tier == ModelTier.POWERFUL

    def test_auto_downgrade_disabled(self):
        config = RoutingConfig(auto_downgrade=False)
        router = ModelRouter(config)
        decision = router.route("hello")
        # Without auto-downgrade, simple messages still use balanced
        assert decision.model.tier == ModelTier.BALANCED

    def test_budget_enforcement(self):
        config = RoutingConfig(per_request_budget=0.0001)  # Very low budget
        router = ModelRouter(config)
        decision = router.route("Tell me about quantum physics in detail")
        # Should try to find a cheaper model
        assert decision.estimated_cost <= 0.001  # Should be very cheap

    def test_decision_has_fallbacks(self, router_default):
        decision = router_default.route("hello")
        # May or may not have fallbacks depending on tier
        assert isinstance(decision.fallback_models, list)

    def test_daily_spend_tracking(self, router_default):
        assert router_default.daily_spend == 0.0
        router_default.record_usage(0.05)
        assert router_default.daily_spend == 0.05
        assert router_default.budget_remaining == 4.95
        router_default.reset_daily_spend()
        assert router_default.daily_spend == 0.0

    def test_estimate_cost(self, router_default):
        decision = router_default.route("hello")
        assert decision.estimated_cost >= 0

    def test_tools_required(self, router_default):
        decision = router_default.route("Send an email to Mike", has_tools=True)
        assert decision.model.supports_tools


# ==================================================================
# Storage Tests
# ==================================================================


class TestStorage:
    def test_default_config(self, storage):
        config = storage.get_config()
        assert config.daily_budget == 5.0
        assert config.per_request_budget == 0.50

    def test_save_and_load_config(self, storage):
        config = RoutingConfig(daily_budget=10.0, preferred_provider="openai")
        storage.save_config(config)
        loaded = storage.get_config()
        assert loaded.daily_budget == 10.0
        assert loaded.preferred_provider == "openai"

    def test_record_usage(self, storage):
        record = UsageRecord(
            model_id="anthropic:claude-sonnet",
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            input_tokens=100,
            output_tokens=200,
            cost=0.0045,
            latency_ms=1500,
            complexity="moderate",
        )
        rid = storage.record_usage(record)
        assert rid

        records = storage.get_usage()
        assert len(records) == 1
        assert records[0].cost == 0.0045

    def test_cost_summary(self, storage):
        for i in range(3):
            storage.record_usage(UsageRecord(
                model_id=f"anthropic:claude-{'haiku' if i < 2 else 'sonnet'}",
                provider="anthropic",
                model="test",
                cost=0.01 * (i + 1),
            ))

        summary = storage.get_cost_summary()
        assert summary["request_count"] == 3
        assert summary["total_cost"] > 0
        assert "anthropic" in summary["by_provider"]

    def test_daily_cost(self, storage):
        storage.record_usage(UsageRecord(
            model_id="anthropic:claude-sonnet",
            provider="anthropic",
            model="test",
            cost=0.05,
        ))
        daily = storage.get_daily_cost()
        assert daily == 0.05

    def test_filter_by_provider(self, storage):
        storage.record_usage(UsageRecord(
            model_id="anthropic:claude-sonnet",
            provider="anthropic",
            model="test",
            cost=0.01,
        ))
        storage.record_usage(UsageRecord(
            model_id="openai:gpt-4o",
            provider="openai",
            model="test",
            cost=0.02,
        ))
        anthropic_records = storage.get_usage(provider="anthropic")
        assert len(anthropic_records) == 1
        assert anthropic_records[0].provider == "anthropic"
