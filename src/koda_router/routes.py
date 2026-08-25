"""FastAPI routes for multi-model routing and usage tracking.

Endpoints:
    GET    /api/v1/routing/models           List available models
    GET    /api/v1/routing/providers         List providers
    GET    /api/v1/routing/config            Get routing config
    PUT    /api/v1/routing/config            Update routing config
    POST   /api/v1/routing/estimate          Estimate model for a message
    GET    /api/v1/routing/usage             Get usage records
    GET    /api/v1/routing/usage/summary     Get cost summary
    GET    /api/v1/routing/budget            Get budget status
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from koda_router.catalog import ALL_MODELS, get_model, list_models, list_providers
from koda_router.models import LatencyTier, RoutingConfig
from koda_router.router import ModelRouter
from koda_router.storage import RoutingStorage

logger = logging.getLogger("koda_router.routes")

router = APIRouter(prefix="/api/v1/routing", tags=["routing"])

_storage: RoutingStorage | None = None
_router: ModelRouter | None = None


def init_routing_routes(storage: RoutingStorage, model_router: ModelRouter) -> None:
    global _storage, _router
    _storage = storage
    _router = model_router


def _get_storage() -> RoutingStorage:
    if _storage is None:
        raise RuntimeError("Routing storage not initialized")
    return _storage


def _get_router() -> ModelRouter:
    if _router is None:
        raise RuntimeError("Model router not initialized")
    return _router


# Request models

class EstimateRequest(BaseModel):
    message: str = Field(..., min_length=1)
    has_tools: bool = False
    has_images: bool = False
    latency: str = "realtime"
    force_model: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    daily_budget: Optional[float] = None
    per_request_budget: Optional[float] = None
    preferred_provider: Optional[str] = None
    preferred_tier: Optional[str] = None
    auto_downgrade: Optional[bool] = None
    auto_upgrade: Optional[bool] = None
    fallback_enabled: Optional[bool] = None
    fallback_order: Optional[list[str]] = None


# Routes

@router.get("/models")
async def get_models(provider: Optional[str] = None, tier: Optional[str] = None) -> dict:
    """List all available models with pricing."""
    models = list_models(provider=provider, tier=tier)
    return {
        "models": [m.model_dump() for m in models],
        "count": len(models),
    }


@router.get("/providers")
async def get_providers() -> dict:
    """List available providers."""
    return {"providers": list_providers()}


@router.get("/config")
async def get_config() -> dict:
    """Get the current routing configuration."""
    storage = _get_storage()
    config = storage.get_config()
    return {"config": config.model_dump()}


@router.put("/config")
async def update_config(request: ConfigUpdateRequest) -> dict:
    """Update routing configuration."""
    storage = _get_storage()
    model_router = _get_router()

    config = storage.get_config()
    update_data = request.model_dump(exclude_none=True)

    # Apply updates
    for key, value in update_data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    storage.save_config(config)
    model_router.config = config

    return {"message": "Routing config updated", "config": config.model_dump()}


@router.post("/estimate")
async def estimate_model(request: EstimateRequest) -> dict:
    """Estimate which model would be selected for a message."""
    model_router = _get_router()

    try:
        latency = LatencyTier(request.latency)
    except ValueError:
        latency = LatencyTier.REALTIME

    decision = model_router.route(
        message=request.message,
        has_tools=request.has_tools,
        has_images=request.has_images,
        latency=latency,
        force_model=request.force_model,
    )

    return {
        "model": decision.model.model_dump(),
        "reason": decision.reason,
        "complexity": decision.complexity.value,
        "estimated_cost": decision.estimated_cost,
        "fallback_models": decision.fallback_models,
    }


@router.get("/usage")
async def get_usage(
    since: Optional[str] = None,
    provider: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Get usage records."""
    storage = _get_storage()
    records = storage.get_usage(since=since, provider=provider, limit=limit)
    return {
        "records": [r.model_dump(mode="json") for r in records],
        "count": len(records),
    }


@router.get("/usage/summary")
async def get_usage_summary(since: Optional[str] = None) -> dict:
    """Get aggregated cost summary."""
    storage = _get_storage()
    return storage.get_cost_summary(since=since)


@router.get("/budget")
async def get_budget_status() -> dict:
    """Get current budget status."""
    storage = _get_storage()
    model_router = _get_router()
    config = storage.get_config()
    daily_cost = storage.get_daily_cost()

    return {
        "daily_budget": config.daily_budget,
        "daily_spend": round(daily_cost, 6),
        "remaining": round(max(0, config.daily_budget - daily_cost), 6),
        "per_request_budget": config.per_request_budget,
        "utilization_pct": round(
            (daily_cost / config.daily_budget * 100) if config.daily_budget > 0 else 0, 1
        ),
    }
