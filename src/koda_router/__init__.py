"""Multi-model routing — intelligent model selection based on task attributes.

Routes requests to the best model/provider based on:
- Task complexity (simple → fast cheap model, complex → powerful model)
- Cost budget (respect per-request and daily spending limits)
- Latency requirements (real-time chat vs. background tasks)
- Provider availability (fallback chain)
- User preferences (forced provider override)
"""
