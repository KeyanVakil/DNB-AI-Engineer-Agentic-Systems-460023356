"""Per-model token cost table."""

from __future__ import annotations

_COSTS: dict[str, tuple[float, float]] = {
    # (cost_per_1k_in_usd, cost_per_1k_out_usd)
    "llama3.1:8b": (0.0, 0.0),
    "llama3.1:70b": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "claude-haiku-20240307": (0.00025, 0.00125),
    "claude-sonnet-4-6": (0.003, 0.015),
    "fake-cheap": (0.001, 0.002),
    "fake-premium": (0.01, 0.03),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_cost, out_cost = _COSTS.get(model, (0.0, 0.0))
    return (prompt_tokens / 1000) * in_cost + (completion_tokens / 1000) * out_cost


def cost_per_1k(model: str) -> tuple[float, float]:
    return _COSTS.get(model, (0.0, 0.0))
