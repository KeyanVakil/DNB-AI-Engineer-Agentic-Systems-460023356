"""LLM provider abstraction.

All agent code calls `llm.complete(...)` on the object returned by `get_llm()`.
In tests this is replaced via dependency_overrides with a FakeLLM.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.observability.otel import get_tracer

_llm: Any | None = None


def get_llm() -> Any:
    """FastAPI dependency — replaced in tests via dependency_overrides."""
    global _llm
    if _llm is None:
        from app.settings import get_settings

        settings = get_settings()
        _llm = _build_llm(settings)
    return _llm


def _build_llm(settings: Any) -> Any:
    provider = settings.llm_provider
    if provider == "fake":
        return _FakeLLM()
    if provider == "ollama":
        return OllamaLLM(base_url=settings.ollama_base_url, model=settings.llm_model)
    if provider == "openai":
        return LiteLLMProvider(model=f"openai/{settings.llm_model}")
    if provider == "anthropic":
        return LiteLLMProvider(model=f"anthropic/{settings.llm_model}")
    if provider == "bedrock":
        return LiteLLMProvider(model=f"bedrock/{settings.llm_model}")
    raise ValueError(f"Unknown LLM provider: {provider!r}")


class _FakeLLM:
    """Minimal non-strict fake for production fallback (normally overridden in tests)."""

    async def complete(self, *, model: str, messages: list[dict], **_: Any) -> dict[str, Any]:
        return {
            "content": "{}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "model": model,
            "prompt_hash": "",
        }


class OllamaLLM:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def complete(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        agent: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import httpx

        effective_model = model or self._model
        prompt_hash = _hash_messages(messages)

        tracer = get_tracer()
        with tracer.start_as_current_span("llm.completion") as span:
            span.set_attribute("llm.model", effective_model)
            span.set_attribute("llm.prompt_hash", prompt_hash)

            payload: dict[str, Any] = {
                "model": effective_model,
                "messages": messages,
                "stream": False,
            }
            if response_format:
                payload["format"] = "json"

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
                resp.raise_for_status()

            data = resp.json()
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
            content = data["message"]["content"]

            span.set_attribute("llm.tokens.in", usage["prompt_tokens"])
            span.set_attribute("llm.tokens.out", usage["completion_tokens"])
            _record_llm_metrics(effective_model, usage)

            return {
                "content": content,
                "usage": usage,
                "model": effective_model,
                "prompt_hash": prompt_hash,
            }


class LiteLLMProvider:
    def __init__(self, model: str) -> None:
        self._model = model

    async def complete(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        agent: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "litellm is required for openai/anthropic/bedrock providers. "
                "Install with: pip install litellm"
            ) from exc

        effective_model = model or self._model
        prompt_hash = _hash_messages(messages)

        tracer = get_tracer()
        with tracer.start_as_current_span("llm.completion") as span:
            span.set_attribute("llm.model", effective_model)
            span.set_attribute("llm.prompt_hash", prompt_hash)

            kw: dict[str, Any] = {}
            if response_format:
                kw["response_format"] = response_format

            resp = await litellm.acompletion(model=effective_model, messages=messages, **kw)
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
            content = resp.choices[0].message.content or ""

            span.set_attribute("llm.tokens.in", usage["prompt_tokens"])
            span.set_attribute("llm.tokens.out", usage["completion_tokens"])
            _record_llm_metrics(effective_model, usage)

            return {
                "content": content,
                "usage": usage,
                "model": effective_model,
                "prompt_hash": prompt_hash,
            }


def _hash_messages(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True).encode()
    ).hexdigest()[:16]


def _record_llm_metrics(model: str, usage: dict[str, Any]) -> None:
    try:
        from app.llm.cost import estimate_cost
        from app.observability.metrics import llm_cost_usd_total, llm_tokens_total

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        llm_tokens_total.labels(kind="in", model=model).inc(prompt_tokens)
        llm_tokens_total.labels(kind="out", model=model).inc(completion_tokens)
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        if cost:
            llm_cost_usd_total.labels(model=model).inc(cost)
    except Exception:
        pass
