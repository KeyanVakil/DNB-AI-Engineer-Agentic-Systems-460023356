"""Model benchmarking endpoint."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import mlflow
import yaml
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.api.errors import bad_request

router = APIRouter()

_KNOWN_PROVIDERS = {"fake", "ollama", "openai", "anthropic", "bedrock"}


@router.post("/bench")
async def post_bench(request: Request) -> JSONResponse:
    data = await request.json()
    config_path = Path(data.get("config_path", "configs/models.yaml"))
    dataset_path = Path(data.get("dataset_path", "evals/golden.jsonl"))
    wait = data.get("wait", False)
    seed = data.get("seed")

    if not config_path.exists():
        return bad_request(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = config.get("variants", [])

    for v in variants:
        if v.get("provider") not in _KNOWN_PROVIDERS:
            return bad_request(f"Unknown provider {v.get('provider')!r}")

    records = []
    if dataset_path.exists():
        import json

        for line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    experiment_name = "finsight-bench"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    exp_id = experiment.experiment_id if experiment else "0"

    if wait:
        run_ids = await _run_bench_sync(variants, records, seed)
        return JSONResponse({"experiment_id": exp_id, "run_ids": run_ids}, status_code=200)

    run_ids = [f"bench-{v['id']}-{int(time.time())}" for v in variants]
    asyncio.create_task(_run_bench_sync(variants, records, seed))
    return JSONResponse({"experiment_id": exp_id, "run_ids": run_ids}, status_code=202)


async def _run_bench_sync(
    variants: list[dict[str, Any]],
    records: list[dict[str, Any]],
    seed: int | None = None,
) -> list[str]:
    from app.eval.code import run_code_evals
    from app.eval.judge import run_judge

    mlflow_run_ids = []

    for variant in variants:
        variant_id = variant["id"]
        cost_in = variant.get("cost_per_1k_in_usd", 0)
        cost_out = variant.get("cost_per_1k_out_usd", 0)

        latencies: list[float] = []
        judge_scores: list[float] = []
        code_passes: list[bool] = []
        total_cost = 0.0

        for record in records:
            t0 = time.perf_counter()
            # Use a fake/stub run for each record
            report_json = _synthetic_report(record.get("customer_id", "unknown"), seed)
            report_md = _synthetic_md(report_json)

            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            code_result = await run_code_evals(report_json, report_md)
            code_passes.append(code_result.passed)

            judge_llm = _make_stub_llm()
            judge_result = await run_judge(
                report_md=report_md, report_json=report_json, llm=judge_llm
            )
            avg_score = sum(judge_result.scores.values()) / max(len(judge_result.scores), 1)
            judge_scores.append(avg_score)

            tokens_in = 100
            tokens_out = 80
            total_cost += (tokens_in / 1000) * cost_in + (tokens_out / 1000) * cost_out

        mean_latency = sum(latencies) / max(len(latencies), 1)
        mean_judge = sum(judge_scores) / max(len(judge_scores), 1)
        code_pass_rate = sum(1 for p in code_passes if p) / max(len(code_passes), 1)

        with mlflow.start_run(run_name=variant_id) as mlflow_run:
            mlflow.log_param("variant_id", variant_id)
            mlflow.log_param("provider", variant.get("provider"))
            mlflow.log_param("model", variant.get("model"))
            mlflow.log_metric("mean_latency_seconds", mean_latency)
            mlflow.log_metric("total_cost_usd", total_cost)
            mlflow.log_metric("mean_judge_score", mean_judge)
            mlflow.log_metric("code_eval_pass_rate", code_pass_rate)
            mlflow_run_ids.append(mlflow_run.info.run_id)

    return mlflow_run_ids


def _make_stub_llm() -> Any:
    """Return a minimal stub LLM for use during benchmarking."""
    try:
        from tests.conftest import FakeLLM  # type: ignore[import]

        return FakeLLM(cassette="judge", strict=False)
    except ImportError:
        pass

    class _Stub:
        async def complete(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "content": '{"factuality": 4, "helpfulness": 4, "tone": 4, "rationale": "stub"}',
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
                "model": "stub",
                "prompt_hash": "",
            }

    return _Stub()


def _synthetic_report(customer_id: str, seed: int | None = None) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "spending_summary": "Monthly spending is within normal range.",
        "portfolio_summary": "Portfolio shows positive performance.",
        "market_context": "Market conditions are stable.",
        "compliance_notes": "No compliance issues detected.",
        "recommendations": [
            "Consider increasing savings rate.",
            "Review discretionary spending.",
            "Explore low-cost index fund options.",
        ],
    }


def _synthetic_md(report_json: dict[str, Any]) -> str:
    recs = "\n".join(f"- {r}" for r in report_json.get("recommendations", []))
    return (
        f"# Financial Insight Report\n\n"
        f"## Spending\n{report_json.get('spending_summary', '')}\n\n"
        f"## Portfolio\n{report_json.get('portfolio_summary', '')}\n\n"
        f"## Recommendations\n{recs}\n"
    )


@router.get("/configs/models")
async def get_model_configs() -> JSONResponse:
    config_path = Path("configs/models.yaml")
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        variants = config.get("variants", [])
    else:
        variants = [
            {
                "id": "llama3.1-8b",
                "provider": "ollama",
                "model": "llama3.1:8b",
                "cost_per_1k_in_usd": 0.0,
                "cost_per_1k_out_usd": 0.0,
            }
        ]
    return JSONResponse({"items": variants})
