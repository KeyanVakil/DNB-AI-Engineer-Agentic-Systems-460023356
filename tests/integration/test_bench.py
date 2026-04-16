"""Model comparison / benchmarking (Feature F4).

Acceptance criteria:
- `configs/models.yaml` declares variants with per-token cost
- `finsight bench --config configs/models.yaml --dataset evals/golden.jsonl`
  runs the eval set against each variant
- Per-variant metrics (mean latency, total cost, mean judge score,
  code-eval pass rate) land in MLflow and in the UI comparison table
- Reproducible: same dataset + same config seed → same MLflow metrics
  within tolerance
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture()
def bench_config(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "variants": [
                    {
                        "id": "fake-cheap",
                        "provider": "fake",
                        "model": "fake-cheap",
                        "cost_per_1k_in_usd": 0.001,
                        "cost_per_1k_out_usd": 0.002,
                    },
                    {
                        "id": "fake-premium",
                        "provider": "fake",
                        "model": "fake-premium",
                        "cost_per_1k_in_usd": 0.01,
                        "cost_per_1k_out_usd": 0.03,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def bench_dataset(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    records = [
        json.dumps({"customer_id": "c-0001", "expectations": {"must_mention": "savings"}}),
        json.dumps({"customer_id": "c-0001", "expectations": {"must_mention": "dividend"}}),
    ]
    path.write_text("\n".join(records), encoding="utf-8")
    return path


async def test_post_bench_creates_mlflow_experiment(
    client, bench_config, bench_dataset, seed_fixtures
):
    response = await client.post(
        "/bench",
        json={"config_path": str(bench_config), "dataset_path": str(bench_dataset)},
    )

    assert response.status_code == 202
    body = response.json()
    assert "experiment_id" in body
    assert "run_ids" in body
    assert len(body["run_ids"]) == 2  # one per variant


async def test_bench_logs_per_variant_metrics(
    client, bench_config, bench_dataset, seed_fixtures
):
    import mlflow

    response = await client.post(
        "/bench",
        json={
            "config_path": str(bench_config),
            "dataset_path": str(bench_dataset),
            "wait": True,
        },
    )
    assert response.status_code == 200  # synchronous variant
    body = response.json()

    mlflow_client = mlflow.tracking.MlflowClient()
    for variant_run_id in body["run_ids"]:
        mlflow_run = mlflow_client.get_run(variant_run_id)
        metrics = mlflow_run.data.metrics
        for expected in (
            "mean_latency_seconds",
            "total_cost_usd",
            "mean_judge_score",
            "code_eval_pass_rate",
        ):
            assert expected in metrics, f"missing metric {expected!r}"


def test_finsight_bench_cli_runs_all_variants(bench_config, bench_dataset, tmp_path: Path):
    output = tmp_path / "bench.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "bench",
            "--config",
            str(bench_config),
            "--dataset",
            str(bench_dataset),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    variants = {v["id"] for v in payload["variants"]}
    assert variants == {"fake-cheap", "fake-premium"}


async def test_bench_is_reproducible_with_fixed_seed(
    client, bench_config, bench_dataset, seed_fixtures
):
    import mlflow

    def _run():
        return client.post(
            "/bench",
            json={
                "config_path": str(bench_config),
                "dataset_path": str(bench_dataset),
                "seed": 42,
                "wait": True,
            },
        )

    first = (await _run()).json()
    second = (await _run()).json()

    mlflow_client = mlflow.tracking.MlflowClient()
    for a, b in zip(first["run_ids"], second["run_ids"]):
        m_a = mlflow_client.get_run(a).data.metrics
        m_b = mlflow_client.get_run(b).data.metrics
        for key in ("total_cost_usd", "code_eval_pass_rate"):
            assert m_a[key] == pytest.approx(m_b[key], rel=0.01), (
                f"metric {key!r} not reproducible: {m_a[key]!r} vs {m_b[key]!r}"
            )


async def test_bench_rejects_unknown_variant_provider(client, tmp_path: Path, bench_dataset):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"variants": [{"id": "x", "provider": "martian", "model": "x", "cost_per_1k_in_usd": 0}]}
        ),
        encoding="utf-8",
    )

    response = await client.post(
        "/bench",
        json={"config_path": str(bad), "dataset_path": str(bench_dataset)},
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
