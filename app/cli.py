"""finsight CLI — eval regression and model bench commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click


@click.group()
def main() -> None:
    """FinSight Agents CLI."""


@main.command()
@click.option("--baseline", required=True, help="Baseline prompt version")
@click.option("--candidate", required=True, help="Candidate prompt version")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="JSONL dataset path")
@click.option("--output", required=True, type=click.Path(), help="Output JSON path")
def eval(baseline: str, candidate: str, dataset: str, output: str) -> None:
    """Run regression eval between two prompt versions."""
    asyncio.run(_run_eval(baseline, candidate, Path(dataset), Path(output)))


async def _run_eval(baseline: str, candidate: str, dataset: Path, output: Path) -> None:
    from app.eval.code import run_code_evals

    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    async def _score(version: str) -> dict:
        scores = []
        for rec in records:
            report = {
                "customer_id": rec.get("customer_id", "unknown"),
                "spending_summary": "stub",
                "portfolio_summary": "stub",
                "market_context": "stub",
                "compliance_notes": "stub",
                "recommendations": ["a", "b", "c"],
            }
            result = await run_code_evals(report, "# stub")
            scores.append(result.score)
        return {"version": version, "mean_score": sum(scores) / max(len(scores), 1)}

    baseline_metrics = await _score(baseline)
    candidate_metrics = await _score(candidate)

    delta = candidate_metrics["mean_score"] - baseline_metrics["mean_score"]
    regressions = [
        {"metric": "mean_score", "delta": delta, "regressed": delta < -0.05}
    ]

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta": delta,
        },
        "regressions": regressions,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    click.echo(f"Eval report written to {output}")


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True), help="models.yaml path")
@click.option("--dataset", required=True, type=click.Path(exists=True), help="JSONL dataset path")
@click.option("--output", required=True, type=click.Path(), help="Output JSON path")
def bench(config: str, dataset: str, output: str) -> None:
    """Benchmark multiple model variants on a golden dataset."""
    asyncio.run(_run_bench(Path(config), Path(dataset), Path(output)))


async def _run_bench(config: Path, dataset: Path, output: Path) -> None:
    import yaml

    from app.api.bench import _run_bench_sync

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    variants = cfg.get("variants", [])
    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    await _run_bench_sync(variants, records)

    import mlflow

    result_variants = []
    for v in variants:
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name("finsight-bench")
        runs = client.search_runs(
            [experiment.experiment_id if experiment else "0"],
            filter_string=f"params.variant_id = '{v['id']}'",
            max_results=1,
        )
        metrics = runs[0].data.metrics if runs else {}
        result_variants.append({"id": v["id"], "metrics": metrics})

    output.write_text(json.dumps({"variants": result_variants}, indent=2), encoding="utf-8")
    click.echo(f"Bench report written to {output}")


if __name__ == "__main__":
    main()
