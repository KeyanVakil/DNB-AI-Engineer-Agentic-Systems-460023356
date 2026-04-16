"""Evaluation harness tests (Feature F3).

Acceptance criteria:
- Code evals: schema compliance, 2-5 recommendations, no PII in markdown
- LLM-judge eval: {factuality, helpfulness, tone} scored 1-5 with rationale
- Human review queue present
- All three eval types logged to MLflow under one experiment per prompt version
- CLI `finsight eval --baseline v1 --candidate v2` produces a regression report
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.eval]


# ---------------------------------------------------------------------------
# Code eval
# ---------------------------------------------------------------------------


async def test_code_eval_passes_for_valid_report(completed_run, db_session):
    from app.eval.code import run_code_evals

    result = await run_code_evals(completed_run["report"]["json"], completed_run["report"]["markdown"])

    assert result.passed is True
    assert result.score == 1.0
    assert result.checks["schema"] is True
    assert result.checks["recommendation_count"] is True
    assert result.checks["no_pii"] is True


async def test_code_eval_fails_when_recommendations_too_few():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "x",
        "portfolio_summary": "y",
        "market_context": "z",
        "compliance_notes": "ok",
        "recommendations": ["only one"],
    }
    result = await run_code_evals(report_json, "# Report\n- only one")

    assert result.passed is False
    assert result.checks["recommendation_count"] is False


async def test_code_eval_fails_when_recommendations_too_many():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "x",
        "portfolio_summary": "y",
        "market_context": "z",
        "compliance_notes": "ok",
        "recommendations": [f"rec {i}" for i in range(6)],
    }
    result = await run_code_evals(report_json, "# Report")

    assert result.passed is False
    assert result.checks["recommendation_count"] is False


async def test_code_eval_flags_pii_in_markdown():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "x",
        "portfolio_summary": "y",
        "market_context": "z",
        "compliance_notes": "ok",
        "recommendations": ["a", "b", "c"],
    }
    markdown_with_pii = "# Report\nfnr=12345678901 leaked here"

    result = await run_code_evals(report_json, markdown_with_pii)

    assert result.passed is False
    assert result.checks["no_pii"] is False


async def test_code_eval_fails_when_report_json_schema_invalid():
    from app.eval.code import run_code_evals

    # Missing `recommendations`
    result = await run_code_evals({"customer_id": "c-0001"}, "# Report")

    assert result.passed is False
    assert result.checks["schema"] is False


# ---------------------------------------------------------------------------
# LLM-judge eval
# ---------------------------------------------------------------------------


async def test_judge_eval_returns_scored_result(completed_run):
    from app.eval.judge import run_judge
    from tests.conftest import FakeLLM

    judge_llm = FakeLLM(cassette="judge", strict=False)
    result = await run_judge(
        report_md=completed_run["report"]["markdown"],
        report_json=completed_run["report"]["json"],
        llm=judge_llm,
    )

    for dim in ("factuality", "helpfulness", "tone"):
        assert dim in result.scores
        assert 1 <= result.scores[dim] <= 5
    assert result.rationale


async def test_judge_eval_uses_separate_llm_call(completed_run, fake_llm):
    """The judge must call the LLM, not reuse a cached agent response."""
    from app.eval.judge import run_judge
    from tests.conftest import FakeLLM

    judge_llm = FakeLLM(cassette="judge", strict=False)

    await run_judge(
        report_md=completed_run["report"]["markdown"],
        report_json=completed_run["report"]["json"],
        llm=judge_llm,
    )

    assert len(judge_llm.calls) >= 1


# ---------------------------------------------------------------------------
# Human review queue
# ---------------------------------------------------------------------------


async def test_human_review_queue_enqueues_every_run(client, completed_run):
    response = await client.get("/reviews/queue")

    assert response.status_code == 200
    queue = response.json()
    run_ids = [item["run_id"] for item in queue["items"]]
    assert completed_run["run_id"] in run_ids


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


async def test_all_three_eval_kinds_logged_to_mlflow(completed_run):
    import mlflow

    client = mlflow.tracking.MlflowClient()
    experiment_name = f"finsight-prompt-{completed_run.get('prompt_version', 'default')}"
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None, f"no mlflow experiment {experiment_name!r}"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.finsight_run_id = '{completed_run['run_id']}'",
    )
    assert runs
    tags = {t.key: t.value for t in runs[0].data.tags.values()} if hasattr(runs[0].data.tags, "values") else dict(runs[0].data.tags)
    assert tags.get("eval_kinds")
    kinds = set(tags["eval_kinds"].split(","))
    assert {"code", "judge"}.issubset(kinds)


async def test_eval_results_rows_written_per_kind(db_session, completed_run):
    from sqlalchemy import text

    result = await db_session.execute(
        text("SELECT kind FROM eval_results WHERE run_id = :rid"),
        {"rid": completed_run["run_id"]},
    )
    kinds = {row[0] for row in result}
    assert "code" in kinds
    assert "judge" in kinds


# ---------------------------------------------------------------------------
# CLI: finsight eval --baseline v1 --candidate v2
# ---------------------------------------------------------------------------


def test_finsight_eval_cli_produces_regression_report(tmp_path: Path):
    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"customer_id": "c-0001", "expectations": {"must_mention": "savings"}})
            for _ in range(3)
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "eval",
            "--baseline",
            "v1",
            "--candidate",
            "v2",
            "--dataset",
            str(dataset),
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["baseline"] == "v1"
    assert payload["candidate"] == "v2"
    assert "metrics" in payload
    assert "regressions" in payload
