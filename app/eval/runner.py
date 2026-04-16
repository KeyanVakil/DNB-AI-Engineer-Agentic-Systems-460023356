"""Orchestrates code + judge evals and logs everything to MLflow."""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import mlflow


async def run_evals(
    *,
    run_id: uuid.UUID,
    report_json: dict[str, Any],
    report_md: str,
    llm: Any,
    prompt_version: str = "default",
    db_session: Any | None = None,
) -> dict[str, dict[str, Any]]:
    from app.eval.code import run_code_evals
    from app.eval.judge import run_judge

    code_result = await run_code_evals(report_json, report_md)
    judge_result = await run_judge(report_md=report_md, report_json=report_json, llm=llm)

    experiment_name = f"finsight-prompt-{prompt_version}"
    mlflow.set_experiment(experiment_name)

    eval_kinds = "code,judge"
    with mlflow.start_run(tags={"finsight_run_id": str(run_id), "eval_kinds": eval_kinds}):
        mlflow.log_metric("code_eval_score", code_result.score)
        mlflow.log_metric("code_eval_passed", int(code_result.passed))
        avg_judge = sum(judge_result.scores.values()) / max(len(judge_result.scores), 1)
        mlflow.log_metric("judge_score", avg_judge)
        for dim, val in judge_result.scores.items():
            mlflow.log_metric(f"judge_{dim}", val)

    if db_session is not None:
        await _persist_eval_results(db_session, run_id, code_result, judge_result)

    return {
        "code": {
            "passed": code_result.passed,
            "score": code_result.score,
            "checks": code_result.checks,
        },
        "judge": {
            "passed": judge_result.passed,
            "score": sum(judge_result.scores.values()) / max(len(judge_result.scores), 1),
            "scores": judge_result.scores,
            "rationale": judge_result.rationale,
        },
    }


async def _persist_eval_results(
    session: Any,
    run_id: uuid.UUID,
    code_result: Any,
    judge_result: Any,
) -> None:
    from app.memory.models import EvalResult

    session.add(
        EvalResult(
            id=uuid.uuid4(),
            run_id=run_id,
            kind="code",
            score=Decimal(str(round(code_result.score, 4))),
            passed=code_result.passed,
            payload={"checks": code_result.checks},
        )
    )
    avg = sum(judge_result.scores.values()) / max(len(judge_result.scores), 1)
    session.add(
        EvalResult(
            id=uuid.uuid4(),
            run_id=run_id,
            kind="judge",
            score=Decimal(str(round(avg, 4))),
            passed=judge_result.passed,
            payload={"scores": judge_result.scores, "rationale": judge_result.rationale},
        )
    )
    await session.flush()
