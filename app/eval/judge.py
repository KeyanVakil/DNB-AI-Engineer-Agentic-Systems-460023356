"""LLM-as-judge evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.agents.base import parse_json_response, run_llm


@dataclass
class JudgeResult:
    scores: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    passed: bool = True


async def run_judge(
    *,
    report_md: str,
    report_json: dict[str, Any],
    llm: Any,
) -> JudgeResult:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an impartial evaluator of financial insight reports. "
                "Score the report on three dimensions: factuality, helpfulness, tone. "
                "Each score is an integer from 1 (poor) to 5 (excellent). "
                'Return JSON: '
                '{"factuality": int, "helpfulness": int, "tone": int, "rationale": "..."}.'
            ),
        },
        {
            "role": "user",
            "content": f"Report JSON:\n{report_json}\n\nReport Markdown:\n{report_md}",
        },
    ]

    resp = await run_llm(llm, model=None, messages=messages, agent="judge")

    try:
        data = parse_json_response(resp["content"])
    except Exception:
        data = {}

    scores: dict[str, int] = {}
    for dim in ("factuality", "helpfulness", "tone"):
        raw = data.get(dim, 3)
        scores[dim] = max(1, min(5, int(raw)))

    rationale = data.get("rationale", "")
    avg = sum(scores.values()) / len(scores) if scores else 0.0

    try:
        from app.observability.metrics import eval_judge_score
        for dim, val in scores.items():
            eval_judge_score.labels(dimension=dim).set(val)
    except Exception:
        pass

    return JudgeResult(
        scores=scores,
        rationale=rationale,
        passed=avg >= 3.0,
    )
