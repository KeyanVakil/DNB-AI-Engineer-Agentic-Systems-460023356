"""Code-based eval: schema checks, recommendation count, PII detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_PII_PATTERNS = [
    re.compile(r"fnr=", re.IGNORECASE),
    re.compile(r"\bSSN\b"),
    re.compile(r"\+47\s*9\d{7}"),
]

_REQUIRED_KEYS = {
    "customer_id",
    "spending_summary",
    "portfolio_summary",
    "market_context",
    "compliance_notes",
    "recommendations",
}


@dataclass
class CodeEvalResult:
    passed: bool
    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


async def run_code_evals(
    report_json: dict[str, Any],
    report_md: str,
) -> CodeEvalResult:
    checks: dict[str, bool] = {}

    # Schema check
    missing = _REQUIRED_KEYS - set(report_json.keys())
    checks["schema"] = len(missing) == 0

    # Recommendation count
    recs = report_json.get("recommendations", [])
    checks["recommendation_count"] = isinstance(recs, list) and 2 <= len(recs) <= 5

    # PII check
    full_text = report_md + " " + str(report_json)
    checks["no_pii"] = not any(p.search(full_text) for p in _PII_PATTERNS)

    passed = all(checks.values())
    score = sum(1 for v in checks.values() if v) / len(checks)

    return CodeEvalResult(
        passed=passed,
        score=score,
        checks=checks,
        details={"missing_keys": list(missing), "rec_count": len(recs)},
    )
