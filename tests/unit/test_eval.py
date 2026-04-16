"""Unit tests for code-based evals (no database required)."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_code_eval_passes_for_complete_valid_report():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "Monthly spend 12 000 NOK.",
        "portfolio_summary": "Diversified equity portfolio.",
        "market_context": "OSEBX up 0.8%.",
        "compliance_notes": "No issues detected.",
        "recommendations": ["Save more", "Diversify", "Review annually"],
    }
    result = await run_code_evals(report_json, "# Report\n\nContent here.")
    assert result.passed
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_code_eval_catches_missing_required_key():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "Monthly spend 12 000 NOK.",
        "portfolio_summary": "Diversified equity portfolio.",
        "market_context": "OSEBX up 0.8%.",
        # Missing compliance_notes and recommendations
    }
    result = await run_code_evals(report_json, "# Report")
    assert not result.passed
    assert result.checks["schema"] is False


@pytest.mark.asyncio
async def test_code_eval_catches_too_few_recommendations():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "x",
        "portfolio_summary": "y",
        "market_context": "z",
        "compliance_notes": "ok",
        "recommendations": ["just one"],
    }
    result = await run_code_evals(report_json, "# R")
    assert not result.passed
    assert result.checks["recommendation_count"] is False


@pytest.mark.asyncio
async def test_code_eval_detects_pii_in_markdown():
    from app.eval.code import run_code_evals

    report_json = {
        "customer_id": "c-0001",
        "spending_summary": "x",
        "portfolio_summary": "y",
        "market_context": "z",
        "compliance_notes": "ok",
        "recommendations": ["a", "b", "c"],
    }
    md_with_pii = "# Report\nSSN found in report."
    result = await run_code_evals(report_json, md_with_pii)
    assert not result.passed
    assert result.checks["no_pii"] is False


def test_parse_json_response_strips_markdown_fences():
    from app.agents.base import parse_json_response

    text = '```json\n{"key": "value"}\n```'
    result = parse_json_response(text)
    assert result == {"key": "value"}


def test_parse_json_response_handles_plain_json():
    from app.agents.base import parse_json_response

    result = parse_json_response('{"a": 1}')
    assert result == {"a": 1}
