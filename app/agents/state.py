"""Shared state types for the FinSight LangGraph."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, field_validator
from typing_extensions import TypedDict


def _list_merge(a: list, b: list) -> list:
    return a + b


class GraphState(TypedDict, total=False):
    customer_id: str
    customer_profile: dict[str, Any]
    transaction_result: dict[str, Any]
    portfolio_result: dict[str, Any]
    market_result: dict[str, Any]
    compliance_result: dict[str, Any]
    report_json: dict[str, Any]
    report_md: str
    revision_count: int
    status: str
    error: str | None
    errors: Annotated[list[Any], _list_merge]
    thread_id: str | None


class FinSightReport(BaseModel):
    customer_id: str
    spending_summary: str
    portfolio_summary: str
    market_context: str
    compliance_notes: str
    recommendations: list[str]

    @field_validator("recommendations")
    @classmethod
    def check_rec_count(cls, v: list[str]) -> list[str]:
        if not (2 <= len(v) <= 5):
            raise ValueError(f"recommendations must have 2–5 items, got {len(v)}")
        return v
