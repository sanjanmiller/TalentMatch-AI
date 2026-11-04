
"""
Purpose:
    Define the shared state used by the recruitment LangGraph workflow.

The state carries candidate evaluations, HR decisions, and email delivery
results between graph nodes.
"""

from typing import Any, TypedDict


class RecruitmentState(TypedDict, total=False):
    """Store information shared between recruitment workflow nodes."""

    job_description: str
    job_title: str
    candidate_limit: int

    # Candidate evaluation results converted into dictionaries.
    candidates: list[dict[str, Any]]
    recruitment_summary: dict[str, Any]

    # HR approval request and recorded decision.
    approval_request: dict[str, Any]
    hr_decision: dict[str, Any]

    # One delivery result per candidate selected by HR.
    email_results: list[dict[str, Any]]

    # Current workflow status and user-facing message.
    status: str
    message: str
