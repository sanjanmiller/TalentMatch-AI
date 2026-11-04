
"""
Purpose:
    Implement the nodes used by the recruitment LangGraph workflow.

The existing RecruitmentWorkflow handles candidate retrieval and LLM
evaluation. This file connects that service to HR approval, shortlisting
email delivery, and SQLite email tracking.
"""

from typing import Any

from langgraph.types import interrupt

from src.schemas.evaluation_schema import CandidateAssessment
from src.services import db_service
from src.services import email_tracking_service as tracking
from src.services.email_service import (
    EmailConfigurationError,
    EmailDeliveryError,
    send_shortlisting_email,
)
from src.services.recruitment_workflow import RecruitmentWorkflow
from src.graph.state import RecruitmentState


def retrieve_and_evaluate_candidates(
    state: RecruitmentState,
) -> dict[str, Any]:
    """Run the existing recruitment service and prepare results for HR."""

    job_description = state.get("job_description", "").strip()
    candidate_pool_size = state.get("candidate_limit", 8)

    workflow = RecruitmentWorkflow(candidate_pool_size=candidate_pool_size)
    result = workflow.run(job_description)

    candidates: list[dict[str, Any]] = []

    for candidate in result.candidates:
        identity = (
            db_service.get_candidate_identity(candidate.candidate_id) or {}
        )
        evaluation = candidate.evaluation

        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": (
                    identity.get("full_name") or "Name unavailable"
                ),
                "email": identity.get("email"),
                "retrieval_distance": candidate.retrieval_distance,
                "matched_sections": candidate.matched_sections,
                "rank": candidate.rank,
                "provider_name": candidate.provider_name,
                "evaluation": (
                    evaluation.model_dump(mode="json")
                    if isinstance(evaluation, CandidateAssessment)
                    else None
                ),
                "overall_score": (
                    evaluation.overall_score
                    if evaluation is not None
                    else None
                ),
                "error": candidate.error,
            }
        )

    summary = {
        "success": result.success,
        "message": result.message,
        "candidate_pool_size": result.candidate_pool_size,
        "top_n": result.top_n,
        "retrieved_chunk_count": result.retrieved_chunk_count,
        "unique_candidate_count": result.unique_candidate_count,
        "evaluated_candidate_count": result.evaluated_candidate_count,
    }

    return {
        "candidates": candidates,
        "recruitment_summary": summary,
        "status": (
            "awaiting_hr_approval"
            if candidates
            else "no_candidates_found"
        ),
        "message": (
            "Candidates are ready for HR review."
            if candidates
            else result.message
        ),
    }


def request_hr_approval(state: RecruitmentState) -> dict[str, Any]:
    """
    Pause the graph and wait for the HR decision supplied by Streamlit.

    Expected approval response:
        {
            "decision": "approve",
            "selected_candidate_ids": ["CAND-001", "CAND-003"]
        }

    To reject all candidates:
        {
            "decision": "reject",
            "selected_candidate_ids": []
        }
    """

    candidates = state.get("candidates", [])
    candidate_ids = {
        candidate["candidate_id"]
        for candidate in candidates
        if candidate.get("candidate_id")
    }

    # Pause execution until HR supplies a decision.
    hr_response = interrupt(
        {
            "type": "hr_approval_required",
            "message": "Select candidates to receive the shortlisting email.",
            "candidates": candidates,
        }
    )

    if not isinstance(hr_response, dict):
        raise ValueError("HR approval must be supplied as a dictionary.")

    decision = hr_response.get("decision")
    selected_ids = hr_response.get("selected_candidate_ids", [])

    if decision not in {"approve", "reject"}:
        raise ValueError("HR decision must be 'approve' or 'reject'.")

    if not isinstance(selected_ids, list):
        raise ValueError("Selected candidate IDs must be a list.")

    if any(not isinstance(item, str) for item in selected_ids):
        raise ValueError("Every selected candidate ID must be a string.")

    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("A candidate cannot be selected more than once.")

    # HR can only select candidates returned by this workflow run.
    if not set(selected_ids).issubset(candidate_ids):
        raise ValueError("The selection contains an unknown candidate ID.")

    if decision == "approve" and not selected_ids:
        raise ValueError("Select at least one candidate or reject all.")

    if decision == "reject":
        selected_ids = []

    return {
        "hr_decision": {
            "decision": decision,
            "selected_candidate_ids": selected_ids,
        },
        "status": (
            "approved_pending_email"
            if decision == "approve"
            else "rejected_no_emails"
        ),
        "message": (
            "HR approved the selected candidates."
            if decision == "approve"
            else "HR rejected the email request. No emails will be sent."
        ),
    }


def record_approval(state: RecruitmentState) -> dict[str, Any]:
    """Record HR approval for compatibility with the existing workflow."""

    selected_ids = state.get("hr_decision", {}).get(
        "selected_candidate_ids", []
    )

    return {
        "status": "approved_pending_email",
        "message": f"HR approved {len(selected_ids)} candidate(s).",
    }


def _record_unselected_candidates(
    state: RecruitmentState,
    job_title: str,
) -> None:
    """
    Mark unselected candidates as Not Shortlisted.

    Preserve existing sent or uncertain email attempts so a later workflow
    run cannot accidentally make those candidates eligible for auto-resending.
    """

    if not job_title:
        return

    existing_records = tracking.get_email_tracking_records()
    existing_statuses = {
        (record["candidate_id"], record["job_title"]): record["status"]
        for record in existing_records
    }

    for candidate in state.get("candidates", []):
        candidate_id = candidate.get("candidate_id")

        if not candidate_id:
            continue

        key = (candidate_id, job_title)
        previous_status = existing_statuses.get(key)

        # Do not overwrite a previous send or uncertain send attempt.
        if previous_status in {
            tracking.SHORTLISTED_MAIL_SENT,
            tracking.SHORTLISTED_MAIL_NOT_SENT,
        }:
            continue

        identity = db_service.get_candidate_identity(candidate_id) or {}

        tracking.record_candidate_status(
            candidate_id=candidate_id,
            candidate_name=(
                candidate.get("candidate_name")
                or identity.get("full_name")
                or "Name unavailable"
            ),
            job_title=job_title,
            recipient_email=candidate.get("email"),
            status=tracking.NOT_SHORTLISTED,
        )


def record_rejection(state: RecruitmentState) -> dict[str, Any]:
    """Record HR rejection without sending any emails."""

    _record_unselected_candidates(
        state,
        state.get("job_title", "").strip(),
    )

    return {
        "status": "rejected_no_emails",
        "message": "HR declined. No emails were sent.",
    }


def send_approved_candidate_emails(
    state: RecruitmentState,
) -> dict[str, Any]:
    """
    Send shortlisting emails only to candidates explicitly approved by HR.

    Tracking is prepared before delivery. Successful SMTP submissions are
    marked Mail Sent; failures remain Mail Not Sent. Existing send attempts
    are not automatically retried.
    """

    hr_decision = state.get("hr_decision", {})
    decision = hr_decision.get("decision")
    selected_ids = hr_decision.get("selected_candidate_ids", [])

    # Fail closed unless HR explicitly approves candidates.
    if decision != "approve" or not selected_ids:
        return {
            "email_results": [],
            "status": "rejected_no_emails",
            "message": "No approved candidates. No emails were sent.",
        }

    candidates = state.get("candidates", [])
    candidates_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
        if candidate.get("candidate_id")
    }

    job_title = state.get("job_title", "").strip()

    if not job_title:
        return {
            "email_results": [
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "message": "Job title is missing.",
                }
                for candidate_id in selected_ids
            ],
            "status": "email_delivery_issues",
            "message": (
                "Email sending was skipped because the job title is missing."
            ),
        }

    selected_id_set = set(selected_ids)

    # Track candidates HR did not select.
    unselected_state = {
        **state,
        "candidates": [
            candidate
            for candidate in candidates
            if candidate.get("candidate_id") not in selected_id_set
        ],
    }
    _record_unselected_candidates(unselected_state, job_title)

    email_results: list[dict[str, Any]] = []

    for candidate_id in selected_ids:
        candidate = candidates_by_id.get(candidate_id)

        if candidate is None:
            email_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "message": "Candidate was not found in this workflow run.",
                }
            )
            continue

        candidate_name = (
            candidate.get("candidate_name") or "Name unavailable"
        ).strip()
        candidate_email = candidate.get("email")

        # Check tracking before attempting email delivery.
        should_send = tracking.prepare_shortlisting_email(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            job_title=job_title,
            recipient_email=candidate_email or "",
        )

        if not should_send:
            email_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "message": (
                        "A previous email attempt is recorded. "
                        "Automatic retry was skipped."
                    ),
                }
            )
            continue

        if not candidate_email:
            email_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "message": "Candidate email address is unavailable.",
                }
            )
            continue

        try:
            send_shortlisting_email(
                candidate_name=candidate_name,
                candidate_email=candidate_email,
                job_title=job_title,
            )

            # Update tracking only after the SMTP call returns successfully.
            tracking.record_candidate_status(
                candidate_id=candidate_id,
                candidate_name=candidate_name,
                job_title=job_title,
                recipient_email=candidate_email,
                status=tracking.SHORTLISTED_MAIL_SENT,
            )

            email_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "sent",
                    "message": "Email accepted by the SMTP server.",
                }
            )

        except (
            EmailConfigurationError,
            EmailDeliveryError,
            ValueError,
        ):
            # The prepared record remains Mail Not Sent.
            # Do not expose SMTP credentials or raw exception details.
            email_results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "message": (
                        "Email could not be confirmed as sent. "
                        "Automatic retry is disabled."
                    ),
                }
            )

    sent_count = sum(
        result["status"] == "sent"
        for result in email_results
    )
    failed_count = sum(
        result["status"] == "failed"
        for result in email_results
    )
    skipped_count = sum(
        result["status"] == "skipped"
        for result in email_results
    )

    if failed_count == 0 and skipped_count == 0:
        status = "emails_sent"
        message = f"Successfully sent {sent_count} shortlisting email(s)."
    else:
        status = "email_delivery_issues"
        message = (
            f"Sent {sent_count}; failed {failed_count}; "
            f"skipped {skipped_count} email attempt(s)."
        )
    return {
        "email_results": email_results,
        "status": status,
        "message": message,
    }
