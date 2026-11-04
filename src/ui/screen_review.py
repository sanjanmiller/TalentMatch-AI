"""
Purpose:
    Screen 2 "HR Review" of TalentMatch AI: the complete Human-in-the-
    Loop approval process.

What this file does:
    1. Shows every candidate the interrupted workflow presented for
       review, with ACTUAL names/emails (via the shared display) and
       the complete LLM assessment.
    2. Captures one explicit Approve/Reject decision per candidate in
       the session state; HR can freely change any decision until
       they submit.
    3. Shows a final summary (counts + names) and requires an
       explicit confirmation checkbox plus a submit button, so no
       email is ever sent before a deliberate confirmation.
    4. On submit, resumes the PAUSED LangGraph run via
       resume_with_hr_decision() — using the SAME thread ID, so the
       SQLite checkpointer restores the interrupted state and no RAG
       or LLM work is repeated. Only approved candidate IDs are
       passed to the existing email node.
    5. Guards against duplicate submission with the
       "hr_decision_submitted" session flag: a Streamlit rerun (or an
       accidental second click) can never resume the graph twice.
    6. Presents the final email results per candidate, using the
       tracked status wording and "SMTP accepted at" timestamps from
       the existing email_tracking table.

Interacts with:
    - src/ui/session.py (resume_with_hr_decision)
    - src/ui/jd_display.py (ID -> name/email display)
    - src/ui/components.py (shared status labels)
    - src/services/email_tracking_service.py (SMTP accepted-at values)
    - src/services/run_history_service.py (records run completion)

Side effects:
    - Submitting resumes the paused graph and (for approvals) sends
      real shortlist emails through the existing email service.
"""

import streamlit as st

from src.services import run_history_service
from src.services.email_tracking_service import (
    SHORTLISTED_MAIL_SENT,
    get_email_tracking_records,
)
from src.ui import session as session_module
from src.ui.components import (
    EMAIL_STATUS_MISSING,
    NOT_SHORTLISTED_LABEL,
    email_status_label,
)
from src.ui.jd_display import (
    display_candidate_result,
    get_candidate_display_name,
    resolve_candidate_identity,
)

# Actions offered per candidate. The wording is unambiguous and every
# candidate's DEFAULT is "Reject" — HR must actively approve someone
# for an email to be sent.
DECISION_OPTION_APPROVE = "Approve for shortlist email"
DECISION_OPTION_REJECT = "Reject"


def _collect_decisions(
    candidates: list[dict],
) -> tuple[dict[str, bool], dict[str, str]]:
    """Render decision widgets and collect the current decisions.

    Each candidate gets its own unambiguous radio control, keyed by
    the candidate ID so Streamlit keeps the individual selections
    stable across reruns. The widget values are written back into
    session_state["hr_decisions"], which is what the confirmation
    summary and the resume payload are built from.

    Args:
        candidates: All candidate dictionaries awaiting review.

    Returns:
        Tuple of:
            - decisions: candidate_id -> True(approved)/False(rejected)
            - labels:    candidate_id -> HR-facing display label
    """

    decisions: dict[str, bool] = {}
    labels: dict[str, str] = {}

    existing_decisions = st.session_state.get("hr_decisions") or {}

    for candidate in candidates:
        candidate_id = (candidate.get("candidate_id") or "").strip()
        display_name = get_candidate_display_name(candidate, candidates)
        labels[candidate_id] = display_name

        st.markdown(f"#### Candidate for review: {display_name}")

        # Full LLM analysis inside the standard expander, identical to
        # the New Recruitment screen (no duplicated rendering logic).
        display_candidate_result(
            candidate,
            display_name=display_name,
        )

        # Failed evaluations are still reviewable (HR knows the
        # identity), but the missing assessment is visible above.
        previous_decision = existing_decisions.get(candidate_id, False)

        chosen_option = st.radio(
            "Decision",
            [DECISION_OPTION_APPROVE, DECISION_OPTION_REJECT],
            index=0 if previous_decision else 1,
            key=f"decision_{candidate_id}",
            horizontal=True,
        )

        decisions[candidate_id] = chosen_option == DECISION_OPTION_APPROVE

        st.divider()

    # Store the freshly collected decisions back into the session so
    # the summary, the resume call and future reruns all agree.
    st.session_state["hr_decisions"] = decisions

    return decisions, labels


def _render_decision_summary(
    decisions: dict[str, bool],
    labels: dict[str, str],
) -> None:
    """Render the live confirmation summary before HR confirms.

    HR sees exactly who will be emailed and who will not, BEFORE any
    submission. Nothing is sent from this function.

    Args:
        decisions: candidate_id -> approved True/False.
        labels: candidate_id -> display label.
    """

    st.markdown("#### Final confirmation")

    approved_ids = [
        candidate_id for candidate_id, ok in decisions.items() if ok
    ]
    rejected_ids = [
        candidate_id for candidate_id, ok in decisions.items() if not ok
    ]

    st.markdown(
        f"**Approved candidates:** {len(approved_ids)} "
        f"(will receive the shortlist email)"
    )
    for candidate_id in approved_ids:
        st.markdown(f"- ✅ {labels[candidate_id]}")

    st.markdown(
        f"**Rejected candidates:** {len(rejected_ids)} "
        f"(marked '{NOT_SHORTLISTED_LABEL}', no email)"
    )
    for candidate_id in rejected_ids:
        st.markdown(f"- ⛔ {labels[candidate_id]}")

    st.caption(
        "Emails are sent only after you confirm below. Rejected "
        "candidates never receive an email."
    )


def _resume_and_finish(
    decisions: dict[str, bool],
    candidates: list[dict],
    labels: dict[str, str],
) -> None:
    """Confirm + submit handler: resume the graph and record history.

    Called ONLY from the Submit button. The
    "hr_decision_submitted" session flag is the duplicate-submission
    guard: after the first successful resume this screen switches to
    its results view and can never call the graph again.

    Args:
        decisions: candidate_id -> approved True/False.
        candidates: Candidates that were presented for review.
        labels: candidate_id -> display label (for log context).
    """

    if st.session_state.get("hr_decision_submitted"):
        # Belt-and-braces guard; the UI normally hides the button.
        st.warning(
            "The HR decision for this run has already been submitted.",
            icon="⚠️",
        )
        st.rerun()

    thread_id = st.session_state.get("thread_id")
    job_title = st.session_state.get("job_title") or ""

    if not thread_id or not job_title:
        st.error(
            "The recruitment run context is incomplete. Please start "
            "a new run on the New Recruitment screen.",
            icon="⛔",
        )
        return

    with st.spinner(
        "Applying HR decisions and processing shortlist emails..."
        if any(decisions.values())
        else "Applying HR decisions..."
    ):
        try:
            resume_result = session_module.resume_with_hr_decision(
                decisions,
                thread_id=thread_id,
            )
        except ValueError as exc:
            # Decision validation errors from the backend contract.
            st.error(f"The decision could not be submitted: {exc}", icon="⛔")
            return
        except Exception as exc:
            # Any backend failure: HR sees a friendly message and the
            # submission stays open (the run can be resumed again).
            print(f"HR resume failed: {exc}")
            st.error(
                "The decision could not be submitted because of a "
                "backend error. No emails were sent in this attempt. "
                "Please review the decisions and try again.",
                icon="⛔",
            )
            return

    final_state = resume_result["state"]

    # Mark submitted IMMEDIATELY after the successful resume, before
    # any rendering — this is what makes reruns harmless.
    st.session_state["hr_decision_submitted"] = True
    st.session_state["workflow_completed"] = True
    st.session_state["workflow_status"] = final_state.get("status")
    st.session_state["workflow_message"] = final_state.get("message")
    st.session_state["email_results"] = final_state.get(
        "email_results",
        [],
    )

    # Persist the run completion in the additive history table. A
    # failure here must never retroactively undo the emails, so it is
    # logged and surfaced as a warning only.
    try:
        final_status = final_state.get("status") or "completed"
        run_history_service.record_run_completion(
            thread_id,
            job_title,
            decisions=decisions,
            presented_candidates=candidates,
            email_results=st.session_state["email_results"],
            final_status=final_status,
        )
    except Exception as exc:
        print(f"Run history completion could not be recorded: {exc}")
        st.warning(
            "The recruitment run was processed, but its history could "
            "not be saved to the dashboard database.",
            icon="⚠️",
        )

    # Technical details go to the console log only; never shown in UI.
    approved_names = ", ".join(
        labels.get(candidate_id, candidate_id)
        for candidate_id, ok in decisions.items()
        if ok
    ) or "none"
    print(
        f"HR decision submitted for thread {thread_id}: "
        f"approved {approved_names}"
    )

    st.rerun()


def _render_final_email_results(
    candidates: list[dict],
) -> None:
    """Render the end-of-run view after submission (read-only).

    Shows the overall outcome, per-candidate email statuses with the
    tracked wording, and the "SMTP accepted at" timestamp straight
    from the existing email_tracking records. No timestamp is presented
    as a guaranteed inbox delivery.

    Args:
        candidates: Candidates that were presented for review.
    """

    st.subheader("HR decision submitted")

    status = st.session_state.get("workflow_status")
    message = st.session_state.get("workflow_message")

    if status == "rejected_no_emails":
        st.info(
            "All candidates were rejected — no shortlist emails were "
            "sent.",
            icon="ℹ️",
        )
    else:
        st.success(
            "The HR decisions were applied and the shortlist email "
            "workflow has finished.",
            icon="✅",
        )
    if message:
        st.caption(str(message))

    email_results = {
        result.get("candidate_id"): result
        for result in st.session_state.get("email_results") or []
        if result.get("candidate_id")
    }

    # SMTP accepted-at values come from the EXISTING tracking table —
    # they record when the mail server accepted the submission.
    tracking_by_id: dict[str, dict] = {}
    job_title = st.session_state.get("job_title")
    try:
        for record in get_email_tracking_records():
            if (
                record.get("job_title") == job_title
                and record.get("status") == SHORTLISTED_MAIL_SENT
                and record.get("candidate_id")
            ):
                tracking_by_id[record["candidate_id"]] = record
    except Exception as exc:
        print(f"Email tracking could not be read: {exc}")

    st.markdown("#### Outcome per candidate")

    for candidate in candidates:
        candidate_id = (candidate.get("candidate_id") or "").strip()
        name, email = resolve_candidate_identity(candidate)
        approved = (st.session_state.get("hr_decisions") or {}).get(
            candidate_id
        )

        with st.container(border=True):
            st.markdown(f"**{name}**" + (f" ({email})" if email else ""))

            if approved is False:
                st.markdown(f"- Decision: ⛔ {NOT_SHORTLISTED_LABEL}")
                continue

            result = email_results.get(candidate_id)
            if result is None:
                st.markdown("- Decision: ✅ Approved")
                st.markdown(
                    f"- Email status: {EMAIL_STATUS_MISSING} "
                    "(no email attempt recorded)"
                )
                continue

            if result.get("status") == "sent":
                st.markdown("- Decision: ✅ Approved")
                st.success(
                    "Shortlist email sent. Status: "
                    f"{email_status_label(result.get('status'))}",
                    icon="📨",
                )
                tracking_record = tracking_by_id.get(candidate_id)
                if tracking_record and tracking_record.get("updated_at"):
                    st.caption(
                        "SMTP accepted at: "
                        f"{tracking_record['updated_at']} "
                        "(SMTP acceptance does not guarantee final "
                        "inbox delivery)"
                    )
            else:
                st.markdown("- Decision: ✅ Approved")
                # "skipped" is the existing duplicate-prevention
                # safeguard (not a failure), so it reads as a warning.
                if result.get("status") == "skipped":
                    st.warning(
                        "Email status: "
                        f"{email_status_label(result.get('status'))} "
                        f"— {result.get('message') or ''}",
                        icon="⚠️",
                    )
                else:
                    st.error(
                        "Email status: "
                        f"{email_status_label(result.get('status'))} "
                        f"— {result.get('message') or ''}",
                        icon="❌",
                    )

    st.caption(
        "You can start a new recruitment run at any time on the "
        "New Recruitment screen. This run is now read-only."
    )

    with st.expander("Finished run context", expanded=False):
        st.write(f"Thread: {st.session_state.get('thread_id') or '—'}")
        st.write(f"Job title: {st.session_state.get('job_title') or '—'}")
        st.write(f"Status: {st.session_state.get('workflow_status') or '—'}")


def render_hr_review_screen() -> None:
    """Render the complete HR Review screen."""

    st.subheader("HR Review")

    candidates = st.session_state.get("candidate_results") or []

    if not candidates:
        st.info(
            "No recruitment run is awaiting review. Start one on the "
            "**New Recruitment** screen first — candidate discovery "
            "must complete before HR decisions can be made.",
            icon="ℹ️",
        )
        return

    # After the final submission this screen is read-only.
    if st.session_state.get("hr_decision_submitted"):
        _render_final_email_results(candidates)
        return

    st.caption(
        "The AI has ranked these candidates. HR makes the final "
        "shortlist decision: approve to send the shortlist email, "
        "reject to take no action. Decisions can be changed until "
        "you confirm."
    )

    decisions, labels = _collect_decisions(candidates)

    _render_decision_summary(decisions, labels)

    confirm_checked = st.checkbox(
        "I confirm these final decisions",
        key="hr_confirm_checkbox",
    )

    submit_disabled = not confirm_checked

    if st.button(
        "Submit final HR decision",
        type="primary",
        disabled=submit_disabled,
        width="stretch",
        help=(
            "Resumes the paused workflow with only the approved "
            "candidate IDs. No email is sent before this."
            if not submit_disabled
            else "Tick the confirmation box first."
        ),
    ):
        _resume_and_finish(decisions, candidates, labels)
