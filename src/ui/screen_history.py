"""
Purpose:
    The "Dashboard" and "Past JDs" screens of TalentMatch AI — a
    read-only view of recruitment activity and of every recorded run
    with its candidate-level outcomes.

What this file does:
    - render_dashboard_screen(): summary metric cards computed ONLY
      from actually stored data (the additive recruitment_runs table
      and the existing email_tracking table), plus a recent-activity
      overview.
    - render_past_jds_screen(): the historical recruitment list — job
      title, JD source, date, candidate counts, emails and overall
      status — with a per-run drill-down that shows actual candidate
      name/email, rank, score, HR decision, email status and the
      "SMTP accepted at" timestamp from the tracking record.
    - Both screens are strictly READ-ONLY: opening an old run can
      never restart RAG, LLM evaluation or email sending, because
      these functions only read from run_history_service.

    The two screens share the same rendering helpers, so there is one
    implementation of the run details (no duplication).

Interacts with:
    - src/services/run_history_service.py (additive recruitment_runs)
    - src/services/email_tracking_service.py (emails sent metric)

Side effects:
    - Database reads only; no writes, no emails, no LLM calls.
"""

import streamlit as st

from src.services import run_history_service
from src.services.email_tracking_service import (
    SHORTLISTED_MAIL_SENT,
    get_email_tracking_records,
)


def _render_summary_metrics(runs: list[dict]) -> None:
    """Render the summary metric cards from stored records only.

    Args:
        runs: All recorded runs (newest first), as returned by
            run_history_service.get_recruitment_runs().
    """

    total_runs = len(runs)
    total_evaluated = sum(
        run.get("candidate_count") or 0 for run in runs
    )
    total_approved = sum(
        run.get("approved_count") or 0 for run in runs
    )
    total_rejected = sum(
        run.get("rejected_count") or 0 for run in runs
    )

    # Emails actually sent are counted from the EXISTING tracking
    # table (the single source of truth for email activity).
    emails_sent = sum(
        1
        for record in get_email_tracking_records()
        if record.get("status") == SHORTLISTED_MAIL_SENT
    )

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric("Recruitment runs", total_runs)
    metric_b.metric("Candidates evaluated", total_evaluated)
    metric_c.metric("Approved candidates", total_approved)
    metric_d.metric("Rejected candidates", total_rejected)
    metric_e.metric("Shortlist emails sent", emails_sent)


def _render_recent_activity(runs: list[dict]) -> None:
    """Render a compact overview of the most recent runs.

    Args:
        runs: All recorded runs, newest first.
    """

    st.markdown("#### Recent recruitment activity")

    recent_runs = runs[:5]

    for run in recent_runs:
        created_date = str(run.get("created_at") or "")[:10]
        st.markdown(
            f"- **{run['job_title']}** — {created_date} · "
            f"{run.get('candidate_count') or 0} evaluated · "
            f"{run.get('approved_count') or 0} approved · "
            f"{run.get('rejected_count') or 0} rejected · "
            f"Status: {run.get('final_status') or 'Unknown'}"
        )

    if len(runs) > len(recent_runs):
        st.caption(
            "The full list of past recruitments is on the "
            "**Past JDs** screen."
        )


def _render_run_selector(runs: list[dict]) -> None:
    """Render the historical run list and the selected run's details.

    Args:
        runs: All recorded runs, newest first.
    """

    options = {
        f"Run #{run['run_id']} — {run['job_title']} "
        f"({str(run.get('created_at') or '')[:10]})": run
        for run in runs
    }

    selected_label = st.selectbox(
        "Select a recruitment run to view its details",
        list(options.keys()),
        key="history_selected_run",
    )

    selected_run = options.get(selected_label)
    if selected_run is None:
        return

    with st.container(border=True):
        st.markdown(
            f"##### Run #{selected_run['run_id']} — "
            f"{selected_run['job_title']}"
        )

        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.write(f"**Thread ID:** {selected_run.get('thread_id', '—')}")
            st.write(
                "**JD source:** "
                f"{selected_run.get('jd_source') or '—'}"
            )
            st.write(
                "**Recruitment date:** "
                f"{selected_run.get('created_at') or '—'}"
            )
        with detail_right:
            st.write(
                "**Completed:** "
                f"{selected_run.get('completed_at') or 'Not completed'}"
            )
            st.write(
                "**Overall status:** "
                f"{selected_run.get('final_status') or 'Unknown'}"
            )
            st.write(
                "**Counts:** "
                f"{selected_run.get('candidate_count') or 0} evaluated, "
                f"{selected_run.get('approved_count') or 0} approved, "
                f"{selected_run.get('rejected_count') or 0} rejected, "
                "emails: see candidate details below"
            )

        _render_run_candidates(selected_run)


def _render_run_candidates(selected_run: dict) -> None:
    """Render the candidate-level details of one historical run.

    Uses the stored snapshot (actual names/emails/ranks/decisions and
    the SMTP accepted-at timestamp exactly as tracked). Read-only.

    Args:
        selected_run: The selected run row dictionary.
    """

    st.markdown("###### Candidate results")

    try:
        candidate_details = run_history_service.get_run_candidates(
            selected_run["run_id"]
        )
    except Exception as exc:
        print(f"Run candidates could not be loaded: {exc}")
        st.warning(
            "The candidate details for this run could not be loaded "
            "from the database.",
            icon="⚠️",
        )
        return

    if not candidate_details:
        if not selected_run.get("completed_at"):
            st.caption(
                "This run has not been completed by an HR decision "
                "yet, so no candidate history is stored."
            )
        else:
            st.caption("No candidate records are stored for this run.")
        return

    # Structured table view for quick scanning. All display columns
    # are strings so the Arrow-backed table can mix real values and
    # "—" placeholders.
    st.dataframe(
        [
            {
                "Name": entry.get("name") or "Candidate identity unavailable",
                "Email": entry.get("email") or "—",
                "Rank": (
                    str(entry.get("rank"))
                    if entry.get("rank") is not None
                    else "—"
                ),
                "Score": (
                    f"{entry.get('overall_score')}/10"
                    if entry.get("overall_score") is not None
                    else "—"
                ),
                "HR decision": entry.get("decision") or "—",
                "Email status": entry.get("email_status") or "—",
                "SMTP accepted at": (
                    entry.get("smtp_accepted_at") or "—"
                ),
            }
            for entry in candidate_details
        ],
        width="stretch",
        hide_index=True,
    )

    # Plain-language lines for accessibility and quick reading.
    for entry in candidate_details:
        name = entry.get("name") or "Candidate identity unavailable"
        rank_value = entry.get("rank")
        score_value = entry.get("overall_score")
        line = (
            f"Rank {rank_value if rank_value is not None else '—'} · "
            f"{name} ({entry.get('email') or 'email unavailable'}) — "
            f"Score: {score_value if score_value is not None else '—'}/10 · "
            f"Decision: {entry.get('decision') or '—'} · "
            f"Email status: {entry.get('email_status') or '—'}"
        )
        if entry.get("smtp_accepted_at"):
            line += f" · SMTP accepted at: {entry['smtp_accepted_at']}"
        st.caption(line)


def render_dashboard_screen() -> None:
    """Render the Dashboard screen (metrics + recent activity)."""

    st.subheader("Dashboard")

    try:
        runs = run_history_service.get_recruitment_runs()
    except Exception as exc:
        print(f"Recruitment history could not be read: {exc}")
        st.error(
            "The recruitment history could not be loaded from the "
            "database.",
            icon="⛔",
        )
        return

    if not runs:
        st.info(
            "No recruitment runs are recorded yet. Complete a "
            "recruitment run on the New Recruitment screen and its "
            "summary will appear here.",
            icon="ℹ️",
        )
        return

    _render_summary_metrics(runs)

    st.divider()

    _render_recent_activity(runs)

    st.caption(
        "Detailed candidate outcomes for each run are available on "
        "the **Past JDs** screen. Historical records are read-only — "
        "they never restart the workflow or send emails."
    )


def render_past_jds_screen() -> None:
    """Render the Past JDs screen (historical run inspection)."""

    st.subheader("Past JDs")

    try:
        runs = run_history_service.get_recruitment_runs()
    except Exception as exc:
        print(f"Recruitment history could not be read: {exc}")
        st.error(
            "The recruitment history could not be loaded from the "
            "database.",
            icon="⛔",
        )
        return

    if not runs:
        st.info(
            "No recruitment runs are recorded yet. Completed "
            "recruitments will appear here with their job "
            "descriptions and candidate results.",
            icon="ℹ️",
        )
        return

    _render_run_selector(runs)

    st.caption(
        "Historical records are read-only. Opening a past run never "
        "restarts the workflow or sends emails."
    )
