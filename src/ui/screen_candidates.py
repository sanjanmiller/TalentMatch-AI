"""
Purpose:
    The "Candidates" screen of TalentMatch AI — a READ-ONLY candidate
    directory built entirely from EXISTING database records.

What this file does:
    - Lists every candidate with their actual name, email address and
      resume file name (via the additive read-only
      db_service.get_candidate_directory over candidate_identity +
      resume_documents).
    - Per candidate, an expandable section shows:
        * the stored professional profile (skills, experience,
          projects, education, certifications) via the existing
          db_service.get_resume_profile;
        * the candidate's shortlist/email activity from the existing
          email_tracking_service records (including the tracked
          status wording and "SMTP accepted at" timestamps);
        * which recruitment runs the candidate appeared in, resolved
          from the existing run_history_service records.
    - HR can filter the directory by name or email using a simple
      search box.

Rules obeyed here:
    - Nothing is created or duplicated: no new tables, no candidate
      record writes, no scoring/recalculations. When information is
      missing from the database, a clear placeholder is shown instead
      of an invented value.
    - Internal candidate IDs are not used as the primary identity;
      they appear only as discreet reference text in the placeholders
      where the full name is unavailable.
    - No emails are sent and no workflow is started from this screen.

Interacts with:
    - src/services/db_service.py (directory lookup, profile lookup)
    - src/services/email_tracking_service.py (per-candidate history)
    - src/services/run_history_service.py (per-candidate recruitment
      appearances)

Side effects:
    - Database reads only.
"""

import streamlit as st

from src.services import db_service
from src.services import run_history_service
from src.services.email_tracking_service import (
    get_email_tracking_records,
)

# Shown when the identity details cannot be resolved. Never replaced
# by an invented value.
IDENTITY_UNAVAILABLE = "Candidate identity unavailable"


def _load_profiles_directory() -> list[dict]:
    """Load every candidate's identity + resume file details.

    Returns:
        List of directory dictionaries; empty (with a friendly error
        shown) on any database problem.
    """

    try:
        directory = db_service.get_candidate_directory()
    except Exception as exc:
        print(f"Candidate directory could not be read: {exc}")
        st.error(
            "The candidate directory could not be loaded from the "
            "database.",
            icon="⛔",
        )
        return []

    return directory


def _render_profile_section(
    candidate_id: str,
    full_name: str | None,
) -> None:
    """Render the stored professional profile for one candidate.

    Uses profile data exactly as stored by the existing ingestion
    process; missing sections are reported rather than filled in.

    Args:
        candidate_id: The stable candidate ID.
        full_name: Candidate display name (used for console logs).
    """

    try:
        profile = db_service.get_resume_profile(candidate_id)
    except Exception as exc:
        print(f"Profile for {candidate_id} could not be read: {exc}")
        profile = None

    if not profile:
        st.caption("No professional profile is stored for this candidate.")
        return

    technical_skills = profile.get("technical_skills") or []
    if technical_skills:
        st.markdown(
            " ".join(f"`{skill}`" for skill in technical_skills)
        )
    else:
        st.caption("No technical skills recorded.")

    with st.expander("Professional details", expanded=False):
        work_experience = profile.get("work_experience") or []
        st.markdown("**Experience**")
        if work_experience:
            for experience in work_experience:
                role_parts = [experience.get("job_title") or "Role —"]
                employer = experience.get("employer")
                if employer:
                    role_parts.append(f"at {employer}")
                dates = " – ".join(
                    part
                    for part in (
                        experience.get("start_date"),
                        experience.get("end_date"),
                    )
                    if part
                )
                if dates:
                    role_parts.append(f"({dates})")
                st.markdown(f"- **{' '.join(role_parts)}**")
                for responsibility in (
                    experience.get("responsibilities") or []
                ):
                    st.markdown(f"  - {responsibility}")
        else:
            st.caption("None recorded.")

        projects = profile.get("projects") or []
        st.markdown("**Projects**")
        if projects:
            for project in projects:
                project_line = f"- **{project.get('project_name') or '—'}**"
                description = project.get("description")
                if description:
                    project_line += f" — {description}"
                st.markdown(project_line)
                technologies = project.get("technologies") or []
                if technologies:
                    st.markdown(
                        f"  Technologies: {', '.join(technologies)}"
                    )
        else:
            st.caption("None recorded.")

        education = profile.get("education") or []
        st.markdown("**Education**")
        if education:
            for entry in education:
                st.markdown(f"- {entry.get('qualification') or '—'}")
        else:
            st.caption("None recorded.")

        certifications = profile.get("certifications") or []
        st.markdown(
            "**Certifications**: "
            + (", ".join(certifications) if certifications else "None recorded.")
        )

        achievements = profile.get("achievements") or []
        st.markdown("**Achievements**")
        if achievements:
            for achievement in achievements:
                st.markdown(f"- {achievement}")
        else:
            st.caption("None recorded.")


def _render_email_history(candidate_id: str, job_title_default: str = "—") -> None:
    """Render this candidate's shortlist/email tracking rows.

    Reads only from the existing email_tracking table. The tracking
    status wording and the "SMTP accepted at" timestamp are shown
    exactly as recorded (SMTP acceptance is not presented as a
    guaranteed inbox delivery).

    Args:
        candidate_id: The stable candidate ID.
        job_title_default: Unused label placeholder (kept simple).
    """

    records = [
        record
        for record in get_email_tracking_records()
        if record.get("candidate_id") == candidate_id
    ]

    st.markdown("**Shortlist & email activity**")
    if not records:
        st.caption(
            "This candidate has not appeared in any shortlist "
            "decisions yet."
        )
        return

    for record in records:
        job_title = record.get("job_title") or job_title_default
        status = record.get("status") or "Unknown"
        line = f"- **{job_title}**: {status}"
        if record.get("updated_at"):
            # The tracking record's timestamp is when SMTP accepted
            # the email submission, not a delivery confirmation.
            line += f" · SMTP accepted at: {record['updated_at']}"
        st.markdown(line)


def _render_recruitment_appearances(candidate_id: str) -> None:
    """List the recruitment runs this candidate appeared in.

    Derived from the stored run-history snapshots (read-only). Runs
    whose snapshot does not include the candidate are skipped.

    Args:
        candidate_id: The stable candidate ID.
    """

    st.markdown("**Recruitment run appearances**")

    appearances: list[str] = []

    try:
        runs = run_history_service.get_recruitment_runs()
    except Exception as exc:
        print(f"Run history could not be read: {exc}")
        runs = []

    for run in runs:
        try:
            entries = run_history_service.get_run_candidates(
                run["run_id"]
            )
        except Exception:
            entries = []

        for entry in entries:
            if entry.get("candidate_id") != candidate_id:
                continue

            run_line = (
                f"- **{run.get('job_title') or '—'}** "
                f"(run #{run.get('run_id')}, "
                f"{str(run.get('created_at') or '')[:10]})"
            )
            if entry.get("rank") is not None:
                run_line += f", rank {entry['rank']}"
            if entry.get("decision"):
                run_line += f", decision: {entry['decision']}"
            appearances.append(run_line)
            break

    if not appearances:
        st.caption("No completed recruitment runs include this candidate.")
        return

    for run_line in appearances:
        st.markdown(run_line)


def render_candidates_screen() -> None:
    """Render the complete Candidates directory screen."""

    st.subheader("Candidates")

    directory = _load_profiles_directory()

    if not directory:
        st.info(
            "No candidates are recorded yet. Candidates appear here "
            "once their resumes have been added to the system.",
            icon="ℹ️",
        )
        return

    # Simple search filter (name or email), applied client-side to
    # the already-loaded directory rows.
    search_text = st.text_input(
        "Search by name or email",
        key="candidates_search_input",
        help="Leave empty to show every candidate.",
    )

    def _matches(entry: dict) -> bool:
        if not search_text.strip():
            return True

        needle = search_text.strip().casefold()
        name = (entry.get("full_name") or "").casefold()
        email = (entry.get("email") or "").casefold()
        return needle in name or needle in email

    matching_candidates = [
        entry for entry in directory if _matches(entry)
    ]

    st.caption(f"{len(matching_candidates)} of {len(directory)} candidates")

    for entry in matching_candidates:
        candidate_id = entry.get("candidate_id") or ""
        full_name = entry.get("full_name") or IDENTITY_UNAVAILABLE
        email = entry.get("email")

        with st.container(border=True):
            header_name = full_name if full_name != IDENTITY_UNAVAILABLE else (
                f"{IDENTITY_UNAVAILABLE} (ID: {candidate_id})"
            )
            st.markdown(f"##### {header_name}")

            if email:
                st.markdown(f"📧 {email}")
            else:
                st.caption("Email address unavailable.")

            resume_filename = entry.get("resume_filename")
            if resume_filename:
                st.markdown(f"📄 Resume file: {resume_filename}")
            else:
                st.caption("Resume file details unavailable.")

            with st.expander("Profile & history", expanded=False):
                _render_profile_section(candidate_id, full_name)
                st.divider()
                _render_email_history(candidate_id)
                st.divider()
                _render_recruitment_appearances(candidate_id)
