"""
Purpose:
    HR-facing display components for candidate results in TalentMatch AI.

Why this file is needed:
    The LangGraph retrieval node returns candidate dictionaries whose
    primary key is the internal candidate ID (for example "CAND-001").
    HR must see ACTUAL candidate names and email addresses instead, so
    this module maps IDs to identity information from the existing
    SQLite database and renders the complete LLM assessment for review.

Critical rules implemented here:
    - Candidate NAMES and emails are the HR-facing identity; IDs are
      used internally and only shown discreetly when two candidates
      share the same name.
    - Only information actually returned by the backend is displayed.
      Nothing is fabricated and no score is ever recalculated; the
      LLM's output is shown exactly as validated by the backend.
    - When identity information or an evaluation is missing, a clear
      fallback message is shown instead of an invented value.

Interacts with:
    - src/services/db_service.py (get_candidate_identity for the
      candidate_id -> name/email mapping)
    - The candidate dictionaries produced by the retrieval node
      (src/graph/nodes.py), which already embed the resolved name and
      email; the database lookup is only a fallback for raw results.
"""

import streamlit as st

from src.services import db_service


# Shown when neither the graph state nor the database can provide a
# candidate's identity. Never replaced by an invented name.
IDENTITY_UNAVAILABLE = "Candidate identity unavailable"


def resolve_candidate_identity(candidate: dict) -> tuple[str, str | None]:
    """Return (display name, email) for one candidate result dict.

    The graph already fills `candidate_name` and `email` inside the
    candidate dictionaries (src/graph/nodes.py), so those are preferred.
    For raw result dictionaries that carry only a `candidate_id`, the
    existing identity lookup (db_service.get_candidate_identity) is
    used as a fallback.

    Args:
        candidate: One candidate dictionary from the backend.

    Returns:
        Tuple of (name, email). The name is the real full name, or the
        IDENTITY_UNAVAILABLE fallback when it cannot be determined.
        The email is the address on file, or None when unavailable.
    """

    candidate_id = (candidate.get("candidate_id") or "").strip()
    name = candidate.get("candidate_name")
    email = candidate.get("email")

    # Look the identity up only when something is actually missing and
    # a usable ID exists, so existing information is never replaced.
    if (
        (not name or not email)
        and candidate_id
    ):
        identity = db_service.get_candidate_identity(candidate_id) or {}

        if not name:
            name = identity.get("full_name")

        if not email:
            email = identity.get("email")

    if not name:
        name = IDENTITY_UNAVAILABLE

    email = email if email else None

    return name, email


def get_candidate_display_name(
    candidate: dict,
    all_candidates: list[dict] | None = None,
) -> str:
    """Build the HR-facing candidate label for one candidate.

    The NAME is the primary identifier. The internal candidate ID is
    appended discreetly ONLY when two or more shortlisted candidates
    share the same name, so HR can always tell them apart safely.

    Args:
        candidate: One candidate dictionary from the backend.
        all_candidates: The full shortlist (used to detect duplicate
            names). Omit when the list is not available.

    Returns:
        The display label, for example "Rahul Sharma" or, for a
        duplicate name, "Rahul Sharma (ID: CAND-007)".
    """

    name, _ = resolve_candidate_identity(candidate)

    if name == IDENTITY_UNAVAILABLE:
        return name

    if not all_candidates:
        return name

    # Count how often this exact display name appears in the shortlist.
    name_occurrences = sum(
        1
        for other in all_candidates
        if resolve_candidate_identity(other)[0] == name
    )

    if name_occurrences > 1:
        candidate_id = (candidate.get("candidate_id") or "").strip()
        return f"{name} (ID: {candidate_id})"

    return name


def select_top_candidates(
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Select the top-N evaluated candidates in LLM ranking order.

    The backend retrieves and evaluates a pool of up to eight
    candidates, but the New Recruitment screen presents only the top
    five, exactly as ranked by the automated evaluation. No re-scoring
    happens here; candidates without a rank (failed evaluations) are
    excluded and reported separately by the display function.

    Args:
        candidates: All candidate dictionaries from the backend.
        top_n: How many ranked candidates to return (default 5).

    Returns:
        The top-N candidates, sorted by rank (1 = best first).
    """

    ranked_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("rank") is not None
    ]

    ranked_candidates.sort(key=lambda candidate: candidate["rank"])

    return ranked_candidates[:top_n]


def _render_bullet_section(container, heading: str, items: list | None) -> None:
    """Render one heading with bullet lines, or 'None reported.'."""

    container.markdown(f"**{heading}**")

    if items:
        for item in items:
            container.markdown(f"- {item}")
    else:
        container.caption("None reported.")


def display_candidate_result(
    candidate: dict,
    display_name: str | None = None,
    container=None,
) -> None:
    """Render one candidate's complete assessment as an expander.

    Everything shown comes directly from the backend dictionary:
    name, email, rank, overall score, LLM explanation, evidence lists
    and the provider that evaluated the candidate. Missing optional
    fields are rendered as 'N/A' or 'None reported.' — never invented.

    Args:
        candidate: One candidate dictionary from the backend.
        display_name: Optional pre-computed HR-facing label. When
            omitted it is derived from the candidate dictionary.
        container: Optional Streamlit container (for example an
            expander or column) to render into. Defaults to `st`.
    """

    target = container if container is not None else st

    name, email = resolve_candidate_identity(candidate)
    label = display_name or get_candidate_display_name(candidate)

    evaluation = candidate.get("evaluation")
    rank = candidate.get("rank")
    overall_score = candidate.get("overall_score")

    # NOTE: the provider that produced the evaluation stays in the
    # graph state and internal logs only — it is deliberately NOT
    # displayed to HR (no "Evaluated by" field anywhere in the UI).

    expander_title = (
        f"Rank {rank if rank is not None else '—'} — {label}"
    )

    with target.expander(expander_title, expanded=False):
        score_left, rank_middle = st.columns(2)
        score_left.metric(
            "Overall score",
            (
                f"{overall_score}/10"
                if overall_score is not None
                else "N/A"
            ),
        )
        rank_middle.metric(
            "Ranking",
            rank if rank is not None else "N/A",
        )

        if email:
            st.markdown(f"📧 Email: **{email}**")
        else:
            st.caption("Email address unavailable.")

        # A candidate whose evaluation failed has evaluation=None plus
        # an error message. HR still sees the identity and a friendly
        # explanation; the technical error text is console-logged only
        # (it may reference internal providers/wording, never shown).
        if evaluation is None:
            st.warning(
                "No automated evaluation is available for this "
                "candidate.",
                icon="⚠️",
            )
            error_message = candidate.get("error")
            if error_message:
                print(
                    f"Candidate {candidate.get('candidate_id')} "
                    f"evaluation failed: {error_message}"
                )
                st.caption(
                    "This candidate could not be evaluated. Review the "
                    "available information before deciding."
                )
            return

        st.markdown("**Detailed Assessment**")
        st.write(evaluation.get("explanation") or "Not provided.")

        _render_bullet_section(
            st,
            "Relevant Experience",
            evaluation.get("relevant_experience"),
        )
        _render_bullet_section(
            st,
            "Strengths",
            evaluation.get("strengths"),
        )
        _render_bullet_section(
            st,
            "Skill Gaps",
            evaluation.get("skill_gaps"),
        )
        _render_bullet_section(
            st,
            "Missing Evidence",
            evaluation.get("missing_evidence"),
        )
        _render_bullet_section(
            st,
            "Supporting Resume Evidence",
            evaluation.get("supporting_quotes"),
        )


def display_candidate_results(
    candidates: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Display the top-N evaluated candidates for the results area.

    Args:
        candidates: All candidate dictionaries from the backend.
        top_n: How many ranked candidates to show (default 5).

    Returns:
        The list of candidates that was actually displayed, so callers
        (tests or later phases) can assert on the selection.
    """

    displayed_candidates = select_top_candidates(candidates, top_n)

    if not displayed_candidates:
        st.info(
            "No evaluated candidates to display yet. "
            "Start a recruitment run first.",
            icon="ℹ️",
        )
        return displayed_candidates

    for candidate in displayed_candidates:
        display_candidate_result(
            candidate,
            display_name=get_candidate_display_name(
                candidate,
                candidates,
            ),
        )

    # Failed evaluations are reported transparently, without hiding
    # them or inventing an assessment for them.
    failed_count = sum(
        1
        for candidate in candidates
        if candidate.get("rank") is None and candidate.get("error")
    )

    if failed_count:
        st.caption(
            f"{failed_count} pooled candidate(s) could not be "
            "evaluated and are not shown above."
        )

    return displayed_candidates
