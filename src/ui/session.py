"""
Purpose:
    Shared Streamlit session-state and backend-plumbing utilities for
    TalentMatch AI.

What this file does:
    - Initialises Streamlit session state safely (never overwriting
      values that already exist between reruns).
    - Generates unique LangGraph thread IDs, one per recruitment run.
    - Creates the recruitment graph wired to the EXISTING persistent
      SQLite checkpointer (storage/database/checkpoints.sqlite3), so an
      interrupted run can always be resumed in a later phase.
    - Provides `start_recruitment_run()`, the ONE approved way for the
      UI to start the existing recruitment workflow.

Role in the overall recruitment workflow:
    The UI must use `build_recruitment_graph(checkpointer)` (not the
    RecruitmentWorkflow service directly, which would bypass the HR
    approval interrupt and the email workflow). This module is the
    only place where the UI builds that graph.

Interacts with:
    - src/graph/workflow.py (build_recruitment_graph)
    - langgraph SqliteSaver (persistent checkpoint storage)
    - src/services/job_description_input_service.py (JD validation
      functions are used by the calling screens, not here)

Side effects:
    - `create_recruitment_graph()` opens the SQLite checkpoint
      database and may create it if it does not exist yet.
"""

import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from src.graph.workflow import build_recruitment_graph


# Locate the project root without depending on the terminal's working
# directory (same pattern as the other services).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The EXISTING LangGraph checkpoint database. A paused run can survive
# a Streamlit rerun or even a process restart because its state is
# stored here, not in memory.
CHECKPOINT_DATABASE_PATH = (
    PROJECT_ROOT / "storage" / "database" / "checkpoints.sqlite3"
)


# Default value for every Streamlit session-state key the UI uses.
# setdefault() is used for initialisation, so these defaults are ONLY
# applied when the key does not already exist (Streamlit reruns keep
# their values; user selections are never reset).
DEFAULT_SESSION_STATE: dict[str, Any] = {
    # Sidebar navigation: which screen is currently visible.
    "current_screen": "New Recruitment",
    # LangGraph thread identifier of the current recruitment run.
    # None means no run has been started yet.
    "thread_id": None,
    # HR-facing job title for the current run (auto-extracted or typed).
    "job_title": None,
    # Source description of the JD (file name or "pasted text").
    "jd_source": None,
    # Candidate result dictionaries produced by the retrieval node.
    "candidate_results": [],
    # Workflow status/message from the latest graph invocation.
    "workflow_status": None,
    "workflow_message": None,
    # HR's provisional decisions: candidate_id -> True/False.
    "hr_decisions": {},
    # Whether the final HR decision has already been sent to the graph.
    # This prevents accidental double resume (and double email sends).
    "hr_decision_submitted": False,
    # Whether the current run reached its end (emails sent/rejections
    # recorded). Distinguishes "awaiting HR approval" from "finished".
    "workflow_completed": False,
    # Email outcome entries from the graph after the run is finished:
    # list of {candidate_id, status: sent|failed|skipped, message}.
    "email_results": [],
    # Extra metadata of the current run (started_at, counts, ...).
    "run_metadata": {},
}


def initialize_session_state(
    state: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Initialise the UI session state without overwriting values.

    Args:
        state: The mapping to initialise. Defaults to Streamlit's
            `st.session_state`. A plain dict can be supplied in tests.

    Returns:
        The same mapping, now guaranteed to contain every UI key with
        its default value when it was missing.

    Side effects:
        Writes missing keys into `st.session_state` when used in app
        context. Existing values (user selections, a running thread ID)
        are preserved; this is why setdefault() is used and never
        plain assignment.
    """

    target_state = st.session_state if state is None else state

    for key, default_value in DEFAULT_SESSION_STATE.items():
        target_state.setdefault(key, default_value)

    return target_state


def _sanitize_thread_slug(source: str | None) -> str:
    """Turn a JD title/file name into a short, safe URL-like slug."""

    if not source:
        return ""

    # Keep letters/digits/hyphens; replace everything else with a
    # hyphen so the thread ID stays readable and valid.
    slug = re.sub(r"[^A-Za-z0-9]+", "-", source).strip("-").lower()

    return slug[:40]


def new_thread_id(source: str | None = None) -> str:
    """Create a unique LangGraph thread ID for one recruitment run.

    Every NEW run gets a fresh thread ID so that it resolves its own
    HR approval interrupt and never touches a previous run's paused
    workflow. Resuming an existing run must reuse the SAME thread ID
    (stored in session state), never a newly generated one.

    Args:
        source: Optional job title or JD file name, used to make the
            thread ID human-readable.

    Returns:
        A unique thread ID such as
        `talentmatch-junior-data-scientist-20261103-153000-ab12cd34`.
    """

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Suffix from a random UUID guarantees uniqueness even when two
    # runs are started within the same second.
    random_suffix = uuid.uuid4().hex[:8]

    slug = _sanitize_thread_slug(source)

    if slug:
        return f"talentmatch-{slug}-{timestamp}-{random_suffix}"

    return f"talentmatch-{timestamp}-{random_suffix}"


def create_recruitment_graph(
    db_path: str | Path | None = None,
) -> Any:
    """Build the recruitment graph on the persistent SQLite saver.

    The graph is the EXISTING LangGraph workflow (retrieval -> HR
    approval interrupt -> email sending). The SQLite checkpointer
    saves the graph state, so the interrupt can be resumed later even
    after Streamlit reruns.

    Args:
        db_path: Optional alternative checkpoint database path. Tests
            can supply a temporary path; the app uses the existing
            storage/database/checkpoints.sqlite3.

    Returns:
        The compiled LangGraph app (CompiledStateGraph).

    Side effects:
        Opens (and creates if needed) the checkpoint database and runs
        the saver's setup() once, which creates its tables safely.

    Connection lifecycle notes:
        - check_same_thread=False: Streamlit may touch the session
          from different threads between reruns.
        - The sqlite3 connection is held open by the SqliteSaver, so
          the caller does not need to close it while the app runs.
    """

    checkpoint_path = (
        Path(db_path) if db_path else CHECKPOINT_DATABASE_PATH
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(checkpoint_path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    # setup() creates the saver's tables the first time it is used,
    # and is safe to call repeatedly.
    checkpointer.setup()

    return build_recruitment_graph(checkpointer)


@st.cache_resource(show_spinner=False)
def get_cached_recruitment_graph() -> Any:
    """Return the shared recruitment graph, building it once per app.

    st.cache_resource keeps ONE graph + SQLite connection across
    Streamlit reruns and user interactions, so the app does not open
    a new checkpoint database connection on every rerun. Building
    the graph does NOT call any LLM; it only wires the checkpointer
    and graph nodes.
    """

    return create_recruitment_graph()


def start_recruitment_run(
    job_description: str,
    job_title: str,
    *,
    graph: Any = None,
    thread_id: str | None = None,
    candidate_limit: int = 8,
) -> dict[str, Any]:
    """Start ONE recruitment run on the existing LangGraph workflow.

    IMPORTANT:
        This wrapper must only be called from an explicit user action
        (for example a button press). It must NEVER be called
        automatically during a Streamlit rerun, because Streamlit
        reruns the script on every widget interaction and that would
        silently re-run RAG retrieval and LLM evaluation.

    The resumed HR approval flow is NOT part of this function; it is
    implemented in a later phase.

    Args:
        job_description: The validated, non-empty JD text.
        job_title: The non-empty job title used for emails/tracking.
        graph: Pre-built LangGraph app. Defaults to the cached shared
            graph in app context; tests supply a mock.
        thread_id: Optional fixed thread ID (used by tests). A NEW
            unique thread ID is generated when omitted.
        candidate_limit: Candidate pool size; the backend default is 8.

    Returns:
        Dictionary with:
            "thread_id": the thread ID used for this run,
            "state": the graph's returned paused/ended state dict
                (contains "__interrupt__" when HR approval is pending).

    Raises:
        ValueError: When any input is empty or invalid. The caller
            must show the message to HR instead of silently
            continuing.
    """

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    if not isinstance(job_title, str) or not job_title.strip():
        raise ValueError("Job title cannot be empty.")

    if (
        isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit < 1
    ):
        raise ValueError(
            "Candidate limit must be a positive integer."
        )

    if graph is None:
        graph = get_cached_recruitment_graph()

    if thread_id is None:
        thread_id = new_thread_id(job_title)

    # Verified backend contract (src/graph/nodes.py + tests):
    #   - job_description and job_title are strings,
    #   - candidate_limit selects the pool size (default 8),
    #   - the thread_id keeps a paused run resumable later on.
    initial_state = {
        "job_description": job_description.strip(),
        "job_title": job_title.strip(),
        "candidate_limit": candidate_limit,
    }

    config = {"configurable": {"thread_id": thread_id}}

    paused_state = graph.invoke(initial_state, config=config)

    return {
        "thread_id": thread_id,
        "state": paused_state,
    }


def resume_with_hr_decision(
    decisions: dict[str, bool],
    *,
    graph: Any = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Resume an interrupted run with HR's final decisions.

    HOW THE EXISTING INTERRUPT/RESUME CONTRACT IS HONOURED:

        The paused graph is waiting inside `interrupt()` (see
        src/graph/nodes.py). To continue it, the graph must be invoked
        again with a `Command(resume=...)` payload AND the SAME thread
        ID it was paused with, so the SQLite checkpointer can restore
        the saved state instead of starting over. That is what this
        function does; no RAG retrieval or LLM evaluation is repeated.

        The verified backend contract for the resume payload:
            {"decision": "approve" | "reject",
             "selected_candidate_ids": [candidate_id, ...]}
        - "approve" emails the selected candidates,
        - "reject" (empty selection) marks every candidate as
          'Not Shortlisted' and sends nothing.

    Args:
        decisions: candidate_id -> True (approved) / False (rejected),
            exactly as captured by the HR Review screen widgets.
        graph: Pre-built LangGraph app. Defaults to the cached shared
            graph in app context; tests supply a mock.
        thread_id: The thread ID the run was PAUSED with (stored in
            session state after start_recruitment_run).

    Returns:
        Dictionary with:
            "thread_id": the same thread ID that was passed in,
            "state": the graph's FINAL state (contains "status",
                "message" and "email_results").

    Raises:
        ValueError: When the thread ID is missing or no decisions
            were supplied. The caller (HR Review screen) must validate
            before calling and must guard against double submission
            via the "hr_decision_submitted" session flag; this
            function intentionally stays free of Streamlit state so
            it can be unit tested.
    """

    if not isinstance(decisions, dict) or not decisions:
        raise ValueError("No HR decisions were supplied.")

    if thread_id is None or not str(thread_id).strip():
        raise ValueError(
            "A thread ID is required to resume a paused run."
        )

    # Only explicitly APPROVED candidates may reach the email node.
    approved_candidate_ids = [
        candidate_id
        for candidate_id, approved in decisions.items()
        if approved
    ]

    if approved_candidate_ids:
        resume_payload = {
            "decision": "approve",
            "selected_candidate_ids": approved_candidate_ids,
        }
    else:
        resume_payload = {
            "decision": "reject",
            "selected_candidate_ids": [],
        }

    if graph is None:
        graph = get_cached_recruitment_graph()

    # Invoking with Command(resume=...) restores the paused state for
    # this thread (that is the LangGraph interrupt/resume mechanism)
    # and runs only the remaining nodes: decision validation and the
    # email workflow.
    config = {"configurable": {"thread_id": thread_id}}

    final_state = graph.invoke(Command(resume=resume_payload), config=config)

    return {
        "thread_id": thread_id,
        "state": final_state,
    }
