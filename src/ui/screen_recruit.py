"""
Purpose:
    Screen 1 "New Recruitment" of TalentMatch AI: everything HR needs
    to provide a job description, launch the existing LangGraph
    candidate-discovery pipeline, and see the top-5 evaluated
    candidates.

What this file does:
    1. Lets HR supply the JD in ONE of two clearly separated ways:
       a file upload (PDF / TXT / DOCX) or pasted text. The existing
       input service (src/services/job_description_input_service.py)
       does all parsing/validation — nothing is duplicated here.
    2. Extracts the job title with the existing backend function and
       falls back to a manual text input when none is found (a title
       is never invented).
    3. Runs the workflow ONLY on the exact button press
       "Find relevant candidates" via start_recruitment_run() — never
       on Streamlit reruns, navigation or refreshes.
    4. Stores the returned paused state (top candidates, thread ID)
       in the session state and records the run start in the additive
       run history table.
    5. Renders the candidate results with the shared HR-facing
       display (actual names and emails, complete LLM assessment).

Interacts with:
    - src/ui/session.py (start_recruitment_run, cached graph)
    - src/ui/jd_display.py (ID -> name/email display)
    - src/services/job_description_input_service.py (JD validation)
    - src/services/run_history_service.py (additive run history)

Side effects:
    - Button press executes the real RAG + LLM workflow and records
      the run in recruitment_runs (idempotent on thread_id).
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.services import run_history_service
from src.services.job_description_input_service import (
    extract_job_title,
    load_job_description_from_file,
    load_job_description_from_text,
)
from src.ui import session as session_module
from src.ui.jd_display import display_candidate_results
from src.ui.components import SCREEN_HR_REVIEW, SCREEN_PAST_JDS

# Message shown for parser errors raised by the existing input service.
_AVAILABLE_EXTS = "PDF, TXT and DOCX"

# The two clearly separated JD input methods shown to HR.
INPUT_METHOD_UPLOAD = "Upload a file (PDF, TXT or DOCX)"
INPUT_METHOD_PASTE = "Paste job description text"


def _parse_uploaded_file(uploaded_file) -> str | None:
    """Extract and validate the text of an uploaded JD file.

    Streamlit gives us the file's CONTENT, but the existing input
    service selects its parser by file EXTENSION. So the upload is
    written to a temporary file with the original extension first,
    and the service is called with that path. The temporary file is
    deleted again immediately after parsing.

    Args:
        uploaded_file: A Streamlit UploadedFile (name + bytes).

    Returns:
        The cleaned JD text, or None when parsing failed (an error
        message is shown to HR instead of raising).
    """

    if uploaded_file is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".txt"

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            delete=False,
        ) as temp_handle:
            temp_handle.write(uploaded_file.getbuffer())
            temp_path = Path(temp_handle.name)

        try:
            return load_job_description_from_file(temp_path)
        finally:
            # The temporary copy is no longer needed once parsed.
            temp_path.unlink(missing_ok=True)
    except Exception as exc:
        # Includes DocumentParsingError and ValueError from the
        # existing service. HR sees a friendly message; the technical
        # detail goes into console logs only.
        print(f"JD file parsing failed: {exc}")
        st.error(
            f"Could not read the job description file "
            f"({uploaded_file.name}). Supported formats are "
            f"{_AVAILABLE_EXTS} and the file must contain extractable "
            "text (a scanned PDF cannot be parsed).",
            icon="📄",
        )
        return None


def _render_jd_input() -> tuple[str | None, str | None]:
    """Render the JD input block and return (validated text, source).

    Only ONE input method can be active because the two branches are
    mutually exclusive (radio-controlled), which is what prevents
    conflicting inputs. The returned text has already passed the
    existing validation; None means "nothing valid provided yet".
    """

    st.markdown("#### Job Description")

    input_method = st.radio(
        "Job description source",
        [INPUT_METHOD_UPLOAD, INPUT_METHOD_PASTE],
        key="jd_input_method",
        horizontal=True,
        help=(
            "Only one input method is active at a time, so conflicting "
            "inputs cannot be submitted together."
        ),
    )

    jd_text: str | None = None
    jd_source: str | None = None

    if input_method == INPUT_METHOD_UPLOAD:
        uploaded_file = st.file_uploader(
            "Upload a job description file",
            type=["pdf", "txt", "docx"],
            key="jd_file_uploader",
        )

        if uploaded_file is not None:
            jd_text = _parse_uploaded_file(uploaded_file)
            jd_source = uploaded_file.name

            if jd_text:
                st.caption(f"Using uploaded file: {uploaded_file.name}")
                with st.expander("Preview the parsed job description"):
                    st.write(jd_text[:2000])
    else:
        pasted_text = st.text_area(
            "Paste the complete job description",
            height=320,
            key="jd_paste_area",
            placeholder=(
                "Paste the full job description here, including the "
                "job title, responsibilities and required skills."
            ),
        )

        if pasted_text and pasted_text.strip():
            try:
                jd_text = load_job_description_from_text(pasted_text)
                jd_source = "pasted text"

                st.caption("Pasted job description validated.")
                with st.expander("Preview the job description"):
                    st.write(jd_text[:2000])
            except ValueError as exc:
                # Empty/whitespace-only text: show a friendly error.
                st.error(f"Invalid job description text: {exc}", icon="✏️")

    return jd_text, jd_source


def _render_job_title_step(jd_text: str | None) -> str | None:
    """Render the job-title block and return the resolved title.

    The title is auto-extracted with the existing backend function.
    Only when extraction fails is a manual text input shown — and
    whatever HR typed is used as-is. Nothing is invented.

    Args:
        jd_text: The validated JD text (or None when not provided).

    Returns:
        The non-empty job title to use, or None when still missing
        (the start button stays disabled in that case).
    """

    st.markdown("#### Job Title")

    if not jd_text:
        st.caption("Provide a job description first.")
        return None

    extracted_title = extract_job_title(jd_text)

    if extracted_title:
        st.caption(
            f"Auto-extracted from the job description: "
            f"**{extracted_title}**"
        )
        job_title = st.text_input(
            "Job title (edit if required)",
            value=extracted_title,
            key="job_title_input",
        )
    else:
        st.caption(
            "The job title could not be extracted automatically. "
            "Please enter it manually — it is used for candidate "
            "emails and tracking."
        )
        job_title = st.text_input(
            "Job title (enter manually)",
            key="job_title_manual_input",
        )

    return job_title.strip() if job_title else None


def _store_run_results(
    run_thread_id: str,
    paused_state: dict,
    job_title: str,
    jd_source: str | None,
) -> bool:
    """Store one finished discovery invocation into the session state.

    Args:
        run_thread_id: The thread ID the run used.
        paused_state: The graph's returned state (paused or ended).
        job_title: Job title of this run.
        jd_source: JD source label of this run.

    Returns:
        True when discovery finished with candidates awaiting HR
        approval; False for any other completion (e.g. no candidates).
    """

    candidates = paused_state.get("candidates") or []
    interrupts = paused_state.get("__interrupt__") or []
    has_interrupt = bool(interrupts)

    st.session_state["thread_id"] = run_thread_id
    st.session_state["job_title"] = job_title
    st.session_state["jd_source"] = jd_source
    st.session_state["candidate_results"] = candidates
    st.session_state["hr_decisions"] = {}
    st.session_state["hr_decision_submitted"] = False
    st.session_state["email_results"] = []

    if has_interrupt:
        st.session_state["workflow_status"] = (
            paused_state.get("status") or "awaiting_hr_approval"
        )
        st.session_state["workflow_message"] = paused_state.get("message")
        st.session_state["workflow_completed"] = False
    else:
        # The run ended without pausing (no / not enough candidates).
        st.session_state["workflow_status"] = (
            paused_state.get("status") or "no_candidates_found"
        )
        st.session_state["workflow_message"] = paused_state.get("message")
        st.session_state["workflow_completed"] = True

        # Nothing to review, so this run's history row is completed
        # immediately (zero candidates, no emails).
        try:
            run_history_service.record_run_completion(
                run_thread_id,
                job_title,
                decisions={},
                presented_candidates=[],
                email_results=[],
                final_status="no_candidates_found",
            )
        except Exception as exc:
            # History problems never block the UI.
            print(f"Run history completion could not be recorded: {exc}")

    st.session_state["run_metadata"] = {
        "started_at": paused_state.get("started_at"),
        "jd_source": jd_source,
        "job_title": job_title,
        "candidate_count": len(candidates),
    }

    return has_interrupt


def _handle_find_candidates(
    jd_text: str,
    job_title: str,
    jd_source: str | None,
) -> None:
    """Execute the discovery workflow for one button press.

    Called ONLY from the "Find relevant candidates" button handler,
    which is what guarantees Streamlit reruns can never re-run RAG or
    LLM evaluation by themselves.

    Args:
        jd_text: Validated non-empty JD text.
        job_title: Non-empty job title.
        jd_source: File name or "pasted text" (for run history).
    """

    # HR-friendly processing feedback: a spinner with a plain summary
    # line plus a short explanation caption. The real technical
    # details (RAG retrieval, LLM evaluation, fallbacks) stay inside
    # the backend and console logs; no artificial progress percentage
    # is shown because the backend does not report progress.
    with st.spinner("Finding relevant candidates..."):
        st.caption(
            "We're reviewing the job requirements and identifying "
            "candidates whose experience and skills match the "
            "position. This can take a few minutes."
        )
        try:
            run_result = session_module.start_recruitment_run(
                jd_text,
                job_title,
            )
        except ValueError as exc:
            st.error(
                f"Candidate discovery could not start: {exc}",
                icon="⛔",
            )
            return
        except Exception as exc:
            # Covers retrieval/evaluation failures. The technical
            # message goes to console logs only — never to HR, so no
            # internal provider or fallback wording is exposed.
            print(f"Candidate discovery failed: {exc}")
            st.error(
                "Candidate discovery failed. Please try again in a "
                "moment; if the problem persists, contact your "
                "administrator.",
                icon="⛔",
            )
            return

    paused_state = run_result["state"]
    run_thread_id = run_result["thread_id"]

    # Persist the run start BEFORE anything else, so the dashboard
    # records that this run happened. Idempotent on thread_id, so an
    # accidental re-record cannot create duplicate rows.
    candidates = paused_state.get("candidates") or []
    try:
        run_history_service.record_run_started(
            run_thread_id,
            job_title,
            jd_source,
            candidate_count=len(candidates),
        )
    except Exception as exc:
        print(f"Run history start could not be recorded: {exc}")

    awaiting_review = _store_run_results(
        run_thread_id,
        paused_state,
        job_title,
        jd_source,
    )

    if awaiting_review:
        # The candidates are already stored in the session state. A
        # rerun lets the whole screen render cleanly from it.
        st.rerun()

    # No-candidate runs finish immediately, so the message is shown
    # INLINE (a rerun would discard it, because Streamlit status
    # messages do not survive st.rerun()).
    st.info(
        paused_state.get("message")
        or "The recruitment run finished, but no evaluated candidates "
        "were found for this job description. The run is recorded in "
        "the dashboard history with zero candidates.",
        icon="ℹ️",
    )


def _navigate_to(next_screen: str):
    """Build a button callback that switches to another screen.

    The callback runs BEFORE Streamlit's automatic rerun, which is the
    only safe way to change the navigation radio's session key
    (changing a widget's session value after the widget has been
    instantiated in the same run is not allowed).

    Args:
        next_screen: One of the screen-name constants.

    Returns:
        The no-argument callback to pass to st.button(on_click=...).
    """

    def _callback() -> None:
        st.session_state["current_screen"] = next_screen

    return _callback


def _render_results_area() -> None:
    """Render the results section from session state (safe on reruns).

    Nothing here touches the backend — it only reads what
    _store_run_results saved after the one allowed invocation.
    """

    st.markdown("#### Candidate Results")

    candidate_results = st.session_state.get("candidate_results") or []
    workflow_completed = st.session_state.get("workflow_completed")
    hr_decision_submitted = st.session_state.get("hr_decision_submitted")

    if not candidate_results:
        if (
            st.session_state.get("thread_id")
            and st.session_state.get("workflow_completed")
        ):
            # A finished run that found no candidates: report it from
            # the stored state so it also survives later reruns.
            st.info(
                st.session_state.get("workflow_message")
                or "No evaluated candidates were found for this job "
                "description.",
                icon="ℹ️",
            )
        st.write(
            "After discovery, the top 5 evaluated candidates will be "
            "shown here with their name, email, score, rank and the "
            "complete assessment."
        )
        st.caption(
            "Start a run to continue to the HR review stage."
        )
        return

    # The top 5 evaluated candidates, exactly as ranked by the
    # automated evaluation.
    display_candidate_results(candidate_results, top_n=5)

    st.markdown("#### Shortlisting")

    if workflow_completed or hr_decision_submitted:
        st.success(
            "This run is finished. See the decisions and email statuses "
            "on the HR Review screen, and the history on the Past JDs "
            "screen.",
            icon="✅",
        )
        if st.button(
            "Go to Past JDs",
            key="results_goto_dashboard_button",
            on_click=_navigate_to(SCREEN_PAST_JDS),
        ):
            # The navigation itself happens in the on_click callback.
            pass
    else:
        st.info(
            "This run is awaiting HR approval. Open the "
            "**HR Review** screen to accept or reject candidates and "
            "send the shortlist emails.",
            icon="⏳",
        )
        if st.button(
            "Go to HR Review",
            key="results_goto_review_button",
            on_click=_navigate_to(SCREEN_HR_REVIEW),
        ):
            # The navigation itself happens in the on_click callback.
            pass


def render_new_recruitment_screen() -> None:
    """Render the complete New Recruitment screen."""

    st.subheader("New Recruitment")

    # Show the context of the current/last run so HR always knows
    # which recruitment they are looking at.
    current_thread = st.session_state.get("thread_id")
    if current_thread:
        with st.expander("Current run context", expanded=False):
            st.write(f"Thread: {current_thread}")
            st.write(f"Job title: {st.session_state.get('job_title') or '—'}")
            st.write(f"JD source: {st.session_state.get('jd_source') or '—'}")
            st.caption(
                "Starting a new run below creates a NEW thread and "
                "never reuses this one."
            )

    jd_text, jd_source = _render_jd_input()

    job_title = _render_job_title_step(jd_text)

    st.markdown("#### Find Relevant Candidates")

    ready_to_start = bool(jd_text and job_title)

    if not ready_to_start:
        st.caption(
            "Provide a job description and a job title to enable "
            "candidate discovery."
        )

    # The exact action label required for this screen. The workflow
    # runs ONLY inside this button's if-block, never during reruns.
    if st.button(
        "Find relevant candidates",
        disabled=not ready_to_start,
        type="primary",
        width="stretch",
        help=(
            "Searches the candidate pool and evaluates how well each "
            "candidate matches the job requirements."
            if ready_to_start
            else "Provide the JD and job title first."
        ),
    ):
        _handle_find_candidates(jd_text, job_title, jd_source)

    _render_results_area()
