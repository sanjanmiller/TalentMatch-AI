"""
Purpose:
    Persist and retrieve COMPLETED recruitment-run history for the
    TalentMatch AI dashboard.

Why this file exists:
    The existing `email_tracking` table is keyed by
    (candidate_id, job_title) and records only email activity. It cannot
    answer questions such as "how many candidates were evaluated in run
    X?" or "which candidates were presented to HR in run Y?". This
    module adds ONE additive SQLite table (`recruitment_runs`) in the
    SAME database the other services already use, plus a small set of
    read/write helpers used by the Streamlit screens.

Design rules obeyed here:
    - ADDITIVE ONLY: nothing in the existing schema is changed; the new
      table is created with CREATE TABLE IF NOT EXISTS, so existing
      rows and databases are never touched.
    - IDEMPOTENT WRITES: a run start is inserted with
      ON CONFLICT(thread_id) DO NOTHING and the completion is an
      UPDATE, so a Streamlit rerun can never create duplicate history
      rows.
    - PARAMETERIZED SQL only; no string-built SQL anywhere.
    - NOTHING IS FABRICATED: names, emails, scores and email statuses
      all come from the graph state, HR's decisions, and the existing
      email_tracking table. Historical rows are immutable through the
      UI (this module never deletes or rewrites completed runs).

Interacts with:
    - src/services/db_service.py (get_connection for SQLite access)
    - src/services/email_tracking_service.py (SMTP accepted-at lookup,
      so the dashboard can label timestamps exactly as tracked)

This service does not start runs, send emails, or evaluate candidates.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services import db_service
from src.services.email_tracking_service import (
    NOT_SHORTLISTED,
    SHORTLISTED_MAIL_NOT_SENT,
    SHORTLISTED_MAIL_SENT,
    get_email_tracking_records,
)

# Reuse the shared display fallback so the dashboard never invents a
# name for a candidate whose identity could not be resolved.
IDENTITY_UNAVAILABLE = "Candidate identity unavailable"

# What is stored in recruitment_runs.results_json: one snapshot entry
# per candidate that was presented for HR review. Kept as JSON so the
# whole shortlist (assessments, decisions, email outcomes) survives in
# a single row without depending on other tables' keys.
RESULT_SNAPSHOT_COLUMNS = (
    "candidate_id",
    "name",
    "email",
    "rank",
    "overall_score",
    "decision",
    "email_status",
    "smtp_accepted_at",
)


def initialize_run_history_table(
    db_path: Path | str | None = None,
) -> None:
    """Create the additive `recruitment_runs` table if it is missing.

    Safe to call repeatedly; existing rows and existing tables are
    never modified or deleted.

    Args:
        db_path: Optional alternative database path (used by tests to
            run against a temporary database). Defaults to the shared
            storage/database/recruitment.db.
    """

    with db_service.get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recruitment_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL UNIQUE,
                job_title TEXT NOT NULL,
                jd_source TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                final_status TEXT,
                candidate_count INTEGER DEFAULT 0,
                approved_count INTEGER DEFAULT 0,
                rejected_count INTEGER DEFAULT 0,
                results_json TEXT DEFAULT '[]'
            )
            """
        )


def record_run_started(
    thread_id: str,
    job_title: str,
    jd_source: str | None,
    candidate_count: int,
    db_path: Path | str | None = None,
) -> None:
    """Insert one history row when a NEW recruitment run starts.

    The insert is deliberately idempotent: if the same thread_id is
    recorded twice (for example by an accidental Streamlit rerun), the
    second insert is silently ignored instead of creating a duplicate
    history row.

    Args:
        thread_id: The unique LangGraph thread ID of the run.
        job_title: The HR-facing job title for this run.
        jd_source: Where the JD came from ("pasted text" or a file
            name), stored for the history display.
        candidate_count: How many candidates were presented for review.
        db_path: Optional alternative database path for tests.

    Raises:
        ValueError: When thread_id or job_title is empty.
    """

    if not (thread_id or "").strip():
        raise ValueError("A thread_id is required to record a run.")

    if not (job_title or "").strip():
        raise ValueError("A job title is required to record a run.")

    created_at = datetime.now(timezone.utc).isoformat()

    # Make sure the additive table exists even if the caller forgot to
    # initialize it explicitly first.
    initialize_run_history_table(db_path)

    with db_service.get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO recruitment_runs (
                thread_id, job_title, jd_source, created_at,
                candidate_count
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO NOTHING
            """,
            (
                thread_id.strip(),
                job_title.strip(),
                jd_source,
                created_at,
                int(candidate_count),
            ),
        )


def record_run_completion(
    thread_id: str,
    job_title: str,
    *,
    decisions: dict[str, bool],
    presented_candidates: list[dict],
    email_results: list[dict] | None = None,
    final_status: str = "completed",
    completed_at: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update a run's history row once HR's decisions have been applied.

    Called exactly once per run, right after the interrupted LangGraph
    graph was resumed. The update (not insert) makes a repeated call —
    for instance a Streamlit rerun — idempotent: the same row is
    rewritten with the same values instead of duplicating history.

    Args:
        thread_id: The thread ID of this run (the row created earlier
            by record_run_started).
        job_title: Job title used for the email_tracking lookup that
            resolves the SMTP accepted-at timestamps.
        decisions: candidate_id -> True(approved) / False(rejected).
        presented_candidates: The candidate dictionaries that were
            shown to HR (as embedded by the retrieval node).
        email_results: The graph's `email_results` list, where each
            entry is {candidate_id, status: sent|failed|skipped,
            message}.
        final_status: Overall outcome label ("emails_sent",
            "rejected_no_emails", "no_candidates_found", ...).
        completed_at: Optional ISO timestamp override (tests).
        db_path: Optional alternative database path for tests.

    Raises:
        ValueError: When thread_id or job_title is empty.
    """

    if not (thread_id or "").strip():
        raise ValueError("A thread_id is required to record a run.")

    if not (job_title or "").strip():
        raise ValueError("A job title is required to record a run.")

    completed_at = completed_at or datetime.now(timezone.utc).isoformat()

    results_snapshot = _build_results_snapshot(
        job_title=job_title,
        decisions=decisions,
        presented_candidates=presented_candidates,
        email_results=email_results or [],
        db_path=db_path,
    )

    approved_count = sum(
        1 for approved in decisions.values() if approved
    )
    rejected_count = sum(
        1 for approved in decisions.values() if not approved
    )
    candidate_count = len(presented_candidates)

    initialize_run_history_table(db_path)

    with db_service.get_connection(db_path) as connection:
        connection.execute(
            """
            UPDATE recruitment_runs
            SET completed_at = ?,
                final_status = ?,
                candidate_count = ?,
                approved_count = ?,
                rejected_count = ?,
                results_json = ?
            WHERE thread_id = ?
            """,
            (
                completed_at,
                final_status,
                candidate_count,
                approved_count,
                rejected_count,
                json.dumps(results_snapshot, ensure_ascii=False),
                thread_id.strip(),
            ),
        )


def _resolve_candidate_identity(
    candidate: dict,
    db_path: Path | str | None = None,
) -> tuple[str, str | None]:
    """Return (name, email) for a candidate result dictionary.

    The retrieval node embeds `candidate_name` and `email`, which are
    preferred. For raw dictionaries carrying only a candidate_id, the
    existing identity lookup is the fallback. The shared
    IDENTITY_UNAVAILABLE message is used when nothing is known — a
    name is never invented.

    Args:
        candidate: One candidate dictionary from the graph state.
        db_path: Optional database path for the identity fallback.

    Returns:
        Tuple of (display name, email or None).
    """

    candidate_id = (candidate.get("candidate_id") or "").strip()
    name = candidate.get("candidate_name")
    email = candidate.get("email")

    if (not name or not email) and candidate_id:
        identity = db_service.get_candidate_identity(
            candidate_id,
            db_path,
        ) or {}

        if not name:
            name = identity.get("full_name")
        if not email:
            email = identity.get("email")

    if not name:
        name = IDENTITY_UNAVAILABLE

    return name, (email or None)


def _build_results_snapshot(
    job_title: str,
    decisions: dict[str, bool],
    presented_candidates: list[dict],
    email_results: list[dict],
    db_path: Path | str | None = None,
) -> list[dict]:
    """Build the per-candidate JSON snapshot stored with the run.

    Email status labels mirror the tracked email word-for-word from the
    existing email_tracking service, and the SMTP accepted-at value is
    taken from the tracking record itself — never generated here.
    Rejects are labelled 'Not Shortlisted' (no email attempted).

    Args:
        job_title: Job title used when emails were (or would be) sent.
        decisions: candidate_id -> approved True/False.
        presented_candidates: Candidate dicts presented to HR.
        email_results: Graph email outcome entries.
        db_path: Optional database path for tests.

    Returns:
        A list of snapshot dictionaries (see RESULT_SNAPSHOT_COLUMNS).
    """

    # Index the graph's email outcomes by candidate ID.
    email_status_by_id: dict[str, str] = {
        result.get("candidate_id"): result.get("status")
        for result in email_results
        if result.get("candidate_id")
    }

    # SMTP accepted-at timestamps come straight from the EXISTING
    # tracking table for (candidate_id, job_title) rows that were
    # actually accepted by SMTP.
    smtp_accepted_by_id: dict[str, str] = {}
    for record in get_email_tracking_records(db_path):
        is_sent = record.get("status") == SHORTLISTED_MAIL_SENT
        is_this_job = record.get("job_title") == job_title
        if is_sent and is_this_job and record.get("candidate_id"):
            smtp_accepted_by_id[record["candidate_id"]] = (
                record.get("updated_at")
            )

    snapshot: list[dict] = []

    for candidate in presented_candidates:
        candidate_id = (candidate.get("candidate_id") or "").strip()
        name, email = _resolve_candidate_identity(candidate, db_path)

        approved = decisions.get(candidate_id)
        raw_status = email_status_by_id.get(candidate_id)

        if approved is True:
            if raw_status == "sent":
                email_status = SHORTLISTED_MAIL_SENT
            elif raw_status == "failed":
                email_status = SHORTLISTED_MAIL_NOT_SENT
            elif raw_status == "skipped":
                # Skipped means the existing UNIQUE(candidate_id,
                # job_title) safeguard found a previous tracking row.
                email_status = (
                    "Shortlisted – Mail Not Sent (skipped - "
                    "duplicate prevention)"
                )
            else:
                email_status = (
                    "Shortlisted – Mail Not Sent "
                    "(no email attempt recorded)"
                )
            smtp_accepted_at = smtp_accepted_by_id.get(candidate_id)
        elif approved is False:
            email_status = NOT_SHORTLISTED
            smtp_accepted_at = None
        else:
            email_status = "No decision recorded"
            smtp_accepted_at = None

        snapshot.append(
            {
                "candidate_id": candidate_id,
                "name": name,
                "email": email,
                "rank": candidate.get("rank"),
                "overall_score": candidate.get("overall_score"),
                "decision": (
                    "Approved"
                    if approved is True
                    else "Rejected" if approved is False else None
                ),
                "email_status": email_status,
                "smtp_accepted_at": smtp_accepted_at,
            }
        )

    # Rank order (1 first) keeps the dashboard readable; unranked
    # (failed) evaluations keep their presentation order at the end.
    def _rank_key(entry: dict) -> tuple[int, int]:
        return (
            0 if entry["rank"] is not None else 1,
            entry["rank"] if entry["rank"] is not None else 0,
        )

    snapshot.sort(key=_rank_key)

    return snapshot


def get_recruitment_runs(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return all recorded runs, newest first.

    Args:
        db_path: Optional alternative database path for tests.

    Returns:
        A list of run dictionaries (run_id, thread_id, job_title,
        jd_source, created_at, completed_at, final_status and the
        count columns) sorted by run_id descending.
    """

    initialize_run_history_table(db_path)

    with db_service.get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                run_id,
                thread_id,
                job_title,
                jd_source,
                created_at,
                completed_at,
                final_status,
                candidate_count,
                approved_count,
                rejected_count
            FROM recruitment_runs
            ORDER BY run_id DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_run_candidates(
    run_id: int,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return the per-candidate snapshot of one historical run.

    The stored JSON is parsed and every expected column is guaranteed
    to exist, so the dashboard can render missing values as blanks
    without KeyError risk.

    Args:
        run_id: Primary key of the run row (get_recruitment_runs).
        db_path: Optional alternative database path for tests.

    Returns:
        The parsed candidate snapshot list for that run (empty when
        the run is unknown or completion was not recorded yet).
    """

    initialize_run_history_table(db_path)

    with db_service.get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT results_json
            FROM recruitment_runs
            WHERE run_id = ?
            """,
            (int(run_id),),
        ).fetchone()

    if row is None or not row["results_json"]:
        return []

    try:
        stored_results = json.loads(row["results_json"])
    except (TypeError, ValueError):
        # Corrupt/unreadable JSON is never shown silently — returns
        # an empty snapshot so the UI can display a clear warning.
        return []

    if not isinstance(stored_results, list):
        return []

    complete_results: list[dict[str, Any]] = []
    for entry in stored_results:
        if not isinstance(entry, dict):
            continue
        complete_entry = {
            column: entry.get(column)
            for column in RESULT_SNAPSHOT_COLUMNS
        }
        complete_results.append(complete_entry)

    return complete_results
