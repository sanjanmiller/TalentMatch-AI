"""Track candidate shortlisting and email status in the SQLite database.

This service keeps HR-facing tracking separate from the recruitment workflow
and avoids automatically retrying emails with an uncertain previous outcome.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.services import db_service


SHORTLISTED_MAIL_SENT = "Shortlisted – Mail Sent"
SHORTLISTED_MAIL_NOT_SENT = "Shortlisted – Mail Not Sent"
NOT_SHORTLISTED = "Not Shortlisted"


def _current_timestamp() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def record_candidate_status(
    candidate_id: str,
    candidate_name: str,
    job_title: str,
    recipient_email: str | None,
    status: str,
    db_path: Path | str | None = None,
) -> None:
    """Insert or update a candidate's tracking status for a job."""
    allowed_statuses = {
        SHORTLISTED_MAIL_SENT,
        SHORTLISTED_MAIL_NOT_SENT,
        NOT_SHORTLISTED,
    }

    if status not in allowed_statuses:
        raise ValueError(f"Unsupported tracking status: {status}")

    now = _current_timestamp()

    with db_service.get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO email_tracking (
                candidate_id,
                candidate_name,
                job_title,
                recipient_email,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id, job_title)
            DO UPDATE SET
                candidate_name = excluded.candidate_name,
                recipient_email = excluded.recipient_email,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                candidate_id,
                candidate_name,
                job_title,
                recipient_email,
                status,
                now,
                now,
            ),
        )


def prepare_shortlisting_email(
    candidate_id: str,
    candidate_name: str,
    job_title: str,
    recipient_email: str,
    db_path: Path | str | None = None,
) -> bool:
    """Mark a candidate as awaiting email and decide whether to send.

    Returns True only when a new send attempt is allowed.
    Existing 'Mail Sent' and 'Mail Not Sent' records are not automatically
    retried, because a previous attempt may already have reached the recipient.
    """
    now = _current_timestamp()

    with db_service.get_connection(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")

        existing_record = connection.execute(
            """
            SELECT status
            FROM email_tracking
            WHERE candidate_id = ? AND job_title = ?
            """,
            (candidate_id, job_title),
        ).fetchone()

        if existing_record is not None:
            previous_status = existing_record["status"]

            if previous_status in {
                SHORTLISTED_MAIL_SENT,
                SHORTLISTED_MAIL_NOT_SENT,
            }:
                return False

        connection.execute(
            """
            INSERT INTO email_tracking (
                candidate_id,
                candidate_name,
                job_title,
                recipient_email,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id, job_title)
            DO UPDATE SET
                candidate_name = excluded.candidate_name,
                recipient_email = excluded.recipient_email,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                candidate_id,
                candidate_name,
                job_title,
                recipient_email,
                SHORTLISTED_MAIL_NOT_SENT,
                now,
                now,
            ),
        )

        return True


def get_email_tracking_records(
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return tracking records for the HR tracking view, newest first."""
    with db_service.get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                candidate_id,
                candidate_name,
                job_title,
                recipient_email,
                status,
                created_at,
                updated_at
            FROM email_tracking
            ORDER BY updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]
