
"""
File: db_service.py

Purpose:
Create and manage the SQLite database for the Intelligent Recruitment
System.

SQLite is the structured system of record. Candidate identity is stored
separately from original resume text and structured professional profiles.

This module provides database initialization and basic candidate
persistence. Semantic indexing and candidate retrieval are handled by
separate services.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.schemas.candidate_schema import Candidate
from src.schemas.resume_schema import ResumeProfile


# Locate the project root without relying on the terminal's working folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep the runtime database under storage/, which is excluded from Git.
DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT / "storage" / "database" / "recruitment.db"
)


def get_connection(
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    """
    Open a SQLite connection and enable foreign-key enforcement.

    A custom path can be supplied by tests to use a temporary database.
    """

    database_path = (
        Path(db_path) if db_path is not None else DEFAULT_DATABASE_PATH
    )

    # Create the database folder if it does not exist yet.
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database(
    db_path: str | Path | None = None,
) -> None:
    """
    Create the database tables if they do not already exist.

    Calling this function repeatedly is safe. Existing records are kept.
    """

    with get_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidate_identity (
                candidate_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT,
                phone TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resume_documents (
                candidate_id TEXT PRIMARY KEY,
                resume_filename TEXT NOT NULL,
                original_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (candidate_id)
                    REFERENCES candidate_identity(candidate_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS resume_profiles (
                candidate_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (candidate_id)
                    REFERENCES candidate_identity(candidate_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT,
                event_type TEXT NOT NULL,
                event_details TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (candidate_id)
                    REFERENCES candidate_identity(candidate_id)
                    ON DELETE SET NULL
            );
            
            CREATE TABLE IF NOT EXISTS email_tracking (
                tracking_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                job_title TEXT NOT NULL,
                recipient_email TEXT,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'Shortlisted – Mail Sent',
                        'Shortlisted – Mail Not Sent',
                        'Not Shortlisted'
                    )
                ),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (candidate_id, job_title),
                FOREIGN KEY (candidate_id)
                    REFERENCES candidate_identity(candidate_id)
                    ON DELETE CASCADE
            );

            """
        )


def save_candidate(
    candidate: Candidate,
    db_path: str | Path | None = None,
) -> None:
    """
    Save candidate identity and original resume text separately.

    If the candidate ID already exists, update the identity and current
    resume document without changing the candidate's original creation
    timestamp.

    The stable candidate ID is the database link; names and email
    addresses are not used as identifiers.
    """

    now = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as connection:
        existing = connection.execute(
            """
            SELECT created_at
            FROM candidate_identity
            WHERE candidate_id = ?
            """,
            (candidate.candidate_id,),
        ).fetchone()

        created_at = (
            existing["created_at"]
            if existing is not None
            else candidate.created_at.isoformat()
        )

        # Store identifying details in their own table.
        connection.execute(
            """
            INSERT INTO candidate_identity (
                candidate_id,
                full_name,
                email,
                phone,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                full_name = excluded.full_name,
                email = excluded.email,
                phone = excluded.phone,
                updated_at = excluded.updated_at
            """,
            (
                candidate.candidate_id,
                candidate.full_name,
                candidate.email,
                candidate.phone,
                created_at,
                now,
            ),
        )

        # Store the original parsed text separately from indexed chunks.
        connection.execute(
            """
            INSERT INTO resume_documents (
                candidate_id,
                resume_filename,
                original_text,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                resume_filename = excluded.resume_filename,
                original_text = excluded.original_text,
                updated_at = excluded.updated_at
            """,
            (
                candidate.candidate_id,
                candidate.resume_filename,
                candidate.resume_text,
                created_at,
                now,
            ),
        )


def save_resume_profile(
    candidate_id: str,
    profile: ResumeProfile,
    db_path: str | Path | None = None,
) -> None:
    """
    Save a validated, structured professional resume profile.

    The candidate must already exist in candidate_identity.
    This function does not independently remove personal information
    from free-text profile fields; sanitization must happen upstream.
    """

    profile_json = json.dumps(
        profile.model_dump(mode="json"),
        ensure_ascii=False,
    )

    now = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO resume_profiles (
                candidate_id,
                profile_json,
                updated_at
            )
            VALUES (?, ?, ?)
            ON CONFLICT(candidate_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (candidate_id, profile_json, now),
        )


def get_candidate_identity(
    candidate_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve identity details using the stable candidate ID."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT candidate_id, full_name, email, phone,
                   created_at, updated_at
            FROM candidate_identity
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()

    return dict(row) if row is not None else None


def get_candidate_directory(
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    List every candidate with identity and resume file details.

    This is a purely READ-ONLY convenience for the Candidates screen:
    it joins the stable identity records with the resume document
    records so the UI can show one directory entry per candidate.
    No rows are inserted, updated or deleted here.
    """

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                ci.candidate_id,
                ci.full_name,
                ci.email,
                rd.resume_filename
            FROM candidate_identity AS ci
            LEFT JOIN resume_documents AS rd
                ON rd.candidate_id = ci.candidate_id
            ORDER BY ci.candidate_id
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_original_resume_text(
    candidate_id: str,
    db_path: str | Path | None = None,
) -> str | None:
    """Retrieve the original parsed resume text from the database."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT original_text
            FROM resume_documents
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()

    return row["original_text"] if row is not None else None


def get_resume_profile(
    candidate_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve the stored professional profile as a Python dictionary."""

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT profile_json
            FROM resume_profiles
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()

    if row is None:
        return None

    return json.loads(row["profile_json"])
