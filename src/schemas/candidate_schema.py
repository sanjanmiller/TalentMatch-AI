"""
Purpose:
Define the Candidate data model used by the Intelligent
Recruitment System.

This model describes which details we store for each candidate.
Pydantic checks that the supplied data meets the rules defined
below before the application uses it.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """Store and validate the information associated with one candidate."""

    # This is the internal ID used to connect a candidate's
    # database record with their resume embeddings in ChromaDB.
    candidate_id: str = Field(
        ...,
        min_length=1,
        description="Unique internal identifier for the candidate.",
    )

    # Contact details may be missing from some resumes,
    # so email and phone are optional.
    full_name: str = Field(..., min_length=1)
    email: str | None = None
    phone: str | None = None

    # Keep the source filename so HR can identify the resume
    # from which the candidate's information was extracted.
    resume_filename: str = Field(..., min_length=1)

    # Store the parsed resume text for later reference
    # and evaluation against a job description.
    resume_text: str = Field(..., min_length=1)

    # Record when the candidate entry was created and last updated.
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Reject unexpected fields instead of silently accepting
    # data that does not belong to this model.
    model_config = ConfigDict(extra="forbid")