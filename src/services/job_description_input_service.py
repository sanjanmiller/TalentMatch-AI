"""
Purpose:
    Provide ONE reusable way to obtain a job description for the
    recruitment workflow, whatever way HR supplies it.

Why this file is needed:
    HR can provide a job description in three different ways:

        1. A TXT document upload (read as plain UTF-8 text).
        2. A PDF (or DOCX) document upload (text extracted for them).
        3. Plain text pasted directly into an input box.

    The future Streamlit UI and the manual test scripts must all be
    able to call the same input handling, so the backend produces the
    same result for every method: ONE clean, non-empty job-description
    string that can be passed straight into RecruitmentWorkflow.run().

Role in the overall recruitment workflow:
    Input validation happens HERE, before the workflow starts. The RAG
    workflow in src/services/recruitment_workflow.py only ever receives
    a validated, non-empty string.

Interacts with:
    - src/services/parser_service.py (extracts text from TXT, PDF and
      DOCX documents; this file reuses it and does NOT duplicate it).

This service does not evaluate resumes, query the vector store, send
emails, or write to the database.
"""

from pathlib import Path

from src.services.parser_service import extract_text_from_file


def load_job_description_from_file(file_path: str | Path) -> str:
    """Load a job description from a TXT, PDF or DOCX document.

    The file format is chosen by extension, exactly as the existing
    parser service does:

        - .txt  : read as plain UTF-8 text
        - .pdf  : text extracted with PyMuPDF
        - .docx : text extracted with python-docx

    Args:
        file_path: Path to the job-description document.

    Returns:
        A clean (trimmed), non-empty job-description string, ready to
        pass into the recruitment workflow.

    Raises:
        DocumentParsingError: When the file is missing, has an
            unsupported extension, cannot be read, or contains no
            extractable text (for example a scanned PDF).
        ValueError: When the extracted text is empty after trimming.
    """

    # Extracted text is trimmed so HR never sees leading/trailing
    # whitespace reach the evaluation prompt.
    extracted_text = (
        extract_text_from_file(file_path).strip()
    )

    if not extracted_text:
        # Defensive check for text that is empty apart from whitespace.
        file_name = Path(file_path).name
        raise ValueError(
            f"The job description in {file_name} is empty."
        )

    return extracted_text


def load_job_description_from_text(jd_text: str) -> str:
    """Validate and clean a job description pasted as plain text.

    Args:
        jd_text: The raw text as HR supplied it (for example pasted
            into a text box in the future Streamlit UI).

    Returns:
        A clean (trimmed), non-empty job-description string.

    Raises:
        ValueError: When the input is not text, or is empty or
            whitespace-only. The caller must handle this error visibly
            instead of continuing with an invalid job description.
    """

    if not isinstance(jd_text, str):
        raise ValueError(
            "Job description must be supplied as plain text."
        )

    cleaned_text = jd_text.strip()

    if not cleaned_text:
        raise ValueError(
            "Job description cannot be empty or whitespace-only."
        )

    return cleaned_text


def extract_job_title(job_description: str) -> str | None:
    """Extract the job title from the existing JD text layout.

    The project's job-description files follow a simple pattern:

        Job Title
        ---------
        Junior Data Scientist

    Args:
        job_description: Full job-description text.

    Returns:
        The job title as a string, or None when no title heading can be
        found. In that case callers should ask HR for the title
        explicitly instead of extracting it with an LLM.
    """

    lines = job_description.splitlines()

    for index, line in enumerate(lines):
        # Find the "Job Title" section heading (case-insensitive).
        if line.strip().casefold() != "job title":
            continue

        # The next meaningful line (skipping separator rows made of
        # dashes) contains the actual title.
        for following_line in lines[index + 1 : index + 4]:
            cleaned = following_line.strip()

            if not cleaned:
                continue

            is_separator = set(cleaned) == {"-"}
            if not is_separator:
                return cleaned

    return None
