
"""
Purpose:
Read candidate resumes, extract basic candidate details and resume sections,
validate the extracted information, and save it to SQLite.

Supported formats:
- PDF
- DOCX

Important:
This service uses local, rule-based extraction. It does not use an LLM.
If a detail cannot be identified confidently, it is left empty rather than
being invented. Extracted information should be checked before production use.
"""

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.schemas.candidate_schema import Candidate
from src.schemas.resume_schema import (
    EducationEntry,
    ResumeProfile,
    ResumeProject,
    WorkExperience,
)
from src.services import db_service, parser_service


# These headings help the service locate sections in differently formatted
# resumes. The list can be expanded when we inspect more sample documents.
SECTION_ALIASES = {
    "professional_summary": {
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "career summary",
        "objective",
        "career objective",
    },
    "technical_skills": {
        "technical skills",
        "skills",
        "key skills",
        "core skills",
        "technical expertise",
        "technologies",
        "tools and technologies",
    },
    "work_experience": {
        "work experience",
        "professional experience",
        "employment history",
        "experience",
        "work history",
        "career history",
    },
    "projects": {
        "projects",
        "personal projects",
        "academic projects",
        "key projects",
        "project experience",
    },
    "education": {
        "education",
        "educational background",
        "academic background",
        "qualifications",
    },
    "certifications": {
        "certifications",
        "certificates",
        "professional certifications",
        "licenses and certifications",
    },
    "achievements": {
        "achievements",
        "accomplishments",
        "awards",
        "honors and awards",
    },
}

# Common headings that should not be mistaken for a candidate's name.
NON_NAME_HEADINGS = {
    "resume",
    "curriculum vitae",
    "cv",
    "contact",
    "contact information",
    "professional summary",
    "summary",
    "objective",
    "technical skills",
    "skills",
    "work experience",
    "professional experience",
    "education",
    "projects",
    "certifications",
}


@dataclass(frozen=True)
class IngestionResult:
    """Report the outcome of processing one resume file."""

    source_file: str
    candidate_id: str | None
    status: str
    message: str



def _get_candidate_id(file_path: Path) -> str:
    """
    Return the candidate ID embedded in a sample filename when available.

    For example, CAND-001_Resume.pdf produces CAND-001.
    If the filename has no candidate ID, generate a UUID-based ID.
    """
    match = re.search(
        r"(?<![A-Z0-9])CAND-\d+(?!\d)",
        file_path.stem,
        re.IGNORECASE,
    )

    if match:
        return match.group(0).upper()

    return f"CAND-{uuid.uuid4().hex[:12].upper()}"


def _extract_email(resume_text: str) -> str | None:
    """Find the first email address in the parsed resume text."""
    match = re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        resume_text,
        re.IGNORECASE,
    )
    return match.group(0) if match else None


def _extract_phone(resume_text: str) -> str | None:
    """
    Find a likely phone number.

    This is a basic pattern, not a complete international phone validator.
    It returns None when no sufficiently long number is found.
    """
    for match in re.finditer(r"\+?\d[\d\s().-]{7,}\d", resume_text):
        phone = match.group(0).strip()
        digit_count = sum(character.isdigit() for character in phone)

        if 10 <= digit_count <= 15:
            return phone

    return None


def _extract_candidate_name(resume_text: str) -> str:
    """
    Use a likely name from the opening lines of a resume.

    This heuristic skips contact details and common headings. It cannot
    guarantee that every extracted line is a person's actual name.
    """
    for line in resume_text.splitlines()[:12]:
        candidate_line = line.strip()

        if not candidate_line:
            continue

        if "@" in candidate_line or "://" in candidate_line:
            continue

        if re.search(r"\b(linkedin|github|portfolio)\b", candidate_line, re.I):
            continue

        if sum(character.isdigit() for character in candidate_line) >= 4:
            continue

        normalized_heading = candidate_line.rstrip(":").strip().lower()
        if normalized_heading in NON_NAME_HEADINGS:
            continue

        # A likely name is short and does not look like a sentence.
        if len(candidate_line) <= 80 and not candidate_line.endswith((".", ";")):
            return candidate_line

    return "Unknown Candidate"


def _normalize_heading(line: str) -> str:
    """Normalize a line so it can be compared with known section headings."""
    return re.sub(r"[\s:|_-]+", " ", line.strip().lower()).strip()


def _identify_section_heading(line: str) -> str | None:
    """Return the internal section name if a line matches a known heading."""
    normalized_line = _normalize_heading(line)

    for section_name, aliases in SECTION_ALIASES.items():
        if normalized_line in aliases:
            return section_name

    return None


def _split_resume_into_sections(resume_text: str) -> dict[str, list[str]]:
    """
    Group lines under recognized resume headings.

    Lines before the first recognized heading are kept in an introductory
    section, which can help recover a professional summary.
    """
    sections: dict[str, list[str]] = {"intro": []}
    current_section = "intro"

    for line in resume_text.splitlines():
        section_name = _identify_section_heading(line)

        if section_name:
            current_section = section_name
            sections.setdefault(current_section, [])
            continue

        sections.setdefault(current_section, []).append(line.strip())

    return sections


def _clean_lines(lines: list[str]) -> list[str]:
    """Remove blank lines and normalize surrounding whitespace."""
    return [line.strip(" \t•*-") for line in lines if line.strip(" \t•*-")]


def _extract_summary(sections: dict[str, list[str]]) -> str | None:
    """Extract a summary from a recognized section or the opening lines."""
    summary_lines = _clean_lines(sections.get("professional_summary", []))

    if not summary_lines:
        summary_lines = _clean_lines(sections.get("intro", []))

    if not summary_lines:
        return None

    # Do not copy the candidate's likely name or contact details into summary.
    summary_lines = [
        line
        for line in summary_lines
        if "@" not in line
        and not re.search(r"https?://|www\.", line, re.IGNORECASE)
    ]

    summary = " ".join(summary_lines).strip()
    return summary or None


def _extract_list_items(lines: list[str]) -> list[str]:
    """Convert a section into a simple list of non-empty text items."""
    items: list[str] = []

    for line in _clean_lines(lines):
        # Split skill lists such as "Python, SQL, Excel" into individual items.
        if "," in line and len(line) < 250:
            items.extend(part.strip() for part in line.split(",") if part.strip())
        else:
            items.append(line)

    # Preserve order while removing exact duplicates.
    unique_items: list[str] = []
    seen_items: set[str] = set()

    for item in items:
        key = item.casefold()
        if key not in seen_items:
            unique_items.append(item)
            seen_items.add(key)

    return unique_items


def _extract_work_experience(lines: list[str]) -> list[WorkExperience]:
    """
    Convert recognizable experience lines into conservative entries.

    This initial implementation does not infer employer names, dates, or job
    titles from ambiguous text. Each non-empty line is retained as a
    responsibility under a neutral entry when it cannot be parsed reliably.
    """
    responsibilities = _clean_lines(lines)

    if not responsibilities:
        return []

    return [
        WorkExperience(
            job_title="Role details extracted from resume",
            employer=None,
            start_date=None,
            end_date=None,
            responsibilities=responsibilities,
        )
    ]


def _extract_projects(lines: list[str]) -> list[ResumeProject]:
    """
    Preserve project-section content without guessing project boundaries.

    Until project-heading recognition is improved, the section is represented
    as one project description when it contains text.
    """
    project_lines = _clean_lines(lines)

    if not project_lines:
        return []

    return [
        ResumeProject(
            project_name="Projects listed in resume",
            description=" ".join(project_lines),
            technologies=[],
        )
    ]


def _extract_education(lines: list[str]) -> list[EducationEntry]:
    """
    Preserve education text so it can be chunked and indexed later.

    The education lines are stored exactly as written, with no guessed
    degrees or institutions. Storing the text (instead of a placeholder
    label) lets the chunking and indexing services create a meaningful
    Education chunk in ChromaDB.

    Earlier versions stored only the placeholder label
    "Education details listed in resume" and discarded the section
    text, which caused the Education section to disappear from the
    vector store even when it existed in the saved profile.
    """
    education_lines = _clean_lines(lines)

    if not education_lines:
        return []

    return [
        EducationEntry(
            qualification=" ".join(education_lines),
        )
    ]


def _build_resume_profile(resume_text: str) -> ResumeProfile:
    """Build and validate a structured profile from the parsed resume text."""
    sections = _split_resume_into_sections(resume_text)

    return ResumeProfile(
        professional_summary=_extract_summary(sections),
        technical_skills=_extract_list_items(
            sections.get("technical_skills", [])
        ),
        work_experience=_extract_work_experience(
            sections.get("work_experience", [])
        ),
        projects=_extract_projects(sections.get("projects", [])),
        education=_extract_education(sections.get("education", [])),
        certifications=_extract_list_items(
            sections.get("certifications", [])
        ),
        achievements=_extract_list_items(sections.get("achievements", [])),
    )


def ingest_resume_file(file_path: str | Path) -> IngestionResult:
    """
    Parse, validate, and save one PDF or DOCX resume to SQLite.

    The database service stores candidate identity separately from the
    original resume text and structured resume profile.
    """
    path = Path(file_path)

    try:
        if not path.is_file():
            raise FileNotFoundError(f"Resume file does not exist: {path}")

        if path.suffix.lower() not in {".pdf", ".docx"}:
            raise ValueError("Only PDF and DOCX resumes are supported.")

        resume_text = parser_service.extract_text_from_file(path).strip()

        if not resume_text:
            raise ValueError("The resume contains no extractable text.")

        candidate_id = _get_candidate_id(path)
        candidate = Candidate(
            candidate_id=candidate_id,
            full_name=_extract_candidate_name(resume_text),
            email=_extract_email(resume_text),
            phone=_extract_phone(resume_text),
            resume_filename=path.name,
            resume_text=resume_text,
        )
        resume_profile = _build_resume_profile(resume_text)

        # Validate the profile explicitly before writing any candidate data.
        resume_profile = ResumeProfile.model_validate(
            resume_profile.model_dump()
        )

        db_service.save_candidate(candidate)
        db_service.save_resume_profile(candidate_id, resume_profile)

        return IngestionResult(
            source_file=path.name,
            candidate_id=candidate_id,
            status="success",
            message="Resume parsed, validated, and saved to SQLite.",
        )

    except (
        OSError,
        ValueError,
        ValidationError,
        parser_service.DocumentParsingError,
    ) as error:
        return IngestionResult(
            source_file=path.name,
            candidate_id=None,
            status="failed",
            message=str(error),
        )


def ingest_resume_directory(directory_path: str | Path) -> list[IngestionResult]:
    """
    Ingest every PDF and DOCX resume found in a directory and its subfolders.

    Processing continues when one file fails, so a single bad document does
    not prevent the remaining resumes from being ingested.
    """
    directory = Path(directory_path)

    if not directory.is_dir():
        raise NotADirectoryError(f"Resume directory does not exist: {directory}")

    db_service.initialize_database()

    resume_files = sorted(
        file_path
        for file_path in directory.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in {".pdf", ".docx"}
    )

    results = [ingest_resume_file(file_path) for file_path in resume_files]

    return results
