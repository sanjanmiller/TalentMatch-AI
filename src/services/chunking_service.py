
"""
Purpose:
Split a structured resume profile into smaller, meaningful professional
sections for future embedding and semantic retrieval.

Each chunk contains professional information and metadata that links it
to the correct candidate in SQLite.

Privacy rules:
- Candidate names and contact fields are not accepted by this service.
- Email addresses and phone-like numbers found in professional text
  are redacted before the text is returned.
- The candidate ID is stored as metadata, never in the chunk text.

Important:
This service prepares text for retrieval. It does not generate embeddings,
save vectors to ChromaDB, evaluate candidates, or calculate scores.
"""

from dataclasses import dataclass
import re

from src.schemas.resume_schema import ResumeProfile


# A chunk should remain small enough to retrieve as focused evidence.
DEFAULT_MAX_CHUNK_CHARS = 1200


@dataclass(frozen=True)
class ResumeChunk:
    """Represent one professional resume chunk and its source metadata."""

    chunk_id: str
    candidate_id: str
    section: str
    chunk_index: int
    text: str

    def to_metadata(self) -> dict[str, str | int]:
        """Return metadata suitable for a vector database record."""
        return {
            "chunk_id": self.chunk_id,
            "candidate_id": self.candidate_id,
            "section": self.section,
            "chunk_index": self.chunk_index,
        }


def _redact_personal_contact_details(text: str) -> str:
    """Remove email addresses and phone-like numbers from text."""

    # Redact standard email address patterns.
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "[EMAIL REDACTED]",
        text,
        flags=re.IGNORECASE,
    )

    # Redact phone-like numbers containing at least eight digits.
    # This avoids redacting ordinary short numbers such as graduation years.
    phone_pattern = re.compile(
        r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)"
    )

    # A year range such as "2020 - 2024" commonly marks a study or work
    # period. It is not a contact detail, so it is kept as written.
    year_range_pattern = re.compile(r"\d{4}\s*[-\u2013\u2014]\s*\d{4}")

    def replace_phone(match: re.Match[str]) -> str:
        """Redact a match only when it contains enough digits."""
        digit_count = sum(character.isdigit() for character in match.group())

        stripped_match = match.group().strip("() \t")

        if digit_count >= 8 and not year_range_pattern.fullmatch(
            stripped_match
        ):
            return "[PHONE REDACTED]"

        return match.group()

    return phone_pattern.sub(replace_phone, text)


def _clean_text(text: str) -> str:
    """Normalize whitespace and redact contact details."""
    text = _redact_personal_contact_details(text)
    return " ".join(text.split()).strip()


def _is_placeholder(text: str) -> bool:
    """Identify placeholder labels that are not genuine resume evidence."""
    normalized = text.strip().casefold()

    placeholder_phrases = {
        "role details extracted from resume",
        "projects listed in resume",
        "education details listed in resume",
    }

    return normalized in placeholder_phrases


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split long text into word-preserving pieces within the size limit."""

    words = text.split()
    pieces: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        # A single unusually long word cannot be split naturally.
        if len(word) > max_chars:
            if current_words:
                pieces.append(" ".join(current_words))
                current_words = []
                current_length = 0

            pieces.append(word)
            continue

        extra_length = len(word) if not current_words else len(word) + 1

        if current_length + extra_length > max_chars:
            pieces.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)
        else:
            current_words.append(word)
            current_length += extra_length

    if current_words:
        pieces.append(" ".join(current_words))

    return pieces


def _build_section_texts(profile: ResumeProfile) -> list[tuple[str, list[str]]]:
    """Convert available profile fields into named professional sections."""

    sections: list[tuple[str, list[str]]] = []

    # Professional summary
    if profile.professional_summary:
        summary = _clean_text(profile.professional_summary)

        if summary:
            sections.append(("professional_summary", [summary]))

    # Technical skills
    skills = [
        _clean_text(skill)
        for skill in profile.technical_skills
        if _clean_text(skill)
    ]

    if skills:
        sections.append(("technical_skills", skills))

    # Work experience
    experience_items: list[str] = []

    for experience in profile.work_experience:
        details: list[str] = []

        # Do not index a placeholder job title as if it were real evidence.
        if experience.job_title and not _is_placeholder(experience.job_title):
            details.append(f"Job title: {experience.job_title}")

        if experience.employer:
            details.append(f"Employer: {experience.employer}")

        if experience.start_date:
            details.append(f"Start date: {experience.start_date}")

        if experience.end_date:
            details.append(f"End date: {experience.end_date}")

        details.extend(experience.responsibilities)

        cleaned_details = [
            _clean_text(detail)
            for detail in details
            if detail and _clean_text(detail)
        ]

        if cleaned_details:
            experience_items.append(" ".join(cleaned_details))

    if experience_items:
        sections.append(("work_experience", experience_items))

    # Projects
    project_items: list[str] = []

    for project in profile.projects:
        details: list[str] = []

        if project.project_name and not _is_placeholder(project.project_name):
            details.append(f"Project: {project.project_name}")

        if project.description:
            details.append(project.description)

        if project.technologies:
            details.append(
                "Technologies: " + ", ".join(project.technologies)
            )

        cleaned_details = [
            _clean_text(detail)
            for detail in details
            if detail and _clean_text(detail)
        ]

        if cleaned_details:
            project_items.append(" ".join(cleaned_details))

    if project_items:
        sections.append(("projects", project_items))

    # Education
    education_items: list[str] = []

    for education in profile.education:
        details: list[str] = []

        if (
            education.qualification
            and not _is_placeholder(education.qualification)
        ):
            details.append(f"Qualification: {education.qualification}")

        if education.institution:
            details.append(f"Institution: {education.institution}")

        if education.graduation_year:
            details.append(f"Graduation year: {education.graduation_year}")

        cleaned_details = [
            _clean_text(detail)
            for detail in details
            if detail and _clean_text(detail)
        ]

        if cleaned_details:
            education_items.append(" ".join(cleaned_details))

    if education_items:
        sections.append(("education", education_items))

    # Certifications
    certifications = [
        _clean_text(certification)
        for certification in profile.certifications
        if _clean_text(certification)
    ]

    if certifications:
        sections.append(("certifications", certifications))

    # Achievements
    achievements = [
        _clean_text(achievement)
        for achievement in profile.achievements
        if _clean_text(achievement)
    ]

    if achievements:
        sections.append(("achievements", achievements))

    return sections


def chunk_resume_profile(
    candidate_id: str,
    profile: ResumeProfile,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[ResumeChunk]:
    """
    Create section-based chunks from a validated resume profile.

    Args:
        candidate_id: Stable internal ID used to link chunks to SQLite.
        profile: Validated professional resume profile.
        max_chunk_chars: Preferred maximum text length for each chunk.

    Returns:
        A list of chunks with professional text and source metadata.

    Raises:
        ValueError: If the candidate ID or chunk size is invalid.
    """
    if not candidate_id or not candidate_id.strip():
        raise ValueError("candidate_id cannot be empty.")

    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be at least 1.")

    section_texts = _build_section_texts(profile)
    chunks: list[ResumeChunk] = []

    for section_name, items in section_texts:
        section_chunk_number = 0
        current_items: list[str] = []
        current_length = 0

        for item in items:
            # Split a long item first so it cannot overwhelm a chunk.
            pieces = _split_long_text(item, max_chunk_chars)

            for piece in pieces:
                additional_length = (
                    len(piece) if not current_items else len(piece) + 1
                )

                if (
                    current_items
                    and current_length + additional_length > max_chunk_chars
                ):
                    chunk_text = " ".join(current_items)
                    chunk_id = (
                        f"{candidate_id}_{section_name}_{section_chunk_number}"
                    )

                    chunks.append(
                        ResumeChunk(
                            chunk_id=chunk_id,
                            candidate_id=candidate_id,
                            section=section_name,
                            chunk_index=section_chunk_number,
                            text=chunk_text,
                        )
                    )

                    section_chunk_number += 1
                    current_items = []
                    current_length = 0

                current_items.append(piece)
                current_length += (
                    len(piece) if len(current_items) == 1 else len(piece) + 1
                )

        # Save any remaining text in this section.
        if current_items:
            chunk_id = (
                f"{candidate_id}_{section_name}_{section_chunk_number}"
            )

            chunks.append(
                ResumeChunk(
                    chunk_id=chunk_id,
                    candidate_id=candidate_id,
                    section=section_name,
                    chunk_index=section_chunk_number,
                    text=" ".join(current_items),
                )
            )

    return chunks
