
"""
Purpose:
Define the structured professional information extracted from a resume.

This file describes the information our recruitment system expects,
such as skills, work experience, projects, education, and certifications.

It does not read PDF or DOCX files. The parser service will read those
files, and the extraction process will populate these data models.

Important privacy rule:
Candidate names, email addresses, and phone numbers do not belong in
these models. Professional information is handled separately from
candidate identity and contact details.
"""

from pydantic import BaseModel, ConfigDict, Field


class WorkExperience(BaseModel):
    """Represent one work experience entry from a candidate's resume."""

    # The role held by the candidate, such as Software Engineer.
    job_title: str = Field(..., min_length=1)

    # The employer's name, when it is available in the resume.
    employer: str | None = None

    # Keep dates as written or normalized by the extraction process.
    # Strings accommodate formats such as "Jan 2022" or "2022–2024".
    start_date: str | None = None
    end_date: str | None = None

    # Store responsibilities and accomplishments relevant to the role.
    # Some resumes may not provide a detailed description.
    responsibilities: list[str] = Field(default_factory=list)

    # Reject fields that are not part of this model.
    model_config = ConfigDict(extra="forbid")


class ResumeProject(BaseModel):
    """Represent a project described in the candidate's resume."""

    # The project name or title.
    project_name: str = Field(..., min_length=1)

    # A short description of the project, if provided.
    description: str | None = None

    # Technologies or tools explicitly associated with the project.
    technologies: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EducationEntry(BaseModel):
    """Represent one educational qualification from a resume."""

    # For example, Bachelor of Technology or Master of Science.
    qualification: str = Field(..., min_length=1)

    # The educational institution, if available.
    institution: str | None = None

    # The graduation year or study period, if stated.
    graduation_year: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResumeProfile(BaseModel):
    """
    Store the structured professional information extracted from a resume.

    This model excludes candidate identity and contact details. Its
    professional fields can support retrieval and evidence-based evaluation.
    """

    # A summary is optional because not every resume includes one.
    professional_summary: str | None = None

    # Skills explicitly identified in the resume.
    technical_skills: list[str] = Field(default_factory=list)

    # Each entry represents one job or work experience.
    work_experience: list[WorkExperience] = Field(default_factory=list)

    # Projects can provide additional evidence of practical skills.
    projects: list[ResumeProject] = Field(default_factory=list)

    # Educational qualifications listed in the resume.
    education: list[EducationEntry] = Field(default_factory=list)

    # Certifications relevant to the candidate's qualifications.
    certifications: list[str] = Field(default_factory=list)

    # Achievements explicitly mentioned in the resume.
    achievements: list[str] = Field(default_factory=list)

    # Reject unexpected top-level fields, including accidental
    # additions of identity or contact information.
    model_config = ConfigDict(extra="forbid")
