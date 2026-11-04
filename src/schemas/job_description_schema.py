
"""
File: job_description_schema.py

Purpose:
Define validated, reusable data models for job descriptions in the
Intelligent Recruitment System.

The schema supports different professions. It does not assume that a job
requires a particular programming language, degree, certification, or
number of years of experience.

This file only defines and validates structured data. It does not parse
documents, evaluate candidates, or calculate candidate scores.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------
# Reject unexpected field names instead of silently accepting mistakes.
# This helps us catch spelling errors and incorrect data structures early.


class StrictSchema(BaseModel):
    """Base model that rejects fields not defined in the schema."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# 1. Technical skills
# ---------------------------------------------------------------------


class TechnicalSkillRequirement(StrictSchema):
    """Represent a skill required or preferred for a particular role."""

    skill_name: str = Field(
        min_length=1,
        description="Name of the technical or professional skill.",
    )

    proficiency_level: str | None = Field(
        default=None,
        description=(
            "Required proficiency, using the wording from the JD "
            "when available."
        ),
    )

    requirement_type: Literal[
        "mandatory",
        "preferred",
        "not_specified",
    ] = Field(
        default="not_specified",
        description="Whether the skill is mandatory or preferred.",
    )

    requirement_details: str | None = Field(
        default=None,
        description="Additional context about the skill requirement.",
    )


# ---------------------------------------------------------------------
# 2. Experience
# ---------------------------------------------------------------------


class ExperienceRequirements(StrictSchema):
    """Represent the experience requirements stated in a job description."""

    minimum_years: float | None = Field(
        default=None,
        ge=0,
        description="Minimum required years of experience, if specified.",
    )

    maximum_years: float | None = Field(
        default=None,
        ge=0,
        description="Maximum years in the stated experience range, if any.",
    )

    accepted_experience_types: list[str] = Field(
        default_factory=list,
        description=(
            "Qualifying experience types, such as employment, internships, "
            "projects, volunteering, or other relevant experience."
        ),
    )

    additional_requirements: list[str] = Field(
        default_factory=list,
        description="Other experience-related conditions stated in the JD.",
    )

    @model_validator(mode="after")
    def validate_experience_range(self):
        """Ensure the maximum is not lower than the minimum."""

        if (
            self.minimum_years is not None
            and self.maximum_years is not None
            and self.maximum_years < self.minimum_years
        ):
            raise ValueError(
                "maximum_years cannot be less than minimum_years."
            )

        return self


# ---------------------------------------------------------------------
# 3. Education
# ---------------------------------------------------------------------


class EducationRequirements(StrictSchema):
    """Represent required or preferred educational qualifications."""

    degree_types: list[str] = Field(
        default_factory=list,
        description="Degree or qualification types mentioned in the JD.",
    )

    relevant_disciplines: list[str] = Field(
        default_factory=list,
        description="Fields of study relevant to the role, if specified.",
    )

    alternative_qualification_details: list[str] = Field(
        default_factory=list,
        description=(
            "Accepted equivalent qualifications or alternative pathways."
        ),
    )

    additional_requirements: list[str] = Field(
        default_factory=list,
        description="Other education-related requirements from the JD.",
    )


# ---------------------------------------------------------------------
# 4. Certifications
# ---------------------------------------------------------------------


class CertificationRequirements(StrictSchema):
    """Keep mandatory and preferred certifications separate."""

    mandatory_certifications: list[str] = Field(
        default_factory=list,
        description="Certifications explicitly required by the JD.",
    )

    preferred_certifications: list[str] = Field(
        default_factory=list,
        description="Certifications described as desirable or preferred.",
    )


# ---------------------------------------------------------------------
# 5. Employment details
# ---------------------------------------------------------------------


class EmploymentDetails(StrictSchema):
    """Represent the employment arrangements described in the JD."""

    location: str | None = Field(
        default=None,
        description="Job location, if specified.",
    )

    employment_type: str | None = Field(
        default=None,
        description="Employment type, such as full-time or contract.",
    )

    work_arrangement: str | None = Field(
        default=None,
        description="Work arrangement, such as onsite, hybrid, or remote.",
    )

    in_office_days_per_week: int | None = Field(
        default=None,
        ge=0,
        le=7,
        description="Required office days per week, if specified.",
    )


# ---------------------------------------------------------------------
# Main job description schema
# ---------------------------------------------------------------------


class JobDescription(StrictSchema):
    """
    Represent a complete, structured job description.

    All nine standard JD sections are supported. Fields can remain empty
    when the source JD does not provide the relevant information.
    """

    # Section 1: Job title
    job_title: str = Field(
        min_length=1,
        description="Title of the advertised position.",
    )

    company_name: str | None = Field(
        default=None,
        description="Company or employer name, if provided.",
    )

    company_overview: str | None = Field(
        default=None,
        description="Company background stated in the JD.",
    )

    role_summary: str | None = Field(
        default=None,
        description="Short summary of the role and its purpose.",
    )

    # Section 2: Technical or professional skills
    required_technical_skills: list[TechnicalSkillRequirement] = Field(
        default_factory=list,
        description=(
            "Technical or professional skills mentioned in the JD. "
            "The list can be empty when no skills are specified."
        ),
    )

    # Section 3: Experience
    required_experience: ExperienceRequirements = Field(
        default_factory=ExperienceRequirements,
        description="Experience requirements stated in the JD.",
    )

    # Section 4: Education
    education: EducationRequirements = Field(
        default_factory=EducationRequirements,
        description="Educational requirements stated in the JD.",
    )

    # Section 5: Certifications
    certifications: CertificationRequirements = Field(
        default_factory=CertificationRequirements,
        description="Mandatory and preferred certifications.",
    )

    # Section 6: Responsibilities
    key_responsibilities: list[str] = Field(
        default_factory=list,
        description="Main responsibilities listed in the JD.",
    )

    # Section 7: Preferred qualifications
    preferred_qualifications: list[str] = Field(
        default_factory=list,
        description="Additional desirable qualifications.",
    )

    # Section 8: Evaluation priorities
    evaluation_priorities: list[str] = Field(
        default_factory=list,
        description=(
            "Priorities explicitly stated in the JD to guide candidate "
            "evaluation."
        ),
    )

    # Section 9: Employment details
    employment_details: EmploymentDetails = Field(
        default_factory=EmploymentDetails,
        description="Location and employment arrangements.",
    )

    # Preserve the original parsed JD text for traceability and later checks.
    original_jd_text: str | None = Field(
        default=None,
        description="Original text extracted from the job description.",
    )
