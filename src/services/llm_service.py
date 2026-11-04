
"""
Purpose:
    Generate structured candidate evaluations using configurable LLM providers.

Provider order:
    1. Primary Gemini
    2. Secondary Gemini
    3. Groq

Responsibilities:
    - Load provider configuration from environment variables.
    - Build the batch candidate evaluation prompt.
    - Call configured LLM providers.
    - Validate responses using RankedCandidateEvaluations.
    - Validate that the response covers exactly the supplied candidates.
    - Fall back to the next provider if a provider fails.
    - Return an explicit error if all providers fail.

Important:
    - API keys and raw provider responses are never logged.
    - The LLM supplies the overall scores and the ranking. Python
      validates the response but must not recalculate, modify, or
      override any score or ranking position.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from google import genai
from groq import Groq
from pydantic import ValidationError

from src.prompts.evaluation_prompt import (
    GROQ_SYSTEM_PROMPT,
    build_evaluation_prompt,
)
from src.schemas.evaluation_schema import RankedCandidateEvaluations


# Load environment variables from .env.
load_dotenv()


class LLMServiceError(RuntimeError):
    """Raised when all configured LLM providers fail."""


# Keep Gemini requests bounded so fallback can happen promptly.
GEMINI_HTTP_TIMEOUT_MS = 30_000
GEMINI_HTTP_RETRY_ATTEMPTS = 2


@dataclass(frozen=True)
class ProviderConfig:
    """Store the configuration for one LLM provider."""

    name: str
    api_key: str
    model: str


def _get_provider_configs() -> list[ProviderConfig]:
    """Load providers in the required fallback order."""

    providers = [
        ProviderConfig(
            name="primary_gemini",
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            model=os.getenv("GEMINI_MODEL", "").strip(),
        ),
        ProviderConfig(
            name="secondary_gemini",
            api_key=os.getenv("GEMINI_SECONDARY_API_KEY", "").strip(),
            model=os.getenv("GEMINI_SECONDARY_MODEL", "").strip(),
        ),
        ProviderConfig(
            name="groq",
            api_key=os.getenv("GROQ_API_KEY", "").strip(),
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        ),
    ]

    # Use only providers with both an API key and model configured.
    return [
        provider
        for provider in providers
        if provider.api_key and provider.model
    ]


def _build_gemini_client(api_key: str) -> genai.Client:
    """Create a Gemini client with bounded request and retry settings."""

    return genai.Client(
        api_key=api_key,
        http_options={
            "timeout": GEMINI_HTTP_TIMEOUT_MS,
            "retry_options": {"attempts": GEMINI_HTTP_RETRY_ATTEMPTS},
        },
    )


def _build_gemini_config(model: str) -> dict:
    """Build the generation settings for the configured Gemini model."""

    config = {"response_mime_type": "application/json"}

    # Gemini 2.x models support temperature; Gemini 3.x models do not need it.
    if re.match(r"(?:models/)?gemini-2\.", model):
        config["temperature"] = 0

    return config


def _validate_input(
    job_description: str,
    candidate_contexts: dict[str, str],
) -> None:
    """Validate inputs before making an API request."""

    if not isinstance(job_description, str) or not job_description.strip():
        raise ValueError("Job description cannot be empty.")

    if not isinstance(candidate_contexts, dict):
        raise ValueError(
            "Candidate contexts must be supplied as a dictionary "
            "of candidate IDs to resume chunk text."
        )

    if not candidate_contexts:
        raise ValueError("At least one candidate context is required.")

    for candidate_id, resume_text in candidate_contexts.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Candidate IDs cannot be empty.")

        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError(
                f"Resume context for candidate {candidate_id.strip()} "
                "cannot be empty."
            )


def _extract_json(response_text: str) -> dict:
    """Extract and parse a JSON object from the provider response."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Provider returned an empty response.")

    cleaned_text = response_text.strip()

    # Remove optional Markdown code fences.
    cleaned_text = re.sub(
        r"\A```(?:json)?\s*|\s*```\Z",
        "",
        cleaned_text,
        flags=re.IGNORECASE,
    ).strip()

    try:
        response_data = json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Provider returned invalid JSON.") from exc

    if not isinstance(response_data, dict):
        raise ValueError("Provider response must be a JSON object.")

    return response_data


def _validate_response_covers_pool(
    ranked_evaluations: RankedCandidateEvaluations,
    supplied_candidate_ids: list[str],
) -> None:
    """
    Check the response covers exactly the candidates supplied for
    evaluation, using their exact candidate IDs.
    """

    returned_ids = {
        assessment.candidate_id for assessment in ranked_evaluations.candidates
    }
    supplied_ids = set(supplied_candidate_ids)

    missing_ids = supplied_ids - returned_ids

    if missing_ids:
        raise ValueError(
            "Provider response is missing candidates: "
            + ", ".join(sorted(missing_ids))
        )

    unexpected_ids = returned_ids - supplied_ids

    if unexpected_ids:
        raise ValueError(
            "Provider response contains unknown candidates: "
            + ", ".join(sorted(unexpected_ids))
        )


def _call_gemini(config: ProviderConfig, prompt: str) -> str:
    """Send the evaluation request to Gemini."""

    client = _build_gemini_client(config.api_key)

    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config=_build_gemini_config(config.model),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text


def _call_groq(config: ProviderConfig, prompt: str) -> str:
    """Send the evaluation request to Groq."""

    client = Groq(api_key=config.api_key)

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "system",
                "content": GROQ_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    response_text = response.choices[0].message.content

    if not response_text:
        raise ValueError("Groq returned an empty response.")

    return response_text


def _call_provider(config: ProviderConfig, prompt: str) -> str:
    """Call the appropriate provider based on its configuration."""

    if config.name in {"primary_gemini", "secondary_gemini"}:
        return _call_gemini(config, prompt)

    if config.name == "groq":
        return _call_groq(config, prompt)

    raise ValueError("Unknown LLM provider.")


def evaluate_candidates(
    candidate_contexts: dict[str, str],
    job_description: str,
    *,
    provider_configs: list[ProviderConfig] | None = None,
    provider_call: Callable[[ProviderConfig, str], str] | None = None,
) -> tuple[RankedCandidateEvaluations, str]:
    """
    Evaluate and rank the supplied candidates using the first provider
    that returns valid data.

    Args:
        candidate_contexts: Mapping of candidate IDs to their
            consolidated retrieved resume chunk text.
        job_description: The job description all candidates are
            evaluated against.

    Returns:
        A tuple containing the validated ranked evaluation and the
        successful provider name.

    Raises:
        ValueError: If the input is invalid.
        LLMServiceError: If all configured providers fail.
    """

    _validate_input(job_description, candidate_contexts)

    candidate_contexts = {
        candidate_id.strip(): resume_text.strip()
        for candidate_id, resume_text in candidate_contexts.items()
    }

    configs = (
        provider_configs
        if provider_configs is not None
        else _get_provider_configs()
    )

    if not configs:
        raise LLMServiceError("No LLM providers are configured.")

    prompt = build_evaluation_prompt(
        job_description=job_description.strip(),
        candidate_contexts=candidate_contexts,
    )

    call_provider = provider_call or _call_provider
    failure_reasons = []

    # Try providers in their configured order.
    for config in configs:
        try:
            response_text = call_provider(config, prompt)
            response_data = _extract_json(response_text)

            ranked_evaluations = (
                RankedCandidateEvaluations.model_validate(response_data)
            )

            # Ensure the response covers exactly the supplied candidates.
            _validate_response_covers_pool(
                ranked_evaluations,
                list(candidate_contexts),
            )

            print(
                f"[LLM SUCCESS] Evaluation completed using {config.name}."
            )

            return ranked_evaluations, config.name

        except Exception as exc:
            # Log only the provider name and exception type.
            # Never log API keys, prompts, or raw provider responses.
            error_detail = type(exc).__name__

            status_code = getattr(exc, "status_code", None)
            if status_code is not None:
                error_detail += f" (HTTP {status_code})"

            print(
                f"[LLM FALLBACK] {config.name} failed: {error_detail}"
            )

            failure_reasons.append(
                f"{config.name}: {error_detail}"
            )

    raise LLMServiceError(
        "All configured LLM providers failed. "
        + "; ".join(failure_reasons)
    )
