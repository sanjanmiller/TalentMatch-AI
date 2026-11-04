
"""
Purpose:
    Coordinate candidate retrieval and LLM evaluation for a job description.

Workflow:
    Job description -> Embedding -> ChromaDB retrieval -> Candidate grouping
    -> Consolidated per-candidate context -> Batch LLM evaluation and
    ranking -> Response validation -> Top candidate selection

The service retrieves the most semantically relevant candidates, builds a
consolidated context of all indexed resume chunks for each candidate, and
asks the LLM to evaluate and rank the whole pool in a single batch call.
Python validates the LLM response but never recalculates scores or
changes the LLM's ranking.

The default pool is eight unique candidates, of which the top five
candidates by LLM ranking are returned. HR remains responsible for
reviewing the results and making recruitment decisions.
"""

from dataclasses import dataclass, field
from math import inf
from typing import Any

from src.services.embedding_service import EmbeddingService
from src.services.llm_service import evaluate_candidates
from src.services.vector_store_service import VectorStoreService
from src.schemas.evaluation_schema import CandidateAssessment


@dataclass
class CandidateWorkflowResult:
    """Store the result for one candidate in the retrieval pool."""

    candidate_id: str
    retrieval_distance: float
    matched_sections: list[str] = field(default_factory=list)
    rank: int | None = None
    provider_name: str | None = None
    evaluation: CandidateAssessment | None = None
    error: str | None = None


@dataclass
class RecruitmentWorkflowResult:
    """Store the overall result for one job-description search."""

    success: bool
    message: str
    candidate_pool_size: int
    top_n: int
    retrieved_chunk_count: int = 0
    unique_candidate_count: int = 0
    evaluated_candidate_count: int = 0
    candidates: list[CandidateWorkflowResult] = field(default_factory=list)


class RecruitmentWorkflow:
    """Retrieve, evaluate, and rank the most relevant candidates for a job."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store_service: VectorStoreService | None = None,
        candidate_pool_size: int = 8,
        top_n: int = 5,
        chunks_per_candidate_slot: int = 6,
    ) -> None:
        """
        Initialise the workflow with the existing project services.

        The retrieval query requests several chunks per candidate slot
        because one candidate may have multiple chunks in the results.

        The candidate pool size and top-N selection are configurable.
        """

        if candidate_pool_size < 1:
            raise ValueError("Candidate pool size must be at least 1.")

        if top_n < 1:
            raise ValueError("Top-N count must be at least 1.")

        if chunks_per_candidate_slot < 1:
            raise ValueError(
                "Chunks per candidate slot must be at least 1."
            )

        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store_service = (
            vector_store_service or VectorStoreService()
        )
        self.candidate_pool_size = candidate_pool_size
        self.top_n = top_n
        self.chunks_per_candidate_slot = chunks_per_candidate_slot

    @staticmethod
    def _group_chunks_by_candidate(
        chunks: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group retrieved chunks by candidate ID, preserving their order."""

        grouped: dict[str, list[dict[str, Any]]] = {}

        for chunk in chunks:
            candidate_id = chunk.get("candidate_id")

            # Ignore malformed retrieval records without a usable ID.
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                continue

            grouped.setdefault(candidate_id, []).append(chunk)

        return grouped

    @staticmethod
    def _get_distance(chunk: dict[str, Any]) -> float:
        """Return a valid cosine distance, sorting invalid values last."""

        distance = chunk.get("distance")

        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            return float(distance)

        return inf

    @staticmethod
    def _build_candidate_context(
        chunks: list[dict[str, Any]],
    ) -> str:
        """
        Combine a candidate's resume chunks into consolidated LLM context.

        Chunks are ordered by retrieval distance, with the closest
        matching chunk appearing first. Every available section is
        included; there are no section-specific restrictions.
        """

        sorted_chunks = sorted(
            chunks,
            key=RecruitmentWorkflow._get_distance,
        )

        context_parts = []

        for chunk in sorted_chunks:
            section = chunk.get("section", "general")
            text = chunk.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            context_parts.append(
                f"RESUME SECTION: {section}\n{text.strip()}"
            )

        return "\n\n".join(context_parts)

    def _get_consolidated_candidate_chunks(
        self,
        candidate_id: str,
        job_embedding: Any,
        initially_retrieved: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Collect all indexed chunks for one candidate.

        A candidate-specific ChromaDB query returns every chunk indexed
        for this candidate. These are merged with the chunks from the
        initial global retrieval, de-duplicated by their stable chunk
        IDs. No section filtering is applied.
        """

        candidate_chunk_count = self.vector_store_service.count_chunks(
            candidate_id
        )

        # Fetch all of this candidate's indexed chunks in one query.
        candidate_chunks = (
            self.vector_store_service.query_by_embedding(
                job_embedding,
                n_results=candidate_chunk_count,
                candidate_id=candidate_id,
            )
            if candidate_chunk_count > 0
            else []
        )

        # Merge by chunk ID so every indexed chunk appears exactly once.
        combined_chunks: dict[str, dict[str, Any]] = {}

        for chunk in [*initially_retrieved, *candidate_chunks]:
            chunk_id = chunk.get("chunk_id")

            if isinstance(chunk_id, str) and chunk_id and chunk_id not in combined_chunks:
                combined_chunks[chunk_id] = chunk

        return list(combined_chunks.values())

    def run(
        self,
        job_description: str,
        candidate_pool_size: int | None = None,
        top_n: int | None = None,
    ) -> RecruitmentWorkflowResult:
        """
        Retrieve, evaluate, and rank candidates for the job description.

        All pooled candidates are evaluated and ranked by the LLM in one
        batch call. A single batch failure after provider fallback marks
        every pooled candidate with a transparent error rather than an
        invented evaluation.
        """

        if not isinstance(job_description, str) or not job_description.strip():
            raise ValueError("Job description cannot be empty.")

        pool_size = (
            self.candidate_pool_size
            if candidate_pool_size is None
            else candidate_pool_size
        )

        result_top_n = self.top_n if top_n is None else top_n

        if (
            not isinstance(pool_size, int)
            or isinstance(pool_size, bool)
            or pool_size < 1
        ):
            raise ValueError(
                "Candidate pool size must be a positive integer."
            )

        if (
            not isinstance(result_top_n, int)
            or isinstance(result_top_n, bool)
            or result_top_n < 1
        ):
            raise ValueError("Top-N count must be a positive integer.")

        # 1. Convert the job description into an embedding.
        job_embedding = self.embedding_service.embed_job_description(
            job_description.strip()
        )

        # 2. Retrieve resume chunks using semantic similarity.
        # Multiple chunks may belong to one candidate.
        chunk_limit = pool_size * self.chunks_per_candidate_slot

        retrieved_chunks = self.vector_store_service.query_by_embedding(
            job_embedding,
            n_results=chunk_limit,
        )

        # 3. Group chunks and pre-select the pool by best matching chunk.
        # Similarity is a retrieval mechanism only; it never produces the
        # final suitability score or ranking.
        grouped_chunks = self._group_chunks_by_candidate(retrieved_chunks)

        pool_candidates = sorted(
            grouped_chunks.items(),
            key=lambda item: min(
                self._get_distance(chunk) for chunk in item[1]
            ),
        )[:pool_size]

        if not pool_candidates:
            return RecruitmentWorkflowResult(
                success=True,
                message="No candidates were found for this job description.",
                candidate_pool_size=pool_size,
                top_n=result_top_n,
                retrieved_chunk_count=len(retrieved_chunks),
                unique_candidate_count=0,
            )

        # 4. Build one CandidateWorkflowResult and one consolidated
        # context per pooled candidate.
        candidate_results: dict[str, CandidateWorkflowResult] = {}
        candidate_contexts: dict[str, str] = {}

        for candidate_id, candidate_chunks in pool_candidates:
            best_distance = min(
                self._get_distance(chunk) for chunk in candidate_chunks
            )

            result = CandidateWorkflowResult(
                candidate_id=candidate_id,
                retrieval_distance=best_distance,
            )

            candidate_results[candidate_id] = result

            # Fetch all indexed chunks for this candidate so the context
            # covers every available resume section.
            consolidated_chunks = self._get_consolidated_candidate_chunks(
                candidate_id,
                job_embedding,
                candidate_chunks,
            )

            result.matched_sections = list(
                dict.fromkeys(
                    chunk["section"]
                    for chunk in consolidated_chunks
                    if isinstance(chunk.get("section"), str)
                    and chunk["section"].strip()
                )
            )

            candidate_context = self._build_candidate_context(
                consolidated_chunks
            )

            if candidate_context.strip():
                candidate_contexts[candidate_id] = candidate_context
            else:
                result.error = (
                    "Candidate evaluation failed: ValueError: "
                    "No usable resume chunks were retrieved "
                    "for this candidate."
                )

        # 5. Evaluate and rank the whole pool in one batch LLM call.
        results = list(candidate_results.values())

        if candidate_contexts:
            try:
                ranked_evaluations, provider_name = evaluate_candidates(
                    candidate_contexts=candidate_contexts,
                    job_description=job_description.strip(),
                )

                for assessment in ranked_evaluations.candidates:
                    result = candidate_results[assessment.candidate_id]
                    result.rank = assessment.rank
                    result.provider_name = provider_name
                    result.evaluation = assessment

            except Exception as exc:
                # Keep the failure explicit on every pooled candidate;
                # never invent a score or ranking position.
                error_message = (
                    "Candidate evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                for result in results:
                    if result.evaluation is None:
                        result.error = error_message

        # 6. Return the top-N candidates in the LLM's ranking order.
        evaluated_results = sorted(
            (
                result
                for result in results
                if result.rank is not None
            ),
            key=lambda result: result.rank,
        )

        top_results = evaluated_results[:result_top_n]

        failed_results = [
            result
            for result in results
            if result.rank is None
        ]

        if not evaluated_results:
            message = (
                "Candidate evaluation did not complete for any "
                "retrieved candidate. See the recorded errors for "
                "details."
            )
        else:
            message = (
                f"Evaluated {len(evaluated_results)} of the "
                f"{pool_size} requested candidate slots; returning the "
                f"top {len(top_results)} by LLM ranking."
            )

            if failed_results:
                message += (
                    f" {len(failed_results)} retrieved candidate(s) "
                    "could not be evaluated; see the recorded errors."
                )

        return RecruitmentWorkflowResult(
            success=True,
            message=message,
            candidate_pool_size=pool_size,
            top_n=result_top_n,
            retrieved_chunk_count=len(retrieved_chunks),
            unique_candidate_count=len(grouped_chunks),
            evaluated_candidate_count=len(evaluated_results),
            candidates=[*top_results, *failed_results],
        )
