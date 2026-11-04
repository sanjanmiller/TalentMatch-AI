
"""
File: indexing_service.py

Purpose:
Coordinate the indexing of stored resume profiles into ChromaDB.

Workflow:
1. Read candidate IDs from the SQLite resume_profiles table.
2. Load and validate each stored professional profile.
3. Split the profile into section-based chunks.
4. Generate embeddings using the existing embedding service.
5. Replace the candidate's indexed chunks in ChromaDB.

Candidate identity and contact details remain in SQLite.
This module does not calculate candidate scores or make hiring decisions.
"""

import json
from pathlib import Path
from typing import Any

from src.schemas.resume_schema import ResumeProfile
from src.services.chunking_service import chunk_resume_profile
from src.services.db_service import (
    DEFAULT_DATABASE_PATH,
    get_connection,
    get_resume_profile,
)
from src.services.embedding_service import EmbeddingService
from src.services.vector_store_service import VectorStoreService


# Resolve the project root independently of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Keep the persistent vector store in the project's storage directory.
DEFAULT_VECTOR_STORE_PATH = PROJECT_ROOT / "storage" / "vector_store"


def index_candidate(
    candidate_id: str,
    db_path: str | Path | None = None,
    vector_store: VectorStoreService | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """
    Index one candidate's stored resume profile into ChromaDB.

    This runs the same steps as the candidate part of the full indexing
    workflow: load the profile, chunk it, embed every chunk, and replace
    that candidate's previous chunks in the vector store.

    Args:
        candidate_id: Stable candidate ID linking SQLite and ChromaDB.
        db_path: Optional SQLite database path, mainly useful for testing.
        vector_store: Optional existing vector-store service.
        embedding_service: Optional existing embedding service.

    Returns:
        A result dictionary containing the candidate ID, chunk count,
        and the sorted list of indexed section names.

    Raises:
        ValueError: If no stored profile exists for the candidate.
    """

    database_path = (
        Path(db_path) if db_path is not None else DEFAULT_DATABASE_PATH
    )

    # Reuse one service instance across every call in a run.
    if vector_store is None:
        vector_store = VectorStoreService(
            persist_directory=DEFAULT_VECTOR_STORE_PATH
        )

    if embedding_service is None:
        embedding_service = EmbeddingService()

    # Retrieve the stored profile as a Python dictionary.
    profile_data = get_resume_profile(
        candidate_id,
        db_path=database_path,
    )

    if profile_data is None:
        raise ValueError(
            "No stored resume profile was found for this candidate."
        )

    # Validate the dictionary, including nested profile entries.
    profile = ResumeProfile.model_validate(profile_data)

    # Build privacy-conscious, section-based chunks.
    chunks = chunk_resume_profile(candidate_id, profile)

    # Generate vectors in the same order as the chunks.
    embeddings = embedding_service.embed_chunks(chunks)

    # Save the vectors and remove stale chunks for this candidate.
    vector_store.replace_candidate_chunks(
        candidate_id=candidate_id,
        chunks=chunks,
        embeddings=embeddings,
    )

    return {
        "candidate_id": candidate_id,
        "chunks": len(chunks),
        "sections": sorted({chunk.section for chunk in chunks}),
    }


def index_all_candidates(
    db_path: str | Path | None = None,
    vector_store: VectorStoreService | None = None,
    embedding_service: EmbeddingService | None = None,
) -> dict[str, Any]:
    """
    Index all candidates with stored resume profiles.

    Args:
        db_path: Optional SQLite database path, mainly useful for testing.
        vector_store: Optional existing vector-store service.
        embedding_service: Optional existing embedding service.

    Returns:
        A summary containing candidate counts, chunk counts, and
        individual candidate results.

    Notes:
        The embedding model is initialized at most once per run.
        Individual candidate failures are recorded without stopping
        the processing of other candidates.
    """
    database_path = (
        Path(db_path) if db_path is not None else DEFAULT_DATABASE_PATH
    )

    # Read only candidate IDs. Identity and contact details are not
    # needed for embedding or vector-store metadata.
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT candidate_id
            FROM resume_profiles
            ORDER BY candidate_id
            """
        ).fetchall()

    candidate_ids = [row["candidate_id"] for row in rows]

    summary: dict[str, Any] = {
        "candidates_found": len(candidate_ids),
        "indexed_candidates": 0,
        "failed_candidates": 0,
        "total_chunks_indexed": 0,
        "results": [],
    }

    # Avoid loading the embedding model when there is nothing to index.
    if not candidate_ids:
        return summary

    # Reuse one service instance throughout the indexing run.
    if vector_store is None:
        vector_store = VectorStoreService(
            persist_directory=DEFAULT_VECTOR_STORE_PATH
        )

    if embedding_service is None:
        embedding_service = EmbeddingService()

    for candidate_id in candidate_ids:
        try:
            result = index_candidate(
                candidate_id,
                db_path=database_path,
                vector_store=vector_store,
                embedding_service=embedding_service,
            )

            summary["indexed_candidates"] += 1
            summary["total_chunks_indexed"] += result["chunks"]

            summary["results"].append(
                {
                    **result,
                    "status": "indexed",
                }
            )

        except Exception as exc:
            # Record the error and continue with the next candidate.
            summary["failed_candidates"] += 1

            summary["results"].append(
                {
                    "candidate_id": candidate_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return summary


def main() -> None:
    """Run the indexing workflow and print a readable summary."""
    summary = index_all_candidates()

    print("\nResume indexing summary")
    print("=" * 40)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if summary["failed_candidates"] > 0:
        print(
            "\nSome candidates could not be indexed. "
            "Review the errors above before continuing."
        )


if __name__ == "__main__":
    main()
