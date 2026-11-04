
"""
Persistent ChromaDB vector storage for the Intelligent Recruitment System.

This module stores resume chunks and their precomputed embeddings, supports
candidate-specific resume updates, and retrieves relevant chunks by semantic
similarity. Candidate identity and contact information remain in SQLite.

Embedding generation is deliberately handled by embedding_service.py rather
than by this module.
"""

from pathlib import Path
from typing import Any, Sequence

import chromadb

from src.services.chunking_service import ResumeChunk


DEFAULT_COLLECTION_NAME = "resume_chunks"
DEFAULT_STORAGE_PATH = Path("storage/vector_store")


class VectorStoreService:
    """Manage persistent resume-chunk embeddings in ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_STORAGE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        """Initialize a persistent ChromaDB client and resume collection."""
        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Cosine distance is suitable for normalized text embeddings.
        # ChromaDB persists the collection automatically.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _validate_embedding(embedding: Sequence[float]) -> list[float]:
        """Validate one embedding and convert it to ordinary Python floats."""
        if isinstance(embedding, (str, bytes)) or not isinstance(
            embedding, Sequence
        ):
            raise ValueError("Each embedding must be a numeric sequence.")

        if len(embedding) == 0:
            raise ValueError("Embedding cannot be empty.")

        validated: list[float] = []

        for value in embedding:
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise ValueError("Embedding values must be numeric.")

            numeric_value = float(value)

            if not __import__("math").isfinite(numeric_value):
                raise ValueError("Embedding values must be finite.")

            validated.append(numeric_value)

        return validated

    @staticmethod
    def _validate_chunks_and_embeddings(
        chunks: Sequence[ResumeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> tuple[list[ResumeChunk], list[list[float]]]:
        """Validate matching chunks and embeddings before writing to ChromaDB."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        validated_chunks = list(chunks)
        validated_embeddings = [
            VectorStoreService._validate_embedding(embedding)
            for embedding in embeddings
        ]

        if not validated_chunks:
            return validated_chunks, validated_embeddings

        chunk_ids: set[str] = set()
        expected_dimension = len(validated_embeddings[0])

        for chunk, embedding in zip(
            validated_chunks, validated_embeddings
        ):
            if not isinstance(chunk, ResumeChunk):
                raise ValueError("Every item must be a ResumeChunk.")

            if not chunk.chunk_id or not chunk.chunk_id.strip():
                raise ValueError("Chunk IDs cannot be empty.")

            if not chunk.candidate_id or not chunk.candidate_id.strip():
                raise ValueError("Candidate IDs cannot be empty.")

            if not chunk.section or not chunk.section.strip():
                raise ValueError("Chunk sections cannot be empty.")

            if not chunk.text or not chunk.text.strip():
                raise ValueError("Chunk text cannot be empty.")

            if chunk.chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID supplied: {chunk.chunk_id}"
                )

            chunk_ids.add(chunk.chunk_id)

            if len(embedding) != expected_dimension:
                raise ValueError(
                    "All embeddings in a batch must have the same dimension."
                )

        return validated_chunks, validated_embeddings

    @staticmethod
    def _metadata_for_chunk(chunk: ResumeChunk) -> dict[str, str | int]:
        """Keep only the non-contact metadata needed for retrieval."""
        # Do not blindly store every field returned by to_metadata().
        # An explicit allowlist prevents accidental metadata expansion.
        return {
            "candidate_id": chunk.candidate_id,
            "section": chunk.section,
            "chunk_index": int(chunk.chunk_index),
        }

    def upsert_chunks(
        self,
        chunks: Sequence[ResumeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Insert new chunks or update chunks with matching stable IDs.

        Returns the number of chunks submitted. This method does not remove
        older chunks that are absent from the supplied batch.
        """
        validated_chunks, validated_embeddings = (
            self._validate_chunks_and_embeddings(chunks, embeddings)
        )

        if not validated_chunks:
            return 0

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in validated_chunks],
            documents=[chunk.text for chunk in validated_chunks],
            metadatas=[
                self._metadata_for_chunk(chunk)
                for chunk in validated_chunks
            ],
            embeddings=validated_embeddings,
        )

        return len(validated_chunks)

    def replace_candidate_chunks(
        self,
        candidate_id: str,
        chunks: Sequence[ResumeChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Replace all indexed chunks for one candidate with the supplied set.

        Existing chunks not present in the new set are removed. New chunks
        and embeddings are fully validated before any stored chunks are
        changed. Pass empty lists to remove all indexed chunks for a candidate.
        """
        if not candidate_id or not candidate_id.strip():
            raise ValueError("Candidate ID cannot be empty.")

        validated_chunks, validated_embeddings = (
            self._validate_chunks_and_embeddings(chunks, embeddings)
        )

        for chunk in validated_chunks:
            if chunk.candidate_id != candidate_id:
                raise ValueError(
                    "Every replacement chunk must belong to "
                    f"candidate {candidate_id!r}."
                )

        # Find the previous chunk IDs for this candidate.
        existing = self.collection.get(
            where={"candidate_id": candidate_id},
            include=["metadatas"],
        )
        old_ids = set(existing["ids"])
        new_ids = {chunk.chunk_id for chunk in validated_chunks}

        # Upsert first, then delete stale IDs. This avoids deleting the old
        # chunks before replacement inputs have been validated and written.
        if validated_chunks:
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in validated_chunks],
                documents=[chunk.text for chunk in validated_chunks],
                metadatas=[
                    self._metadata_for_chunk(chunk)
                    for chunk in validated_chunks
                ],
                embeddings=validated_embeddings,
            )

        stale_ids = sorted(old_ids - new_ids)

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return len(validated_chunks)

    def query_by_embedding(
        self,
        query_embedding: Sequence[float],
        n_results: int = 10,
        candidate_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the closest resume chunks for a supplied query embedding.

        If candidate_id is provided, search only that candidate's chunks.
        Results contain chunk text, candidate ID, section, chunk index,
        and cosine distance. A lower distance indicates greater similarity.
        """
        validated_embedding = self._validate_embedding(query_embedding)

        if isinstance(n_results, bool) or not isinstance(n_results, int):
            raise ValueError("n_results must be a positive integer.")

        if n_results < 1:
            raise ValueError("n_results must be a positive integer.")

        if candidate_id is not None and (
            not isinstance(candidate_id, str) or not candidate_id.strip()
        ):
            raise ValueError("Candidate ID must be a non-empty string.")

        collection_size = self.collection.count()

        if collection_size == 0:
            return []

        query_options: dict[str, Any] = {
            "query_embeddings": [validated_embedding],
            "n_results": min(n_results, collection_size),
            "include": ["documents", "metadatas", "distances"],
        }

        if candidate_id is not None:
            query_options["where"] = {"candidate_id": candidate_id}

        results = self.collection.query(**query_options)

        # ChromaDB returns one result list per query embedding. We submit
        # exactly one embedding, so unpack the first result list.
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        matches: list[dict[str, Any]] = []

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if document is None or metadata is None or distance is None:
                continue

            matches.append(
                {
                    "chunk_id": chunk_id,
                    "candidate_id": metadata["candidate_id"],
                    "section": metadata["section"],
                    "chunk_index": metadata["chunk_index"],
                    "text": document,
                    "distance": float(distance),
                }
            )

        return matches

    def delete_candidate(self, candidate_id: str) -> int:
        """Delete every indexed chunk belonging to one candidate."""
        if not candidate_id or not candidate_id.strip():
            raise ValueError("Candidate ID cannot be empty.")

        existing = self.collection.get(
            where={"candidate_id": candidate_id},
            include=["metadatas"],
        )
        chunk_ids = existing["ids"]

        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

        return len(chunk_ids)

    def count_chunks(self, candidate_id: str | None = None) -> int:
        """Return the total chunk count, optionally for one candidate."""
        if candidate_id is None:
            return self.collection.count()

        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Candidate ID must be a non-empty string.")

        return len(
            self.collection.get(
                where={"candidate_id": candidate_id},
                include=["metadatas"],
            )["ids"]
        )
