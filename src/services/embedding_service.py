
"""
Embedding service for the Intelligent Recruitment System.

This module converts resume chunks and job descriptions into numeric
vectors using one reusable Sentence Transformers model.

The service is independent of ChromaDB. It only generates and validates
embeddings; storing and searching vectors will be handled separately.
"""

import math
import os
from collections.abc import Sequence
from typing import Any

from sentence_transformers import SentenceTransformer


# A compact model suitable for the initial local prototype.
# Override this through the EMBEDDING_MODEL_NAME environment variable.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate and validate embeddings for recruitment documents."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
    ) -> None:
        """
        Initialize the embedding service.

        Args:
            model_name: Optional Sentence Transformers model identifier.
            model: Optional preloaded or mocked model, mainly useful for
                testing without downloading model weights.
        """
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL_NAME")
            or DEFAULT_MODEL_NAME
        )

        # Reuse the model rather than loading it for every embedding call.
        if model is not None:
            self.model = model
        else:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Check the model name, installed dependencies, and "
                    "model-download/network access."
                ) from exc

        self._embedding_dimension: int | None = None

        # Prefer the current API name to avoid a FutureWarning.
        # Fall back to the old name for compatibility with older versions.
        dimension_method = getattr(
            self.model,
            "get_embedding_dimension",
            None,
        )

        if not callable(dimension_method):
            dimension_method = getattr(
                self.model,
                "get_sentence_embedding_dimension",
                None,
            )

        # Validate the dimension when the model provides it.
        if callable(dimension_method):
            dimension = dimension_method()

            if dimension is not None:
                if (
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension <= 0
                ):
                    raise ValueError(
                        "The embedding model reported an invalid "
                        "embedding dimension."
                    )

                self._embedding_dimension = dimension

    @staticmethod
    def _validate_text(text: str) -> None:
        """Reject values that are not non-empty strings."""
        if not isinstance(text, str):
            raise TypeError("Embedding input must be a string.")

        if not text.strip():
            raise ValueError(
                "Embedding input cannot be empty or whitespace."
            )

    def embed_text(self, text: str) -> list[float]:
        """
        Generate an embedding for one text.

        Returns:
            A list of finite floating-point numbers.
        """
        return self.embed_texts([text])[0]

    def embed_job_description(self, description: str) -> list[float]:
        """Embed a job description using the same model as resumes."""
        return self.embed_text(description)

    def embed_chunks(
        self,
        chunks: Sequence[Any],
    ) -> list[list[float]]:
        """
        Generate embeddings for ResumeChunk-like objects.

        Each object must have a text attribute. The input objects, their
        IDs, and their metadata are not modified.

        Returns:
            Embeddings in the same order as the supplied chunks.
        """
        if isinstance(chunks, (str, bytes)):
            raise TypeError(
                "Pass a sequence of chunk objects, not a single string."
            )

        texts: list[str] = []

        for index, chunk in enumerate(chunks):
            if not hasattr(chunk, "text"):
                raise TypeError(
                    f"Chunk at index {index} must have a 'text' attribute."
                )

            texts.append(chunk.text)

        return self.embed_texts(texts)

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        An empty batch returns an empty list without calling the model.
        Invalid entries cause an error rather than being silently skipped.
        """
        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "Pass a sequence of strings to embed_texts(), "
                "not a single string."
            )

        if not isinstance(texts, Sequence):
            raise TypeError("Batch input must be a sequence of strings.")

        if len(texts) == 0:
            return []

        # Validate all entries before making a model call.
        for index, text in enumerate(texts):
            try:
                self._validate_text(text)
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"Invalid text at batch index {index}: {exc}"
                ) from exc

        try:
            raw_embeddings = self.model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "The embedding model failed to encode text."
            ) from exc

        # Convert NumPy arrays or tensor-like results into ordinary lists.
        if hasattr(raw_embeddings, "detach"):
            raw_embeddings = raw_embeddings.detach().cpu()

        if hasattr(raw_embeddings, "tolist"):
            raw_embeddings = raw_embeddings.tolist()

        if not isinstance(raw_embeddings, (list, tuple)):
            raise ValueError(
                "The model returned an unsupported output format."
            )

        if len(raw_embeddings) != len(texts):
            raise ValueError(
                "The model returned a different number of embeddings "
                "than the number of input texts."
            )

        validated_embeddings: list[list[float]] = []

        for row_index, raw_vector in enumerate(raw_embeddings):
            if hasattr(raw_vector, "tolist"):
                raw_vector = raw_vector.tolist()

            if not isinstance(raw_vector, (list, tuple)) or not raw_vector:
                raise ValueError(
                    f"Embedding at index {row_index} is not a "
                    "non-empty vector."
                )

            vector: list[float] = []

            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(
                    value, (int, float)
                ):
                    raise ValueError(
                        f"Embedding at index {row_index} contains "
                        "a non-numeric value."
                    )

                number = float(value)

                if not math.isfinite(number):
                    raise ValueError(
                        f"Embedding at index {row_index} contains "
                        "a non-finite value."
                    )

                vector.append(number)

            if self._embedding_dimension is None:
                self._embedding_dimension = len(vector)
            elif len(vector) != self._embedding_dimension:
                raise ValueError(
                    f"Embedding at index {row_index} has dimension "
                    f"{len(vector)}; expected "
                    f"{self._embedding_dimension}."
                )

            validated_embeddings.append(vector)

        return validated_embeddings
