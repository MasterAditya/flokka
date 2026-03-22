"""Embedding service using sentence-transformers with a deterministic CI fallback."""

import hashlib
import logging
import math

from app.core.config import settings
from app.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Produces dense vector embeddings for text chunks.

    When *simulate* is ``True`` (or when sentence-transformers cannot be
    loaded), embeddings are computed via a deterministic SHA-256 hash.  The
    hash-based fallback is structurally correct — it produces normalised
    float vectors of the configured dimension — but has no semantic meaning.
    It exists solely to let the full pipeline run in CI without model
    downloads or GPU access.
    """

    def __init__(self, model_name: str | None = None, simulate: bool = False) -> None:
        self._model_name = model_name or settings.embedding_model
        self._dim = settings.embedding_dim
        self._simulate = simulate
        self._model = None  # lazy-loaded on first real embed call

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Attach embeddings to each chunk in-place and return the list."""
        texts = [c.text for c in chunks]
        embeddings = self._encode(texts)
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk.embedding = embedding
        logger.info(
            "Embedded %d chunks (model=%s, simulated=%s)",
            len(chunks),
            self._model_name,
            self._simulate,
        )
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Produce a single query embedding for nearest-neighbour retrieval."""
        return self._encode([query])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._simulate:
            return [self._hash_embedding(t) for t in texts]

        model = self._load_model()
        if model is None:
            return [self._hash_embedding(t) for t in texts]

        return model.encode(texts, show_progress_bar=False).tolist()

    def _load_model(self):
        """Lazy-load the sentence-transformers model; fall back to simulation on error."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load sentence-transformers model %r (%s). "
                "Falling back to hash-based simulation.",
                self._model_name,
                exc,
            )
            self._simulate = True
        return self._model

    def _hash_embedding(self, text: str) -> list[float]:
        """
        Deterministic pseudo-embedding derived from SHA-256.

        Produces a normalised vector of ``self._dim`` floats in [-1, 1].
        Not suitable for semantic retrieval — only for structural testing.
        """
        digest = hashlib.sha256(text.encode()).digest()  # 32 bytes
        raw = [(b / 127.5) - 1.0 for b in digest]
        repeated = (raw * (self._dim // len(raw) + 1))[: self._dim]
        norm = math.sqrt(sum(v * v for v in repeated)) or 1.0
        return [v / norm for v in repeated]
