"""Embedding service using sentence-transformers with a local fallback."""

import hashlib
import logging
import math

from app.core.config import settings
from app.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings for text chunks.

    Uses sentence-transformers when available; falls back to a
    deterministic hash-based simulation so the pipeline can run
    without GPU/model downloads in CI/testing environments.
    """

    def __init__(self, model_name: str | None = None, simulate: bool = False) -> None:
        self._model_name = model_name or settings.embedding_model
        self._model = None
        self._use_simulation = simulate
        self._dim = settings.embedding_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Attach embeddings to a list of DocumentChunk objects in-place."""
        model = self._get_model()
        texts = [c.text for c in chunks]

        if self._use_simulation:
            embeddings = [self._simulated_embedding(t) for t in texts]
        else:
            embeddings = model.encode(texts, show_progress_bar=False).tolist()

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        logger.info(
            "Generated %d embeddings (model=%s, simulated=%s)",
            len(chunks),
            self._model_name,
            self._use_simulation,
        )
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string for retrieval."""
        model = self._get_model()
        if self._use_simulation:
            return self._simulated_embedding(query)
        return model.encode([query], show_progress_bar=False)[0].tolist()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None or self._use_simulation:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded sentence-transformers model: %s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load sentence-transformers (%s). Using simulated embeddings.",
                exc,
            )
            self._use_simulation = True

        return self._model

    def _simulated_embedding(self, text: str) -> list[float]:
        """
        Deterministic pseudo-embedding for testing/CI.

        Hashes the text to produce a stable, normalised vector.
        NOT suitable for semantic search – only for structural testing.
        """
        digest = hashlib.sha256(text.encode()).digest()
        raw = [(b / 127.5) - 1.0 for b in digest]  # 32 values in [-1, 1]

        # Stretch/truncate to embedding_dim
        repeated = (raw * (self._dim // len(raw) + 1))[: self._dim]

        # L2 normalise
        norm = math.sqrt(sum(v * v for v in repeated)) or 1.0
        return [v / norm for v in repeated]
