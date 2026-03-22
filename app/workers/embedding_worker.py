"""Embedding worker: dense vector generation for chunks (pipeline stage 2)."""

import logging

from app.models.chunk import DocumentChunk
from app.models.document import JobStatus
from app.services.embedder import EmbeddingService
from app.utils.timing import measure_ms
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.embed_chunks", bind=True, max_retries=3)
def embed_chunks(self, job_id: str, chunks_data: list[dict]) -> dict:
    """
    Stage 2 — encode each chunk into a dense float vector.

    Chains to ``workers.index_chunks`` on success.
    """
    logger.info(
        "Embedding started: job=%s chunks=%d", job_id, len(chunks_data)
    )

    try:
        chunks = [DocumentChunk(**c) for c in chunks_data]

        embedder = EmbeddingService()
        embedded_chunks, embed_ms = measure_ms(embedder.embed_chunks, chunks)
        logger.info(
            "Embedding complete: job=%s chunks=%d duration_ms=%.2f",
            job_id,
            len(embedded_chunks),
            embed_ms,
        )

        celery_app.send_task(
            "workers.index_chunks",
            kwargs={
                "job_id": job_id,
                "chunks_data": [c.model_dump() for c in embedded_chunks],
            },
        )

        return {
            "job_id": job_id,
            "status": JobStatus.EMBEDDING,
            "chunk_count": len(embedded_chunks),
            "embed_ms": embed_ms,
        }

    except Exception as exc:
        logger.exception("Embedding failed (will retry): job=%s", job_id)
        raise self.retry(exc=exc, countdown=5)
