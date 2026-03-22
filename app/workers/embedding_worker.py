"""Embedding worker: generate embeddings for chunks."""

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
    Stage 2 – Generate embeddings for chunks.

    Chains to index_chunks on success.
    """
    logger.info("Starting embedding for job %s (%d chunks)", job_id, len(chunks_data))

    try:
        chunks = [DocumentChunk(**c) for c in chunks_data]

        embedder = EmbeddingService()
        embedded_chunks, embed_ms = measure_ms(embedder.embed_chunks, chunks)
        logger.info("Embedded %d chunks in %.2f ms", len(embedded_chunks), embed_ms)

        embedded_data = [c.model_dump() for c in embedded_chunks]

        # Chain to indexing worker
        index_chunks.delay(job_id=job_id, chunks_data=embedded_data)

        return {
            "job_id": job_id,
            "status": JobStatus.EMBEDDING,
            "chunk_count": len(embedded_chunks),
            "embed_ms": embed_ms,
        }

    except Exception as exc:
        logger.exception("Embedding failed for job %s: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=5)


# Avoid circular import
from app.workers.indexing_worker import index_chunks  # noqa: E402
