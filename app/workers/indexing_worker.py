"""Indexing worker: upsert embedded chunks into ChromaDB (pipeline stage 3)."""

import logging

from app.models.chunk import DocumentChunk
from app.models.document import JobStatus
from app.services.vector_store import VectorStore
from app.utils.timing import measure_ms
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.index_chunks", bind=True, max_retries=3)
def index_chunks(self, job_id: str, chunks_data: list[dict]) -> dict:
    """
    Stage 3 — upsert chunks (text + embedding + metadata) into ChromaDB.

    Upserts are idempotent: re-running this task with the same chunk IDs
    is safe and will not produce duplicate entries.
    """
    logger.info("Indexing started: job=%s chunks=%d", job_id, len(chunks_data))

    try:
        chunks = [DocumentChunk(**c) for c in chunks_data]

        store = VectorStore()
        stored, index_ms = measure_ms(store.upsert_chunks, chunks)
        logger.info(
            "Indexing complete: job=%s stored=%d duration_ms=%.2f",
            job_id,
            stored,
            index_ms,
        )

        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "chunk_count": stored,
            "index_ms": index_ms,
        }

    except Exception as exc:
        logger.exception("Indexing failed (will retry): job=%s", job_id)
        raise self.retry(exc=exc, countdown=5)
