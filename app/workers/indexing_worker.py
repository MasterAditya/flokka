"""Indexing worker: store embedded chunks in ChromaDB."""

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
    Stage 3 – Upsert embedded chunks into ChromaDB.
    """
    logger.info("Starting indexing for job %s (%d chunks)", job_id, len(chunks_data))

    try:
        chunks = [DocumentChunk(**c) for c in chunks_data]

        store = VectorStore()
        stored, index_ms = measure_ms(store.upsert_chunks, chunks)
        logger.info("Indexed %d chunks in %.2f ms", stored, index_ms)

        return {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "chunk_count": stored,
            "index_ms": index_ms,
        }

    except Exception as exc:
        logger.exception("Indexing failed for job %s: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=5)
