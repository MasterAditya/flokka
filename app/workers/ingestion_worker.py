"""Ingestion worker: text extraction and chunking."""

import logging

from app.models.document import JobStatus
from app.services.chunker import ChunkConfig, TextChunker
from app.services.text_extractor import TextExtractor
from app.utils.timing import measure_ms
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.ingest_document", bind=True, max_retries=3)
def ingest_document(
    self,
    job_id: str,
    document_id: str,
    filename: str,
    content_type: str,
    content_b64: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """
    Stage 1 – Extract text and split into chunks.

    Chains to embed_chunks on success.
    """
    import base64  # noqa: PLC0415

    logger.info("Starting ingestion for job %s (file=%r)", job_id, filename)

    try:
        content = base64.b64decode(content_b64)

        extractor = TextExtractor()
        text, extract_ms = measure_ms(
            extractor.extract, content, filename, content_type
        )
        logger.info("Extracted %d chars in %.2f ms", len(text), extract_ms)

        chunker = TextChunker()
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="overlap",
        )
        chunks, chunk_ms = measure_ms(
            chunker.chunk, text, document_id, job_id, filename, config
        )
        logger.info("Created %d chunks in %.2f ms", len(chunks), chunk_ms)

        # Pass serialised chunks to the next worker
        chunks_data = [c.model_dump() for c in chunks]

        # Chain to embedding worker
        embed_chunks.delay(job_id=job_id, chunks_data=chunks_data)

        return {
            "job_id": job_id,
            "status": JobStatus.CHUNKING,
            "chunk_count": len(chunks),
            "extract_ms": extract_ms,
            "chunk_ms": chunk_ms,
        }

    except Exception as exc:
        logger.exception("Ingestion failed for job %s: %s", job_id, exc)
        raise self.retry(exc=exc, countdown=5)


# Avoid circular import – import here to enable .delay()
from app.workers.embedding_worker import embed_chunks  # noqa: E402
