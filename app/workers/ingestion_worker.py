"""Ingestion worker: text extraction and chunking (pipeline stage 1)."""

import base64
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
    Stage 1 — decode the uploaded file, extract text, and split into chunks.

    Chains to ``workers.embed_chunks`` on success.
    """
    logger.info("Ingestion started: job=%s file=%r", job_id, filename)

    try:
        content = base64.b64decode(content_b64)

        extractor = TextExtractor()
        text, extract_ms = measure_ms(extractor.extract, content, filename, content_type)
        logger.info(
            "Extraction complete: job=%s chars=%d duration_ms=%.2f",
            job_id,
            len(text),
            extract_ms,
        )

        chunker = TextChunker()
        config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="overlap",
        )
        chunks, chunk_ms = measure_ms(
            chunker.chunk, text, document_id, job_id, filename, config
        )
        logger.info(
            "Chunking complete: job=%s chunks=%d duration_ms=%.2f",
            job_id,
            len(chunks),
            chunk_ms,
        )

        celery_app.send_task(
            "workers.embed_chunks",
            kwargs={"job_id": job_id, "chunks_data": [c.model_dump() for c in chunks]},
        )

        return {
            "job_id": job_id,
            "status": JobStatus.CHUNKING,
            "chunk_count": len(chunks),
            "extract_ms": extract_ms,
            "chunk_ms": chunk_ms,
        }

    except (ValueError, ImportError) as exc:
        # Non-retryable: bad file format or missing optional dependency.
        logger.error("Ingestion permanently failed: job=%s reason=%s", job_id, exc)
        raise
    except Exception as exc:
        logger.exception("Ingestion failed (will retry): job=%s", job_id)
        raise self.retry(exc=exc, countdown=5)
