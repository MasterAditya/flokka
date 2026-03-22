"""Document ingestion API endpoints."""

import base64
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.document import IngestionJob, IngestionResponse, JobStatus
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

ALLOWED_CONTENT_TYPES = frozenset(
    {
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)
ALLOWED_EXTENSIONS = frozenset({".txt", ".pdf", ".docx"})

# In-memory job registry (P0: replace with Redis hash before multi-replica deploy).
_jobs: dict[str, IngestionJob] = {}


@router.post(
    "/",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
    description=(
        "Upload a .txt, .pdf or .docx file. "
        "The document is queued for background processing; use the returned "
        "``job_id`` to poll for status."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (.txt, .pdf, .docx)"),
    chunk_size: int = Form(default=512, ge=64, le=4096),
    chunk_overlap: int = Form(default=64, ge=0, le=512),
) -> IngestionResponse:
    """Accept a document upload and enqueue it for background processing."""
    _validate_file_type(file)

    content = await file.read()
    _validate_file_size(content, file.filename or "unknown")

    job = IngestionJob(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
    )
    _jobs[job.job_id] = job

    logger.info(
        "Ingestion job created: job=%s document=%s file=%r bytes=%d",
        job.job_id,
        job.document_id,
        file.filename,
        len(content),
    )

    _enqueue(job, content, chunk_size, chunk_overlap)

    return IngestionResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        filename=job.filename,
        status=job.status,
        message="Document queued for ingestion. Use job_id to poll for status.",
    )


@router.get(
    "/{job_id}",
    response_model=IngestionJob,
    summary="Get ingestion job status",
)
async def get_job_status(job_id: str) -> IngestionJob:
    """Return the current state of an ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )
    return job


def _validate_file_type(file: UploadFile) -> None:
    content_type = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()
    if content_type not in ALLOWED_CONTENT_TYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: content_type={content_type!r}, "
                f"extension={ext!r}. Accepted: .txt, .pdf, .docx"
            ),
        )


def _validate_file_size(content: bytes, filename: str) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File {filename!r} ({len(content)} bytes) exceeds the "
                f"{settings.max_upload_size_mb} MB upload limit."
            ),
        )


def _enqueue(
    job: IngestionJob, content: bytes, chunk_size: int, chunk_overlap: int
) -> None:
    """Send the ingestion task to the Celery broker (best-effort)."""
    try:
        celery_app.send_task(
            "workers.ingest_document",
            kwargs={
                "job_id": job.job_id,
                "document_id": job.document_id,
                "filename": job.filename,
                "content_type": job.content_type,
                "content_b64": base64.b64encode(content).decode(),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )
        job.status = JobStatus.PENDING
        logger.debug("Task enqueued: job=%s", job.job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Broker unavailable — task not enqueued: job=%s error=%s",
            job.job_id,
            exc,
        )
