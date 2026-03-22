"""Document ingestion API endpoint."""

import base64
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.models.document import IngestionJob, IngestionResponse, JobStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingestion"])

ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# In-memory job store (replace with Redis/DB in production)
_jobs: dict[str, IngestionJob] = {}


@router.post(
    "/",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
    description=(
        "Upload a .txt, .pdf or .docx file. "
        "Returns an ingestion job ID for status tracking."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document to ingest (.txt, .pdf, .docx)"),
    chunk_size: int = Form(default=512, ge=64, le=4096),
    chunk_overlap: int = Form(default=64, ge=0, le=512),
) -> IngestionResponse:
    """Accept a document upload and queue it for background processing."""
    _validate_file(file)

    content = await file.read()
    _validate_size(content, file.filename or "unknown")

    job = IngestionJob(
        filename=file.filename or "unknown",
        content_type=file.content_type or "application/octet-stream",
    )
    _jobs[job.job_id] = job

    logger.info(
        "Queued ingestion job %s for file %r (%d bytes)",
        job.job_id,
        file.filename,
        len(content),
    )

    # Dispatch Celery task (best-effort; log and continue if broker unavailable)
    _dispatch_task(job, content, chunk_size, chunk_overlap)

    return IngestionResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        filename=job.filename,
        status=job.status,
        message="Document queued for ingestion. Use job_id to track status.",
    )


@router.get(
    "/{job_id}",
    response_model=IngestionJob,
    summary="Get ingestion job status",
)
async def get_job_status(job_id: str) -> IngestionJob:
    """Return the current status of an ingestion job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found.",
        )
    return job


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _validate_file(file: UploadFile) -> None:
    content_type = file.content_type or ""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()

    # Accept by MIME type or by extension when MIME is octet-stream
    allowed_exts = {".txt", ".pdf", ".docx"}
    if content_type not in ALLOWED_CONTENT_TYPES and ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: {content_type!r}. "
                "Allowed: .txt, .pdf, .docx"
            ),
        )


def _validate_size(content: bytes, filename: str) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File {filename!r} exceeds maximum size of "
                f"{settings.max_upload_size_mb} MB."
            ),
        )


def _dispatch_task(
    job: IngestionJob, content: bytes, chunk_size: int, chunk_overlap: int
) -> None:
    """Send the ingestion task to Celery (non-blocking, best-effort)."""
    try:
        from app.workers.ingestion_worker import ingest_document  # noqa: PLC0415

        content_b64 = base64.b64encode(content).decode()
        ingest_document.delay(
            job_id=job.job_id,
            document_id=job.document_id,
            filename=job.filename,
            content_type=job.content_type,
            content_b64=content_b64,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        job.status = JobStatus.PENDING
        logger.debug("Task dispatched for job %s", job.job_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not dispatch Celery task for job %s: %s. "
            "Processing will not occur until broker is available.",
            job.job_id,
            exc,
        )
