"""Document domain models."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    chunk_count: int = 0
    processing_time_ms: Optional[float] = None


class IngestionRequest(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class IngestionResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: JobStatus
    message: str
