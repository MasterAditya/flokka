"""Chunk domain models."""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    job_id: str
    filename: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    chunk_size: int
    overlap: int
    strategy: str


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    document_id: str
    text: str
    embedding: list[float] | None = None
    metadata: ChunkMetadata

    def to_chroma_dict(self) -> dict[str, Any]:
        """Serialise to the shape expected by ChromaDB's upsert API."""
        return {
            "id": self.chunk_id,
            "document": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata.model_dump(),
        }
