"""Chunk domain models."""

from typing import Any, Optional

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
    chunk_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    document_id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: ChunkMetadata

    def to_chroma_dict(self) -> dict[str, Any]:
        """Convert chunk to ChromaDB insertion format."""
        return {
            "id": self.chunk_id,
            "document": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata.model_dump(),
        }
