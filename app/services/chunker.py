"""Text chunking strategies."""

import logging
from dataclasses import dataclass

from app.models.chunk import ChunkMetadata, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: str = "overlap"  # "fixed" | "overlap"


class TextChunker:
    """
    Splits text into chunks using fixed-size or overlap strategies.

    Fixed-size: non-overlapping chunks of exactly `chunk_size` chars.
    Overlap: sliding window chunks with `chunk_overlap` char overlap.
    """

    def chunk(
        self,
        text: str,
        document_id: str,
        job_id: str,
        filename: str,
        config: ChunkConfig | None = None,
    ) -> list[DocumentChunk]:
        """
        Split *text* into DocumentChunk objects.

        Args:
            text: Source text to split.
            document_id: Parent document identifier.
            job_id: Ingestion job identifier.
            filename: Original filename for metadata.
            config: Chunking configuration; uses defaults if None.

        Returns:
            List of DocumentChunk instances with metadata.
        """
        if config is None:
            config = ChunkConfig()

        if config.strategy == "fixed":
            raw_chunks = self._fixed_chunks(text, config.chunk_size)
        else:
            raw_chunks = self._overlap_chunks(
                text, config.chunk_size, config.chunk_overlap
            )

        total = len(raw_chunks)
        chunks: list[DocumentChunk] = []

        for idx, (start, end, chunk_text) in enumerate(raw_chunks):
            metadata = ChunkMetadata(
                document_id=document_id,
                job_id=job_id,
                filename=filename,
                chunk_index=idx,
                total_chunks=total,
                start_char=start,
                end_char=end,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                strategy=config.strategy,
            )
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    text=chunk_text,
                    metadata=metadata,
                )
            )

        logger.info(
            "Chunked document %r into %d chunks (strategy=%s, size=%d, overlap=%d)",
            document_id,
            total,
            config.strategy,
            config.chunk_size,
            config.chunk_overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Private chunking algorithms
    # ------------------------------------------------------------------

    def _fixed_chunks(self, text: str, size: int) -> list[tuple[int, int, str]]:
        """Non-overlapping fixed-size chunks."""
        results: list[tuple[int, int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            results.append((start, end, text[start:end]))
            start = end
        return results

    def _overlap_chunks(
        self, text: str, size: int, overlap: int
    ) -> list[tuple[int, int, str]]:
        """Sliding-window chunks with configurable overlap."""
        if overlap >= size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than chunk_size ({size})"
            )

        results: list[tuple[int, int, str]] = []
        step = size - overlap
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            results.append((start, end, text[start:end]))
            if end == len(text):
                break
            start += step
        return results
