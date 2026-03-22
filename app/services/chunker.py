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
    Splits plain text into overlapping or fixed-size chunks.

    Fixed-size strategy: non-overlapping windows of exactly *chunk_size*
    characters (last chunk may be shorter).

    Overlap strategy: sliding window of *chunk_size* characters advancing
    by ``chunk_size - chunk_overlap`` characters per step, so each pair of
    consecutive chunks shares *chunk_overlap* characters.
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
        Split *text* into :class:`DocumentChunk` objects with full metadata.

        Args:
            text: Source text to split.
            document_id: Identifier of the parent document.
            job_id: Identifier of the ingestion job.
            filename: Original filename, recorded in chunk metadata.
            config: Chunking parameters; defaults are used when ``None``.

        Returns:
            Ordered list of chunks covering the entire input text.

        Raises:
            ValueError: If ``chunk_overlap >= chunk_size`` for the overlap strategy.
        """
        if config is None:
            config = ChunkConfig()

        if config.strategy == "fixed":
            windows = self._fixed_windows(text, config.chunk_size)
        else:
            windows = self._overlap_windows(text, config.chunk_size, config.chunk_overlap)

        total = len(windows)
        chunks = [
            DocumentChunk(
                document_id=document_id,
                text=chunk_text,
                metadata=ChunkMetadata(
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
                ),
            )
            for idx, (start, end, chunk_text) in enumerate(windows)
        ]

        logger.info(
            "Chunked document %r: chunks=%d strategy=%s size=%d overlap=%d",
            document_id,
            total,
            config.strategy,
            config.chunk_size,
            config.chunk_overlap,
        )
        return chunks

    def _fixed_windows(self, text: str, size: int) -> list[tuple[int, int, str]]:
        """Non-overlapping fixed-size windows."""
        windows: list[tuple[int, int, str]] = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            windows.append((start, end, text[start:end]))
            start = end
        return windows

    def _overlap_windows(
        self, text: str, size: int, overlap: int
    ) -> list[tuple[int, int, str]]:
        """Sliding-window chunks with configurable overlap."""
        if overlap >= size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than chunk_size ({size})"
            )
        windows: list[tuple[int, int, str]] = []
        step = size - overlap
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            windows.append((start, end, text[start:end]))
            if end == len(text):
                break
            start += step
        return windows
