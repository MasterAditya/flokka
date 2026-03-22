"""Tests for the text chunking service."""

import pytest

from app.services.chunker import ChunkConfig, TextChunker


@pytest.fixture
def chunker() -> TextChunker:
    return TextChunker()


@pytest.fixture
def long_text() -> str:
    # ~1000 chars
    return "Hello world. " * 77


class TestFixedChunking:
    def test_produces_correct_number_of_chunks(
        self, chunker: TextChunker, long_text: str
    ) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(long_text, "doc1", "job1", "test.txt", config)
        expected = -(-len(long_text) // 100)  # ceil division
        assert len(chunks) == expected

    def test_chunks_cover_full_text(self, chunker: TextChunker, long_text: str) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(long_text, "doc1", "job1", "test.txt", config)
        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == long_text

    def test_chunk_size_respected(self, chunker: TextChunker, long_text: str) -> None:
        config = ChunkConfig(chunk_size=200, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(long_text, "doc1", "job1", "test.txt", config)
        for chunk in chunks[:-1]:  # Last chunk may be shorter
            assert len(chunk.text) == 200

    def test_empty_text_returns_no_chunks(self, chunker: TextChunker) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk("", "doc1", "job1", "test.txt", config)
        assert chunks == []

    def test_text_shorter_than_chunk_size(self, chunker: TextChunker) -> None:
        config = ChunkConfig(chunk_size=1000, chunk_overlap=0, strategy="fixed")
        text = "Short text."
        chunks = chunker.chunk(text, "doc1", "job1", "test.txt", config)
        assert len(chunks) == 1
        assert chunks[0].text == text


class TestOverlapChunking:
    def test_overlap_produces_more_chunks_than_fixed(
        self, chunker: TextChunker, long_text: str
    ) -> None:
        fixed_cfg = ChunkConfig(chunk_size=200, chunk_overlap=0, strategy="fixed")
        overlap_cfg = ChunkConfig(chunk_size=200, chunk_overlap=50, strategy="overlap")
        fixed_chunks = chunker.chunk(long_text, "doc1", "job1", "test.txt", fixed_cfg)
        overlap_chunks = chunker.chunk(
            long_text, "doc1", "job1", "test.txt", overlap_cfg
        )
        assert len(overlap_chunks) >= len(fixed_chunks)

    def test_consecutive_chunks_share_overlap(self, chunker: TextChunker) -> None:
        text = "A" * 300
        config = ChunkConfig(chunk_size=100, chunk_overlap=20, strategy="overlap")
        chunks = chunker.chunk(text, "doc1", "job1", "test.txt", config)
        for i in range(len(chunks) - 1):
            # End of chunk i should equal beginning of chunk i+1 (overlap region)
            overlap_end = chunks[i].text[-20:]
            overlap_start = chunks[i + 1].text[:20]
            assert overlap_end == overlap_start

    def test_overlap_greater_than_size_raises(self, chunker: TextChunker) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=100, strategy="overlap")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunker.chunk("some text", "doc1", "job1", "test.txt", config)

    def test_default_strategy_is_overlap(self, chunker: TextChunker) -> None:
        config = ChunkConfig()
        assert config.strategy == "overlap"


class TestChunkMetadata:
    def test_metadata_fields_populated(
        self, chunker: TextChunker, long_text: str
    ) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(long_text, "doc42", "job99", "myfile.txt", config)
        meta = chunks[0].metadata
        assert meta.document_id == "doc42"
        assert meta.job_id == "job99"
        assert meta.filename == "myfile.txt"
        assert meta.chunk_index == 0
        assert meta.total_chunks == len(chunks)
        assert meta.strategy == "fixed"

    def test_chunk_indices_are_sequential(
        self, chunker: TextChunker, long_text: str
    ) -> None:
        config = ChunkConfig(chunk_size=100, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(long_text, "doc1", "job1", "test.txt", config)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i

    def test_start_end_char_correct(self, chunker: TextChunker) -> None:
        text = "ABCDEFGHIJ"
        config = ChunkConfig(chunk_size=4, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(text, "doc1", "job1", "test.txt", config)
        assert chunks[0].metadata.start_char == 0
        assert chunks[0].metadata.end_char == 4
        assert chunks[1].metadata.start_char == 4
        assert chunks[1].metadata.end_char == 8
