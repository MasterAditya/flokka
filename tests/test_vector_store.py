"""Tests for the vector store service."""

import pytest

from app.models.chunk import DocumentChunk
from app.services.chunker import ChunkConfig, TextChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStore


@pytest.fixture
def memory_store() -> VectorStore:
    """In-memory ChromaDB store for tests."""
    return VectorStore(host="memory", collection_name="test_collection")


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Pre-built chunks with simulated embeddings."""
    embedder = EmbeddingService()
    chunker = TextChunker()
    text = "The Flokka ingestion engine processes documents efficiently. " * 10
    config = ChunkConfig(chunk_size=100, chunk_overlap=20, strategy="overlap")
    chunks = chunker.chunk(text, "doc-vs-001", "job-vs-001", "sample.txt", config)
    return embedder.embed_chunks(chunks)


class TestVectorStoreUpsert:
    def test_upsert_returns_chunk_count(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        stored = memory_store.upsert_chunks(sample_chunks)
        assert stored == len(sample_chunks)

    def test_upsert_empty_list_returns_zero(self, memory_store: VectorStore) -> None:
        assert memory_store.upsert_chunks([]) == 0

    def test_count_increases_after_upsert(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        before = memory_store.count()
        memory_store.upsert_chunks(sample_chunks)
        after = memory_store.count()
        assert after == before + len(sample_chunks)

    def test_upsert_is_idempotent(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        memory_store.upsert_chunks(sample_chunks)
        count_after_first = memory_store.count()
        memory_store.upsert_chunks(sample_chunks)  # Same chunks, same IDs
        count_after_second = memory_store.count()
        assert count_after_first == count_after_second


class TestVectorStoreQuery:
    def test_query_returns_results(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        memory_store.upsert_chunks(sample_chunks)
        embedder = EmbeddingService()
        query_vec = embedder.embed_query("ingestion engine")
        results = memory_store.query(query_vec, n_results=3)
        assert len(results) <= 3
        assert all("text" in r and "metadata" in r and "distance" in r for r in results)

    def test_query_result_has_expected_keys(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        memory_store.upsert_chunks(sample_chunks)
        embedder = EmbeddingService()
        query_vec = embedder.embed_query("document processing")
        results = memory_store.query(query_vec, n_results=1)
        assert len(results) >= 1
        result = results[0]
        assert "text" in result
        assert "metadata" in result
        assert "distance" in result


class TestVectorStoreDelete:
    def test_delete_removes_document_chunks(
        self, memory_store: VectorStore, sample_chunks: list[DocumentChunk]
    ) -> None:
        memory_store.upsert_chunks(sample_chunks)
        count_before = memory_store.count()
        memory_store.delete_document("doc-vs-001")
        count_after = memory_store.count()
        assert count_after < count_before


class TestEmbeddingService:
    def test_embed_chunks_attaches_embeddings(self) -> None:
        chunker = TextChunker()
        embedder = EmbeddingService()
        text = "Test embedding generation for chunks." * 5
        config = ChunkConfig(chunk_size=50, chunk_overlap=0, strategy="fixed")
        chunks = chunker.chunk(text, "doc1", "job1", "test.txt", config)
        embedded = embedder.embed_chunks(chunks)
        for chunk in embedded:
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0

    def test_embed_query_returns_vector(self) -> None:
        embedder = EmbeddingService()
        vec = embedder.embed_query("sample query")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(v, float) for v in vec)

    def test_simulated_embedding_is_deterministic(self) -> None:
        embedder = EmbeddingService()
        vec1 = embedder._simulated_embedding("hello world")
        vec2 = embedder._simulated_embedding("hello world")
        assert vec1 == vec2

    def test_simulated_embeddings_differ_for_different_texts(self) -> None:
        embedder = EmbeddingService()
        vec1 = embedder._simulated_embedding("hello")
        vec2 = embedder._simulated_embedding("world")
        assert vec1 != vec2
