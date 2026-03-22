"""Tests for the vector store service and embedding service."""

import pytest

from app.models.chunk import DocumentChunk
from app.services.chunker import ChunkConfig, TextChunker
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStore


@pytest.fixture
def store() -> VectorStore:
    """Isolated in-memory ChromaDB collection for each test."""
    return VectorStore(host="memory", collection_name="test_collection")


@pytest.fixture
def embedded_chunks() -> list[DocumentChunk]:
    """A small set of chunks with simulated embeddings for vector store tests."""
    text = "The Flokka ingestion engine processes documents efficiently. " * 10
    config = ChunkConfig(chunk_size=100, chunk_overlap=20, strategy="overlap")
    chunks = TextChunker().chunk(text, "doc-vs-001", "job-vs-001", "sample.txt", config)
    return EmbeddingService(simulate=True).embed_chunks(chunks)


class TestUpsert:
    def test_returns_stored_count(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        assert store.upsert_chunks(embedded_chunks) == len(embedded_chunks)

    def test_empty_list_is_noop(self, store: VectorStore) -> None:
        assert store.upsert_chunks([]) == 0

    def test_collection_grows_after_upsert(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        before = store.count()
        store.upsert_chunks(embedded_chunks)
        assert store.count() == before + len(embedded_chunks)

    def test_upsert_is_idempotent(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        store.upsert_chunks(embedded_chunks)
        count_after_first = store.count()
        store.upsert_chunks(embedded_chunks)
        assert store.count() == count_after_first


class TestQuery:
    def test_returns_at_most_n_results(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        store.upsert_chunks(embedded_chunks)
        query_vec = EmbeddingService(simulate=True).embed_query("ingestion engine")
        results = store.query(query_vec, n_results=3)
        assert len(results) <= 3

    def test_result_shape(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        store.upsert_chunks(embedded_chunks)
        query_vec = EmbeddingService(simulate=True).embed_query("document processing")
        results = store.query(query_vec, n_results=1)
        assert results
        assert {"text", "metadata", "distance"} <= results[0].keys()


class TestDelete:
    def test_removes_chunks_for_document(
        self, store: VectorStore, embedded_chunks: list[DocumentChunk]
    ) -> None:
        store.upsert_chunks(embedded_chunks)
        before = store.count()
        store.delete_document("doc-vs-001")
        assert store.count() < before


class TestEmbeddingService:
    def test_chunks_receive_embeddings(self) -> None:
        text = "Embedding test document content." * 5
        config = ChunkConfig(chunk_size=50, chunk_overlap=0, strategy="fixed")
        chunks = TextChunker().chunk(text, "doc1", "job1", "test.txt", config)
        embedded = EmbeddingService(simulate=True).embed_chunks(chunks)
        for chunk in embedded:
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0

    def test_query_embedding_is_float_vector(self) -> None:
        vec = EmbeddingService(simulate=True).embed_query("sample query")
        assert isinstance(vec, list)
        assert vec
        assert all(isinstance(v, float) for v in vec)

    def test_same_text_produces_identical_embedding(self) -> None:
        svc = EmbeddingService(simulate=True)
        assert svc.embed_query("hello world") == svc.embed_query("hello world")

    def test_different_texts_produce_different_embeddings(self) -> None:
        svc = EmbeddingService(simulate=True)
        assert svc.embed_query("hello") != svc.embed_query("world")
