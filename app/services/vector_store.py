"""ChromaDB vector store integration."""

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around ChromaDB for storing and querying document chunks.

    Initialises an in-memory client when ``host`` is ``"memory"`` or when
    the configured HTTP server is unreachable.  This allows the full pipeline
    to run in environments without a ChromaDB container (tests, local dev).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._host = host or settings.chroma_host
        self._port = port or settings.chroma_port
        self._collection_name = collection_name or settings.chroma_collection
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Insert or update chunks in the vector store.

        Returns the number of chunks written.  Upserts are idempotent by
        ``chunk_id``, so re-running a failed stage is safe.
        """
        if not chunks:
            return 0

        collection = self._get_collection()
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata.model_dump() for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding is not None]

        if len(embeddings) == len(chunks):
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info(
            "Upserted %d chunks into collection %r", len(chunks), self._collection_name
        )
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Return the top-*n_results* chunks nearest to *query_embedding*.

        Each result dict contains ``text``, ``metadata``, and ``distance``.
        """
        collection = self._get_collection()
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to *document_id*."""
        self._get_collection().delete(where={"document_id": document_id})
        logger.info("Deleted all chunks for document %r", document_id)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._get_collection().count()

    def _get_client(self) -> chromadb.ClientAPI:
        if self._client is not None:
            return self._client

        if self._host == "memory":
            self._client = chromadb.Client()
            logger.info("Using in-memory ChromaDB client")
            return self._client

        try:
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            logger.info("Connected to ChromaDB at %s:%d", self._host, self._port)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cannot reach ChromaDB at %s:%d (%s). Falling back to in-memory client.",
                self._host,
                self._port,
                exc,
            )
            self._client = chromadb.Client()

        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is not None:
            return self._collection
        self._collection = self._get_client().get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection
