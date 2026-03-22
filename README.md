# Flokka Ingestion Engine

An async document ingestion pipeline that extracts text from uploaded files, splits it into semantically coherent chunks, generates dense vector embeddings, and persists everything into a vector store for downstream retrieval.

The system is built around the constraint that ingestion is **not a request-scoped operation** — a 50 MB PDF might take 10–30 seconds to process, and that work must survive API restarts, be retryable on failure, and be scalable independently of the HTTP tier.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Ingestion Pipeline](#ingestion-pipeline)
3. [Architecture Decisions](#architecture-decisions)
   - [Why Celery instead of FastAPI BackgroundTasks](#why-celery-instead-of-fastapi-backgroundtasks)
   - [Why Redis](#why-redis)
   - [Why ChromaDB](#why-chromadb)
   - [Why sentence-transformers](#why-sentence-transformers)
   - [Why overlap chunking as the default](#why-overlap-chunking-as-the-default)
4. [Tradeoffs](#tradeoffs)
5. [Scaling Limitations](#scaling-limitations)
6. [Where this fits in a RAG system](#where-this-fits-in-a-rag-system)
7. [Future Improvements](#future-improvements)
8. [Getting Started](#getting-started)
9. [Project Structure](#project-structure)

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Client / API Consumer                        │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │  POST /api/v1/ingest/
                                  │  (multipart: file + chunk params)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         FastAPI  (API tier)                           │
│                                                                       │
│  1. Validate MIME type and file size                                  │
│  2. Assign IngestionJob UUID                                          │
│  3. Enqueue Celery task → Redis                                       │
│  4. Return 202 Accepted + job_id immediately                          │
└─────────────────────────────────┬────────────────────────────────────┘
                                  │  serialised task payload
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         Redis  (broker + result backend)              │
└──────────────────────────────────────────────────────────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ ingestion_worker │──▶│ embedding_worker │──▶│   indexing_worker    │
│                  │   │                  │   │                      │
│ • extract text   │   │ • load model     │   │ • upsert to ChromaDB │
│   (.txt/pdf/docx)│   │   (lazy, cached) │   │   ids, documents,    │
│ • fixed / overlap│   │ • encode chunks  │   │   embeddings,        │
│   chunking       │   │   → float vectors│   │   metadata           │
│ • attach metadata│   │                  │   │                      │
└──────────────────┘   └──────────────────┘   └──────────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │      ChromaDB        │
                                              │   (vector store)     │
                                              │                      │
                                              │  HNSW index          │
                                              │  cosine similarity   │
                                              │  metadata filtering  │
                                              └──────────────────────┘
```

The API tier and worker tier are **fully decoupled**. The API process never touches document bytes after enqueueing — it can be restarted, scaled, or fail entirely without affecting in-flight ingestion jobs. Workers pull tasks from Redis and process them independently.

---

## Ingestion Pipeline

Each Celery task represents one discrete stage. On success, a stage chains directly into the next. On failure, Celery retries with exponential backoff up to three times before marking the task as failed.

| Stage | Task | Input | Output |
|-------|------|-------|--------|
| 1 | `ingest_document` | base64-encoded file bytes | list of `DocumentChunk` dicts |
| 2 | `embed_chunks` | `DocumentChunk` dicts (no vectors) | `DocumentChunk` dicts with `embedding` |
| 3 | `index_chunks` | `DocumentChunk` dicts with `embedding` | stored in ChromaDB |

Passing data between stages as serialised dicts — rather than through a shared database row or shared memory — keeps each worker stateless and makes each stage independently replayable. If the embedding stage fails after extraction has already produced clean chunks, only the embedding task needs to be retried; the extraction work is not repeated.

---

## Architecture Decisions

### Why Celery instead of FastAPI BackgroundTasks

FastAPI's built-in `BackgroundTasks` runs the work function inside the same OS process as the request handler. This is the right tool for fast, cheap, fire-and-forget operations such as writing an audit log entry. It is the wrong model for document ingestion for four reasons:

**No durability.** If the API process restarts mid-ingestion — due to a deployment, an OOM kill, or a crash — the background task is silently dropped. There is no record that it was ever running and no way to resume it.

**No retry semantics.** A failed extraction must be detected and re-triggered by the caller. There is no built-in backoff, dead-letter queue, or visibility timeout. You would have to build all of that yourself.

**No independent scalability.** Embedding is CPU-bound (or GPU-bound). Running it inside the API process ties embedding throughput to HTTP worker count and wastes ASGI concurrency slots on blocking compute. You cannot scale the expensive part without also scaling the cheap part.

**No observability.** There is no task ID, no state machine, no way to query "how many documents are currently being embedded" without building that plumbing yourself.

Celery solves all four problems. The tradeoff is operational complexity: you need a broker, and workers are separate processes to deploy and monitor. For a system where jobs are measured in seconds to minutes and must be reliable, that cost is warranted.

### Why Redis

Redis was chosen as both the Celery broker and result backend for the following reasons:

**Latency.** Redis operates in-memory and delivers sub-millisecond enqueue and dequeue times. For a task queue, broker latency adds directly to the delay between document upload and the start of processing.

**Operational simplicity.** Using Redis as both broker *and* result backend means one fewer infrastructure dependency. The common alternative — RabbitMQ as the broker with a separate Postgres-backed result store — is more powerful but substantially more complex to operate.

**Data structure fit.** Celery's default Redis transport uses Redis Lists as FIFO queues. The semantics map exactly onto a task queue with no impedance mismatch.

**Ecosystem.** Redis is already widely deployed as a cache and session store. In a real system, this engine would share a Redis cluster with other services rather than provisioning dedicated broker infrastructure.

**Considered alternatives:**

| Option | Why not chosen |
|--------|----------------|
| RabbitMQ | Better routing and AMQP protocol support, but operationally heavier and requires a separate result backend |
| AWS SQS | Fully managed with no ops burden, but introduces cloud vendor lock-in and a ~250 ms minimum visibility timeout that adds latency to every task |
| Kafka | Correct choice if ingestion events need to be replayed by multiple consumers or retained long-term, but it is a streaming platform rather than a task queue and adds significant operational overhead for this use case |

Redis becomes a single point of failure without Redis Sentinel or Redis Cluster. That limitation is covered in [Scaling Limitations](#scaling-limitations).

### Why ChromaDB

ChromaDB was chosen because it offers a property no other option in this space provides: the same client code runs against an **in-process ephemeral store** (for tests and local development) and a **persistent HTTP server** (for staging and production), with only a configuration change and no code changes.

Beyond portability:

**HNSW index with cosine similarity** is the right index for normalised sentence embeddings. Cosine similarity is invariant to vector magnitude, which matters because sentence-transformers normalises its output by default. Using Euclidean distance on normalised vectors would give equivalent results but is less semantically principled.

**Native metadata filtering** allows queries to be scoped to a single document or collection without a separate filtering step. `where={"document_id": "..."}` is evaluated at the index level, not as a post-retrieval filter.

**No schema migrations.** Collections are created on first write. There are no migration scripts to manage across environments.

**Considered alternatives:**

| Option | Why not chosen |
|--------|----------------|
| pgvector | Good choice if you already run Postgres and need transactional guarantees — atomically deleting a document and all its chunks, for example. Adds SQL overhead and schema management. Worth reconsidering once a Postgres instance exists in the stack for job state. |
| Pinecone | Fully managed and scales to billions of vectors with no operational burden, but it is a paid external service and introduces a network dependency into the indexing hot path |
| Qdrant | Rust-based, fast, and production-grade with native distributed mode. The strongest alternative for a system that has outgrown ChromaDB's single-node limit. The main cost is adding another unfamiliar service early in the project lifecycle. |
| Weaviate | Feature-rich — multi-modal support, GraphQL API, hybrid search — but the configuration surface area is substantially larger than is warranted at this scale |

### Why sentence-transformers

`sentence-transformers` provides pre-trained bi-encoder models that produce fixed-length dense vectors optimised for semantic similarity tasks. `all-MiniLM-L6-v2` (384 dimensions) was selected because:

- It produces competitive retrieval quality on standard benchmarks (BEIR) for its size class.
- The model weights are approximately 22 MB, small enough to be loaded per worker process without immediately running into memory pressure.
- Inference runs on CPU without a GPU, keeping infrastructure requirements minimal.

The embedding service falls back to a deterministic hash-based simulation when the model cannot be loaded — no network access in CI, or when `simulate=True` is passed explicitly. This keeps the test suite fast and infrastructure-free without sacrificing pipeline coverage.

### Why overlap chunking as the default

Fixed-size chunking cuts text at hard character boundaries, frequently splitting a sentence mid-thought. When that truncated chunk is later retrieved and injected into an LLM prompt, the missing context degrades answer quality.

Overlap chunking uses a sliding window: each chunk shares `chunk_overlap` characters with its neighbours, ensuring that any sentence near a boundary appears fully in at least one chunk. The cost is redundant storage — a 10 000-character document chunked at size=512 with overlap=64 produces approximately 21 chunks rather than 20. That ~5% overhead is acceptable given the retrieval quality improvement for boundary-straddling sentences.

The `fixed` strategy is retained for cases where deduplication is more important than boundary coherence, such as when chunk content hashes are used to detect duplicate ingestion.

---

## Tradeoffs

These are deliberate compromises in the current implementation. They are not oversights — they are accepted costs that are documented here so that any engineer picking this up knows exactly what needs to change before it can be considered production-ready.

**In-memory job store.** Job state is held in the API process's memory in a plain dict. This means job history is lost on API restart and is invisible to other API replicas running behind a load balancer. The fix is a shared store — a Redis hash is the lowest-friction path since Redis is already in the stack. This was deferred to keep the first iteration dependency-free.

**Approximate nearest-neighbour search.** ChromaDB's HNSW index is an approximate algorithm. It does not guarantee finding the globally nearest vectors — it finds vectors that are very likely to be among the nearest, with a configurable accuracy/speed tradeoff (`ef` parameter). For RAG retrieval this is acceptable: a slightly sub-optimal chunk is almost always still useful context for an LLM. It would not be acceptable for exact duplicate detection or security-critical matching.

**Model loaded per worker process.** Each Celery worker loads the sentence-transformers model independently. At four workers and approximately 500 MB of RSS per process, this approaches 2 GB just for the embedding stage. The correct architecture at scale is a dedicated inference service — a single FastAPI process wrapping the model — that all embedding workers call over HTTP. The model loads once; workers are thin HTTP clients. This is a P2 item in the roadmap.

**Celery task chaining couples stage failures.** If the indexing stage fails after embedding has already completed, the retry re-runs embedding from scratch, re-encoding all chunks. This is wasteful but not incorrect — ChromaDB upserts are idempotent by chunk ID. The better design is to persist intermediate results between stages (e.g. store embedded chunks in Redis or S3 before indexing), so that a stage failure only replays that stage. This adds storage and coordination complexity that is not warranted at the current scale.

**No backpressure on the task queue.** If ingestion requests arrive faster than workers can process them, the Redis queue grows unbounded. Redis will eventually exhaust memory and, depending on its `maxmemory-policy`, start evicting keys — which silently drops tasks. The mitigation is to set `maxmemory-policy noeviction` in Redis (errors rather than silent drops), enforce a per-client queue depth limit at the API layer, and expose queue depth as a metric so that autoscaling can respond before the queue becomes a problem.

---

## Scaling Limitations

Understanding where the system breaks is as important as understanding how it works.

**API tier** is stateless and scales horizontally behind any load balancer with no code changes. The only constraint today is the in-memory job store — once that is moved to Redis or Postgres, the API tier has no meaningful bottleneck until you are handling thousands of concurrent uploads.

**Worker tier** scales horizontally, but ingestion and embedding are fundamentally different workload shapes and should not share a worker pool at scale:

- `ingest_document` is I/O-bound: file reading, PDF parsing, and string operations. It benefits from higher concurrency per core.
- `embed_chunks` is CPU-bound (or GPU-bound). Adding more embedding workers only helps if additional CPU cores or GPUs are available. Running 20 embedding workers on a 4-core machine wastes memory loading the model 20 times and gains nothing in throughput.

The fix is to route `embed_chunks` to a dedicated `embedding` Celery queue with a separately sized worker pool. This is a configuration change, not a code change.

**Redis** runs as a single node in this configuration and is a single point of failure. Redis Sentinel adds automatic failover with minimal operational overhead and is the appropriate next step for production use. Redis Cluster enables horizontal sharding but adds client-side complexity that is rarely justified for a task queue unless you are enqueuing millions of tasks per second.

**ChromaDB** runs as a single-node HTTP server with no built-in replication or clustering. This is suitable for deployments up to tens of millions of vectors (roughly a few hundred thousand documents), after which query latency degrades as the HNSW graph grows and memory pressure increases. The migration path at that point is Qdrant or Weaviate, both of which offer distributed deployments with equivalent HNSW indexing semantics and compatible metadata filtering APIs.

**Embedding model memory** is the most immediate per-host bottleneck. Four workers with `all-MiniLM-L6-v2` consume approximately 2 GB of RSS. Larger, higher-quality models — `all-mpnet-base-v2` at 768 dimensions, for example — quadruple memory requirements. A shared inference service resolves this entirely.

---

## Where this fits in a RAG system

This engine is the **ingestion leg** of a retrieval-augmented generation pipeline. A complete RAG system has three distinct phases:

```
Phase 1 — Ingestion (this engine)
  Documents → text extraction → chunking → embedding → vector store

Phase 2 — Retrieval
  User query → query embedding → ANN search → top-k ranked chunks

Phase 3 — Generation
  System prompt + top-k chunks + user query → LLM → grounded response
```

`VectorStore.query()` is already implemented and ready to serve Phase 2. Phases 2 and 3 would be added as a separate `retrieval` service — a thin FastAPI application that accepts a query string, calls the embedding service to produce a query vector, queries ChromaDB, and constructs an LLM prompt from the ranked results. It shares the ChromaDB instance with the ingestion workers but has no other dependency on them.

A key design constraint in RAG is **chunk granularity alignment with retrieval quality**. Chunks that are too large retrieve too much irrelevant context alongside the answer. Chunks that are too small split important context across boundaries that may not be retrieved together. The 512-character default with 64-character overlap is a reasonable starting point; optimal values are workload-specific and should be validated against retrieval benchmarks for any production knowledge base.

---

## Future Improvements

Ordered by impact relative to implementation cost.

### P0 — Required before a second API replica can be deployed

**Persistent job store.** Replace the in-memory `_jobs` dict with a Redis hash keyed by job ID. This is a small change (a few lines in the ingestion route and a Redis client dependency that is already in the stack) with a large impact: job state survives restarts and is visible across all API replicas.

**Separate Celery queues by workload type.** Route `embed_chunks` to a dedicated `embedding` queue consumed by a separately configured worker pool. This prevents slow, CPU-bound embedding jobs from blocking fast, I/O-bound extraction jobs that share the same queue.

### P1 — Required before a public-facing deployment

**Retrieval endpoint.** `POST /api/v1/query` — accepts a query string, embeds it using the same `EmbeddingService`, queries ChromaDB, and returns ranked chunks with scores. The vector store and embedding layers are already implemented; this is a thin API handler.

**Authentication.** API key middleware on all ingestion and retrieval endpoints. Without it, any client can fill the queue or read any document's chunks.

**Rate limiting.** Per-client request rate and per-client queue depth limits. Prevents a single client from exhausting the queue and triggering the backpressure failure mode described above.

**Document deduplication.** Compute a SHA-256 hash of file content at upload time. Reject or skip documents that match an already-ingested hash. Reduces redundant processing and vector store bloat when the same file is uploaded multiple times.

### P2 — Operational maturity

**Metrics.** Expose Prometheus metrics: queue depth per Celery queue, task duration by stage (p50/p95/p99), embedding throughput in chunks per second, ChromaDB collection size. Alert on queue depth exceeding a threshold (a leading indicator of the backpressure failure mode) and on task failure rate exceeding 1%.

**Distributed tracing.** Propagate a trace ID from the HTTP request through all three Celery stages using OpenTelemetry. This allows the full timeline of a single ingestion job — from upload to final indexing — to be reconstructed in a trace viewer such as Jaeger or Grafana Tempo.

**Shared inference service.** Replace per-worker model loading with a dedicated embedding microservice (FastAPI + sentence-transformers, or a Triton inference server). Workers become lightweight HTTP clients; the model loads once regardless of worker count.

**Streaming upload.** Accept chunked multipart uploads for large files instead of buffering the entire file in the API process before enqueueing. Reduces peak memory usage on the API tier for large documents.

**Webhook callbacks.** Allow callers to register a callback URL on upload. POST to that URL when the job reaches `COMPLETED` or `FAILED`, eliminating the need for the caller to poll the status endpoint.

---

## Getting Started

### With Docker Compose

```bash
docker compose up --build
```

Starts four services: FastAPI on `:8000`, a Celery worker, Redis on `:6379`, and ChromaDB on `:8001`. Interactive API docs: `http://localhost:8000/docs`.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env

# Start the API (auto-reload on code changes)
uvicorn app.main:app --reload

# Start a worker in a separate terminal (requires a running Redis)
celery -A app.workers.celery_app.celery_app worker --loglevel=info
```

### Tests

```bash
pytest tests/ --cov=app --cov-report=term-missing -v
```

Tests use an in-memory ChromaDB client and mock the Celery dispatch. No external services are required.

### API Reference

```bash
# Upload a document and receive a job ID
curl -X POST http://localhost:8000/api/v1/ingest/ \
  -F "file=@document.pdf" \
  -F "chunk_size=512" \
  -F "chunk_overlap=64"
# → 202 Accepted  {"job_id": "...", "document_id": "...", "status": "pending"}

# Poll for job status
curl http://localhost:8000/api/v1/ingest/{job_id}
# → 200 OK  {"job_id": "...", "status": "completed", "chunk_count": 42, ...}

# Health check (used by Docker and load balancer probes)
curl http://localhost:8000/health
# → 200 OK  {"status": "ok", "version": "1.0.0"}
```

---

## Project Structure

```
flokka/
├── app/
│   ├── main.py                  # Application factory, middleware, router registration
│   ├── api/routes/
│   │   └── ingestion.py         # POST /ingest, GET /ingest/{job_id}
│   ├── services/
│   │   ├── text_extractor.py    # Format-specific text extraction (txt, pdf, docx)
│   │   ├── chunker.py           # Fixed and overlap chunking strategies
│   │   ├── embedder.py          # sentence-transformers with hash-based CI fallback
│   │   └── vector_store.py      # ChromaDB client (HTTP + in-memory fallback)
│   ├── workers/
│   │   ├── celery_app.py        # Celery application and transport configuration
│   │   ├── ingestion_worker.py  # Stage 1: extract + chunk
│   │   ├── embedding_worker.py  # Stage 2: encode chunks
│   │   └── indexing_worker.py   # Stage 3: upsert to ChromaDB
│   ├── models/
│   │   ├── document.py          # IngestionJob, JobStatus, IngestionResponse
│   │   └── chunk.py             # DocumentChunk, ChunkMetadata
│   ├── core/
│   │   ├── config.py            # Pydantic Settings (env-driven configuration)
│   │   └── logging_config.py    # Log format and third-party logger suppression
│   └── utils/
│       └── timing.py            # measure_ms helper and @timed decorator
├── tests/
│   ├── conftest.py              # Shared fixtures, Celery dispatch mock
│   ├── test_chunker.py          # Chunking correctness and metadata integrity
│   ├── test_ingestion.py        # API contract tests
│   └── test_vector_store.py     # Vector store upsert, query, delete
├── docker/
│   ├── Dockerfile               # API image (python:3.11-slim)
│   └── Dockerfile.worker        # Worker image
├── .github/workflows/ci.yml     # Lint (ruff, black), test, coverage gate
├── docker-compose.yml           # Full local stack
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```
