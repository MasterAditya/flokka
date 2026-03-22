# Flokka Ingestion Engine

A production-style document ingestion pipeline that processes documents into semantic chunks, generates embeddings, and stores them in a vector database for retrieval-augmented generation (RAG) workflows.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client / API Consumer                        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  POST /api/v1/ingest/
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI (Ingestion API)                         │
│  • Validates file type & size                                        │
│  • Creates IngestionJob (UUID)                                       │
│  • Dispatches Celery task → Redis                                    │
│  • Returns job_id immediately (202 Accepted)                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │  Task enqueue
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Redis (Task Broker)                            │
└──────────┬──────────────────────────────────────────────────────────┘
           │  Celery workers consume tasks
           ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│  Ingestion Worker    │───▶│  Embedding Worker    │───▶│  Indexing Worker       │
│                      │    │                      │    │                        │
│  1. Extract text     │    │  3. Load model       │    │  5. Upsert to ChromaDB │
│     (.txt/.pdf/docx) │    │  4. Encode chunks    │    │     - chunk text       │
│  2. Chunk text       │    │     → vectors        │    │     - embedding vector │
│     (fixed/overlap)  │    │                      │    │     - metadata         │
└──────────────────────┘    └──────────────────────┘    └────────────────────────┘
                                                                    │
                                                                    ▼
                                                        ┌────────────────────────┐
                                                        │   ChromaDB             │
                                                        │   (Vector Store)       │
                                                        └────────────────────────┘
```

## Pipeline Explanation

The ingestion pipeline follows a three-stage architecture:

| Stage | Worker | Responsibility |
|-------|--------|----------------|
| **1. Extract** | `ingestion_worker` | Parse .txt, .pdf, .docx → plain text |
| **2. Chunk** | `ingestion_worker` | Split text into fixed or overlap chunks with metadata |
| **3. Embed** | `embedding_worker` | Encode each chunk via sentence-transformers |
| **4. Index** | `indexing_worker` | Upsert chunk text + embedding + metadata into ChromaDB |

Each stage is a separate Celery task that chains to the next on success, enabling independent scaling and retry logic.

## RAG Explanation

**Retrieval-Augmented Generation (RAG)** enhances LLM responses by grounding them in a knowledge base:

1. **Ingestion (this engine)** – Documents are chunked and embedded into a vector store.
2. **Retrieval** – At query time, the user's question is embedded and nearest-neighbour search finds relevant chunks.
3. **Generation** – Retrieved chunks are injected into the LLM prompt as context, producing grounded responses.

This engine handles step 1. The `VectorStore.query()` method is ready to serve step 2.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI** | Modern, async-first Python web framework with OpenAPI out of the box |
| **Celery + Redis** | Industry-standard distributed task queue; Redis is fast and simple |
| **ChromaDB** | Embeddable vector DB with HTTP API; easy to run locally or in Docker |
| **sentence-transformers** | Open-source, high-quality embeddings; no API key required |
| **Overlap chunking default** | Preserves context at boundaries vs hard cuts |
| **Simulated embeddings fallback** | CI/tests run without GPU or model downloads |
| **In-memory ChromaDB fallback** | Tests run without a Docker container |
| **Pydantic v2 models** | Type-safe, serialisable domain objects |
| **Structured logging** | All log lines include timestamp, level, logger name for observability |

## Scaling Explanation

```
Scale the API tier:                Scale the workers:
  docker compose scale api=3         docker compose scale worker=8
```

- **API tier** is stateless → horizontal scaling with a load balancer
- **Workers** are independently scalable per stage using Celery's concurrency and autoscaling
- **Redis Streams / RabbitMQ** can replace the default Redis broker for higher throughput
- **ChromaDB** can be replaced with Pinecone, Weaviate or pgvector for managed scaling

## Future Improvements

- [ ] Persistent job status store (Redis/Postgres instead of in-memory dict)
- [ ] Streaming ingestion for large files (chunked multipart upload)
- [ ] Document deduplication using content hash
- [ ] Webhook callbacks when ingestion completes
- [ ] Retrieval API endpoint (`POST /api/v1/query`)
- [ ] Metrics with Prometheus + Grafana
- [ ] Distributed tracing with OpenTelemetry
- [ ] PDF table extraction with `pdfplumber` or `camelot`
- [ ] Rate limiting and authentication (API keys / JWT)
- [ ] Kubernetes HPA for worker autoscaling

## Getting Started

### With Docker Compose

```bash
# Start all services
docker compose up --build

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env

# Start API
uvicorn app.main:app --reload

# Start worker (separate terminal, requires Redis)
celery -A app.workers.celery_app.celery_app worker --loglevel=info
```

### Running Tests

```bash
pytest tests/ --cov=app --cov-report=term-missing -v
```

### API Usage

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/ingest/ \
  -F "file=@document.txt" \
  -F "chunk_size=512" \
  -F "chunk_overlap=64"

# Check job status
curl http://localhost:8000/api/v1/ingest/{job_id}

# Health check
curl http://localhost:8000/health
```

## Project Structure

```
flokka/
├── app/
│   ├── main.py                  # FastAPI application factory
│   ├── api/routes/
│   │   └── ingestion.py         # Ingestion endpoint
│   ├── services/
│   │   ├── text_extractor.py    # txt/pdf/docx text extraction
│   │   ├── chunker.py           # Fixed & overlap chunking
│   │   ├── embedder.py          # Sentence-transformer embeddings
│   │   └── vector_store.py      # ChromaDB integration
│   ├── workers/
│   │   ├── celery_app.py        # Celery application factory
│   │   ├── ingestion_worker.py  # Extract + chunk stage
│   │   ├── embedding_worker.py  # Embed stage
│   │   └── indexing_worker.py   # Index stage
│   ├── models/
│   │   ├── document.py          # IngestionJob, JobStatus
│   │   └── chunk.py             # DocumentChunk, ChunkMetadata
│   ├── core/
│   │   ├── config.py            # Pydantic Settings
│   │   └── logging_config.py    # Structured logging
│   └── utils/
│       └── timing.py            # Execution timing helpers
├── tests/
│   ├── test_chunker.py          # Chunking logic tests
│   ├── test_ingestion.py        # API endpoint tests
│   └── test_vector_store.py     # Vector insertion tests
├── docker/
│   ├── Dockerfile               # API image
│   └── Dockerfile.worker        # Worker image
├── .github/workflows/ci.yml     # Lint + test + coverage
├── docker-compose.yml           # Full stack
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```
