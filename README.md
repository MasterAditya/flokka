# Flokka Ingestion Engine

> Production-style asynchronous document ingestion pipeline for Retrieval-Augmented Generation (RAG) systems built with **FastAPI, Celery, Redis, ChromaDB, and Sentence Transformers**.

Flokka is designed around a simple principle:

> **Document ingestion is not a request-scoped operation.**

Processing large documents can take several seconds or even minutes. Instead of blocking the API, Flokka distributes work across background workers, enabling reliable, fault-tolerant, and horizontally scalable ingestion.

---

## Features

- 🚀 FastAPI REST API
- ⚡ Asynchronous processing with Celery
- 📦 Redis task broker & result backend
- 🧠 Semantic embeddings using Sentence Transformers
- 📚 ChromaDB vector storage
- 🔄 Automatic retries with exponential backoff
- 📄 Support for TXT, PDF, and DOCX
- 📈 Stateless, horizontally scalable worker architecture
- 🛠 Production-oriented design with documented tradeoffs

---

# Why Flokka?

Most tutorials process uploaded documents using FastAPI's `BackgroundTasks`.

While suitable for lightweight operations, they become problematic for long-running workloads because:

- Background jobs disappear if the API process crashes.
- Failed jobs cannot automatically retry.
- CPU-intensive embedding blocks API workers.
- Workers cannot scale independently.
- Job progress is difficult to monitor.

Flokka treats document ingestion as a distributed workflow rather than a request-scoped operation.

---

# System Architecture

```text
                          Client
                             │
                             │ POST /ingest
                             ▼
                    ┌─────────────────┐
                    │     FastAPI     │
                    └────────┬────────┘
                             │
                    Return 202 Accepted
                             │
                             ▼
                    ┌─────────────────┐
                    │      Redis      │
                    │ Task Queue      │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │ Extract Text   │  │ Generate       │  │ Index into     │
 │ Worker         │  │ Embeddings     │  │ ChromaDB       │
 └────────────────┘  └────────────────┘  └────────────────┘
                             │
                             ▼
                      ┌───────────────┐
                      │   ChromaDB    │
                      │ Vector Store  │
                      └───────────────┘
```

The API never processes the document itself.

Its only responsibility is to:

1. Validate the request
2. Create a Job ID
3. Enqueue a Celery task
4. Return `202 Accepted`

Everything else happens asynchronously.

---

# Ingestion Pipeline

Each document passes through three independent stages.

| Stage | Responsibility |
|-------|----------------|
| Extract | Read TXT, PDF or DOCX and extract text |
| Chunk | Split text into overlapping semantic chunks |
| Embed | Generate dense vector embeddings |
| Index | Store vectors and metadata inside ChromaDB |

Each stage is retryable and independently executable.

---

# Technology Stack

| Layer | Technology |
|--------|------------|
| API | FastAPI |
| Task Queue | Celery |
| Broker | Redis |
| Vector Database | ChromaDB |
| Embedding Model | Sentence Transformers |
| Language | Python |
| Containerization | Docker |

---

# Design Decisions

## Why Celery?

FastAPI BackgroundTasks execute inside the API process.

Celery was chosen because it provides:

- Durable task execution
- Automatic retries
- Independent worker scaling
- Task monitoring
- Fault tolerance

This allows the API layer to remain lightweight while workers handle computationally expensive tasks.

---

## Why Redis?

Redis acts as both:

- Celery broker
- Result backend

Advantages:

- Extremely low latency
- Simple deployment
- Minimal operational overhead
- Native support within Celery

---

## Why ChromaDB?

ChromaDB provides:

- HNSW indexing
- Cosine similarity search
- Metadata filtering
- Easy local development
- HTTP server mode for production

It offers an excellent balance between simplicity and performance for small to medium RAG systems.

---

## Why Sentence Transformers?

The project uses **all-MiniLM-L6-v2** because it provides:

- Strong semantic retrieval quality
- Small model size
- Fast CPU inference
- Low infrastructure requirements

---

## Why Overlap Chunking?

Large language models retrieve context better when adjacent chunks share information.

Compared to fixed chunking, overlap chunking:

- Preserves sentence continuity
- Improves retrieval quality
- Reduces boundary information loss

The storage overhead is small relative to the improvement in retrieval accuracy.

---

# Engineering Tradeoffs

This project intentionally documents its current limitations.

Current compromises include:

- In-memory job tracking
- Single-node Redis deployment
- Single-node ChromaDB
- Per-worker embedding model loading
- Approximate nearest-neighbour search
- No queue backpressure

Each limitation includes a documented migration strategy for production-scale deployments.

---

# Scalability

## Current

✅ Stateless FastAPI API

✅ Independent Celery workers

✅ Redis task broker

✅ ChromaDB vector storage

---

## Future Scaling

- Redis Sentinel
- Dedicated embedding worker queue
- Shared embedding service
- Distributed vector database (Qdrant)
- Kubernetes deployment
- Horizontal autoscaling

---

# Repository Structure

```text
.
├── app/
│   ├── api/
│   ├── services/
│   ├── workers/
│   ├── vectorstore/
│   └── models/
│
├── tests/
├── docker/
├── requirements.txt
└── README.md
```

---

# Getting Started

## Clone the repository

```bash
git clone https://github.com/yourusername/flokka.git
cd flokka
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run with Docker

```bash
docker compose up --build
```

## Open API documentation

```
http://localhost:8000/docs
```

---

# Roadmap

## P0

- Persistent job storage
- Separate Celery queues
- Shared job status

## P1

- Retrieval API
- Authentication
- Rate limiting
- Document deduplication

## P2

- Shared embedding service
- Distributed vector database
- Horizontal autoscaling
- Multi-tenant collections

---

# Where Flokka Fits in a RAG Pipeline

```text
                Documents
                    │
                    ▼
        Flokka Ingestion Engine
                    │
                    ▼
             Vector Database
                    │
                    ▼
              Retrieval API
                    │
                    ▼
             Large Language Model
                    │
                    ▼
               Final Response
```

Flokka is responsible for the **ingestion stage** of a Retrieval-Augmented Generation (RAG) pipeline by converting raw documents into searchable vector representations.

---

# Key Engineering Concepts Demonstrated

- Distributed task processing
- Asynchronous backend architecture
- API-worker decoupling
- Semantic search
- Vector databases
- Production-oriented system design
- Retry mechanisms
- Fault tolerance
- Horizontal scalability
- Backend architecture design

---

# Future Documentation

This README provides a high-level overview of the system.

A detailed **`ARCHITECTURE.md`** document will cover:

- Design rationale
- Technology comparisons
- Scaling considerations
- Failure modes
- Performance tradeoffs
- Production migration strategies

---

## License

This project is intended for educational and portfolio purposes.