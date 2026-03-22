"""Shared pytest fixtures."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def mock_celery(monkeypatch):
    """
    Prevent Celery from attempting a Redis connection during tests.

    Apply this fixture explicitly to tests that exercise the ingest endpoint;
    do not declare it autouse so that unit tests for other services are not
    affected.
    """
    with patch("app.api.routes.ingestion._enqueue", return_value=None):
        yield


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client shared across the session."""
    return TestClient(app)


@pytest.fixture
def sample_text() -> str:
    return ("The quick brown fox jumps over the lazy dog. " * 20).strip()


@pytest.fixture
def sample_document_id() -> str:
    return "doc-test-001"


@pytest.fixture
def sample_job_id() -> str:
    return "job-test-001"
