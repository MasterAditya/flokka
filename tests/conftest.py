"""Shared pytest fixtures."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def mock_celery_dispatch():
    """Prevent tests from trying to connect to Redis via Celery."""
    with patch("app.api.routes.ingestion._dispatch_task", return_value=None):
        yield


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI test client."""
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
