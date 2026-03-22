"""Tests for the ingestion API endpoint."""

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.document import JobStatus


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestIngestEndpoint:
    def test_ingest_txt_returns_202(self, client: TestClient) -> None:
        content = b"Hello, this is a test document.\n" * 30
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")},
            data={"chunk_size": "256", "chunk_overlap": "32"},
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "document_id" in data
        assert data["filename"] == "test.txt"
        assert data["status"] == JobStatus.PENDING

    def test_ingest_returns_job_id(self, client: TestClient) -> None:
        content = b"Sample text for ingestion testing." * 20
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("doc.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 202
        data = response.json()
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0

    def test_ingest_unsupported_type_returns_415(self, client: TestClient) -> None:
        content = b"<html><body>Not a doc</body></html>"
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("page.html", io.BytesIO(content), "text/html")},
        )
        assert response.status_code == 415

    def test_ingest_stores_job_retrievable(self, client: TestClient) -> None:
        content = b"Document for status check." * 10
        post_response = client.post(
            "/api/v1/ingest/",
            files={"file": ("status_test.txt", io.BytesIO(content), "text/plain")},
        )
        assert post_response.status_code == 202
        job_id = post_response.json()["job_id"]

        get_response = client.get(f"/api/v1/ingest/{job_id}")
        assert get_response.status_code == 200
        job_data = get_response.json()
        assert job_data["job_id"] == job_id

    def test_get_unknown_job_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/ingest/nonexistent-job-id")
        assert response.status_code == 404

    def test_ingest_default_chunk_params(self, client: TestClient) -> None:
        content = b"Default params document." * 20
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("defaults.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 202

    def test_message_included_in_response(self, client: TestClient) -> None:
        content = b"Message check document." * 10
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("msg.txt", io.BytesIO(content), "text/plain")},
        )
        assert response.status_code == 202
        assert "message" in response.json()
