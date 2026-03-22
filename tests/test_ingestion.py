"""Tests for the document ingestion API."""

import io

import pytest
from fastapi.testclient import TestClient

from app.models.document import JobStatus


class TestHealthEndpoint:
    def test_returns_200_with_status_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body


class TestIngestEndpoint:
    """Tests for POST /api/v1/ingest/."""

    def test_txt_upload_returns_202(
        self, client: TestClient, mock_celery
    ) -> None:
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("report.txt", io.BytesIO(b"Hello world.\n" * 40), "text/plain")},
            data={"chunk_size": "256", "chunk_overlap": "32"},
        )
        assert response.status_code == 202

    def test_response_contains_required_fields(
        self, client: TestClient, mock_celery
    ) -> None:
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("doc.txt", io.BytesIO(b"Sample text." * 20), "text/plain")},
        )
        assert response.status_code == 202
        body = response.json()
        assert isinstance(body["job_id"], str) and body["job_id"]
        assert isinstance(body["document_id"], str) and body["document_id"]
        assert body["filename"] == "doc.txt"
        assert body["status"] == JobStatus.PENDING
        assert "message" in body

    def test_unsupported_mime_type_returns_415(
        self, client: TestClient, mock_celery
    ) -> None:
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("page.html", io.BytesIO(b"<html/>"), "text/html")},
        )
        assert response.status_code == 415

    def test_job_is_retrievable_after_upload(
        self, client: TestClient, mock_celery
    ) -> None:
        upload = client.post(
            "/api/v1/ingest/",
            files={
                "file": ("check.txt", io.BytesIO(b"Status check document." * 10), "text/plain")
            },
        )
        assert upload.status_code == 202
        job_id = upload.json()["job_id"]

        status_response = client.get(f"/api/v1/ingest/{job_id}")
        assert status_response.status_code == 200
        assert status_response.json()["job_id"] == job_id

    def test_unknown_job_id_returns_404(self, client: TestClient) -> None:
        response = client.get("/api/v1/ingest/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404

    @pytest.mark.parametrize(
        "chunk_size,chunk_overlap",
        [
            (64, 0),
            (512, 64),
            (4096, 512),
        ],
    )
    def test_accepts_valid_chunk_parameters(
        self,
        client: TestClient,
        mock_celery,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        response = client.post(
            "/api/v1/ingest/",
            files={"file": ("param.txt", io.BytesIO(b"Params test." * 20), "text/plain")},
            data={"chunk_size": str(chunk_size), "chunk_overlap": str(chunk_overlap)},
        )
        assert response.status_code == 202
