"""Tests for Veritext REST API endpoints, security, queue, and authentication."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from veritext.api.app import app, sanitize_filename
from veritext.api.auth import (
    check_rate_limit,
    create_api_key,
    hash_key,
    load_api_keys,
    revoke_api_key,
    sign_webhook_payload,
    verify_api_key,
)
from veritext.api.queue import JobQueue


@pytest.fixture
def api_keys_file(tmp_path):
    keys_path = tmp_path / "api_keys.json"
    key, khash = create_api_key("test_user", config_path=keys_path)
    return key, keys_path


def test_sanitize_filename():
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\calc.exe") == "calc.exe"
    assert sanitize_filename("student essay (final).pdf") == "student_essay__final_.pdf"


def test_api_key_auth_and_revocation(tmp_path):
    keys_path = tmp_path / "api_keys.json"
    key, khash = create_api_key("instructor_jane", config_path=keys_path)
    assert key.startswith("vt_live_")

    # Verify valid
    meta = verify_api_key(key, config_path=keys_path)
    assert meta is not None
    assert meta["name"] == "instructor_jane"

    # Verify invalid
    assert verify_api_key("vt_live_invalid12345", config_path=keys_path) is None
    assert verify_api_key("non_prefixed_key", config_path=keys_path) is None

    # Revoke
    revoked = revoke_api_key("instructor_jane", config_path=keys_path)
    assert revoked is True
    assert verify_api_key(key, config_path=keys_path) is None


def test_rate_limiting():
    key_hash = "test_rate_limit_hash"
    # Should allow up to max_requests
    for _ in range(5):
        assert check_rate_limit(key_hash, max_requests=5, window_seconds=60) is True
    # 6th request should fail
    assert check_rate_limit(key_hash, max_requests=5, window_seconds=60) is False


def test_webhook_signing():
    secret = "super_secret_webhook_key"
    payload = b'{"event": "completed", "job_id": "1234"}'
    sig = sign_webhook_payload(secret, payload)
    assert sig.startswith("sha256=")

    # Verify signature reproducibility
    sig2 = sign_webhook_payload(secret, payload)
    assert sig == sig2

    # Different secret gives different signature
    diff_sig = sign_webhook_payload("wrong_secret", payload)
    assert sig != diff_sig


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "worker_running" in data
        assert "jobs_in_queue" in data


def test_submit_without_auth_fails():
    with TestClient(app) as client:
        files = {"file": ("test.txt", b"Some sample text for checking.", "text/plain")}
        resp = client.post("/submit", files=files)
        assert resp.status_code == 401
        assert "Missing required 'X-API-Key' header" in resp.json()["detail"]


def test_submit_with_invalid_extension(api_keys_file):
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        with TestClient(app) as client:
            files = {"file": ("malicious.exe", b"binary data", "application/octet-stream")}
            headers = {"X-API-Key": key}
            resp = client.post("/submit", files=files, headers=headers)
            assert resp.status_code == 400
            assert "Unsupported file extension" in resp.json()["detail"]


def test_submit_empty_file(api_keys_file):
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        with TestClient(app) as client:
            files = {"file": ("empty.txt", b"", "text/plain")}
            headers = {"X-API-Key": key}
            resp = client.post("/submit", files=files, headers=headers)
            assert resp.status_code == 400
            assert "empty" in resp.json()["detail"].lower()


def test_submit_and_check_job(api_keys_file, tmp_path):
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        with TestClient(app) as client:
            files = {"file": ("essay.txt", b"This is a legitimate academic essay submission.", "text/plain")}
            headers = {"X-API-Key": key}
            resp = client.post("/submit", files=files, headers=headers, data={"mode": "fast"})
            assert resp.status_code == 202
            data = resp.json()
            assert data["status"] == "queued"
            job_id = data["job_id"]
            assert job_id

            # Check job endpoint
            status_resp = client.get(f"/jobs/{job_id}", headers=headers)
            assert status_resp.status_code == 200
            job_data = status_resp.json()
            assert job_data["job_id"] == job_id
            assert job_data["status"] in ("queued", "running", "completed")


def test_get_nonexistent_job_returns_404(api_keys_file):
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        with TestClient(app) as client:
            headers = {"X-API-Key": key}
            resp = client.get("/jobs/nonexistent_job_99999", headers=headers)
            assert resp.status_code == 404


def test_sse_events_auth_failure():
    with TestClient(app) as client:
        resp = client.get("/jobs/test_job_123/events")
        assert resp.status_code == 401
        assert "Missing required 'X-API-Key' header" in resp.json()["detail"]


def test_sse_events_nonexistent_job(api_keys_file):
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        with TestClient(app) as client:
            headers = {"X-API-Key": key}
            resp = client.get("/jobs/missing_job_abc/events", headers=headers)
            assert resp.status_code == 404


def test_sse_event_order(api_keys_file):
    import secrets
    key, keys_path = api_keys_file
    with patch("veritext.api.app.verify_api_key", return_value={"name": "test"}):
        from veritext.api.app import queue

        job_id = f"sse_{secrets.token_hex(12)}"
        queue.enqueue_job(
            job_id=job_id,
            file_path="/tmp/test.txt",
            original_filename="test.txt",
            options={},
        )
        queue.add_event(job_id, "progress", {"step": "similarity", "progress": 0.4})
        queue.add_event(job_id, "engine_done", {"engine": "similarity", "progress": 0.6})
        queue.add_event(job_id, "completed", {"job_id": job_id, "status": "completed", "progress": 1.0})

        with TestClient(app) as client:
            headers = {"X-API-Key": key}
            resp = client.get(f"/jobs/{job_id}/events", headers=headers)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            lines = resp.text.strip().split("\n")
            event_types = [
                line.split("event: ")[1].strip()
                for line in lines
                if line.startswith("event: ")
            ]

            assert len(event_types) >= 4
            assert event_types[0] == "status"
            assert event_types[1] == "progress"
            assert event_types[2] == "engine_done"
            assert event_types[3] == "completed"

