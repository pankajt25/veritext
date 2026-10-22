"""Tests for optional CORE source client and credential secrecy."""

import logging
import os
from unittest.mock import MagicMock, patch
import pytest

from veritext.sources.clients.core import (
    get_core_status,
    is_core_configured,
    search as search_core,
)


def test_core_disabled_without_key(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)

    assert is_core_configured() is False
    status = get_core_status()
    assert status["configured"] is False
    assert status["status"] == "not configured"
    assert "not set" in status["notice"]

    # Search should return [] immediately without making network requests
    results = search_core("quantum machine learning")
    assert results == []


def test_core_enabled_with_mocked_key(monkeypatch):
    secret_key = "secret_core_test_token_xyz"
    monkeypatch.setenv("CORE_API_KEY", secret_key)

    assert is_core_configured() is True
    status = get_core_status()
    assert status["configured"] is True
    assert status["status"] == "active"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "id": "123456",
                "title": "Quantum Error Correction Protocols",
                "description": "An analysis of surface codes and transversal gates.",
                "downloadUrl": "https://core.ac.uk/download/pdf/123456.pdf",
                "yearPublished": 2024,
                "authors": [{"name": "Alice Smith"}, {"name": "Bob Jones"}],
            }
        ]
    }

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp

    results = search_core("surface codes", limit=5, client=mock_client)
    assert len(results) == 1
    assert results[0].id == "core_123456"
    assert "Quantum Error Correction" in results[0].title
    assert "Alice Smith" in results[0].metadata["authors"]

    # Verify authorization header was passed
    args, kwargs = mock_client.get.call_args
    headers = kwargs.get("headers", {})
    assert headers.get("Authorization") == f"Bearer {secret_key}"


def test_core_key_never_logged(monkeypatch, caplog):
    secret_key = "super_confidential_key_123456789"
    monkeypatch.setenv("CORE_API_KEY", secret_key)

    mock_client = MagicMock()
    mock_client.get.side_effect = Exception("Simulated connection drop")

    with caplog.at_level(logging.DEBUG):
        search_core("test query", client=mock_client)

    # Key must NEVER appear anywhere in logs
    for record in caplog.records:
        assert secret_key not in record.message
        assert secret_key not in str(record)
