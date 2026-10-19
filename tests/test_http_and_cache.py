"""Unit tests for HTTP caching, rate limiting, and retry backoff."""

from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

from veritext.sources.http import (
    PerHostRateLimiter,
    PoliteHttpClient,
    SQLiteHttpCache,
)


def test_sqlite_cache_set_and_get(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    cache = SQLiteHttpCache(db_file)

    url = "https://api.example.org/search?q=test"
    headers = {"Content-Type": "application/json"}
    body = '{"results": ["a", "b"]}'

    cache.set(cache_key=url, status_code=200, headers=headers, body_text=body, ttl_seconds=3600)

    cached = cache.get(url)
    assert cached is not None
    assert cached.status_code == 200
    assert cached.json() == {"results": ["a", "b"]}
    assert cached.headers["Content-Type"] == "application/json"


def test_sqlite_cache_expiry(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    cache = SQLiteHttpCache(db_file)

    url = "https://api.example.org/temp"
    # Negative TTL simulates expired entry
    cache.set(cache_key=url, status_code=200, headers={}, body_text="expired", ttl_seconds=-10)

    cached = cache.get(url)
    assert cached is None
    # Verify pruned from DB
    stats = cache.stats()
    assert stats["entry_count"] == 0


def test_rate_limiter_interval():
    limiter = PerHostRateLimiter(default_interval=0.05)
    url = "https://api.example.org/item"

    start = time.time()
    limiter.wait(url)
    limiter.wait(url)
    elapsed = time.time() - start

    assert elapsed >= 0.045


def test_polite_http_client_uses_cache(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    client = PoliteHttpClient(cache_db_path=db_file)

    # Pre-populate cache
    url = "https://mock.test/api"
    client.cache.set(url, 200, {}, '{"cached": true}')

    with patch.object(client.session, "get") as mock_get:
        resp = client.get(url, use_cache=True)
        assert resp.status_code == 200
        assert resp.json() == {"cached": True}
        # Real session should not have been called
        mock_get.assert_not_called()


def test_polite_http_client_retry_and_backoff(tmp_path):
    db_file = tmp_path / "test_cache.sqlite"
    client = PoliteHttpClient(cache_db_path=db_file, max_retries=3, backoff_factor=0.01)

    url = "https://mock.test/retry"

    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.headers = {}

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.headers = {}
    mock_resp_200.text = '{"success": true}'

    with patch.object(client.session, "get", side_effect=[mock_resp_503, mock_resp_200]) as mock_get:
        resp = client.get(url, use_cache=False)
        assert resp.status_code == 200
        assert resp.json() == {"success": True}
        assert mock_get.call_count == 2
