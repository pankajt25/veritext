"""Polite HTTP client with per-host rate limiting, exponential backoff, and SQLite caching."""

import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Optional, Union
from urllib.parse import urlparse

import requests


DEFAULT_USER_AGENT = "Veritext/0.2 (college project; github.com/pankajt25/veritext)"
DEFAULT_TIMEOUT = 10.0
DEFAULT_CACHE_DAYS = 7


class CachedResponse:
    """Mock-compatible response object returned when retrieving from HTTP cache."""

    def __init__(self, status_code: int, text: str, headers: Dict[str, str], url: str):
        self.status_code = status_code
        self.text = text
        self.headers = headers
        self.url = url
        self._json_cache: Optional[Any] = None

    def json(self) -> Any:
        if self._json_cache is None:
            self._json_cache = json.loads(self.text)
        return self._json_cache

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


class SQLiteHttpCache:
    """On-disk SQLite response cache for HTTP requests."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            db_path = base_dir / "data" / "cache" / "http_cache.sqlite"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS http_cache (
                    cache_key TEXT PRIMARY KEY,
                    status_code INTEGER,
                    headers_json TEXT,
                    body_text TEXT,
                    created_at REAL,
                    expires_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_expires_at ON http_cache(expires_at)"
            )

    def get(self, cache_key: str) -> Optional[CachedResponse]:
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT status_code, headers_json, body_text, expires_at
                FROM http_cache WHERE cache_key = ?
                """,
                (cache_key,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            status_code, headers_json, body_text, expires_at = row
            if expires_at and expires_at < now:
                # Expired, clean up
                conn.execute(
                    "DELETE FROM http_cache WHERE cache_key = ?", (cache_key,)
                )
                return None
            headers = json.loads(headers_json) if headers_json else {}
            return CachedResponse(status_code, body_text, headers, cache_key)

    def set(
        self,
        cache_key: str,
        status_code: int,
        headers: Dict[str, str],
        body_text: str,
        ttl_seconds: Optional[int] = DEFAULT_CACHE_DAYS * 86400,
    ):
        now = time.time()
        expires_at = (now + ttl_seconds) if ttl_seconds is not None else 0
        headers_json = json.dumps(dict(headers))
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO http_cache
                (cache_key, status_code, headers_json, body_text, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    status_code,
                    headers_json,
                    body_text,
                    now,
                    expires_at,
                ),
            )

    def stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0]
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"entry_count": count, "size_bytes": size_bytes}

    def clear(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM http_cache")


class PerHostRateLimiter:
    """Enforces minimum intervals between requests to the same hostname."""

    def __init__(self, default_interval: float = 1.0):
        self.default_interval = default_interval
        self.host_intervals: Dict[str, float] = {
            "export.arxiv.org": 3.0,
            "api.semanticscholar.org": 2.0,
            "api.openalex.org": 1.0,
            "api.crossref.org": 1.0,
            "en.wikipedia.org": 1.0,
        }
        self.last_request_time: Dict[str, float] = {}

    def wait(self, url: str):
        host = urlparse(url).netloc.lower()
        if not host:
            return
        interval = self.host_intervals.get(host, self.default_interval)
        last_time = self.last_request_time.get(host, 0.0)
        elapsed = time.time() - last_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self.last_request_time[host] = time.time()


class PoliteHttpClient:
    """
    Polite HTTP client equipped with:
    - Custom User-Agent
    - SQLite response caching
    - Per-host rate limiting
    - Exponential-backoff retries for 429 and 5xx errors
    """

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        cache_db_path: Optional[Union[str, Path]] = None,
        rate_limiter: Optional[PerHostRateLimiter] = None,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout
        self.cache = SQLiteHttpCache(cache_db_path)
        self.rate_limiter = rate_limiter or PerHostRateLimiter()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        ttl_days: int = DEFAULT_CACHE_DAYS,
    ) -> CachedResponse:
        """
        Executes a GET request with caching, rate limiting, and exponential backoff retry.
        """
        # Formulate full canonical URL for caching
        req = requests.Request("GET", url, params=params)
        prep = self.session.prepare_request(req)
        full_url = prep.url or url

        if use_cache:
            cached = self.cache.get(full_url)
            if cached is not None:
                return cached

        # Rate limiting
        self.rate_limiter.wait(full_url)

        req_timeout = timeout or self.timeout
        effective_headers = headers or {}

        # Retries with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    headers=effective_headers,
                    timeout=req_timeout,
                )

                # Retry on 429 or 5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries - 1:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after and retry_after.isdigit():
                            wait_time = float(retry_after)
                        else:
                            wait_time = self.backoff_factor * (2**attempt)
                        time.sleep(wait_time)
                        continue

                # Successful response (or final non-retryable response)
                if use_cache and resp.status_code == 200:
                    self.cache.set(
                        cache_key=full_url,
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        body_text=resp.text,
                        ttl_seconds=ttl_days * 86400,
                    )

                return CachedResponse(
                    status_code=resp.status_code,
                    text=resp.text,
                    headers=dict(resp.headers),
                    url=full_url,
                )

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
                else:
                    raise

        if last_error:
            raise last_error

        return CachedResponse(status_code=500, text="", headers={}, url=full_url)


# Global singleton client instance
_default_client: Optional[PoliteHttpClient] = None


def get_http_client() -> PoliteHttpClient:
    """Returns the shared PoliteHttpClient singleton."""
    global _default_client
    if _default_client is None:
        _default_client = PoliteHttpClient()
    return _default_client
