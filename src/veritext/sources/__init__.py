"""Free source search and reference library package for Veritext."""

from veritext.sources.http import (
    CachedResponse,
    PerHostRateLimiter,
    PoliteHttpClient,
    SQLiteHttpCache,
    get_http_client,
)
from veritext.sources.library import (
    index_default_references,
    index_directory,
    index_file,
)
from veritext.sources.models import Query, SourceCandidate
from veritext.sources.query_builder import build_queries, extract_key_phrases
from veritext.sources.retrieval import find_candidates
from veritext.sources.store import ReferenceStore, compute_content_hash

__all__ = [
    "SourceCandidate",
    "Query",
    "PoliteHttpClient",
    "SQLiteHttpCache",
    "CachedResponse",
    "PerHostRateLimiter",
    "get_http_client",
    "ReferenceStore",
    "compute_content_hash",
    "extract_key_phrases",
    "build_queries",
    "index_file",
    "index_directory",
    "index_default_references",
    "find_candidates",
]
