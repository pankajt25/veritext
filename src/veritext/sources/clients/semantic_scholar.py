"""Semantic Scholar search client with graceful rate-limit handling."""

import hashlib
import logging
from typing import List, Optional

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Session rate limit and disable flag
_SESSION_RATE_LIMITED: bool = False
_DISABLED_BY_CONFIG: bool = False


def set_semantic_scholar_disabled(disabled: bool = True) -> None:
    """Configures whether Semantic Scholar is explicitly disabled."""
    global _DISABLED_BY_CONFIG
    _DISABLED_BY_CONFIG = disabled


def reset_semantic_scholar_session() -> None:
    """Resets the in-memory session rate-limit flag."""
    global _SESSION_RATE_LIMITED
    _SESSION_RATE_LIMITED = False


def is_semantic_scholar_available() -> bool:
    """Returns True if Semantic Scholar is active and not rate-limited for this session."""
    return not (_SESSION_RATE_LIMITED or _DISABLED_BY_CONFIG)


def search(
    query: str, limit: int = 5, client: Optional[PoliteHttpClient] = None
) -> List[SourceCandidate]:
    """
    Searches Semantic Scholar API for academic papers.
    Unauthenticated requests may return 429; degrades gracefully and never crashes.
    After the first 429, skips Semantic Scholar for the rest of the session with a single log line.
    """
    global _SESSION_RATE_LIMITED

    if not is_semantic_scholar_available():
        return []

    if not query or not query.strip():
        return []

    http = client or get_http_client()
    params = {
        "query": query.strip(),
        "limit": limit,
        "fields": "title,authors,year,abstract,url,venue",
    }

    try:
        resp = http.get(SEMANTIC_SCHOLAR_API_URL, params=params)

        if resp.status_code == 429:
            _SESSION_RATE_LIMITED = True
            logger.warning(
                "Semantic Scholar returned HTTP 429 (rate-limited). Skipping Semantic Scholar for the remainder of this session."
            )
            return []

        if resp.status_code != 200:
            logger.warning(
                f"Semantic Scholar returned HTTP {resp.status_code} for query: {query}"
            )
            return []

        data = resp.json()
        papers = data.get("data", [])
        candidates: List[SourceCandidate] = []

        for paper in papers:
            title = paper.get("title") or "Untitled Paper"

            authors = []
            for a in paper.get("authors", []):
                name = a.get("name")
                if name:
                    authors.append(name.strip())

            paper_id = paper.get("paperId", "")
            url = (
                paper.get("url")
                or f"https://www.semanticscholar.org/paper/{paper_id}"
            )
            year = paper.get("year")
            abstract = paper.get("abstract", "")
            text = abstract if abstract else title

            candidate_id = "s2_" + (
                paper_id
                if paper_id
                else hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16]
            )

            candidates.append(
                SourceCandidate(
                    id=candidate_id,
                    title=title,
                    authors=authors,
                    url=url,
                    origin="semantic_scholar",
                    year=year,
                    text=text,
                    metadata={"paper_id": paper_id, "venue": paper.get("venue", "")},
                )
            )

        return candidates

    except Exception as e:
        logger.warning(f"Error querying Semantic Scholar for '{query}': {e}")
        return []
