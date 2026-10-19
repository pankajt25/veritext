"""OpenAlex search client with abstract inverted-index reconstruction."""

import hashlib
import logging
from typing import Any, Dict, List, Optional

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

OPENALEX_API_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs clean sequential text from OpenAlex's abstract_inverted_index.
    E.g. {"We": [0], "propose": [1], "a": [2], "method": [3]} -> "We propose a method"
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    positions: List[tuple[int, str]] = []
    for word, pos_list in inverted_index.items():
        if isinstance(pos_list, list):
            for p in pos_list:
                positions.append((p, word))

    positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positions)


def search(
    query: str, limit: int = 5, client: Optional[PoliteHttpClient] = None
) -> List[SourceCandidate]:
    """
    Searches OpenAlex API for scholarly works matching the query.
    Returns a list of SourceCandidate objects.
    """
    if not query or not query.strip():
        return []

    http = client or get_http_client()
    params = {
        "search": query.strip(),
        "per_page": limit,
        "mailto": "info@veritext.org",
    }

    try:
        resp = http.get(OPENALEX_API_URL, params=params)
        if resp.status_code != 200:
            logger.warning(
                f"OpenAlex returned HTTP {resp.status_code} for query: {query}"
            )
            return []

        data = resp.json()
        results = data.get("results", [])
        candidates: List[SourceCandidate] = []

        for item in results:
            title = item.get("display_name") or item.get("title") or "Untitled Work"

            authors = []
            for authorship in item.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name")
                if author_name:
                    authors.append(author_name)

            doi = item.get("doi") or ""
            url = doi or item.get("id") or ""
            year = item.get("publication_year")

            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            text = abstract if abstract else title

            alex_id = item.get("id", "").split("/")[-1]
            candidate_id = "openalex_" + (
                alex_id
                if alex_id
                else hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16]
            )

            candidates.append(
                SourceCandidate(
                    id=candidate_id,
                    title=title,
                    authors=authors,
                    url=url,
                    origin="openalex",
                    year=year,
                    text=text,
                    metadata={
                        "doi": doi,
                        "openalex_id": item.get("id"),
                        "cited_by_count": item.get("cited_by_count", 0),
                    },
                )
            )

        return candidates

    except Exception as e:
        logger.warning(f"Error querying OpenAlex for '{query}': {e}")
        return []
