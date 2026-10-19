"""Crossref search client."""

import hashlib
import logging
import re
from typing import List, Optional

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"


def _clean_jats_tags(text: str) -> str:
    """Strips JATS XML markup tags often present in Crossref abstracts."""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def search(
    query: str, limit: int = 5, client: Optional[PoliteHttpClient] = None
) -> List[SourceCandidate]:
    """
    Searches Crossref API for academic works matching the query.
    Returns a list of SourceCandidate objects.
    """
    if not query or not query.strip():
        return []

    http = client or get_http_client()
    params = {
        "query": query.strip(),
        "rows": limit,
        "mailto": "info@veritext.org",
    }

    try:
        resp = http.get(CROSSREF_API_URL, params=params)
        if resp.status_code != 200:
            logger.warning(
                f"Crossref returned HTTP {resp.status_code} for query: {query}"
            )
            return []

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        candidates: List[SourceCandidate] = []

        for item in items:
            titles = item.get("title", [])
            title = titles[0] if titles else "Untitled Crossref Work"

            authors = []
            for a in item.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    authors.append(name)

            doi = item.get("DOI", "")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")

            # Extract year from published date parts
            year = None
            date_info = (
                item.get("published-print")
                or item.get("published-online")
                or item.get("created")
            )
            if date_info and "date-parts" in date_info:
                parts = date_info["date-parts"]
                if parts and parts[0]:
                    try:
                        year = int(parts[0][0])
                    except (ValueError, TypeError):
                        pass

            abstract = _clean_jats_tags(item.get("abstract", ""))
            text = abstract if abstract else title

            # Generate stable ID
            id_source = doi or url or title
            candidate_id = "crossref_" + hashlib.sha256(
                id_source.encode("utf-8")
            ).hexdigest()[:16]

            candidates.append(
                SourceCandidate(
                    id=candidate_id,
                    title=title,
                    authors=authors,
                    url=url,
                    origin="crossref",
                    year=year,
                    text=text,
                    metadata={"doi": doi, "publisher": item.get("publisher", "")},
                )
            )

        return candidates

    except Exception as e:
        logger.warning(f"Error querying Crossref for '{query}': {e}")
        return []
