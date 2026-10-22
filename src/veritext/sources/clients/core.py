"""Optional CORE (core.ac.uk) open-access repository search client.

Activated strictly if CORE_API_KEY is set in the environment.
If CORE_API_KEY is unset or blank, status is 'not configured' and searches return [].
Security guarantee: CORE_API_KEY is never logged, printed, or saved to any file.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

CORE_API_URL = "https://api.core.ac.uk/v3/search/works"


def is_core_configured() -> bool:
    """Checks whether the CORE API key is configured in the environment."""
    key = os.environ.get("CORE_API_KEY", "").strip()
    return bool(key)


def get_core_status() -> Dict[str, Any]:
    """Returns the operational status of the CORE client without exposing any secret key."""
    configured = is_core_configured()
    return {
        "source": "core",
        "configured": configured,
        "status": "active" if configured else "not configured",
        "notice": (
            "CORE repository client active."
            if configured
            else "CORE client is not configured (CORE_API_KEY environment variable not set)."
        ),
    }


def search(
    query: str,
    limit: int = 5,
    client: Optional[PoliteHttpClient] = None,
) -> List[SourceCandidate]:
    """Searches CORE open research works if CORE_API_KEY is available.

    Never logs or exposes the API key.
    """
    if not query or not query.strip():
        return []

    api_key = os.environ.get("CORE_API_KEY", "").strip()
    if not api_key:
        logger.debug("CORE search skipped: CORE_API_KEY not configured.")
        return []

    http = client or get_http_client()
    # Mask headers from potential loggers
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query.strip(), "limit": min(limit, 20)}

    try:
        resp = http.get(CORE_API_URL, params=params, headers=headers)
        if resp.status_code != 200:
            logger.warning(f"CORE API returned HTTP {resp.status_code} for query.")
            return []

        data = resp.json()
        results = data.get("results", []) if isinstance(data, dict) else []
        candidates: List[SourceCandidate] = []

        for item in results:
            if not isinstance(item, dict):
                continue

            title = item.get("title") or "Untitled CORE Document"
            abstract = item.get("abstract") or item.get("description") or ""

            # Extract URLs
            url = (
                item.get("downloadUrl")
                or (item.get("sourceFulltextUrls", [None])[0] if isinstance(item.get("sourceFulltextUrls"), list) else None)
                or (f"https://doi.org/{item.get('doi')}" if item.get("doi") else "")
                or f"https://core.ac.uk/works/{item.get('id', '')}"
            )

            # Authors
            authors_data = item.get("authors", [])
            authors_list: List[str] = []
            if isinstance(authors_data, list):
                for a in authors_data:
                    if isinstance(a, dict) and a.get("name"):
                        authors_list.append(a["name"])
                    elif isinstance(a, str):
                        authors_list.append(a)

            doc_id = str(item.get("id") or hashlib.sha256(title.encode("utf-8")).hexdigest()[:10])

            candidate = SourceCandidate(
                id=f"core_{doc_id}",
                title=title,
                authors=authors_list[:5],
                url=url,
                origin="core",
                year=item.get("yearPublished"),
                text=abstract[:800],
                metadata={
                    "year": item.get("yearPublished"),
                    "authors": authors_list[:5],
                    "doi": item.get("doi"),
                    "fulltext_available": bool(item.get("downloadUrl")),
                },
            )
            candidates.append(candidate)

        return candidates[:limit]

    except Exception as e:
        logger.warning(f"Error querying CORE API: {e}")
        return []
