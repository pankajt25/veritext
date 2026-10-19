"""Wikipedia search client utilizing MediaWiki Action API for text extracts."""

import hashlib
import logging
import re
from typing import List, Optional

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"


def _clean_html(raw_html: str) -> str:
    """Removes HTML tags from snippet strings."""
    if not raw_html:
        return ""
    clean = re.sub(r"<[^>]+>", " ", raw_html)
    return " ".join(clean.split())


def search(
    query: str, limit: int = 5, client: Optional[PoliteHttpClient] = None
) -> List[SourceCandidate]:
    """
    Searches Wikipedia using the MediaWiki Action API and extracts introductory prose.
    Returns a list of SourceCandidate objects.
    """
    if not query or not query.strip():
        return []

    http = client or get_http_client()
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": limit,
        "format": "json",
    }

    try:
        resp = http.get(WIKIPEDIA_API_URL, params=search_params)
        if resp.status_code != 200:
            logger.warning(
                f"Wikipedia returned HTTP {resp.status_code} for search query: {query}"
            )
            return []

        data = resp.json()
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return []

        page_ids = [str(item["pageid"]) for item in search_results if "pageid" in item]
        snippets = {
            str(item["pageid"]): _clean_html(item.get("snippet", ""))
            for item in search_results
            if "pageid" in item
        }

        # Retrieve plain-text introductory extracts for the found pages
        extracts = {}
        if page_ids:
            extract_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "pageids": "|".join(page_ids),
                "format": "json",
            }
            ext_resp = http.get(WIKIPEDIA_API_URL, params=extract_params)
            if ext_resp.status_code == 200:
                pages = ext_resp.json().get("query", {}).get("pages", {})
                for pid, pdata in pages.items():
                    ext = pdata.get("extract", "").strip()
                    if ext:
                        extracts[str(pid)] = ext

        candidates: List[SourceCandidate] = []
        for item in search_results:
            pid = str(item.get("pageid", ""))
            title = item.get("title", "Wikipedia Article")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            text = extracts.get(pid) or snippets.get(pid) or title

            candidate_id = "wiki_" + (
                pid
                if pid
                else hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16]
            )

            candidates.append(
                SourceCandidate(
                    id=candidate_id,
                    title=title,
                    authors=["Wikipedia Contributors"],
                    url=url,
                    origin="wikipedia",
                    year=None,
                    text=text,
                    metadata={"pageid": pid, "wordcount": item.get("wordcount", 0)},
                )
            )

        return candidates

    except Exception as e:
        logger.warning(f"Error querying Wikipedia for '{query}': {e}")
        return []
