"""arXiv search client parsing Atom XML responses."""

import hashlib
import logging
from typing import List, Optional
import xml.etree.ElementTree as ET

from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def parse_arxiv_xml(xml_content: str) -> List[SourceCandidate]:
    """Parses arXiv Atom XML response into SourceCandidate objects."""
    if not xml_content or not xml_content.strip():
        return []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse arXiv XML: {e}")
        return []

    candidates: List[SourceCandidate] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
        title = " ".join(raw_title.split())

        authors = []
        for author in entry.findall("atom:author", ATOM_NS):
            name = author.findtext("atom:name", default="", namespaces=ATOM_NS)
            if name and name.strip():
                authors.append(name.strip())

        url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
        # Look for alternate HTML link if available
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("rel") == "alternate":
                url = link.attrib.get("href", url)
                break

        # Published year
        published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        year = None
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                pass

        raw_summary = entry.findtext(
            "atom:summary", default="", namespaces=ATOM_NS
        )
        text = " ".join(raw_summary.split())
        if not text:
            text = title

        # ID extraction
        arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else ""
        candidate_id = "arxiv_" + (
            arxiv_id
            if arxiv_id
            else hashlib.sha256((url or title).encode("utf-8")).hexdigest()[:16]
        )

        candidates.append(
            SourceCandidate(
                id=candidate_id,
                title=title,
                authors=authors,
                url=url,
                origin="arxiv",
                year=year,
                text=text,
                metadata={"published": published, "arxiv_id": arxiv_id},
            )
        )

    return candidates


def search(
    query: str, limit: int = 5, client: Optional[PoliteHttpClient] = None
) -> List[SourceCandidate]:
    """
    Searches arXiv API for preprints matching the query.
    Returns a list of SourceCandidate objects.
    """
    if not query or not query.strip():
        return []

    http = client or get_http_client()
    # arXiv search syntax: all:query
    params = {
        "search_query": f"all:{query.strip()}",
        "start": 0,
        "max_results": limit,
    }

    try:
        resp = http.get(ARXIV_API_URL, params=params)
        if resp.status_code != 200:
            logger.warning(
                f"arXiv returned HTTP {resp.status_code} for query: {query}"
            )
            return []

        return parse_arxiv_xml(resp.text)

    except Exception as e:
        logger.warning(f"Error querying arXiv for '{query}': {e}")
        return []
