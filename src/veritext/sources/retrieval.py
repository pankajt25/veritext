"""Unified candidate retrieval coordinator orchestrating online APIs and local library."""

import logging
import re
from typing import Dict, List, Optional, Set

from veritext.ingest.models import Document
from veritext.sources.clients import (
    search_arxiv,
    search_crossref,
    search_openalex,
    search_semantic_scholar,
    search_wikipedia,
)
from veritext.sources.http import PoliteHttpClient, get_http_client
from veritext.sources.models import Query, SourceCandidate
from veritext.sources.query_builder import build_queries
from veritext.sources.store import ReferenceStore, compute_content_hash

logger = logging.getLogger(__name__)


def _extract_word_tokens(text: str) -> Set[str]:
    """Extracts lowercase alphabetic words of length >= 3 for lexical overlap."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def rank_candidates_by_overlap(
    document_text: str, candidates: List[SourceCandidate]
) -> List[SourceCandidate]:
    """
    Ranks candidates based on lexical token overlap with the input document.
    Updates candidate.score and sorts descending.
    """
    doc_tokens = _extract_word_tokens(document_text)
    if not doc_tokens:
        return candidates

    for cand in candidates:
        cand_text = f"{cand.title} {cand.text}"
        cand_tokens = _extract_word_tokens(cand_text)
        if not cand_tokens:
            cand.score = 0.0
            continue

        shared = doc_tokens.intersection(cand_tokens)
        # Jaccard-like score weighted towards shared query vocabulary
        score = len(shared) / (len(cand_tokens) ** 0.5 + 1.0)
        cand.score = round(score, 4)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def find_candidates(
    document: Document,
    use_online: bool = True,
    use_local: bool = True,
    max_sources: int = 30,
    store: Optional[ReferenceStore] = None,
    http_client: Optional[PoliteHttpClient] = None,
    queries: Optional[List[Query]] = None,
) -> List[SourceCandidate]:
    """
    Finds candidate source texts matching the document using local library and/or free online APIs.

    Args:
        document: The ingested Document from Stage 1.
        use_online: If True, queries external public APIs (Crossref, OpenAlex, arXiv, Semantic Scholar, Wikipedia).
        use_local: If True, queries the local SQLite FTS5 reference library.
        max_sources: Maximum number of top candidates to return.
        store: Optional ReferenceStore instance.
        http_client: Optional PoliteHttpClient instance.
        queries: Optional pre-built queries.

    Returns:
        List of de-duplicated, ranked SourceCandidate objects.
    """
    ref_store = store or ReferenceStore()
    client = http_client or get_http_client()

    # 1. Formulate search queries
    search_queries = queries or build_queries(document, max_queries=15)
    candidates: List[SourceCandidate] = []
    seen_hashes: Set[str] = set()

    def add_candidate(cand: SourceCandidate):
        chash = compute_content_hash(cand.title, cand.url, cand.text)
        if chash not in seen_hashes:
            seen_hashes.add(chash)
            candidates.append(cand)

    # 2. Local Library Retrieval via FTS5
    if use_local:
        for q in search_queries:
            search_terms = [q.text]
            if q.keywords:
                search_terms.extend(q.keywords)
            for term in set(search_terms):
                try:
                    local_matches = ref_store.search_fts(term, limit=5)
                    for cand in local_matches:
                        add_candidate(cand)
                except Exception as e:
                    logger.warning(f"Error querying local library for '{term}': {e}")

    # 3. Online Search APIs (executed politely and sequentially)
    if use_online:
        online_clients = [
            ("crossref", search_crossref),
            ("openalex", search_openalex),
            ("arxiv", search_arxiv),
            ("semantic_scholar", search_semantic_scholar),
            ("wikipedia", search_wikipedia),
        ]

        for q in search_queries:
            for name, client_fn in online_clients:
                try:
                    # Query 2-3 items per query to stay polite and fast
                    results = client_fn(q.text, limit=3, client=client)
                    for cand in results:
                        add_candidate(cand)
                except Exception as e:
                    # Individual client failures are logged and never fatal
                    logger.warning(
                        f"Search client '{name}' failed for query '{q.text}': {e}"
                    )

    # 4. Rank candidates by keyword overlap with the document text
    ranked = rank_candidates_by_overlap(document.clean_text, candidates)

    return ranked[:max_sources]
