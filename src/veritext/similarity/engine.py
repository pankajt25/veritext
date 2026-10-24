"""Core plagiarism detection engine running retrieval, normalization, winnowing, and semantic matching."""

import logging
from typing import Any, Dict, List, Optional

from veritext.classrepo.repo import ClassRepositoryManager
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.winnow import winnow
from veritext.ingest.models import Document
from veritext.normalize.normalize import normalize_text
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import SimilarityResult
from veritext.similarity.scorer import compute_similarity_score
from veritext.similarity.semantic import find_semantic_matches
from veritext.sources.fulltext import FullTextRetriever
from veritext.sources.models import SourceCandidate
from veritext.sources.retrieval import find_candidates
from veritext.sources.store import ReferenceStore

logger = logging.getLogger(__name__)


def check_document(
    document: Document,
    use_online: bool = True,
    use_local: bool = True,
    max_sources: int = 30,
    similarity_threshold: float = 0.75,
    store: Optional[ReferenceStore] = None,
    index: Optional[FingerprintIndex] = None,
    options: Optional[Dict[str, Any]] = None,
) -> SimilarityResult:
    """
    Executes the comprehensive plagiarism check on an ingested document:
    1. Normalizes text and tracks anomalies (homoglyphs, zero-width chars).
    2. Retrieves candidate comparison sources via local library and open APIs (Stage 2).
    3. Indexes candidate source passages into winnowing fingerprint inverted index.
    4. Computes winnowing fingerprints and discovers exact / near-verbatim match spans.
    5. Discovers semantic paraphrase matches for non-exact spans using vector search.
    6. Merges intervals and computes overall similarity percentage and per-source breakdown.

    Returns:
        SimilarityResult object.
    """
    opts = options or {}
    ref_store = store or ReferenceStore()
    fp_index = index or FingerprintIndex()

    # Step 1: Text Canonicalisation and Anomaly Tracking
    canonical = normalize_text(document.clean_text)
    anomaly_dicts = [a.to_dict() for a in canonical.anomalies]

    # Step 2: Source Candidate Retrieval
    candidates: List[SourceCandidate] = find_candidates(
        document=document,
        use_online=use_online,
        use_local=use_local,
        max_sources=max_sources,
        store=ref_store,
    )

    # Step 2b: Full-text open-access retrieval for top candidates
    fetch_fulltext = opts.get("fetch_fulltext", use_online)
    fulltexts_fetched = 0
    if fetch_fulltext and candidates:
        try:
            max_ft = opts.get("max_fulltext_fetch", 10)
            ft_retriever = FullTextRetriever(store=ref_store, index=fp_index, max_fetch_per_doc=max_ft)
            candidates, fulltexts_fetched = ft_retriever.fetch_and_index_top_candidates(candidates)
        except Exception as e:
            logger.warning(f"Full-text retrieval step failed gracefully: {e}")

    # Step 2c: Class repository candidate retrieval
    class_name = opts.get("class_name")
    show_names = opts.get("show_names", False)
    if class_name:
        try:
            cmgr = ClassRepositoryManager()
            c_cands, c_fps = cmgr.get_candidates(class_name, show_names=show_names)
            for cc in c_cands:
                candidates.append(cc)
                if cc.id in c_fps:
                    fp_index.add_fingerprints(cc.id, None, c_fps[cc.id])
        except Exception as e:
            logger.warning(f"Class repository '{class_name}' retrieval note: {e}")

    sources_map: Dict[str, SourceCandidate] = {c.id: c for c in candidates}

    # Step 3: Index Candidates in Fingerprint Index
    if use_local:
        fp_index.index_all_library_sources(store=ref_store)
    for cand in candidates:
        fp_index.index_source_candidate(cand)

    # Step 4: Winnowing Fingerprint Matching (Exact Copy)
    doc_fingerprints = winnow(canonical.tokens, k=5, w=4)
    exact_spans = find_exact_matches(
        doc_clean_text=document.clean_text,
        canonical=canonical,
        doc_fingerprints=doc_fingerprints,
        index=fp_index,
        sources_map=sources_map,
        min_words=opts.get("min_words", 8),
    )

    # Step 5: Semantic Paraphrase Matching (Vector Search)
    semantic_spans = []
    min_consec = opts.get("min_consecutive_windows", 2)
    include_single = opts.get("include_single_window", False)

    if candidates and not opts.get("skip_semantic", False):
        try:
            semantic_spans = find_semantic_matches(
                document=document,
                candidates=candidates,
                exact_spans=exact_spans,
                threshold=similarity_threshold,
                progress_callback=opts.get("progress_callback"),
                cancel_check=opts.get("cancel_check"),
                word_ranges=opts.get("word_ranges"),
                min_consecutive_windows=min_consec,
                include_single_window=include_single,
            )
        except Exception as e:
            logger.warning(f"Semantic search encountered an error: {e}")

    # Step 6: Combine Spans and Compute Hybrid Score
    all_spans = exact_spans + semantic_spans
    include_sem_in_score = opts.get("include_semantic_in_score", False)
    result = compute_similarity_score(
        clean_text=document.clean_text,
        spans=all_spans,
        sources_map=sources_map,
        document_path=document.path,
        exclude_small_matches=opts.get("exclude_small_matches", True),
        min_match_chars=opts.get("min_match_chars", 25),
        anomalies=anomaly_dicts,
        include_single_window_semantic=include_single,
        include_semantic_in_score=include_sem_in_score,
    )
    result.metadata["fulltexts_fetched"] = fulltexts_fetched

    return result
