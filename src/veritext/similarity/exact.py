"""Exact and near-copy match span detection using winnowing fingerprint index."""

from typing import Dict, List, Optional, Tuple

from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.kgrams import DEFAULT_K
from veritext.normalize.models import CanonicalText
from veritext.similarity.models import MatchSpan
from veritext.sources.models import SourceCandidate

DEFAULT_MIN_WORDS = 8
DEFAULT_MAX_GAP_TOKENS = 4


def find_exact_matches(
    doc_clean_text: str,
    canonical: CanonicalText,
    doc_fingerprints: List[Tuple[int, int]],
    index: FingerprintIndex,
    sources_map: Dict[str, SourceCandidate],
    min_words: int = DEFAULT_MIN_WORDS,
    max_gap_tokens: int = DEFAULT_MAX_GAP_TOKENS,
    k: int = DEFAULT_K,
) -> List[MatchSpan]:
    """
    Identifies exact and near-copy matching spans by querying document fingerprints against the index.
    Clusters adjacent matching fingerprints into contiguous spans and drops matches below min_words.

    Args:
        doc_clean_text: The clean text of the input document.
        canonical: The CanonicalText of the document containing tokens and offset_map.
        doc_fingerprints: List of (hash, token_pos) pairs for the document.
        index: FingerprintIndex instance.
        sources_map: Mapping of source_id -> SourceCandidate.
        min_words: Minimum token length for a match span (default: 8 words).
        max_gap_tokens: Maximum token gap tolerated when chaining adjacent fingerprints (default: 4).
        k: k-gram size used during winnowing (default: 5).

    Returns:
        List of MatchSpan objects with exact character start/end offsets.
    """
    if not doc_fingerprints or not canonical.tokens or not canonical.offset_map:
        return []

    # Map hash to document token positions
    hash_to_doc_positions: Dict[int, List[int]] = {}
    for h, pos in doc_fingerprints:
        hash_to_doc_positions.setdefault(h, []).append(pos)

    # Lookup all hashes in index
    matches = index.lookup_hashes(list(hash_to_doc_positions.keys()))
    if not matches:
        return []

    # Group matches by source_id: source_id -> list of (doc_pos, source_pos)
    source_hits: Dict[str, List[Tuple[int, int]]] = {}
    for h, hit_list in matches.items():
        doc_positions = hash_to_doc_positions.get(h, [])
        for doc_pos in doc_positions:
            for hit in hit_list:
                s_id = hit["source_id"]
                s_pos = hit["position"]
                source_hits.setdefault(s_id, []).append((doc_pos, s_pos))

    match_spans: List[MatchSpan] = []
    num_doc_tokens = len(canonical.tokens)

    for s_id, hits in source_hits.items():
        if not hits:
            continue

        source_cand = sources_map.get(s_id)
        source_title = source_cand.title if source_cand else s_id
        source_url = source_cand.url if source_cand else ""
        source_full_text = source_cand.text if source_cand else ""

        # Sort hits by document token position
        hits.sort(key=lambda x: (x[0], x[1]))

        # Cluster hits that are contiguous or close (within max_gap_tokens)
        clusters: List[List[Tuple[int, int]]] = []
        current_cluster: List[Tuple[int, int]] = [hits[0]]

        for next_hit in hits[1:]:
            prev_doc_pos, prev_src_pos = current_cluster[-1]
            curr_doc_pos, curr_src_pos = next_hit

            # Check if within token gap in document
            gap = curr_doc_pos - prev_doc_pos
            if 0 <= gap <= max_gap_tokens:
                current_cluster.append(next_hit)
            else:
                clusters.append(current_cluster)
                current_cluster = [next_hit]

        if current_cluster:
            clusters.append(current_cluster)

        # Convert each cluster into a MatchSpan if >= min_words
        for cluster in clusters:
            doc_start_token = min(p[0] for p in cluster)
            # The last k-gram at pos covers up to pos + k tokens
            doc_end_token = min(
                max(p[0] for p in cluster) + k, num_doc_tokens
            )

            span_token_count = doc_end_token - doc_start_token
            if span_token_count < min_words:
                continue

            # Map token bounds back to character offsets in document text
            doc_char_start = canonical.offset_map[doc_start_token][0]
            doc_char_end = canonical.offset_map[doc_end_token - 1][1]
            matched_text = doc_clean_text[doc_char_start:doc_char_end]

            # Approximate source bounds
            src_start_token = min(p[1] for p in cluster)
            src_end_token = max(p[1] for p in cluster) + k

            match_spans.append(
                MatchSpan(
                    type="exact",
                    score=1.0,
                    source_id=s_id,
                    source_title=source_title,
                    source_url=source_url,
                    doc_start=doc_char_start,
                    doc_end=doc_char_end,
                    matched_text=matched_text,
                    source_start=src_start_token,
                    source_end=src_end_token,
                    source_text=source_full_text[:300],
                    metadata={
                        "token_count": span_token_count,
                        "shared_fingerprints": len(cluster),
                    },
                )
            )

    # Sort spans by document start offset
    match_spans.sort(key=lambda s: s.doc_start)
    return match_spans
