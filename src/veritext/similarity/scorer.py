"""Hybrid scoring, interval merging, and source breakdown for plagiarism analysis."""

from typing import Any, Dict, List, Optional, Tuple

from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.sources.models import SourceCandidate


def merge_character_intervals(
    intervals: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Merges overlapping and adjacent character intervals [start, end].
    Returns non-overlapping, sorted intervals.
    """
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [sorted_intervals[0]]

    for current in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end:
            # Overlap or contiguous: merge by extending end
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)

    return merged


def calculate_covered_characters(intervals: List[Tuple[int, int]]) -> int:
    """Calculates total unique characters covered by a list of intervals."""
    merged = merge_character_intervals(intervals)
    return sum(end - start for start, end in merged)


def compute_similarity_score(
    clean_text: str,
    spans: List[MatchSpan],
    sources_map: Optional[Dict[str, SourceCandidate]] = None,
    document_path: Optional[str] = None,
    exclude_small_matches: bool = True,
    min_match_chars: int = 25,
    top_n_spans: int = 30,
    anomalies: Optional[List[Dict[str, Any]]] = None,
) -> SimilarityResult:
    """
    Computes hybrid similarity score by merging character match intervals.

    Args:
        clean_text: Cleaned text string of the document.
        spans: All candidate match spans (exact, near-copy, paraphrase).
        sources_map: Mapping of source_id -> SourceCandidate.
        document_path: Optional path to the document file.
        exclude_small_matches: Exclude very small isolated spans below min_match_chars.
        min_match_chars: Minimum character length for counted matches (default: 25).
        top_n_spans: Maximum number of top spans to retain in the report.
        anomalies: List of text anomalies found during normalization.

    Returns:
        Complete SimilarityResult instance.
    """
    sources_dict = sources_map or {}
    total_clean_chars = max(1, len(clean_text))

    # 1. Filter counted spans
    counted_spans: List[MatchSpan] = []
    for span in spans:
        span_len = span.doc_end - span.doc_start
        if exclude_small_matches and span_len < min_match_chars:
            continue
        counted_spans.append(span)

    # 2. Compute overall document character coverage
    doc_intervals = [(s.doc_start, s.doc_end) for s in counted_spans]
    total_matched_chars = calculate_covered_characters(doc_intervals)
    overall_percentage = min(
        100.0, (total_matched_chars / total_clean_chars) * 100.0
    )

    # 3. Compute per-source breakdown
    spans_by_source: Dict[str, List[MatchSpan]] = {}
    for s in counted_spans:
        spans_by_source.setdefault(s.source_id, []).append(s)

    source_matches: List[SourceMatch] = []
    for s_id, s_spans in spans_by_source.items():
        src_cand = sources_dict.get(s_id)
        src_title = src_cand.title if src_cand else (s_spans[0].source_title or s_id)
        src_url = src_cand.url if src_cand else s_spans[0].source_url
        src_origin = src_cand.origin if src_cand else "unknown"

        src_intervals = [(sp.doc_start, sp.doc_end) for sp in s_spans]
        src_matched_chars = calculate_covered_characters(src_intervals)
        src_pct = min(
            100.0, (src_matched_chars / total_clean_chars) * 100.0
        )

        source_matches.append(
            SourceMatch(
                source_id=s_id,
                source_title=src_title,
                source_url=src_url,
                origin=src_origin,
                matched_chars=src_matched_chars,
                percentage=src_pct,
                match_count=len(s_spans),
                spans=s_spans,
            )
        )

    # Sort sources by percentage descending
    source_matches.sort(key=lambda sm: sm.percentage, reverse=True)

    # 4. Rank top match spans
    ranked_spans = sorted(
        counted_spans,
        key=lambda s: (s.score, (s.doc_end - s.doc_start)),
        reverse=True,
    )

    return SimilarityResult(
        document_path=document_path,
        total_clean_chars=total_clean_chars,
        matched_chars=total_matched_chars,
        similarity_percentage=overall_percentage,
        spans=ranked_spans[:top_n_spans],
        sources=source_matches,
        anomalies=anomalies or [],
        metadata={
            "exact_matches_count": sum(1 for s in counted_spans if s.type == "exact"),
            "paraphrase_matches_count": sum(
                1 for s in counted_spans if s.type in ("near_copy", "paraphrase")
            ),
            "total_matches_count": len(counted_spans),
        },
    )
