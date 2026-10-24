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
    include_single_window_semantic: bool = False,
    include_semantic_in_score: bool = False,
) -> SimilarityResult:
    """
    Computes hybrid similarity score by merging character match intervals.

    By default, the headline similarity percentage is computed ONLY from exact
    and near-copy matches. Semantic matches are never added to it by default.
    They are tracked separately as paraphrase review leads.

    Args:
        clean_text: Cleaned text string of the document.
        spans: All candidate match spans (exact, near-copy, paraphrase).
        sources_map: Mapping of source_id -> SourceCandidate.
        document_path: Optional path to the document file.
        exclude_small_matches: Exclude very small isolated spans below min_match_chars.
        min_match_chars: Minimum character length for counted matches (default: 25).
        top_n_spans: Maximum number of top spans to retain in the report.
        anomalies: List of text anomalies found during normalization.
        include_single_window_semantic: Include isolated single-window semantic matches
            when include_semantic_in_score is True.
        include_semantic_in_score: Opt-in flag to include semantic paraphrase matches
            in the headline score (default: False, exact & near-copy only).

    Returns:
        Complete SimilarityResult instance.
    """
    sources_dict = sources_map or {}
    total_clean_chars = max(1, len(clean_text))

    # 1. Filter counted spans and separate semantic leads
    counted_spans: List[MatchSpan] = []
    semantic_leads: List[MatchSpan] = []

    for span in spans:
        is_exact_or_near = span.type in ("exact", "near_copy")
        if not is_exact_or_near:
            semantic_leads.append(span)

        span_len = span.doc_end - span.doc_start
        if exclude_small_matches and span_len < min_match_chars:
            continue

        if not is_exact_or_near:
            # Semantic matches are never added to headline score by default
            if not include_semantic_in_score:
                continue
            # When user opted into including semantic matches, respect single-window rule
            if span.metadata.get("is_single_window", False) and not include_single_window_semantic:
                continue

        counted_spans.append(span)

    # 2. Compute overall document character coverage for headline score
    doc_intervals = [(s.doc_start, s.doc_end) for s in counted_spans]
    total_matched_chars = calculate_covered_characters(doc_intervals)
    overall_percentage = min(
        100.0, (total_matched_chars / total_clean_chars) * 100.0
    )

    # Calculate separate paraphrase leads metrics
    sem_intervals = [(s.doc_start, s.doc_end) for s in semantic_leads]
    sem_matched_chars = calculate_covered_characters(sem_intervals)
    sem_percentage = min(100.0, (sem_matched_chars / total_clean_chars) * 100.0)
    sem_leads_count = len(semantic_leads)
    strong_leads_cnt = sum(
        1 for s in semantic_leads
        if s.metadata.get("is_strong_lead", False) or s.metadata.get("lead_level") == "strong lead"
    )
    weak_leads_cnt = sem_leads_count - strong_leads_cnt

    score_label = (
        "score includes unverified semantic leads"
        if include_semantic_in_score
        else "exact & near-copy matches only"
    )

    # 3. Compute per-source breakdown
    counted_spans_by_source: Dict[str, List[MatchSpan]] = {}
    for s in counted_spans:
        counted_spans_by_source.setdefault(s.source_id, []).append(s)

    all_spans_by_source: Dict[str, List[MatchSpan]] = {}
    for s in spans:
        all_spans_by_source.setdefault(s.source_id, []).append(s)

    source_matches: List[SourceMatch] = []
    for s_id, s_all_spans in all_spans_by_source.items():
        src_cand = sources_dict.get(s_id)
        src_title = src_cand.title if src_cand else (s_all_spans[0].source_title or s_id)
        src_url = src_cand.url if src_cand else s_all_spans[0].source_url
        src_origin = src_cand.origin if src_cand else "unknown"

        s_counted = counted_spans_by_source.get(s_id, [])
        src_intervals = [(sp.doc_start, sp.doc_end) for sp in s_counted]
        src_matched_chars = calculate_covered_characters(src_intervals)
        src_pct = min(
            100.0, (src_matched_chars / total_clean_chars) * 100.0
        )

        # Source-specific leads breakdown
        s_sem_spans = [sp for sp in s_all_spans if sp.type not in ("exact", "near_copy")]
        s_lead_count = len(s_sem_spans)
        s_strong_count = sum(
            1 for sp in s_sem_spans
            if sp.metadata.get("is_strong_lead", False) or sp.metadata.get("lead_level") == "strong lead"
        )
        s_weak_count = s_lead_count - s_strong_count
        s_lead_intervals = [(sp.doc_start, sp.doc_end) for sp in s_sem_spans]
        s_lead_chars = calculate_covered_characters(s_lead_intervals)
        s_lead_pct = min(100.0, (s_lead_chars / total_clean_chars) * 100.0)

        source_matches.append(
            SourceMatch(
                source_id=s_id,
                source_title=src_title,
                source_url=src_url,
                origin=src_origin,
                matched_chars=src_matched_chars,
                percentage=src_pct,
                match_count=len(s_all_spans),
                spans=s_all_spans,
                lead_count=s_lead_count,
                strong_lead_count=s_strong_count,
                weak_lead_count=s_weak_count,
                lead_percentage=s_lead_pct,
            )
        )

    # Sort sources by headline percentage descending, then by lead count
    source_matches.sort(key=lambda sm: (sm.percentage, sm.lead_count), reverse=True)

    # 4. Rank top match spans (counted spans first, then review spans)
    ranked_spans = sorted(
        spans,
        key=lambda s: (
            s.type in ("exact", "near_copy"),
            not s.metadata.get("is_single_window", False),
            s.score,
            (s.doc_end - s.doc_start),
        ),
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
            "exact_matches_count": sum(1 for s in counted_spans if s.type in ("exact", "near_copy")),
            "paraphrase_matches_count": sum(
                1 for s in counted_spans if s.type in ("near_copy", "paraphrase", "possible paraphrase (review)")
            ),
            "single_window_review_count": sum(1 for s in spans if s.metadata.get("is_single_window", False)),
            "total_matches_count": len(counted_spans),
            "include_single_window_semantic": include_single_window_semantic,
            "include_semantic_in_score": include_semantic_in_score,
            "score_label": score_label,
            "paraphrase_leads_count": sem_leads_count,
            "paraphrase_leads_percentage": round(sem_percentage, 2),
            "strong_leads_count": strong_leads_cnt,
            "weak_leads_count": weak_leads_cnt,
            "explanatory_line": (
                "Similar meaning is not proof of copying; passages on the same topic often match. "
                "Check the linked source."
            ),
            "interpretation_guide": (
                "Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only"
            ),
            "all_original_spans": spans,
        },
    )
