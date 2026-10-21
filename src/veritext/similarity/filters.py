"""Extended exclusion filters and instant similarity recalculation without re-running engines.

Supports:
(a) Exclude by source (in one click)
(b) Exclude assignment-prompt / template text
(c) Persistent allowlist ('reviewed, not a problem' in data/config/allowlist.json)
(d) Built-in list of common academic phrases
(e) Quotes and small matches
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from veritext.similarity.models import MatchSpan, SimilarityResult, SourceMatch
from veritext.similarity.scorer import calculate_covered_characters, merge_character_intervals

logger = logging.getLogger(__name__)

ALLOWLIST_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "config"
    / "allowlist.json"
)

# Built-in list of common academic stock phrases
COMMON_ACADEMIC_PHRASES: List[str] = [
    "in this paper we propose",
    "the remainder of this paper is organized as follows",
    "as shown in table",
    "as shown in figure",
    "experimental results demonstrate that",
    "in order to evaluate the performance",
    "to the best of our knowledge",
    "further research is needed to",
    "it is important to note that",
    "on the other hand",
    "a wide variety of",
    "this paper is structured as follows",
    "plays a crucial role in",
    "have attracted considerable attention",
    "can be seen from the results",
    "is beyond the scope of this study",
    "conflict of interest",
    "financial support for this research",
    "all rights reserved",
    "the rest of the paper is organized",
    "in recent years there has been",
    "has been widely used in",
    "the experimental results show that",
]


def load_allowlist(config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Loads persistent allowlist of reviewed phrases."""
    p = Path(config_path or ALLOWLIST_PATH)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data.get("phrases", [])
            elif isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"Error reading allowlist from {p}: {e}")
    return []


def save_allowlist(phrases: List[str], config_path: Optional[Union[str, Path]] = None):
    """Persists updated allowlist of reviewed phrases."""
    p = Path(config_path or ALLOWLIST_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    clean_phrases = sorted(list({ph.strip() for ph in phrases if ph.strip()}))
    payload = {
        "_description": "Persistent allowlist of phrases marked as reviewed and harmless by the user.",
        "phrases": clean_phrases,
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def add_to_allowlist(phrase: str, config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Adds a phrase to the persistent allowlist and returns the updated list."""
    curr = load_allowlist(config_path)
    if phrase.strip() and phrase.strip() not in curr:
        curr.append(phrase.strip())
        save_allowlist(curr, config_path)
    return curr


def remove_from_allowlist(phrase: str, config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Removes a phrase from the persistent allowlist and returns the updated list."""
    curr = load_allowlist(config_path)
    if phrase.strip() in curr:
        curr = [p for p in curr if p != phrase.strip()]
        save_allowlist(curr, config_path)
    return curr


@dataclass
class ExclusionFilters:
    """Settings controlling live exclusions for similarity calculation."""

    exclude_small_matches: bool = True
    min_match_chars: int = 25
    exclude_quotes: bool = False
    exclude_sources: Set[str] = field(default_factory=set)  # source_ids
    exclude_template: bool = False
    template_text: Optional[str] = None
    exclude_allowlist: bool = False
    allowlist: List[str] = field(default_factory=list)
    exclude_academic_phrases: bool = False
    academic_phrases: List[str] = field(
        default_factory=lambda: list(COMMON_ACADEMIC_PHRASES)
    )
    exclude_ignored_reviews: bool = True
    ignored_span_keys: Set[str] = field(default_factory=set)

    def to_list(self) -> List[str]:
        """Returns readable names of all currently active exclusion filters."""
        active = []
        if self.exclude_small_matches:
            active.append(f"Small matches (<{self.min_match_chars} chars)")
        if self.exclude_quotes:
            active.append("Quoted / cited text")
        if self.exclude_sources:
            active.append(f"Excluded sources ({len(self.exclude_sources)})")
        if self.exclude_template and self.template_text:
            active.append("Assignment prompt / template text")
        if self.exclude_allowlist:
            active.append(f"Allowlist ({len(self.allowlist)} phrase(s))")
        if self.exclude_academic_phrases:
            active.append("Common academic phrases")
        if self.exclude_ignored_reviews and self.ignored_span_keys:
            active.append(f"Ignored review items ({len(self.ignored_span_keys)})")
        return active


def is_span_excluded(span: MatchSpan, filters: ExclusionFilters) -> bool:
    """Checks whether a single MatchSpan is excluded by active filters."""
    # 0. Exclude Ignored Reviews
    if filters.exclude_ignored_reviews and filters.ignored_span_keys:
        span_id = span.metadata.get("id") if span.metadata else None
        span_key1 = f"span_{span.doc_start}_{span.doc_end}"
        span_key2 = f"{span.doc_start}_{span.doc_end}"
        span_key3 = f"{span.source_id}:{span.doc_start}:{span.doc_end}"
        if (
            (span_id and span_id in filters.ignored_span_keys)
            or span_key1 in filters.ignored_span_keys
            or span_key2 in filters.ignored_span_keys
            or span_key3 in filters.ignored_span_keys
        ):
            return True

    span_len = span.doc_end - span.doc_start
    m_text = (span.matched_text or "").strip()
    m_text_lower = m_text.lower()

    # 1. Exclude Small Matches
    if filters.exclude_small_matches and span_len < filters.min_match_chars:
        return True

    # 2. Exclude by Source
    if filters.exclude_sources and span.source_id in filters.exclude_sources:
        return True

    # 3. Exclude Quotes
    if filters.exclude_quotes:
        if span.metadata.get("is_quoted", False):
            return True
        # Check if enclosed in quotes
        if (
            (m_text.startswith('"') and m_text.endswith('"'))
            or (m_text.startswith('“') and m_text.endswith('”'))
            or (m_text.startswith('«') and m_text.endswith('»'))
        ):
            return True

    # 4. Exclude Template / Assignment Prompt Text
    if filters.exclude_template and filters.template_text:
        tpl_clean = filters.template_text.lower()
        if m_text_lower in tpl_clean:
            return True
        # Check token containment if span has >= 5 words
        span_words = set(re.findall(r"\b\w{3,}\b", m_text_lower))
        if span_words:
            tpl_words = set(re.findall(r"\b\w{3,}\b", tpl_clean))
            overlap = len(span_words.intersection(tpl_words)) / len(span_words)
            if overlap >= 0.85:
                return True

    # 5. Exclude Allowlist Phrases
    if filters.exclude_allowlist and filters.allowlist:
        for phrase in filters.allowlist:
            ph_lower = phrase.strip().lower()
            if not ph_lower:
                continue
            if ph_lower in m_text_lower or m_text_lower in ph_lower:
                return True

    # 6. Exclude Common Academic Phrases
    if filters.exclude_academic_phrases:
        phrases = filters.academic_phrases or COMMON_ACADEMIC_PHRASES
        for aph in phrases:
            aph_lower = aph.strip().lower()
            if not aph_lower:
                continue
            if aph_lower in m_text_lower or m_text_lower in aph_lower:
                return True

    return False


def recalculate_similarity(
    similarity: SimilarityResult,
    clean_text: str,
    filters: ExclusionFilters,
) -> SimilarityResult:
    """Instantly recalculates similarity metrics without re-running any engines.

    Merges surviving character match intervals and updates overall and per-source percentages.
    """
    total_clean_chars = max(1, len(clean_text) if clean_text else similarity.total_clean_chars)

    # Use original all_spans cached in metadata if available
    original_spans: List[MatchSpan] = similarity.metadata.get("all_original_spans", similarity.spans)

    surviving_spans: List[MatchSpan] = []
    for s in original_spans:
        if not is_span_excluded(s, filters):
            surviving_spans.append(s)

    # Recompute document character intervals
    doc_intervals = [(s.doc_start, s.doc_end) for s in surviving_spans]
    total_matched_chars = calculate_covered_characters(doc_intervals)
    overall_percentage = min(100.0, (total_matched_chars / total_clean_chars) * 100.0)

    # Per-source breakdown
    spans_by_source: Dict[str, List[MatchSpan]] = {}
    for s in surviving_spans:
        spans_by_source.setdefault(s.source_id, []).append(s)

    source_matches: List[SourceMatch] = []
    # Preserve metadata from existing sources
    existing_src_map = {sm.source_id: sm for sm in similarity.sources}

    for s_id, s_spans in spans_by_source.items():
        ex_src = existing_src_map.get(s_id)
        src_title = ex_src.source_title if ex_src else (s_spans[0].source_title or s_id)
        src_url = ex_src.source_url if ex_src else s_spans[0].source_url
        src_origin = ex_src.origin if ex_src else "unknown"

        src_intervals = [(sp.doc_start, sp.doc_end) for sp in s_spans]
        src_matched_chars = calculate_covered_characters(src_intervals)
        src_pct = min(100.0, (src_matched_chars / total_clean_chars) * 100.0)

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

    source_matches.sort(key=lambda sm: sm.percentage, reverse=True)

    ranked_spans = sorted(
        surviving_spans,
        key=lambda s: (s.score, (s.doc_end - s.doc_start)),
        reverse=True,
    )

    new_meta = dict(similarity.metadata)
    new_meta["all_original_spans"] = original_spans
    new_meta["active_filters"] = filters.to_list()
    new_meta["excluded_spans_count"] = len(original_spans) - len(surviving_spans)

    return SimilarityResult(
        document_path=similarity.document_path,
        total_clean_chars=total_clean_chars,
        matched_chars=total_matched_chars,
        similarity_percentage=overall_percentage,
        spans=ranked_spans[:50],
        sources=source_matches,
        anomalies=similarity.anomalies,
        metadata=new_meta,
    )


def recalculate_report_filters(report: Any, filters: ExclusionFilters) -> Any:
    """Updates FullReport similarity, active filters list, highlights, and summary instantly."""
    if not report.similarity:
        return report

    new_sim = recalculate_similarity(report.similarity, report.clean_text, filters)
    report.similarity = new_sim

    active_filters_list = filters.to_list()
    report.settings["active_filters"] = active_filters_list

    # Update highlight spans
    for h in report.highlights:
        if h.details.get("engine") == "similarity":
            src_id = h.details.get("source_id")
            if src_id and src_id in filters.exclude_sources:
                h.ignored = True
            elif filters.exclude_small_matches and (h.end - h.start) < filters.min_match_chars:
                h.ignored = True
            else:
                span_text = (report.clean_text or report.original_text)[h.start:h.end]
                dummy_span = MatchSpan(
                    type=h.details.get("match_type", "exact"),
                    score=h.details.get("score", 1.0),
                    source_id=src_id or "",
                    source_title=h.source or "",
                    source_url="",
                    doc_start=h.start,
                    doc_end=h.end,
                    matched_text=span_text,
                )
                if is_span_excluded(dummy_span, filters):
                    h.ignored = True
                else:
                    h.ignored = False

    # Update summary string
    sim_pct = new_sim.similarity_percentage
    src_cnt = len(new_sim.sources)
    ft_cnt = new_sim.metadata.get("fulltexts_fetched", 0)
    sim_str = f"Plagiarism: {sim_pct:.1f}% similarity across {src_cnt} source(s)"
    if ft_cnt > 0:
        sim_str += f" ({ft_cnt} full-text retrieved)"

    summary_parts = [sim_str]
    if report.ai_detect:
        ai_prob = getattr(report.ai_detect, "overall_probability", None)
        ai_pct_str = f"{ai_prob:.1f}%" if ai_prob is not None else "N/A"
        ai_conf = getattr(report.ai_detect, "confidence_band", "none").upper()
        summary_parts.append(f"AI Probability: {ai_pct_str} ({ai_conf} confidence)")

    if report.integrity:
        high_cnt = sum(sum(1 for f in r.findings if f.severity == "high") for r in report.integrity.results)
        warn_cnt = sum(sum(1 for f in r.findings if f.severity == "warning") for r in report.integrity.results)
        info_cnt = sum(sum(1 for f in r.findings if f.severity == "info") for r in report.integrity.results)
        summary_parts.append(f"Integrity: {high_cnt} high, {warn_cnt} warning, {info_cnt} info finding(s)")

    report.summary = " | ".join(summary_parts)
    return report


def recalculate_similarity_with_review(
    similarity: SimilarityResult,
    clean_text: str,
    review_session: Any = None,
    filters: Optional[ExclusionFilters] = None,
) -> SimilarityResult:
    """Recalculates similarity taking into account both exclusion filters and reviewed/ignored spans."""
    active_filters = filters or ExclusionFilters()
    if review_session:
        if hasattr(review_session, "get_ignored_target_ids"):
            ignored_ids = review_session.get_ignored_target_ids()
        elif isinstance(review_session, dict):
            reviews = review_session.get("reviews", {})
            ignored_ids = {
                k
                for k, v in reviews.items()
                if (isinstance(v, dict) and v.get("status") == "ignored")
            }
        else:
            ignored_ids = set()
        active_filters.ignored_span_keys.update(ignored_ids)

    return recalculate_similarity(similarity, clean_text, active_filters)

    return report
