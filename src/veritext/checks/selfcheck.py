"""Self-plagiarism integrity check.

Detects recycled or self-duplicated text by indexing past student or researcher
submissions under origin="self" and executing exact winnowing fingerprint match
against the user's local reference library.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.kgrams import DEFAULT_K
from veritext.fingerprint.winnow import winnow
from veritext.ingest.loaders import LOADER_MAP
from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document
from veritext.normalize.normalize import normalize_text
from veritext.similarity.exact import DEFAULT_MAX_GAP_TOKENS, DEFAULT_MIN_WORDS
from veritext.similarity.models import MatchSpan
from veritext.sources.models import SourceCandidate
from veritext.sources.store import ReferenceStore, compute_content_hash

DEFAULT_PREVIOUS_DIR = Path("data/my_previous")


def add_previous(
    target: Union[str, Path],
    db_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Indexes file(s) from a file or directory as prior work (origin='self').

    Args:
        target: File or directory path containing user's earlier work.
        db_path: Optional custom SQLite database path.

    Returns:
        List of successfully indexed source IDs or titles.
    """
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"Target path does not exist: {target}")

    files_to_index: List[Path] = []
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in LOADER_MAP and not p.name.startswith("."):
                files_to_index.append(p)
    else:
        files_to_index.append(path)

    store = ReferenceStore(db_path=db_path)
    index = FingerprintIndex(db_path=db_path)
    indexed_names: List[str] = []

    for fpath in files_to_index:
        try:
            doc = load_document(fpath)
            content_text = doc.clean_text or doc.raw_text
            if not content_text.strip():
                continue

            content_hash = compute_content_hash(fpath.name, text=content_text)
            source_id = f"self_{content_hash[:16]}"

            # Add source candidate with origin="self"
            candidate = SourceCandidate(
                id=source_id,
                title=fpath.name,
                authors=[],
                url=str(fpath.resolve()),
                origin="self",
                year=None,
                text=content_text,
                metadata={"file_path": str(fpath.resolve()), "file_type": doc.file_type},
            )
            store.add_source(candidate)

            # Index fingerprints into inverted index
            canonical = normalize_text(content_text)
            fps = winnow(canonical.tokens, k=DEFAULT_K)
            index.add_fingerprints(source_id=source_id, passage_id=None, fingerprints=fps)
            indexed_names.append(fpath.name)
        except Exception:
            continue

    return indexed_names


def _merge_spans(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merges overlapping or adjacent character intervals."""
    if not intervals:
        return []
    sorted_int = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [sorted_int[0]]
    for start, end in sorted_int[1:]:
        p_start, p_end = merged[-1]
        if start <= p_end:
            merged[-1] = (p_start, max(p_end, end))
        else:
            merged.append((start, end))
    return merged


def check_self_plagiarism(
    document: Union[Document, str, Path],
    db_path: Optional[Union[str, Path]] = None,
    min_words: int = DEFAULT_MIN_WORDS,
    max_gap_tokens: int = DEFAULT_MAX_GAP_TOKENS,
    threshold_pct: float = 1.0,
) -> CheckResult:
    """Compares document against indexed earlier work (origin='self').

    Args:
        document: Document instance or file path/string.
        db_path: Optional custom SQLite database path.
        min_words: Minimum word length for an exact match span.
        max_gap_tokens: Maximum token gap for clustering fingerprints.
        threshold_pct: Minimum overlap percentage to generate warnings.

    Returns:
        CheckResult containing self-plagiarism findings.
    """
    if isinstance(document, (str, Path)):
        doc_path = Path(document)
        if doc_path.exists() and doc_path.is_file():
            doc = load_document(doc_path)
        else:
            doc = Document(
                raw_text=str(document),
                clean_text=str(document),
                sentences=[],
                paragraphs=[],
                removed_sections=[],
                file_type="txt",
                path=None,
            )
    else:
        doc = document

    store = ReferenceStore(db_path=db_path)
    index = FingerprintIndex(db_path=db_path)

    # Fetch all registered self sources
    self_sources: Dict[str, SourceCandidate] = {}
    with store._get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sources WHERE origin = 'self'"
        ).fetchall()
        for r in rows:
            authors = json.loads(r["authors_json"]) if r["authors_json"] else []
            sc = SourceCandidate(
                id=r["id"],
                title=r["title"],
                authors=authors,
                url=r["url"] or "",
                origin=r["origin"],
                text=r["text"],
                year=r["year"],
            )
            self_sources[sc.id] = sc

    if not self_sources:
        return CheckResult(
            check_name="self_plagiarism",
            findings=[],
            summary="No prior work registered under origin='self'. Use 'veritext.checks add-previous' to register earlier documents.",
            passed=True,
            metadata={"self_source_count": 0},
        )

    # Normalize document and generate fingerprints
    doc_text = doc.clean_text or doc.raw_text
    canonical = normalize_text(doc_text)
    if not canonical.tokens or not canonical.offset_map:
        return CheckResult(
            check_name="self_plagiarism",
            findings=[],
            summary="Document contains no text tokens to compare.",
            passed=True,
            metadata={"self_source_count": len(self_sources)},
        )

    doc_fps = winnow(canonical.tokens, k=DEFAULT_K)
    if not doc_fps:
        return CheckResult(
            check_name="self_plagiarism",
            findings=[],
            summary="Document yielded no fingerprints.",
            passed=True,
            metadata={"self_source_count": len(self_sources)},
        )

    # Map hashes to doc positions
    hash_to_doc: Dict[int, List[int]] = {}
    for h, pos in doc_fps:
        hash_to_doc.setdefault(h, []).append(pos)

    # Lookup hashes in SQLite inverted index
    matches = index.lookup_hashes(list(hash_to_doc.keys()))
    if not matches:
        return CheckResult(
            check_name="self_plagiarism",
            findings=[],
            summary=f"No overlap detected with {len(self_sources)} registered earlier work(s).",
            passed=True,
            metadata={"self_source_count": len(self_sources)},
        )

    # Filter matches to origin="self"
    self_hits: Dict[str, List[Tuple[int, int]]] = {}
    for h, hit_list in matches.items():
        doc_positions = hash_to_doc.get(h, [])
        for doc_pos in doc_positions:
            for hit in hit_list:
                s_id = hit["source_id"]
                if s_id in self_sources:
                    self_hits.setdefault(s_id, []).append((doc_pos, hit["position"]))

    findings: List[Finding] = []
    total_matched_doc_spans: List[Tuple[int, int]] = []
    per_source_overlap: Dict[str, float] = {}

    for s_id, hits in self_hits.items():
        if not hits:
            continue

        src_cand = self_sources[s_id]
        src_title = src_cand.title or s_id
        src_text = src_cand.text or ""

        # Cluster hits
        hits.sort(key=lambda x: (x[0], x[1]))
        clusters: List[List[Tuple[int, int]]] = []
        current_cluster: List[Tuple[int, int]] = [hits[0]]

        for next_hit in hits[1:]:
            prev_doc_pos, _ = current_cluster[-1]
            curr_doc_pos, _ = next_hit
            gap = curr_doc_pos - prev_doc_pos
            if 0 <= gap <= max_gap_tokens:
                current_cluster.append(next_hit)
            else:
                clusters.append(current_cluster)
                current_cluster = [next_hit]
        if current_cluster:
            clusters.append(current_cluster)

        # Build spans
        source_intervals: List[Tuple[int, int]] = []
        for cluster in clusters:
            doc_pos_start = cluster[0][0]
            doc_pos_end = cluster[-1][0] + DEFAULT_K - 1
            span_words = doc_pos_end - doc_pos_start + 1

            if span_words < min_words:
                continue

            doc_pos_end = min(doc_pos_end, len(canonical.offset_map) - 1)
            char_start = canonical.offset_map[doc_pos_start][0]
            char_end = canonical.offset_map[doc_pos_end][1]

            matched_snippet = doc_text[char_start:char_end]
            source_intervals.append((char_start, char_end))
            total_matched_doc_spans.append((char_start, char_end))

            findings.append(
                Finding(
                    check="self_plagiarism",
                    severity="warning",
                    message=f"Overlap with previously submitted work '{src_title}': '{matched_snippet[:60]}...'.",
                    start=char_start,
                    end=char_end,
                    details={
                        "source_id": s_id,
                        "source_title": src_title,
                        "word_count": span_words,
                        "matched_text": matched_snippet,
                    },
                )
            )

        # Compute overlap percentage for this source
        merged = _merge_spans(source_intervals)
        matched_chars = sum(e - s for s, e in merged)
        doc_len = max(1, len(doc_text))
        overlap_pct = (matched_chars / doc_len) * 100.0
        per_source_overlap[src_title] = round(overlap_pct, 2)

    # Adjust severity according to total overlap
    all_merged = _merge_spans(total_matched_doc_spans)
    total_matched_chars = sum(e - s for s, e in all_merged)
    total_overlap_pct = (total_matched_chars / max(1, len(doc_text))) * 100.0

    for f in findings:
        f.details["total_self_overlap_pct"] = round(total_overlap_pct, 2)
        if total_overlap_pct < threshold_pct:
            f.severity = "info"

    findings.sort(key=lambda f: f.start)
    passed = not any(f.severity in ("warning", "high") for f in findings)

    if findings:
        sources_str = ", ".join(f"{k} ({v}%)" for k, v in per_source_overlap.items())
        summary = f"Detected {len(findings)} match span(s) across prior work ({total_overlap_pct:.1f}% total overlap): {sources_str}."
    else:
        summary = f"No self-plagiarism overlap detected with {len(self_sources)} registered earlier work(s)."

    return CheckResult(
        check_name="self_plagiarism",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={
            "self_source_count": len(self_sources),
            "total_overlap_pct": round(total_overlap_pct, 2),
            "per_source_overlap": per_source_overlap,
            "span_count": len(findings),
        },
    )
