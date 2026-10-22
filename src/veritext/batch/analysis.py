"""Batch pairwise similarity and collusion matching across multiple submissions.

Computes pairwise winnowing similarity matrix, generates a self-contained HTML heatmap
(no external resources), and identifies collusion clusters above a threshold (default 30%)
with shared passage extraction.
Engineered to be flat-memory safe for 100 submissions on 8 GB RAM.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from veritext.fingerprint.kgrams import generate_kgrams
from veritext.fingerprint.minhash import (
    compute_minhash,
    minhash_jaccard,
    screen_batch_candidates,
)
from veritext.fingerprint.winnow import winnow
from veritext.ingest.loaders import LOADER_MAP
from veritext.ingest.pipeline import load_document
from veritext.normalize.normalize import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class SubmissionFingerprint:
    """Lightweight in-memory fingerprint representation of a submission."""
    id: str
    name: str
    path: Path
    char_count: int
    word_count: int
    fingerprint_set: Set[int]
    hash_positions: Dict[int, List[int]]  # hash -> list of token positions
    text: str = ""  # kept or read on demand for passage extraction


@dataclass
class SharedPassage:
    """Represents a shared passage detected between two colluding submissions."""
    doc_a_name: str
    doc_b_name: str
    text_snippet: str
    approx_words: int


@dataclass
class CollusionPair:
    """Represents a pair of submissions exceeding the similarity threshold."""
    doc_a_id: str
    doc_a_name: str
    doc_b_id: str
    doc_b_name: str
    similarity_percentage: float
    shared_hashes_count: int
    shared_passages: List[SharedPassage] = field(default_factory=list)


@dataclass
class BatchAnalysisResult:
    """Aggregated output of batch pairwise analysis."""
    submissions: List[str]  # names or anonymised IDs
    matrix: List[List[float]]  # NxN similarity percentages (0.0 to 100.0)
    pairs_above_threshold: List[CollusionPair]
    clusters: List[List[str]]  # connected components of colluding submissions
    threshold: float
    csv_content: str = ""
    html_report: str = ""


def _extract_shared_passages(
    sub_a: SubmissionFingerprint,
    sub_b: SubmissionFingerprint,
    k: int = 5,
    max_passages: int = 5,
) -> List[SharedPassage]:
    """Extracts sample shared text passages between two submissions."""
    shared_hashes = sub_a.fingerprint_set.intersection(sub_b.fingerprint_set)
    if not shared_hashes or not sub_a.text:
        return []

    # Map shared hashes to token positions in Doc A
    matched_token_positions: List[int] = []
    for h in shared_hashes:
        matched_token_positions.extend(sub_a.hash_positions.get(h, []))

    if not matched_token_positions:
        return []

    matched_token_positions.sort()

    # Cluster adjacent token positions into passage ranges
    token_ranges: List[Tuple[int, int]] = []
    curr_start = matched_token_positions[0]
    curr_end = curr_start + k

    for pos in matched_token_positions[1:]:
        if pos <= curr_end + 3:
            curr_end = max(curr_end, pos + k)
        else:
            token_ranges.append((curr_start, curr_end))
            curr_start = pos
            curr_end = pos + k
    token_ranges.append((curr_start, curr_end))

    # Sort ranges by length descending
    token_ranges.sort(key=lambda r: (r[1] - r[0]), reverse=True)

    words = sub_a.text.split()
    passages: List[SharedPassage] = []

    for start_idx, end_idx in token_ranges[:max_passages]:
        snippet_words = words[max(0, start_idx) : min(len(words), end_idx)]
        if len(snippet_words) >= 6:
            snippet_str = " ".join(snippet_words)
            passages.append(
                SharedPassage(
                    doc_a_name=sub_a.name,
                    doc_b_name=sub_b.name,
                    text_snippet=snippet_str[:250] + ("..." if len(snippet_str) > 250 else ""),
                    approx_words=len(snippet_words),
                )
            )

    return passages


def generate_heatmap_html(
    submissions: List[str],
    matrix: List[List[float]],
    threshold: float,
    pairs: List[CollusionPair],
    clusters: List[List[str]],
) -> str:
    """Generates a self-contained HTML heatmap with zero external resources."""
    n = len(submissions)

    def get_color(val: float) -> str:
        if val < 5.0:
            return "#f8fafc"
        elif val < 15.0:
            return "#e2e8f0"
        elif val < threshold * 100.0:
            return "#fef9c3"
        elif val < 60.0:
            return "#fed7aa"
        elif val < 85.0:
            return "#fca5a5"
        else:
            return "#f87171"

    def get_text_color(val: float) -> str:
        return "#991b1b" if val >= threshold * 100.0 else "#1e293b"

    table_rows = []
    th_cells = ["<th></th>"]
    for s in submissions:
        short_name = html.escape(s if len(s) <= 15 else s[:12] + "...")
        th_cells.append(f"<th title='{html.escape(s)}'>{short_name}</th>")
    table_rows.append(f"<tr>{''.join(th_cells)}</tr>")

    for i in range(n):
        row_cells = []
        short_name = html.escape(submissions[i] if len(submissions[i]) <= 15 else submissions[i][:12] + "...")
        row_cells.append(f"<th class='row-hdr' title='{html.escape(submissions[i])}'>{short_name}</th>")
        for j in range(n):
            val = matrix[i][j]
            bg = get_color(val)
            fg = get_text_color(val)
            bold = "font-weight: 700;" if (val >= threshold * 100.0 and i != j) else ""
            title = f"{html.escape(submissions[i])} vs {html.escape(submissions[j])}: {val:.1f}%"
            row_cells.append(
                f"<td style='background-color: {bg}; color: {fg}; {bold}' title='{title}'>{val:.1f}%</td>"
            )
        table_rows.append(f"<tr>{''.join(row_cells)}</tr>")

    table_html = "\n".join(table_rows)

    pairs_html_list = []
    if not pairs:
        pairs_html_list.append("<p class='empty-note'>No submission pairs exceeded the similarity threshold.</p>")
    else:
        for idx, p in enumerate(pairs, 1):
            p_html = [
                f"<div class='pair-card'>",
                f"<h4>{idx}. {html.escape(p.doc_a_name)} &amp; {html.escape(p.doc_b_name)} "
                f"<span class='badge'>{p.similarity_percentage:.1f}% Similarity</span></h4>",
                f"<p class='meta'>Shared winnowing hashes: {p.shared_hashes_count}</p>",
            ]
            if p.shared_passages:
                p_html.append("<div class='passages-box'><strong>Sample Shared Passages:</strong>")
                for sp in p.shared_passages[:3]:
                    p_html.append(f"<blockquote>&ldquo;{html.escape(sp.text_snippet)}&rdquo;</blockquote>")
                p_html.append("</div>")
            p_html.append("</div>")
            pairs_html_list.append("\n".join(p_html))

    pairs_section = "\n".join(pairs_html_list)

    clusters_html_list = []
    if clusters:
        for c_idx, cl in enumerate(clusters, 1):
            members = ", ".join(f"<code>{html.escape(m)}</code>" for m in cl)
            clusters_html_list.append(f"<li><strong>Cluster #{c_idx} ({len(cl)} submissions):</strong> {members}</li>")
        clusters_section = f"<ul>{''.join(clusters_html_list)}</ul>"
    else:
        clusters_section = "<p class='empty-note'>No multi-submission clusters detected.</p>"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Veritext Batch Collusion Analysis</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 2rem;
    background-color: #f8fafc;
    color: #0f172a;
    line-height: 1.5;
  }}
  h1, h2, h3 {{ color: #0f172a; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; padding: 2rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .meta-bar {{ margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid #e2e8f0; font-size: 0.95rem; color: #475569; }}
  .heatmap-wrap {{ overflow-x: auto; margin: 1.5rem 0; padding: 1rem 0; }}
  table.heatmap {{ border-collapse: collapse; font-size: 0.82rem; margin: 0 auto; }}
  table.heatmap th, table.heatmap td {{
    border: 1px solid #cbd5e1;
    padding: 0.45rem 0.6rem;
    text-align: center;
    min-width: 54px;
  }}
  table.heatmap th {{ background: #f1f5f9; font-weight: 600; color: #334155; }}
  table.heatmap th.row-hdr {{ text-align: right; background: #f1f5f9; }}
  .pair-card {{
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 1rem;
  }}
  .badge {{
    background: #dc2626;
    color: #ffffff;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.85rem;
    font-weight: 600;
  }}
  blockquote {{
    background: #ffffff;
    border-left: 3px solid #ea580c;
    margin: 0.5rem 0;
    padding: 0.5rem 0.8rem;
    font-style: italic;
    color: #431407;
    font-family: Georgia, serif;
    font-size: 0.9rem;
  }}
  .empty-note {{ color: #64748b; font-style: italic; }}
</style>
</head>
<body>
<div class="container">
  <h1>Veritext Batch Collusion &amp; Cross-Submission Analysis</h1>
  <div class="meta-bar">
    Total Submissions: <strong>{n}</strong> &bull;
    Collusion Threshold: <strong>{threshold * 100.0:.0f}%</strong> &bull;
    Flagged Pairs: <strong>{len(pairs)}</strong> &bull;
    Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
  </div>

  <h2>Pairwise Similarity Heatmap</h2>
  <div class="heatmap-wrap">
    <table class="heatmap">
      {table_html}
    </table>
  </div>

  <h2>Collusion Clusters (&ge; {threshold * 100.0:.0f}%)</h2>
  {clusters_section}

  <h2>Flagged Submission Pairs &amp; Shared Passages</h2>
  {pairs_section}
</div>
</body>
</html>"""
    return html_doc


def find_clusters(pairs: List[CollusionPair], all_names: List[str]) -> List[List[str]]:
    """Finds connected components of submissions that share edges exceeding the threshold."""
    adj: Dict[str, Set[str]] = {name: set() for name in all_names}
    for p in pairs:
        adj[p.doc_a_name].add(p.doc_b_name)
        adj[p.doc_b_name].add(p.doc_a_name)

    visited: Set[str] = set()
    clusters: List[List[str]] = []

    for name in all_names:
        if name not in visited and adj[name]:
            component: List[str] = []
            queue = [name]
            visited.add(name)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(component) >= 2:
                clusters.append(sorted(component))

    clusters.sort(key=len, reverse=True)
    return clusters


def run_batch_analysis(
    folder: Union[str, Path],
    threshold: float = 0.30,
    anonymize: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> BatchAnalysisResult:
    """Executes memory-safe pairwise winnowing similarity analysis across a directory of submissions."""
    fpath = Path(folder).resolve()
    if not fpath.exists() or not fpath.is_dir():
        raise FileNotFoundError(f"Submissions folder not found: {fpath}")

    files = sorted([f for f in fpath.iterdir() if f.is_file() and f.suffix.lower() in LOADER_MAP])
    if len(files) < 2:
        raise ValueError(f"Batch analysis requires at least 2 supported documents; found {len(files)} in {fpath}")

    def report(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    report(f"Loading and fingerprinting {len(files)} documents...", 0.10)

    fingerprints: List[SubmissionFingerprint] = []
    for idx, f in enumerate(files):
        doc = load_document(f)
        clean_text = doc.clean_text or doc.raw_text or ""
        canonical = normalize_text(clean_text)
        fps = winnow(canonical.tokens, k=5, w=4)

        fp_set = {h for h, _ in fps}
        hash_positions: Dict[int, List[int]] = {}
        for h, pos in fps:
            hash_positions.setdefault(h, []).append(pos)

        sub_id = f"sub_{idx + 1:03d}"
        disp_name = sub_id if anonymize else f.name

        fingerprints.append(
            SubmissionFingerprint(
                id=sub_id,
                name=disp_name,
                path=f,
                char_count=len(clean_text),
                word_count=len(clean_text.split()),
                fingerprint_set=fp_set,
                hash_positions=hash_positions,
                text=clean_text,
            )
        )

    n = len(fingerprints)
    names = [sub.name for sub in fingerprints]
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    flagged_pairs: List[CollusionPair] = []

    total_pairs = (n * (n - 1)) // 2
    pairs_done = 0

    report(f"Screening candidate pairs with MinHash/LSH across {n} submissions...", 0.35)

    # Compute MinHash signatures and screen candidate pairs via LSH
    doc_fps = {fingerprints[i].id: fingerprints[i].fingerprint_set for i in range(n)}
    lsh_candidate_pairs, sig_map = screen_batch_candidates(doc_fps)

    # Convert candidate pairs to index pairs
    id_to_idx = {fingerprints[i].id: i for i in range(n)}
    candidate_index_pairs: Set[Tuple[int, int]] = set()

    for ida, idb in lsh_candidate_pairs:
        if ida in id_to_idx and idb in id_to_idx:
            ia, ib = id_to_idx[ida], id_to_idx[idb]
            candidate_index_pairs.add((min(ia, ib), max(ia, ib)))

    # For large cohorts (n > 20), also check MinHash Jaccard margin to ensure zero misses
    # For small cohorts (n <= 20), check all pairs
    check_all = n <= 20

    report(f"Computing winnowing verification across shortlisted pairs...", 0.50)

    for i in range(n):
        matrix[i][i] = 100.0
        sub_a = fingerprints[i]
        set_a = sub_a.fingerprint_set
        sig_a = sig_map.get(sub_a.id)

        for j in range(i + 1, n):
            sub_b = fingerprints[j]
            sig_b = sig_map.get(sub_b.id)

            is_candidate = (
                check_all
                or ((i, j) in candidate_index_pairs)
                or (sig_a and sig_b and minhash_jaccard(sig_a, sig_b) >= threshold * 0.6)
            )

            if not is_candidate:
                matrix[i][j] = 0.0
                matrix[j][i] = 0.0
                pairs_done += 1
                continue

            set_b = sub_b.fingerprint_set
            shared = set_a.intersection(set_b)
            total = len(set_a) + len(set_b)

            if total == 0:
                sim_pct = 0.0
            else:
                sim_frac = (2.0 * len(shared)) / total
                sim_pct = round(min(100.0, sim_frac * 100.0), 1)

            matrix[i][j] = sim_pct
            matrix[j][i] = sim_pct

            if sim_pct >= threshold * 100.0:
                shared_passages = _extract_shared_passages(sub_a, sub_b)
                flagged_pairs.append(
                    CollusionPair(
                        doc_a_id=sub_a.id,
                        doc_a_name=sub_a.name,
                        doc_b_id=sub_b.id,
                        doc_b_name=sub_b.name,
                        similarity_percentage=sim_pct,
                        shared_hashes_count=len(shared),
                        shared_passages=shared_passages,
                    )
                )

            pairs_done += 1
            if pairs_done % 25 == 0 or pairs_done == total_pairs:
                report(f"Evaluated {pairs_done}/{total_pairs} pairs...", 0.50 + 0.35 * (pairs_done / total_pairs))

    flagged_pairs.sort(key=lambda p: p.similarity_percentage, reverse=True)
    clusters = find_clusters(flagged_pairs, names)

    report("Compiling CSV matrix and HTML heatmap report...", 0.90)
    csv_lines = ["," + ",".join(names)]
    for i in range(n):
        row_str = f"{names[i]}," + ",".join(f"{matrix[i][j]:.1f}" for j in range(n))
        csv_lines.append(row_str)
    csv_content = "\n".join(csv_lines)

    html_content = generate_heatmap_html(
        submissions=names,
        matrix=matrix,
        threshold=threshold,
        pairs=flagged_pairs,
        clusters=clusters,
    )

    report("Batch analysis complete!", 1.0)

    return BatchAnalysisResult(
        submissions=names,
        matrix=matrix,
        pairs_above_threshold=flagged_pairs,
        clusters=clusters,
        threshold=threshold,
        csv_content=csv_content,
        html_report=html_content,
    )
