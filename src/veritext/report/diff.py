"""Side-by-side source diff with word-level highlighting and safe HTML rendering.

Provides bounded excerpts (max 1,500 characters), word-level alignment of shared words,
accessibility-compliant styling (not colour alone), safe HTML escaping, and support for
fingerprint-only matches.
"""

from dataclasses import dataclass
import difflib
import html
import re
from typing import Any, Dict, List, Optional, Tuple

MAX_EXCERPT_CHARS = 1500

DIFF_STYLES: Dict[str, Dict[str, str]] = {
    "exact": {
        "bg": "#fee2e2",
        "color": "#991b1b",
        "border": "2px solid #dc2626",
        "badge": "EXACT",
        "badge_bg": "#dc2626",
        "badge_color": "#ffffff",
    },
    "near-copy": {
        "bg": "#ffedd5",
        "color": "#9a3412",
        "border": "2px dashed #ea580c",
        "badge": "NEAR-COPY",
        "badge_bg": "#ea580c",
        "badge_color": "#ffffff",
    },
    "paraphrase": {
        "bg": "#fef3c7",
        "color": "#92400e",
        "border": "2px dotted #d97706",
        "badge": "PARAPHRASE",
        "badge_bg": "#d97706",
        "badge_color": "#ffffff",
    },
}


def normalize_match_type(match_type: str) -> str:
    """Normalizes match category to one of 'exact', 'near-copy', or 'paraphrase'."""
    m = match_type.lower().strip()
    if "exact" in m:
        return "exact"
    if "near" in m or "close" in m:
        return "near-copy"
    return "paraphrase"


def truncate_excerpt(text: Optional[str], max_chars: int = MAX_EXCERPT_CHARS) -> Tuple[str, bool]:
    """Truncates text cleanly to max_chars, returning the truncated text and a boolean flag."""
    if not text:
        return "", False
    if len(text) <= max_chars:
        return text, False
    # Truncate at word boundary if possible
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated.rstrip() + " ...", True


def tokenize_words_with_ws(text: str) -> List[Tuple[str, bool]]:
    """Splits text into a list of (token, is_word) preserving all characters and whitespace."""
    tokens: List[Tuple[str, bool]] = []
    # Match words vs non-words
    for m in re.finditer(r"(\w+|[^\w\s]+|\s+)", text, re.UNICODE):
        tok = m.group(0)
        is_word = bool(re.match(r"^\w+$", tok, re.UNICODE))
        tokens.append((tok, is_word))
    return tokens


def align_word_excerpts(
    target_text: str,
    source_text: str,
    match_type: str = "exact",
) -> Tuple[str, str]:
    """Computes word-level diff and returns safe HTML strings for target and source.

    Shared words receive inline styles with background, border-bottom, and high contrast.
    All text is strictly escaped using html.escape to prevent XSS.
    """
    style_key = normalize_match_type(match_type)
    st = DIFF_STYLES.get(style_key, DIFF_STYLES["paraphrase"])

    target_tokens = tokenize_words_with_ws(target_text)
    source_tokens = tokenize_words_with_ws(source_text)

    # Extract word-only lists for SequenceMatcher
    target_words = [t[0].lower() for t in target_tokens if t[1]]
    source_words = [t[0].lower() for t in source_tokens if t[1]]

    # Map word index back to token index
    target_word_to_tok: List[int] = [i for i, t in enumerate(target_tokens) if t[1]]
    source_word_to_tok: List[int] = [i for i, t in enumerate(source_tokens) if t[1]]

    matcher = difflib.SequenceMatcher(None, target_words, source_words)
    matching_blocks = matcher.get_matching_blocks()

    target_highlighted: Set[int] = set()
    source_highlighted: Set[int] = set()

    for t_start, s_start, length in matching_blocks:
        if length <= 0:
            continue
        for k in range(length):
            t_tok_idx = target_word_to_tok[t_start + k]
            s_tok_idx = source_word_to_tok[s_start + k]
            target_highlighted.add(t_tok_idx)
            source_highlighted.add(s_tok_idx)

    # Render target HTML
    target_html_parts: List[str] = []
    for i, (tok, is_word) in enumerate(target_tokens):
        safe_tok = html.escape(tok)
        if i in target_highlighted:
            target_html_parts.append(
                f'<span style="background:{st["bg"]}; color:{st["color"]}; '
                f'border-bottom:{st["border"]}; font-weight:600;" '
                f'title="{st["badge"]}">{safe_tok}</span>'
            )
        else:
            target_html_parts.append(safe_tok)

    # Render source HTML
    source_html_parts: List[str] = []
    for i, (tok, is_word) in enumerate(source_tokens):
        safe_tok = html.escape(tok)
        if i in source_highlighted:
            source_html_parts.append(
                f'<span style="background:{st["bg"]}; color:{st["color"]}; '
                f'border-bottom:{st["border"]}; font-weight:600;" '
                f'title="{st["badge"]}">{safe_tok}</span>'
            )
        else:
            source_html_parts.append(safe_tok)

    return "".join(target_html_parts), "".join(source_html_parts)


@dataclass
class SourceDiffItem:
    """Represents a side-by-side comparison between target submission and reference source."""

    source_id: str
    source_title: str
    source_url: str
    match_type: str
    score: float
    target_excerpt: str
    source_excerpt: Optional[str]
    is_fingerprint_only: bool = False
    target_truncated: bool = False
    source_truncated: bool = False

    def render_html_details(self) -> str:
        """Renders an accessible native <details> section with side-by-side comparison."""
        st_key = normalize_match_type(self.match_type)
        st = DIFF_STYLES.get(st_key, DIFF_STYLES["paraphrase"])

        safe_title = html.escape(self.source_title or self.source_id or "Reference Source")
        safe_url = html.escape(self.source_url or "")
        safe_url_link = f' &mdash; <a href="{safe_url}" target="_blank" rel="noopener noreferrer" style="color:#2563eb; font-size:12px;">{safe_url}</a>' if safe_url else ""

        # Check fingerprint only
        if self.is_fingerprint_only or self.source_excerpt is None:
            safe_target = html.escape(self.target_excerpt)
            source_content = (
                '<div style="padding:16px; background:#f8fafc; border:1px dashed #94a3b8; '
                'border-radius:6px; color:#64748b; font-style:italic; text-align:center;">'
                'source text not stored (fingerprints only)'
                '</div>'
            )
            target_content = (
                f'<div style="padding:12px; background:#ffffff; border:1px solid #e2e8f0; '
                f'border-radius:6px; font-family:ui-monospace, monospace; font-size:13px; '
                f'line-height:1.5; white-space:pre-wrap; word-break:break-word;">{safe_target}</div>'
            )
        else:
            t_html, s_html = align_word_excerpts(self.target_excerpt, self.source_excerpt, self.match_type)
            target_content = (
                f'<div style="padding:12px; background:#ffffff; border:1px solid #e2e8f0; '
                f'border-radius:6px; font-family:ui-monospace, monospace; font-size:13px; '
                f'line-height:1.5; white-space:pre-wrap; word-break:break-word;">{t_html}</div>'
            )
            source_content = (
                f'<div style="padding:12px; background:#ffffff; border:1px solid #e2e8f0; '
                f'border-radius:6px; font-family:ui-monospace, monospace; font-size:13px; '
                f'line-height:1.5; white-space:pre-wrap; word-break:break-word;">{s_html}</div>'
            )

        trunc_notice_target = ' <span style="font-size:11px; color:#94a3b8;">(truncated to 1,500 chars)</span>' if self.target_truncated else ""
        trunc_notice_source = ' <span style="font-size:11px; color:#94a3b8;">(truncated to 1,500 chars)</span>' if self.source_truncated else ""

        return f"""
<details style="margin-bottom:14px; border:1px solid #cbd5e1; border-radius:8px; background:#ffffff; overflow:hidden;">
  <summary style="padding:10px 14px; background:#f1f5f9; cursor:pointer; font-weight:600; display:flex; justify-content:space-between; align-items:center; user-select:none;">
    <span>
      <span style="display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; background:{st['badge_bg']}; color:{st['badge_color']}; margin-right:8px;">{st['badge']}</span>
      {safe_title}{safe_url_link}
    </span>
    <span style="font-size:12px; color:#475569; font-weight:normal;">Overlap: {int(self.score * 100 if self.score <= 1.0 else self.score)}%</span>
  </summary>
  <div style="padding:14px; background:#fafafa;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
      <div>
        <div style="font-size:12px; font-weight:700; color:#334155; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;">Target Submission{trunc_notice_target}</div>
        {target_content}
      </div>
      <div>
        <div style="font-size:12px; font-weight:700; color:#334155; margin-bottom:6px; text-transform:uppercase; letter-spacing:0.5px;">Matched Source Reference{trunc_notice_source}</div>
        {source_content}
      </div>
    </div>
  </div>
</details>
"""


def generate_source_diff(
    target_text: str,
    source_text: Optional[str],
    match_type: str = "exact",
    source_title: str = "Reference Source",
    source_url: str = "",
    source_id: str = "",
    score: float = 1.0,
    is_fingerprint_only: bool = False,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> SourceDiffItem:
    """Constructs a bounded SourceDiffItem with word-level alignment and safe styling."""
    t_bounded, t_trunc = truncate_excerpt(target_text, max_chars=max_chars)
    s_bounded, s_trunc = truncate_excerpt(source_text, max_chars=max_chars) if source_text else ("", False)

    return SourceDiffItem(
        source_id=source_id,
        source_title=source_title,
        source_url=source_url,
        match_type=match_type,
        score=score,
        target_excerpt=t_bounded,
        source_excerpt=s_bounded if (source_text is not None and not is_fingerprint_only) else None,
        is_fingerprint_only=is_fingerprint_only or (source_text is None),
        target_truncated=t_trunc,
        source_truncated=s_trunc,
    )


def extract_report_diffs(
    report_dict_or_obj: Any,
    max_matches: int = 10,
    max_chars: int = MAX_EXCERPT_CHARS,
) -> List[SourceDiffItem]:
    """Extracts top match pairs from FullReport or report dict into SourceDiffItem instances."""
    data = report_dict_or_obj if isinstance(report_dict_or_obj, dict) else report_dict_or_obj.to_dict()
    sim_data = data.get("similarity") or {}
    spans = sim_data.get("spans") or []

    # Sort spans by matched text length or score descending
    sorted_spans = sorted(
        spans,
        key=lambda s: (len(s.get("matched_text", "")), s.get("score", 0.0)),
        reverse=True,
    )

    diffs: List[SourceDiffItem] = []
    seen_sources: Set[str] = set()

    for s in sorted_spans:
        source_id = s.get("source_id", "")
        # Deduplicate by source or allow multiple if distinct spans
        key = f"{source_id}:{s.get('doc_start')}"
        if key in seen_sources:
            continue
        seen_sources.add(key)

        matched_text = s.get("matched_text", "")
        if not matched_text.strip():
            continue

        source_text = s.get("source_text")
        meta = s.get("metadata", {}) or {}
        is_fp = meta.get("fingerprints_only", False) or (source_text is None and "fingerprint" in source_id.lower())

        item = generate_source_diff(
            target_text=matched_text,
            source_text=source_text,
            match_type=s.get("type", "exact"),
            source_title=s.get("source_title", source_id),
            source_url=s.get("source_url", ""),
            source_id=source_id,
            score=s.get("score", 1.0),
            is_fingerprint_only=is_fp,
            max_chars=max_chars,
        )
        diffs.append(item)
        if len(diffs) >= max_matches:
            break

    return diffs
