"""Document text highlighting with HTML escaping, overlap resolution, and accessibility styling."""

import html
from typing import Dict, List, Optional, Set, Tuple

from veritext.report.models import HighlightSpan

# Colour-blind friendly palette: distinct background, high-contrast text,
# and distinct border-bottom line styles (solid, dashed, dotted, wavy).
CATEGORY_STYLES: Dict[str, Dict[str, str]] = {
    "exact match": {
        "bg": "#fee2e2",
        "color": "#991b1b",
        "border": "2px solid #dc2626",
        "label": "Exact Match",
        "desc": "Verbatim copy from reference source (solid red underline)",
    },
    "paraphrase": {
        "bg": "#ffedd5",
        "color": "#9a3412",
        "border": "2px dashed #ea580c",
        "label": "Paraphrase",
        "desc": "Paraphrased or near-copy text (dashed orange underline)",
    },
    "likely AI": {
        "bg": "#ede9fe",
        "color": "#5b21b6",
        "border": "2px solid #7c3aed",
        "label": "Likely AI",
        "desc": "Low burstiness / high likelihood AI sentence (solid purple underline)",
    },
    "hidden content": {
        "bg": "#fce7f3",
        "color": "#9d174d",
        "border": "2px dotted #db2777",
        "label": "Hidden Content",
        "desc": "Homoglyphs, invisible Unicode, or font tricks (dotted pink underline)",
    },
    "citation issue": {
        "bg": "#fef9c3",
        "color": "#854d0e",
        "border": "2px wavy #ca8a04",
        "label": "Citation Issue",
        "desc": "Uncited claim, missing reference, or long quote (wavy yellow underline)",
    },
    "self-overlap": {
        "bg": "#e0f2fe",
        "color": "#075985",
        "border": "2px solid #0284c7",
        "label": "Self-Overlap",
        "desc": "Recycled text from user's prior work (solid blue underline)",
    },
    "text spinner": {
        "bg": "#ccfbf1",
        "color": "#115e59",
        "border": "2px dashed #0d9488",
        "label": "Text Spinner",
        "desc": "Suspected automated paraphrasing or tortured phrase (dashed teal underline)",
    },
}

DEFAULT_STYLE = {
    "bg": "#f1f5f9",
    "color": "#334155",
    "border": "2px solid #64748b",
    "label": "Other Finding",
    "desc": "General finding",
}


def render_legend_html() -> str:
    """Generates a self-contained, accessible HTML legend for all highlight categories."""
    items_html = []
    for cat_name, style in CATEGORY_STYLES.items():
        sample_badge = (
            f'<span style="background-color: {style["bg"]}; color: {style["color"]}; '
            f'border-bottom: {style["border"]}; font-weight: 600; padding: 2px 6px; '
            f'border-radius: 3px; font-size: 0.82rem; margin-right: 8px;">'
            f'{html.escape(style["label"])}'
            f"</span>"
        )
        desc_html = f'<span style="font-size: 0.85rem; color: #475569;">{html.escape(style["desc"])}</span>'
        items_html.append(f'<div style="margin: 4px 0;">{sample_badge}{desc_html}</div>')

    return (
        '<div style="background-color: #f8fafc; border: 1px solid #e2e8f0; '
        'border-radius: 6px; padding: 12px 16px; margin: 12px 0; font-family: sans-serif;">'
        '<div style="font-weight: 700; margin-bottom: 8px; font-size: 0.9rem; color: #1e293b;">'
        "Highlight Legend & Accessibility Guide"
        "</div>"
        f'{"".join(items_html)}'
        "</div>"
    )


def render_highlighted_html(
    text: str,
    spans: List[HighlightSpan],
    ignored_ids: Optional[Set[str]] = None,
) -> str:
    """Renders text with accessible HTML highlights, resolving overlaps and escaping markup.

    Args:
        text: Original unescaped document text.
        spans: List of HighlightSpan objects with start and end offsets.
        ignored_ids: Set of span IDs that should not be highlighted.

    Returns:
        Safe HTML fragment representing highlighted text.
    """
    if not text:
        return ""

    ignored = ignored_ids or set()
    active_spans = [
        s for s in spans
        if s.id not in ignored
        and not s.ignored
        and 0 <= s.start < s.end <= len(text)
    ]

    if not active_spans:
        # No highlights: preserve paragraphs while escaping content
        escaped = html.escape(text).replace("\n", "<br/>")
        return f'<div class="veritext-document-view" style="font-family: Georgia, serif; line-height: 1.7; font-size: 1.02rem;">{escaped}</div>'

    # Collect all unique segment boundary positions
    boundaries = {0, len(text)}
    for s in active_spans:
        boundaries.add(s.start)
        boundaries.add(s.end)

    sorted_bounds = sorted(boundaries)
    rendered_parts: List[str] = []

    for i in range(len(sorted_bounds) - 1):
        b_start = sorted_bounds[i]
        b_end = sorted_bounds[i + 1]
        raw_slice = text[b_start:b_end]

        # Find all spans covering this interval
        covering = [s for s in active_spans if s.start <= b_start and s.end >= b_end]

        if not covering:
            # Unhighlighted slice
            rendered_parts.append(html.escape(raw_slice).replace("\n", "<br/>"))
        else:
            # Determine primary style from highest severity / first category
            # High severity takes precedence over warning, then info
            covering.sort(
                key=lambda s: (
                    {"high": 0, "warning": 1, "info": 2}.get(s.severity, 3),
                    s.start,
                )
            )
            primary_span = covering[0]
            cat = primary_span.category
            style = CATEGORY_STYLES.get(cat, DEFAULT_STYLE)

            # Build tooltip title combining all covering spans (properly escaped for attribute safety)
            tooltip_lines = []
            for sp in covering:
                src_note = f" [{sp.source}]" if sp.source else ""
                clean_msg = html.escape(f"{sp.category.upper()}: {sp.message}{src_note}", quote=True)
                tooltip_lines.append(clean_msg)
            tooltip_str = " &#10; ".join(tooltip_lines)

            # Combine categories for CSS class
            classes = " ".join(f"hl-{html.escape(s.category.replace(' ', '-'), quote=True)}" for s in covering)
            escaped_slice = html.escape(raw_slice).replace("\n", "<br/>")

            # Inline styles: distinct background color, text color, and border-bottom indicator
            mark_tag = (
                f'<mark class="{classes}" title="{tooltip_str}" '
                f'style="background-color: {style["bg"]}; color: {style["color"]}; '
                f'border-bottom: {style["border"]}; padding: 1px 2px; border-radius: 2px; '
                f'cursor: help;">'
                f"{escaped_slice}"
                f"</mark>"
            )
            rendered_parts.append(mark_tag)

    full_html = "".join(rendered_parts)
    return (
        '<div class="veritext-document-view" style="font-family: Georgia, serif; '
        'line-height: 1.75; font-size: 1.02rem; color: #1e293b; background: #ffffff; '
        f'padding: 18px 24px; border: 1px solid #cbd5e1; border-radius: 6px;">'
        f"{full_html}"
        f"</div>"
    )
