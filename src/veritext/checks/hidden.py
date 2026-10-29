"""Hidden content and text obfuscation checker.

Detects:
1. Text-level evasion anomalies:
   - Homoglyphs (Cyrillic and Greek lookalikes of Latin letters)
   - Mixed-script words (e.g. Latin words containing Cyrillic characters)
   - Zero-width and bidirectional control characters
   - Benign typographic characters (soft hyphens, non-breaking spaces) classified as info
2. DOCX-level hidden content:
   - Runs formatted with white or near-white font colour on white, light, or unset backgrounds
     (white text on dark/coloured table headers and banners is resolved and not flagged)
   - Runs formatted with tiny font size (< 2 pt) containing visible text exceeding character threshold
   - Runs formatted with the vanish (hidden text) property (flagged as warning and grouped)
3. PDF-level hidden content:
   - Characters formatted with white or near-white fill colour
   - Characters formatted with tiny font size (< 2 pt)
"""

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.ingest.models import Document
from veritext.normalize.models import Anomaly, CanonicalText
from veritext.normalize.normalize import HOMOGLYPH_MAP, normalize_text

# Cyrillic and Greek unicode ranges
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
GREEK_PATTERN = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")

# Genuine text evasion zero-width and bidi control characters
GENUINE_ZERO_WIDTH_CHARS: Set[str] = {
    "\u200b",  # Zero-width space
    "\u200c",  # Zero-width non-joiner
    "\u200d",  # Zero-width joiner
    "\u200e",  # Left-to-right mark
    "\u200f",  # Right-to-left mark
    "\u2060",  # Word joiner
    "\ufeff",  # Zero-width no-break space / BOM
}

# Benign typographic characters that should never produce high severity findings
BENIGN_TYPOGRAPHIC_CHARS: Set[str] = {
    "\u00ad",  # Soft hyphen
    "\u00a0",  # Non-breaking space
    "\u201c",  # Left double curly quote
    "\u201d",  # Right double curly quote
    "\u2018",  # Left single curly quote
    "\u2019",  # Right single curly quote
    "\u2013",  # En-dash
    "\u2014",  # Em-dash
    "\u2022",  # Bullet
}


def _find_word_bounds(text: str, pos: int) -> Tuple[int, int]:
    """Finds the character bounds of the word containing the character at pos."""
    n = len(text)
    if pos < 0 or pos >= n:
        return max(0, pos), min(n, pos + 1)

    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1

    end = pos + 1
    while end < n and not text[end].isspace():
        end += 1

    # Strip surrounding punctuation from bounds if pos remains within
    while start < pos and text[start] in '.,;:!?()[]{}"\'`«»':
        start += 1
    while end > pos + 1 and text[end - 1] in '.,;:!?()[]{}"\'`«»':
        end -= 1

    return start, end


def _is_mixed_script_word(word: str) -> bool:
    """Checks whether a word mixes Latin characters with Cyrillic or Greek lookalikes."""
    has_latin = bool(LATIN_PATTERN.search(word))
    has_cyrillic = bool(CYRILLIC_PATTERN.search(word))
    has_greek = bool(GREEK_PATTERN.search(word))
    has_homoglyph = any(ch in HOMOGLYPH_MAP for ch in word)

    return bool(has_latin and (has_cyrillic or has_greek or has_homoglyph))


def _is_dark_or_colored_bg(fill: Optional[str]) -> bool:
    """Checks whether an XML shading fill represents a dark or colored background.

    Returns True if the background has sufficient contrast for white text to be visible.
    Returns False if unset, auto, transparent, white, or light.
    """
    if not fill:
        return False
    clean = fill.strip().lstrip("#").upper()
    if clean in ("AUTO", "NONE", "TRANSPARENT", ""):
        return False
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return False
    try:
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
    except ValueError:
        return False

    # Perceived relative brightness (sRGB standard coefficients)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    # Any effective background darker than light off-white (brightness < 185) makes white text visible
    return brightness < 185


def _get_element_shd_fill(element: Any, qn_func: Any) -> Optional[str]:
    """Retrieves w:shd w:fill attribute value from an oxml element if present."""
    if element is None:
        return None
    shd = element.find(qn_func("w:shd"))
    if shd is not None:
        val = shd.get(qn_func("w:fill"))
        if val:
            return str(val).strip()
    return None


def _resolve_effective_background(
    run: Any,
    para: Any,
    cell: Optional[Any] = None,
    table: Optional[Any] = None,
    doc: Optional[Any] = None,
    qn_func: Optional[Any] = None,
) -> Optional[str]:
    """Resolves the cascading effective background shading for a DOCX run.

    Priority:
    1. Run shading (w:rPr/w:shd)
    2. Paragraph shading (w:pPr/w:shd)
    3. Table cell shading (w:tcPr/w:shd)
    4. Table row / table shading (w:trPr/w:shd, w:tblPr/w:shd)
    5. Document page background (w:background)
    """
    if qn_func is None:
        return None

    # 1. Run shading
    if run._r is not None and run._r.rPr is not None:
        fill = _get_element_shd_fill(run._r.rPr, qn_func)
        if fill:
            return fill

    # 2. Paragraph shading
    if para is not None and getattr(para, "_p", None) is not None and para._p.pPr is not None:
        fill = _get_element_shd_fill(para._p.pPr, qn_func)
        if fill:
            return fill

    # 3. Table cell shading
    if cell is not None and getattr(cell, "_tc", None) is not None and cell._tc.tcPr is not None:
        fill = _get_element_shd_fill(cell._tc.tcPr, qn_func)
        if fill:
            return fill

    # 4. Table properties shading
    if table is not None and getattr(table, "_tbl", None) is not None and table._tbl.tblPr is not None:
        fill = _get_element_shd_fill(table._tbl.tblPr, qn_func)
        if fill:
            return fill

    # 5. Document page background color
    if doc is not None and hasattr(doc, "_element"):
        bg_els = doc._element.xpath("//w:background/@w:color")
        if bg_els:
            return str(bg_els[0]).strip()

    return None


def _group_findings(findings: List[Finding]) -> List[Finding]:
    """Groups repeated identical findings into a single consolidated Finding.

    For example, identical header cells or repeated vanish markers are grouped into
    ONE finding with a total count and list of character locations.
    """
    if not findings:
        return []

    groups: Dict[Tuple[str, str, str], List[Finding]] = defaultdict(list)
    for f in findings:
        src = f.details.get("source", "text")
        reasons_key = "; ".join(f.details.get("reasons", []))
        if not reasons_key:
            reasons_key = f.details.get("category", f.message.split(":")[0])

        group_key = (f.check, f.severity, f"{src}::{reasons_key}")
        groups[group_key].append(f)

    grouped_findings: List[Finding] = []
    for (chk, sev, _sig), items in groups.items():
        if len(items) == 1:
            grouped_findings.append(items[0])
        else:
            first = items[0]
            count = len(items)
            min_start = min(it.start for it in items)
            max_end = max(it.end for it in items)
            src = first.details.get("source", "text")
            cat = first.details.get("category", "")
            reasons = first.details.get("reasons", [])
            primary_reason = reasons[0] if reasons else (cat or "formatting anomaly")

            first_text = (first.details.get("word") or first.details.get("text") or first.message)[:60].strip()

            if cat == "mixed_script_word":
                msg = (
                    f"Mixed-script words containing Cyrillic/Greek homoglyphs detected across {count} locations: "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )
            elif cat == "homoglyph_word":
                msg = (
                    f"Words containing Cyrillic/Greek homoglyphs detected across {count} locations: "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )
            elif cat == "invisible_characters":
                msg = (
                    f"Zero-width and invisible characters detected across {count} locations: "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )
            elif src == "docx_run":
                msg = (
                    f"DOCX hidden content detected across {count} locations ({primary_reason}): "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )
            elif src == "pdf_char":
                msg = (
                    f"PDF hidden content detected across {count} locations ({primary_reason}): "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )
            else:
                msg = (
                    f"Text obfuscation detected across {count} locations ({primary_reason}): "
                    f"'{first_text}' (and {count - 1} other occurrence(s))."
                )

            grouped_details = dict(first.details)
            grouped_details.update({
                "count": count,
                "grouped": True,
                "locations": [(it.start, it.end) for it in items],
                "occurrences": [
                    {
                        "start": it.start,
                        "end": it.end,
                        "text": it.details.get("text", "")[:60],
                    }
                    for it in items
                ],
            })

            grouped_findings.append(
                Finding(
                    check=chk,
                    severity=sev,
                    message=msg,
                    start=min_start,
                    end=max_end,
                    details=grouped_details,
                )
            )

    grouped_findings.sort(key=lambda f: f.start)
    return grouped_findings


def _check_docx_hidden_runs(file_path: Union[str, Path], raw_text: str) -> List[Finding]:
    """Inspects a DOCX document for white font colour, tiny fonts, or vanished runs."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".docx":
        return []

    try:
        import docx
        from docx.oxml.ns import qn
    except ImportError:
        return []

    try:
        doc = docx.Document(path)
    except Exception:
        return []

    findings: List[Finding] = []
    search_offset = 0

    # Temporary storage for tiny-font runs to check total character threshold
    candidate_tiny_runs: List[Dict[str, Any]] = []

    def inspect_run(
        run: Any,
        para_idx: int,
        para: Any,
        cell: Optional[Any] = None,
        table: Optional[Any] = None,
    ) -> None:
        nonlocal search_offset
        text = run.text
        if not text:
            return

        reasons = []

        # 1. Check vanish (hidden) property - Word uses this for legitimate comments/tags
        is_hidden = False
        if getattr(run.font, "hidden", None) is True:
            is_hidden = True
        elif run._r is not None and run._r.rPr is not None:
            vanish_el = run._r.rPr.find(qn("w:vanish"))
            if vanish_el is not None:
                val = vanish_el.get(qn("w:val"), "true").lower()
                if val not in ("0", "false", "off"):
                    is_hidden = True

        if is_hidden and text.strip():
            reasons.append("vanish (hidden text) property enabled")

        # 2. Check font size under 2 pt
        font_size_pt = None
        if run.font.size is not None:
            font_size_pt = run.font.size.pt
        elif run._r is not None and run._r.rPr is not None:
            sz_el = run._r.rPr.find(qn("w:sz"))
            if sz_el is not None:
                val = sz_el.get(qn("w:val"))
                if val and val.isdigit():
                    font_size_pt = int(val) / 2.0  # sz is in half-points

        is_tiny = False
        if font_size_pt is not None and font_size_pt < 2.0:
            # Check if run contains visible non-whitespace characters (not spaces, tabs, or spacers)
            visible_chars = [
                c for c in text
                if not c.isspace() and c not in "\t\r\n\u00a0\u200b\u200c\u200d\ufeff\u2060"
            ]
            if visible_chars:
                is_tiny = True

        # 3. Check font colour (white or near-white)
        is_white_raw = False
        if run.font.color and run.font.color.rgb:
            rgb = run.font.color.rgb
            if rgb[0] >= 240 and rgb[1] >= 240 and rgb[2] >= 240:
                is_white_raw = True
        elif run._r is not None and run._r.rPr is not None:
            color_el = run._r.rPr.find(qn("w:color"))
            if color_el is not None:
                cval = (color_el.get(qn("w:val")) or "").strip().upper()
                if cval in ("FFFFFF", "FEFEFE", "FDFDFD", "FCFCFC", "FAFAFA", "F0F0F0") or cval.startswith("FFF"):
                    is_white_raw = True

        # Resolve effective background: white text on dark/coloured background is visible and NOT suspicious
        is_white_suspicious = False
        if is_white_raw and text.strip():
            eff_bg = _resolve_effective_background(
                run=run,
                para=para,
                cell=cell,
                table=table,
                doc=doc,
                qn_func=qn,
            )
            if not _is_dark_or_colored_bg(eff_bg):
                # Effective background is white, light, or unset
                is_white_suspicious = True
                reasons.append("white or near-white font colour on light background")

        # Stash tiny-font run for global character threshold evaluation
        if is_tiny and not (is_hidden or is_white_suspicious):
            candidate_tiny_runs.append({
                "run": run,
                "text": text,
                "para_idx": para_idx,
                "font_size_pt": font_size_pt,
            })
            return

        if is_tiny:
            reasons.append(f"font size under 2 pt ({font_size_pt:.1f} pt)")

        if reasons:
            # Find offset in raw document text
            pos = raw_text.find(text, search_offset)
            if pos != -1:
                start_off = pos
                end_off = pos + len(text)
                search_offset = end_off
            else:
                pos = raw_text.find(text)
                start_off = pos if pos != -1 else 0
                end_off = start_off + len(text)

            clean_snippet = text.strip()[:60]
            # Severity: vanish stays warning; white text / tiny text is high
            severity = "high" if (is_white_suspicious or is_tiny) else "warning"

            findings.append(
                Finding(
                    check="hidden_content",
                    severity=severity,
                    message=f"DOCX hidden content detected ('{clean_snippet}'): {'; '.join(reasons)}.",
                    start=start_off,
                    end=end_off,
                    details={
                        "source": "docx_run",
                        "paragraph_index": para_idx,
                        "reasons": reasons,
                        "text": text,
                        "font_size_pt": font_size_pt,
                        "is_white": is_white_suspicious,
                        "is_hidden": is_hidden,
                    },
                )
            )

    # Inspect paragraphs
    for p_idx, para in enumerate(doc.paragraphs):
        for run in para.runs:
            inspect_run(run, p_idx, para=para)

    # Inspect tables
    for t_idx, table in enumerate(doc.tables):
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        inspect_run(run, t_idx, para=para, cell=cell, table=table)

    # Evaluate candidate tiny-font runs: flag only if total characters exceed threshold (>= 20 chars)
    total_tiny_chars = sum(len(item["text"].strip()) for item in candidate_tiny_runs)
    if total_tiny_chars >= 20:
        for item in candidate_tiny_runs:
            t_text = item["text"]
            pos = raw_text.find(t_text)
            start_off = pos if pos != -1 else 0
            end_off = start_off + len(t_text)
            clean_snippet = t_text.strip()[:60]
            f_size = item["font_size_pt"]
            reasons = [f"font size under 2 pt ({f_size:.1f} pt)"]

            findings.append(
                Finding(
                    check="hidden_content",
                    severity="high",
                    message=f"DOCX hidden content detected ('{clean_snippet}'): {'; '.join(reasons)}.",
                    start=start_off,
                    end=end_off,
                    details={
                        "source": "docx_run",
                        "paragraph_index": item["para_idx"],
                        "reasons": reasons,
                        "text": t_text,
                        "font_size_pt": f_size,
                        "is_white": False,
                        "is_hidden": False,
                    },
                )
            )

    return findings


def _is_pdf_color_near_white(color: Any) -> bool:
    """Checks whether a PDF non_stroking_color represents white or near-white."""
    if color is None:
        return False

    # Grayscale
    if isinstance(color, (int, float)):
        return color > 0.95 if isinstance(color, float) else color >= 240
    if isinstance(color, (list, tuple)):
        if len(color) == 1:
            val = color[0]
            return val > 0.95 if isinstance(val, float) else val >= 240
        # RGB
        if len(color) == 3:
            r, g, b = color
            if isinstance(r, float):
                return r >= 0.95 and g >= 0.95 and b >= 0.95
            return r >= 240 and g >= 240 and b >= 240
        # CMYK (0, 0, 0, 0 is white)
        if len(color) == 4:
            c, m, y, k = color
            return c <= 0.05 and m <= 0.05 and y <= 0.05 and k <= 0.05

    return False


def _check_pdf_hidden_characters(file_path: Union[str, Path], raw_text: str) -> List[Finding]:
    """Inspects a PDF document for tiny fonts or white/near-white fill colours."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []

    try:
        import pdfplumber
    except ImportError:
        return []

    try:
        pdf = pdfplumber.open(path)
    except Exception:
        return []

    findings: List[Finding] = []
    suspicious_runs: List[Dict[str, Any]] = []

    try:
        for page_idx, page in enumerate(pdf.pages):
            chars = page.chars or []
            current_run: List[Dict[str, Any]] = []

            for char in chars:
                text_ch = char.get("text", "")
                if not text_ch:
                    continue

                size = char.get("size", 10.0)
                height = char.get("height", 10.0)
                color = char.get("non_stroking_color")

                is_tiny = (size < 2.0 or height < 2.0)
                is_white = _is_pdf_color_near_white(color)

                if (is_tiny or is_white) and not text_ch.isspace():
                    current_run.append({
                        "text": text_ch,
                        "page": page_idx + 1,
                        "is_tiny": is_tiny,
                        "is_white": is_white,
                        "size": size,
                    })
                else:
                    if current_run:
                        suspicious_runs.append({
                            "page": current_run[0]["page"],
                            "text": "".join(c["text"] for c in current_run),
                            "is_tiny": any(c["is_tiny"] for c in current_run),
                            "is_white": any(c["is_white"] for c in current_run),
                            "min_size": min(c["size"] for c in current_run),
                        })
                        current_run = []

            if current_run:
                suspicious_runs.append({
                    "page": current_run[0]["page"],
                    "text": "".join(c["text"] for c in current_run),
                    "is_tiny": any(c["is_tiny"] for c in current_run),
                    "is_white": any(c["is_white"] for c in current_run),
                    "min_size": min(c["size"] for c in current_run),
                })
    finally:
        pdf.close()

    search_pos = 0
    for run in suspicious_runs:
        run_text = run["text"]
        pos = raw_text.find(run_text, search_pos)
        if pos != -1:
            start_off = pos
            end_off = pos + len(run_text)
            search_pos = end_off
        else:
            pos = raw_text.find(run_text)
            start_off = pos if pos != -1 else 0
            end_off = start_off + len(run_text)

        reasons = []
        if run["is_white"]:
            reasons.append("white or near-white fill colour")
        if run["is_tiny"]:
            reasons.append(f"font size under 2 pt ({run['min_size']:.1f} pt)")

        clean_snippet = run_text.strip()[:60]
        findings.append(
            Finding(
                check="hidden_content",
                severity="high",
                message=f"PDF hidden content on page {run['page']} ('{clean_snippet}'): {'; '.join(reasons)}.",
                start=start_off,
                end=end_off,
                details={
                    "source": "pdf_char",
                    "page": run["page"],
                    "reasons": reasons,
                    "text": run_text,
                    "min_size": run["min_size"],
                    "is_white": run["is_white"],
                    "is_tiny": run["is_tiny"],
                },
            )
        )

    return findings


def check_hidden_content(
    document: Union[Document, str, Path],
    canonical: Optional[CanonicalText] = None,
) -> CheckResult:
    """Performs hidden content, adversarial evasion, and formatting obfuscation checks.

    Args:
        document: Document instance, or path / text string.
        canonical: Pre-computed CanonicalText if available.

    Returns:
        CheckResult containing consolidated findings and post-grouping metrics.
    """
    if isinstance(document, (str, Path)):
        from veritext.ingest.pipeline import is_existing_file, load_document
        if is_existing_file(document):
            doc = load_document(Path(document))
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

    raw_text = doc.raw_text or ""
    if canonical is None:
        canonical = normalize_text(raw_text)

    raw_findings: List[Finding] = []

    # 1. Text-level anomalies: homoglyphs, zero-width chars, control chars
    anomalies = canonical.anomalies or []
    homoglyphs = [a for a in anomalies if a.type == "homoglyph"]
    controls = [a for a in anomalies if a.type == "control_character"]

    # Differentiate genuine zero-width evasion from benign typographic characters (soft hyphens, nbsp)
    genuine_zero_width = [
        a for a in anomalies
        if a.type == "zero_width" and a.character in GENUINE_ZERO_WIDTH_CHARS
    ]
    benign_typographic = [
        a for a in anomalies
        if a.type == "zero_width" and a.character in BENIGN_TYPOGRAPHIC_CHARS
    ]

    total_genuine_invisible = len(genuine_zero_width) + len(controls)

    # Group anomalies by affected word bounds in original text
    word_to_anomalies: Dict[Tuple[int, int], List[Anomaly]] = defaultdict(list)
    for anom in anomalies:
        start, end = _find_word_bounds(raw_text, anom.position)
        word_to_anomalies[(start, end)].append(anom)

    mixed_script_count = 0

    for (w_start, w_end), anom_list in word_to_anomalies.items():
        word_text = raw_text[w_start:w_end]
        word_homoglyphs = [a for a in anom_list if a.type == "homoglyph"]
        word_genuine_invis = [
            a for a in anom_list
            if (a.type == "zero_width" and a.character in GENUINE_ZERO_WIDTH_CHARS)
            or a.type == "control_character"
        ]
        word_benign = [
            a for a in anom_list
            if (a.type == "zero_width" and a.character in BENIGN_TYPOGRAPHIC_CHARS)
        ]

        # Case A: Mixed-script word (Cyrillic/Greek in Latin context)
        if _is_mixed_script_word(word_text):
            mixed_script_count += 1
            h_details = ", ".join(
                f"U+{ord(a.character):04X} ('{a.character}' -> '{a.replacement}')"
                for a in word_homoglyphs
            )
            raw_findings.append(
                Finding(
                    check="hidden_content",
                    severity="high",
                    message=f"Mixed-script word '{word_text}' contains Cyrillic/Greek homoglyphs ({h_details}).",
                    start=w_start,
                    end=w_end,
                    details={
                        "category": "mixed_script_word",
                        "word": word_text,
                        "homoglyph_count": len(word_homoglyphs),
                        "anomalies": [a.to_dict() for a in word_homoglyphs],
                    },
                )
            )
        elif word_homoglyphs and len(word_homoglyphs) >= 2:
            # Word with multiple homoglyphs (non-mixed)
            h_details = ", ".join(f"U+{ord(a.character):04X}" for a in word_homoglyphs)
            raw_findings.append(
                Finding(
                    check="hidden_content",
                    severity="warning",
                    message=f"Word '{word_text}' contains Cyrillic/Greek homoglyph substitutions ({h_details}).",
                    start=w_start,
                    end=w_end,
                    details={
                        "category": "homoglyph_word",
                        "word": word_text,
                        "homoglyph_count": len(word_homoglyphs),
                        "anomalies": [a.to_dict() for a in word_homoglyphs],
                    },
                )
            )

        # Case B: Genuine zero-width / bidi evasion characters
        if word_genuine_invis:
            sev = "high" if total_genuine_invisible >= 3 else "info"
            invis_names = ", ".join(set(a.description for a in word_genuine_invis))
            raw_findings.append(
                Finding(
                    check="hidden_content",
                    severity=sev,
                    message=f"Invisible / zero-width characters in '{word_text}' ({invis_names}, {len(word_genuine_invis)} char(s)).",
                    start=w_start,
                    end=w_end,
                    details={
                        "category": "invisible_characters",
                        "word": word_text,
                        "count": len(word_genuine_invis),
                        "anomalies": [a.to_dict() for a in word_genuine_invis],
                    },
                )
            )

        # Case C: Soft hyphens, non-breaking spaces, and standard typographic symbols
        # Classified strictly as 'info', never high or warning
        elif word_benign:
            benign_names = ", ".join(set(a.description for a in word_benign))
            raw_findings.append(
                Finding(
                    check="hidden_content",
                    severity="info",
                    message=f"Standard typographic character in '{word_text}' ({benign_names}).",
                    start=w_start,
                    end=w_end,
                    details={
                        "category": "typographic_characters",
                        "word": word_text,
                        "count": len(word_benign),
                        "anomalies": [a.to_dict() for a in word_benign],
                    },
                )
            )

    # 2. File-specific checks for DOCX and PDF
    if doc.path:
        file_path = Path(doc.path)
        if file_path.suffix.lower() == ".docx":
            docx_findings = _check_docx_hidden_runs(file_path, raw_text)
            raw_findings.extend(docx_findings)
        elif file_path.suffix.lower() == ".pdf":
            pdf_findings = _check_pdf_hidden_characters(file_path, raw_text)
            raw_findings.extend(pdf_findings)

    # 3. Group repeated identical findings into unified findings
    findings = _group_findings(raw_findings)
    findings.sort(key=lambda f: f.start)

    # Severity counts after grouping
    high_count = sum(1 for f in findings if f.severity == "high")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    info_count = sum(1 for f in findings if f.severity == "info")
    passed = not (high_count > 0 or warning_count > 0)

    summary_parts = []
    if mixed_script_count:
        summary_parts.append(f"{mixed_script_count} mixed-script word(s)")
    if total_genuine_invisible:
        summary_parts.append(f"{total_genuine_invisible} invisible/control character(s)")
    if len(benign_typographic):
        summary_parts.append(f"{len(benign_typographic)} typographic character(s)")

    docx_pdf_count = sum(1 for f in findings if f.details.get("source") in ("docx_run", "pdf_char"))
    if docx_pdf_count:
        summary_parts.append(f"{docx_pdf_count} formatting finding(s)")

    if not findings:
        summary = "No hidden content, homoglyphs, or obfuscation detected."
    else:
        summary = (
            f"Detected {len(findings)} finding(s) after grouping "
            f"({high_count} high, {warning_count} warning, {info_count} info): {', '.join(summary_parts)}."
        )

    return CheckResult(
        check_name="hidden_content",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={
            "homoglyph_count": len(homoglyphs),
            "mixed_script_words": mixed_script_count,
            "zero_width_count": len(genuine_zero_width),
            "control_count": len(controls),
            "benign_typographic_count": len(benign_typographic),
            "total_findings": len(findings),
            "raw_findings_count": len(raw_findings),
            "high_count": high_count,
            "warning_count": warning_count,
            "info_count": info_count,
        },
    )
