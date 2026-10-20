"""Hidden content and text obfuscation checker.

Detects:
1. Text-level evasion anomalies:
   - Homoglyphs (Cyrillic and Greek lookalikes of Latin letters)
   - Mixed-script words (e.g. Latin words containing Cyrillic characters)
   - Zero-width, soft hyphens, and bidirectional control characters
2. DOCX-level hidden content:
   - Runs formatted with white or near-white font colour
   - Runs formatted with tiny font size (< 2 pt)
   - Runs formatted with the vanish (hidden text) property
3. PDF-level hidden content:
   - Characters formatted with white or near-white fill colour
   - Characters formatted with tiny font size (< 2 pt)
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.ingest.models import Document
from veritext.normalize.models import Anomaly, CanonicalText
from veritext.normalize.normalize import HOMOGLYPH_MAP, normalize_text

# Cyrillic and Greek unicode ranges
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF\u0500-\u052F]")
GREEK_PATTERN = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")


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

    return (has_latin and (has_cyrillic or has_greek or has_homoglyph))


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

    def inspect_run(run: Any, para_idx: int) -> None:
        nonlocal search_offset
        text = run.text
        if not text or not text.strip():
            return

        reasons = []

        # 1. Check vanish (hidden) property
        is_hidden = False
        if getattr(run.font, "hidden", None) is True:
            is_hidden = True
        elif run._r is not None and run._r.rPr is not None:
            vanish_el = run._r.rPr.find(qn("w:vanish"))
            if vanish_el is not None:
                val = vanish_el.get(qn("w:val"), "true").lower()
                if val not in ("0", "false", "off"):
                    is_hidden = True

        if is_hidden:
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

        if font_size_pt is not None and font_size_pt < 2.0:
            reasons.append(f"font size under 2 pt ({font_size_pt:.1f} pt)")

        # 3. Check font colour (white or near-white)
        is_white = False
        if run.font.color and run.font.color.rgb:
            rgb = run.font.color.rgb
            if rgb[0] >= 240 and rgb[1] >= 240 and rgb[2] >= 240:
                is_white = True
        elif run._r is not None and run._r.rPr is not None:
            color_el = run._r.rPr.find(qn("w:color"))
            if color_el is not None:
                cval = (color_el.get(qn("w:val")) or "").strip().upper()
                if cval in ("FFFFFF", "FEFEFE", "FDFDFD", "FCFCFC", "FAFAFA", "F0F0F0") or cval.startswith("FFF"):
                    is_white = True

        if is_white:
            reasons.append("white or near-white font colour")

        if reasons:
            # Find offset in raw document text
            pos = raw_text.find(text, search_offset)
            if pos != -1:
                start_off = pos
                end_off = pos + len(text)
                search_offset = end_off
            else:
                pos = raw_text.find(text)
                if pos != -1:
                    start_off = pos
                    end_off = pos + len(text)
                else:
                    start_off = 0
                    end_off = len(text)

            clean_snippet = text.strip()[:60]
            findings.append(
                Finding(
                    check="hidden_content",
                    severity="high",
                    message=f"DOCX hidden content detected ('{clean_snippet}'): {'; '.join(reasons)}.",
                    start=start_off,
                    end=end_off,
                    details={
                        "source": "docx_run",
                        "paragraph_index": para_idx,
                        "reasons": reasons,
                        "text": text,
                        "font_size_pt": font_size_pt,
                        "is_white": is_white,
                        "is_hidden": is_hidden,
                    },
                )
            )

    for p_idx, para in enumerate(doc.paragraphs):
        for run in para.runs:
            inspect_run(run, p_idx)

    for t_idx, table in enumerate(doc.tables):
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        inspect_run(run, t_idx)

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
        CheckResult containing all detected findings and summary.
    """
    if isinstance(document, (str, Path)):
        doc_path = Path(document)
        if doc_path.exists() and doc_path.is_file():
            from veritext.ingest.pipeline import load_document
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

    raw_text = doc.raw_text or ""
    if canonical is None:
        canonical = normalize_text(raw_text)

    findings: List[Finding] = []

    # 1. Text-level anomalies: homoglyphs, zero-width chars, control chars
    anomalies = canonical.anomalies or []
    homoglyphs = [a for a in anomalies if a.type == "homoglyph"]
    zero_width = [a for a in anomalies if a.type == "zero_width"]
    controls = [a for a in anomalies if a.type == "control_character"]

    # Group anomalies by affected word bounds in original text
    word_to_anomalies: Dict[Tuple[int, int], List[Anomaly]] = {}
    for anom in anomalies:
        start, end = _find_word_bounds(raw_text, anom.position)
        word_to_anomalies.setdefault((start, end), []).append(anom)

    mixed_script_count = 0
    invisible_count = len(zero_width) + len(controls)

    for (w_start, w_end), anom_list in word_to_anomalies.items():
        word_text = raw_text[w_start:w_end]
        word_homoglyphs = [a for a in anom_list if a.type == "homoglyph"]
        word_invisible = [a for a in anom_list if a.type in ("zero_width", "control_character")]

        # Case A: Mixed-script word
        if _is_mixed_script_word(word_text):
            mixed_script_count += 1
            h_details = ", ".join(f"U+{ord(a.character):04X} ('{a.character}' -> '{a.replacement}')" for a in word_homoglyphs)
            findings.append(
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
            # Word with multiple homoglyphs
            h_details = ", ".join(f"U+{ord(a.character):04X}" for a in word_homoglyphs)
            findings.append(
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

        # Case B: Invisible characters in word
        if word_invisible:
            sev = "high" if invisible_count >= 3 else "info"
            invis_names = ", ".join(set(a.description for a in word_invisible))
            findings.append(
                Finding(
                    check="hidden_content",
                    severity=sev,
                    message=f"Invisible / zero-width characters in '{word_text}' ({invis_names}, {len(word_invisible)} char(s)).",
                    start=w_start,
                    end=w_end,
                    details={
                        "category": "invisible_characters",
                        "word": word_text,
                        "count": len(word_invisible),
                        "anomalies": [a.to_dict() for a in word_invisible],
                    },
                )
            )

    # 2. File-specific checks for DOCX and PDF
    if doc.path:
        file_path = Path(doc.path)
        if file_path.suffix.lower() == ".docx":
            docx_findings = _check_docx_hidden_runs(file_path, raw_text)
            findings.extend(docx_findings)
        elif file_path.suffix.lower() == ".pdf":
            pdf_findings = _check_pdf_hidden_characters(file_path, raw_text)
            findings.extend(pdf_findings)

    # Sort findings by start offset
    findings.sort(key=lambda f: f.start)

    # Aggregated summary & status
    has_high = any(f.severity == "high" for f in findings)
    has_warning = any(f.severity == "warning" for f in findings)
    passed = not (has_high or has_warning)

    summary_parts = []
    if mixed_script_count:
        summary_parts.append(f"{mixed_script_count} mixed-script word(s)")
    if invisible_count:
        summary_parts.append(f"{invisible_count} invisible/control character(s)")
    docx_pdf_count = sum(1 for f in findings if f.details.get("source") in ("docx_run", "pdf_char"))
    if docx_pdf_count:
        summary_parts.append(f"{docx_pdf_count} formatting obfuscation(s) (white text / tiny font / hidden)")

    if not findings:
        summary = "No hidden content, homoglyphs, or obfuscation detected."
    else:
        summary = f"Detected {len(findings)} suspicious item(s): {', '.join(summary_parts)}."

    return CheckResult(
        check_name="hidden_content",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={
            "homoglyph_count": len(homoglyphs),
            "mixed_script_words": mixed_script_count,
            "zero_width_count": len(zero_width),
            "control_count": len(controls),
            "total_findings": len(findings),
        },
    )
