"""Citation and quotation integrity checker.

Verifies:
1. In-text citation detection across numeric ([1], [1-3]) and author-year styles ((Smith, 2020), Smith et al. (2020)).
2. Bibliography / references parsing and consistency:
   - Missing references (in-text citation points to reference not in bibliography)
   - Uncited references (reference in bibliography never cited in body text)
   - Duplicate references
   - Inconsistent citation style within a single document
3. Likely uncited claims:
   - Sentences asserting statistics, percentages, or research claims ("studies show", "according to")
     without a citation in the sentence or the immediately following one.
4. Quotation and verbatim copy integrity:
   - Long quotes (> 25 words) lacking nearby citations
   - Verbatim similarity match spans lacking quotation marks or citations
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.ingest.cleaner import REFERENCES_HEADER_PATTERN
from veritext.ingest.models import Document, Sentence
from veritext.ingest.splitter import split_sentences

# Regex for numeric in-text citations: [1], [1, 2], [1-4], [1, 3-5, 8]
NUMERIC_CITATION_REGEX = re.compile(
    r"\[\s*(\d+(?:\s*(?:[-–—]|,)\s*\d+)*)\s*\]"
)

# Regex for parenthetical author-year citation groups: (Smith, 2020), (Smith & Lee, 2020; Jones, 2021)
PAREN_AUTHOR_YEAR_GROUP_REGEX = re.compile(
    r"\(\s*([A-Z][^()]*?\b(?:19\d\d|20\d\d)[a-z]?(?:\s*;\s*[A-Z][^()]*?\b(?:19\d\d|20\d\d)[a-z]?)*)\s*\)"
)

# Regex for individual author-year inside parentheses: Smith, 2020 or Smith & Lee, 2020 or Smith et al., 2020
SINGLE_PAREN_AUTHOR_YEAR_REGEX = re.compile(
    r"([A-Z][a-zA-Z'\-]+(?:\s+(?:&|and)\s+[A-Z][a-zA-Z'\-]+|\s+et\s+al\.?)?)[,\s]+(\d{4}[a-z]?)"
)

# Regex for narrative author-year: Smith et al. (2020) or Smith and Lee (2020) or Smith (2020)
NARRATIVE_AUTHOR_YEAR_REGEX = re.compile(
    r"\b([A-Z][a-zA-Z'\-]+(?:\s+(?:&|and)\s+[A-Z][a-zA-Z'\-]+|\s+et\s+al\.?)?)\s+\((\d{4}[a-z]?)\)"
)

# Factual / empirical assertion trigger phrases
UNVALIDATED_CLAIM_PHRASES = [
    r"\bstudies\s+(?:have\s+)?(?:show|shown|demonstrated|indicated|suggested)\b",
    r"\bresearch\s+(?:has\s+)?(?:shown|shows|indicates|demonstrated|suggests)\b",
    r"\baccording\s+to\b",
    r"\bit\s+is\s+well\s+known\b",
    r"\bdata\s+(?:shows|indicates|reveals|demonstrates)\b",
    r"\bevidence\s+(?:suggests|shows|indicates|demonstrates)\b",
    r"\bsurveys?\s+(?:indicate|indicates|show|shows|found)\b",
    r"\bexperiments?\s+(?:demonstrate|demonstrates|show|shows|revealed)\b",
    r"\bscholars?\s+(?:have\s+)?(?:argued|asserted|claimed|suggested)\b",
]

# Patterns detecting statistics, percentages, and numerical empirical assertions
STATS_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*%",
    r"\b\d+(?:\.\d+)?\s+percent\b",
    r"\bpercentage\s+of\b",
    r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand|fold|x)\b",
    r"\b(?:increased|decreased|dropped|grew|fell|rose)\s+by\s+\d+",
]


@dataclass
class InTextCitation:
    """Parsed in-text citation span."""
    style: str  # 'numeric' or 'author_year'
    text: str
    keys: List[str]  # e.g. ['1', '2'] or ['smith_2020']
    start: int
    end: int


@dataclass
class ReferenceEntry:
    """Parsed bibliographic reference entry."""
    key: str  # numeric string ('1') or author_year string ('smith_2020')
    author_surname: str
    year: Optional[str]
    title: str
    raw_text: str
    start: int
    end: int


def _parse_numeric_citation_numbers(expr: str) -> List[str]:
    """Expands numeric citation string like '1-3, 5' into ['1', '2', '3', '5']."""
    keys = []
    parts = re.split(r"[,;]", expr)
    for p in parts:
        p = p.strip()
        range_match = re.match(r"^(\d+)\s*[-–—]\s*(\d+)$", p)
        if range_match:
            start_num = int(range_match.group(1))
            end_num = int(range_match.group(2))
            if start_num <= end_num and end_num - start_num <= 50:
                for n in range(start_num, end_num + 1):
                    keys.append(str(n))
            else:
                keys.append(str(start_num))
                keys.append(str(end_num))
        elif re.match(r"^\d+$", p):
            keys.append(p)
    return keys


def detect_in_text_citations(text: str) -> List[InTextCitation]:
    """Finds all in-text citations (numeric and author-year) with character offsets."""
    citations: List[InTextCitation] = []

    # 1. Numeric citations: [1], [1, 2], [1-3]
    for m in NUMERIC_CITATION_REGEX.finditer(text):
        inner = m.group(1)
        keys = _parse_numeric_citation_numbers(inner)
        if keys:
            citations.append(
                InTextCitation(
                    style="numeric",
                    text=m.group(0),
                    keys=keys,
                    start=m.start(),
                    end=m.end(),
                )
            )

    # 2. Parenthetical author-year groups: (Smith, 2020; Johnson, 2021)
    for m in PAREN_AUTHOR_YEAR_GROUP_REGEX.finditer(text):
        inner = m.group(1)
        sub_keys = []
        for sub in SINGLE_PAREN_AUTHOR_YEAR_REGEX.finditer(inner):
            author = sub.group(1).split()[0].rstrip(",").lower()
            year = sub.group(2)
            sub_keys.append(f"{author}_{year}")

        if sub_keys:
            citations.append(
                InTextCitation(
                    style="author_year",
                    text=m.group(0),
                    keys=sub_keys,
                    start=m.start(),
                    end=m.end(),
                )
            )

    # 3. Narrative author-year: Smith et al. (2020)
    for m in NARRATIVE_AUTHOR_YEAR_REGEX.finditer(text):
        author = m.group(1).split()[0].rstrip(",").lower()
        year = m.group(2)
        key = f"{author}_{year}"
        # Avoid duplicates if already covered
        if not any(c.start <= m.start() and c.end >= m.end() for c in citations):
            citations.append(
                InTextCitation(
                    style="author_year",
                    text=m.group(0),
                    keys=[key],
                    start=m.start(),
                    end=m.end(),
                )
            )

    citations.sort(key=lambda c: c.start)
    return citations


def parse_reference_section(
    ref_text: str, base_offset: int = 0
) -> List[ReferenceEntry]:
    """Parses reference list into ReferenceEntry objects."""
    if not ref_text or not ref_text.strip():
        return []

    entries: List[ReferenceEntry] = []
    # Strip any header like "References" or "Bibliography"
    header_match = REFERENCES_HEADER_PATTERN.search(ref_text)
    content_start = header_match.end() if header_match else 0
    ref_body = ref_text[content_start:]
    current_base = base_offset + content_start

    # Split into entries: detect entry start patterns
    # e.g. [1] or 1. or newline followed by author name
    entry_splits = []
    split_pattern = re.compile(
        r"(?m)(?:^(?:\[\d+\]|\d+\.|\*|\-)\s+)|(?:\n\s*\n+)"
    )

    matches = list(split_pattern.finditer(ref_body))
    if not matches:
        # Fall back to splitting by non-empty lines
        lines = [line.strip() for line in ref_body.splitlines() if line.strip()]
        line_offset = 0
        for line in lines:
            pos = ref_body.find(line, line_offset)
            if pos != -1:
                entries.append(_parse_single_reference_entry(line, current_base + pos))
                line_offset = pos + len(line)
        return [e for e in entries if e is not None]

    indices = [m.start() for m in matches]
    if indices[0] > 0 and ref_body[: indices[0]].strip():
        indices.insert(0, 0)
    indices.append(len(ref_body))

    for i in range(len(indices) - 1):
        seg_start = indices[i]
        seg_end = indices[i + 1]
        raw_entry = ref_body[seg_start:seg_end].strip()
        if not raw_entry or len(raw_entry) < 6:
            continue

        entry_obj = _parse_single_reference_entry(raw_entry, current_base + seg_start)
        if entry_obj:
            entries.append(entry_obj)

    return entries


def _parse_single_reference_entry(raw: str, offset: int) -> Optional[ReferenceEntry]:
    """Extracts author, year, title, and key from a single reference entry string."""
    clean_raw = raw.strip()
    if not clean_raw:
        return None

    # Check for numeric index: [1] or 1.
    num_match = re.match(r"^\[?(\d+)\]?\.?\s*", clean_raw)
    numeric_id = num_match.group(1) if num_match else None
    remaining = clean_raw[num_match.end():] if num_match else clean_raw

    # Find 4-digit year (1800-2099)
    year_match = re.search(r"\b(19\d\d|20\d\d)[a-z]?\b", remaining)
    year = year_match.group(1) if year_match else None

    # Author surname estimation
    author_surname = ""
    author_part = remaining[: year_match.start()] if year_match else remaining[:60]
    author_words = re.findall(r"\b[A-Z][a-zA-Z'\-]+\b", author_part)
    if author_words:
        author_surname = author_words[0].lower()

    # Title estimation
    title = ""
    if year_match:
        after_year = remaining[year_match.end():].lstrip(" .)(").strip()
        # Title is typically the sentence or phrase following year
        title_parts = re.split(r"[.\n]", after_year)
        title = title_parts[0].strip() if title_parts else after_year[:80]
    else:
        title = remaining[:80].strip()

    # Key formulation
    if numeric_id:
        key = numeric_id
    elif author_surname and year:
        key = f"{author_surname}_{year}"
    else:
        key = re.sub(r"\W+", "_", author_part.lower().strip())[:20]

    return ReferenceEntry(
        key=key,
        author_surname=author_surname,
        year=year,
        title=title,
        raw_text=clean_raw,
        start=offset,
        end=offset + len(clean_raw),
    )


def check_citations_and_quotes(
    document: Union[Document, str, Path],
    similarity_result: Optional[Any] = None,
) -> CheckResult:
    """Runs citation, bibliography, quote, and uncited claims integrity checks.

    Args:
        document: Document instance or file path/string.
        similarity_result: Optional Stage 3 SimilarityResult with exact match spans.

    Returns:
        CheckResult containing all findings.
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
    findings: List[Finding] = []

    # 1. Detect all in-text citations in the body text (excluding references section)
    ref_sections = [r for r in doc.removed_sections if r.type == "references"]
    ref_start_pos = None
    if ref_sections and ref_sections[0].start is not None:
        ref_start_pos = ref_sections[0].start
    else:
        m = REFERENCES_HEADER_PATTERN.search(raw_text)
        if m:
            ref_start_pos = m.start()

    body_text = raw_text[:ref_start_pos] if ref_start_pos is not None else raw_text
    citations = detect_in_text_citations(body_text)
    numeric_citations = [c for c in citations if c.style == "numeric"]
    author_year_citations = [c for c in citations if c.style == "author_year"]

    # Check for citation style inconsistency
    if numeric_citations and author_year_citations:
        first_num = numeric_citations[0]
        first_ay = author_year_citations[0]
        findings.append(
            Finding(
                check="citations",
                severity="warning",
                message=f"Inconsistent citation styles detected: document mixes numeric ({first_num.text}) and author-year ({first_ay.text}) formats.",
                start=first_ay.start,
                end=first_ay.end,
                details={
                    "issue": "style_inconsistency",
                    "numeric_count": len(numeric_citations),
                    "author_year_count": len(author_year_citations),
                },
            )
        )

    # 2. Extract and parse references section
    ref_sections = [r for r in doc.removed_sections if r.type == "references"]
    ref_entries: List[ReferenceEntry] = []

    if ref_sections:
        for r_sec in ref_sections:
            base_off = r_sec.start if r_sec.start is not None else 0
            ref_entries.extend(parse_reference_section(r_sec.content, base_offset=base_off))
    else:
        # Fallback: look for references header in raw text directly
        match = REFERENCES_HEADER_PATTERN.search(raw_text)
        if match:
            ref_content = raw_text[match.start():]
            ref_entries = parse_reference_section(ref_content, base_offset=match.start())

    # Build reference lookup keys
    ref_keys = {e.key.lower(): e for e in ref_entries}
    ref_numbers = {e.key: e for e in ref_entries if e.key.isdigit()}
    ref_author_years = {
        f"{e.author_surname}_{e.year}".lower(): e
        for e in ref_entries
        if e.author_surname and e.year
    }

    # Track cited reference keys
    cited_keys: Set[str] = set()

    # Check missing references
    if ref_entries:
        for cit in citations:
            for k in cit.keys:
                k_lower = k.lower()
                is_found = False

                if cit.style == "numeric":
                    if k in ref_numbers or k_lower in ref_keys:
                        is_found = True
                        cited_keys.add(k)
                else:
                    if k_lower in ref_author_years or k_lower in ref_keys:
                        is_found = True
                        cited_keys.add(k_lower)
                    else:
                        # Partial match: surname and year
                        parts = k_lower.split("_")
                        if len(parts) == 2:
                            surname, yr = parts
                            for ref_obj in ref_entries:
                                if ref_obj.year == yr and surname in ref_obj.raw_text.lower():
                                    is_found = True
                                    cited_keys.add(ref_obj.key.lower())
                                    break

                if not is_found:
                    findings.append(
                        Finding(
                            check="citations",
                            severity="warning",
                            message=f"In-text citation '{cit.text}' references '{k}' which is missing from the bibliography.",
                            start=cit.start,
                            end=cit.end,
                            details={
                                "issue": "missing_reference",
                                "citation": cit.text,
                                "unmatched_key": k,
                            },
                        )
                    )

        # Check uncited references in the bibliography
        for entry in ref_entries:
            e_key = entry.key.lower()
            e_ay = f"{entry.author_surname}_{entry.year}".lower() if entry.author_surname and entry.year else ""

            was_cited = (
                e_key in cited_keys
                or (e_ay and e_ay in cited_keys)
                or (entry.key in cited_keys)
            )

            if not was_cited:
                preview = entry.raw_text[:70].replace("\n", " ")
                findings.append(
                    Finding(
                        check="citations",
                        severity="info",
                        message=f"Reference listed in bibliography is never cited in document: '{preview}...'.",
                        start=entry.start,
                        end=entry.end,
                        details={
                            "issue": "uncited_reference",
                            "reference_key": entry.key,
                            "raw_text": entry.raw_text,
                        },
                    )
                )

        # Check duplicate references
        seen_entries: Dict[str, ReferenceEntry] = {}
        for entry in ref_entries:
            norm_title = re.sub(r"\W+", "", entry.title.lower())[:35]
            dup_id = f"{entry.author_surname}_{entry.year}_{norm_title}"
            if len(dup_id) > 6 and dup_id in seen_entries:
                prev = seen_entries[dup_id]
                findings.append(
                    Finding(
                        check="citations",
                        severity="warning",
                        message=f"Duplicate reference detected in bibliography: '{entry.title or entry.raw_text[:60]}'.",
                        start=entry.start,
                        end=entry.end,
                        details={
                            "issue": "duplicate_reference",
                            "first_start": prev.start,
                            "duplicate_start": entry.start,
                        },
                    )
                )
            else:
                seen_entries[dup_id] = entry

    # 3. Check for likely uncited claims
    # Sentences asserting statistics or research claims without a citation in current or next sentence
    sentences = doc.sentences
    if not sentences and raw_text:
        # Generate sentences if not loaded
        sentences = split_sentences(raw_text)

    # Pre-calculate which sentences contain citations
    def sentence_has_citation(s: Sentence) -> bool:
        return any(c.start >= s.start and c.end <= s.end for c in citations)

    sent_citation_flags = [sentence_has_citation(s) for s in sentences]
    num_sentences = len(sentences)

    claim_regexes = [re.compile(p, re.IGNORECASE) for p in UNVALIDATED_CLAIM_PHRASES]
    stats_regexes = [re.compile(p, re.IGNORECASE) for p in STATS_PATTERNS]

    for i, sent in enumerate(sentences):
        if getattr(sent, "is_heading", False):
            continue

        # False-positive avoidance: if current OR next sentence has a citation, do not flag
        has_cit = sent_citation_flags[i] or (i + 1 < num_sentences and sent_citation_flags[i + 1])
        if has_cit:
            continue

        stext = sent.text
        # Check claim phrases
        matched_phrase = next((r.pattern for r in claim_regexes if r.search(stext)), None)
        # Check stats
        matched_stat = next((r.pattern for r in stats_regexes if r.search(stext)), None)

        if matched_phrase or matched_stat:
            reason = "research trigger phrase" if matched_phrase else "statistical data / percentage"
            clean_snippet = stext.strip()[:80]
            findings.append(
                Finding(
                    check="citations",
                    severity="warning",  # strictly "warning", never "high"
                    message=f"Likely uncited claim ({reason}) without in-text citation in this or the next sentence: '{clean_snippet}...'.",
                    start=sent.start,
                    end=sent.end,
                    details={
                        "issue": "uncited_claim",
                        "sentence_text": stext,
                        "matched_reason": reason,
                    },
                )
            )

    # 4. Check long quotes without nearby citations
    quotes_to_check: List[Tuple[str, int, int]] = []
    for q_sec in doc.removed_sections:
        if q_sec.type == "quote":
            q_text = q_sec.content.strip()
            q_start = q_sec.start if q_sec.start is not None else raw_text.find(q_text)
            q_end = q_sec.end if q_sec.end is not None else (q_start + len(q_text) if q_start != -1 else 0)
            if q_start != -1:
                quotes_to_check.append((q_text, q_start, q_end))

    # Also detect quotes in body_text that may span newlines or were not filtered
    raw_quote_pattern = re.compile(
        r'("[^"]{40,}"|[\u201c\u201f][^\u201d\u201e]{40,}[\u201d\u201e]|«[^»]{40,}»)',
        re.DOTALL,
    )
    for m in raw_quote_pattern.finditer(body_text):
        q_text = m.group(0).strip()
        if not any(abs(q[1] - m.start()) < 5 for q in quotes_to_check):
            quotes_to_check.append((q_text, m.start(), m.end()))

    for q_text, q_start, q_end in quotes_to_check:
        words = q_text.split()
        if len(words) > 25:
            # Check if there is a citation nearby (within 150 characters before or after)
            search_min = max(0, q_start - 150)
            search_max = min(len(raw_text), q_end + 150)
            has_nearby_cit = any(
                search_min <= c.start <= search_max or search_min <= c.end <= search_max
                for c in citations
            )

            if not has_nearby_cit:
                findings.append(
                    Finding(
                        check="citations",
                        severity="warning",
                        message=f"Long quotation ({len(words)} words) lacks a nearby in-text citation.",
                        start=max(0, q_start),
                        end=max(0, q_end),
                        details={
                            "issue": "long_uncited_quote",
                            "word_count": len(words),
                            "quote_preview": q_text[:70],
                        },
                    )
                )

    # 5. Check verbatim matches from Stage 3 exact results
    if similarity_result is not None:
        sources = getattr(similarity_result, "sources", [])
        for src in sources:
            spans = getattr(src, "spans", [])
            for sp in spans:
                if getattr(sp, "type", "") in ("exact", "near_copy"):
                    m_text = getattr(sp, "matched_text", "")
                    doc_start = getattr(sp, "doc_start", 0)
                    doc_end = getattr(sp, "doc_end", 0)
                    word_len = len(m_text.split())

                    if word_len >= 8:
                        # Check if span is enclosed in quotation marks in raw_text
                        pre_ctx = raw_text[max(0, doc_start - 3):doc_start]
                        post_ctx = raw_text[doc_end:min(len(raw_text), doc_end + 3)]
                        in_quotes = (
                            ('"' in pre_ctx and '"' in post_ctx)
                            or ('“' in pre_ctx or '”' in post_ctx)
                            or ("'" in pre_ctx and "'" in post_ctx)
                        )

                        # Check nearby citation (within 150 characters)
                        has_nearby_cit = any(
                            max(0, doc_start - 150) <= c.start <= min(len(raw_text), doc_end + 150)
                            for c in citations
                        )

                        if not in_quotes and not has_nearby_cit:
                            findings.append(
                                Finding(
                                    check="citations",
                                    severity="warning",
                                    message="possible uncited copied text; add quotation marks and a citation, or rewrite in your own words",
                                    start=doc_start,
                                    end=doc_end,
                                    details={
                                        "issue": "uncited_verbatim_copy",
                                        "source_title": getattr(sp, "source_title", "Unknown"),
                                        "matched_words": word_len,
                                        "matched_text": m_text,
                                    },
                                )
                            )

    findings.sort(key=lambda f: f.start)

    passed = not any(f.severity in ("warning", "high") for f in findings)
    warning_count = sum(1 for f in findings if f.severity == "warning")
    info_count = sum(1 for f in findings if f.severity == "info")

    summary = (
        f"Citations and references validated: {len(citations)} citation(s), {len(ref_entries)} reference(s). "
        f"Flagged {warning_count} warning(s) and {info_count} informational item(s)."
    )

    return CheckResult(
        check_name="citations",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={
            "total_citations": len(citations),
            "numeric_citations": len(numeric_citations),
            "author_year_citations": len(author_year_citations),
            "total_references": len(ref_entries),
            "total_findings": len(findings),
            "warning_count": warning_count,
            "info_count": info_count,
        },
    )
