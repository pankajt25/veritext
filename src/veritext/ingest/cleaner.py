"""Document cleaner to remove references and quoted text while tracking removed segments."""

import re
from typing import List, Tuple
from veritext.ingest.models import RemovedSection


# Patterns for bibliography / references headers
REFERENCES_HEADER_PATTERN = re.compile(
    r"(?m)^(?:\s*(?:#{1,6}\s+|(?:\d+\.?\s+)?)(?:References|Bibliography|Works Cited|Literature Cited)\s*:?\s*$)",
    re.IGNORECASE,
)

# Optional section after references that might terminate the reference block
APPENDIX_HEADER_PATTERN = re.compile(
    r"(?m)^(?:\s*(?:#{1,6}\s+|(?:\d+\.?\s+)?)(?:Appendix|Appendices)\b.*$)",
    re.IGNORECASE,
)

# LaTeX thebibliography environment
LATEX_BIB_PATTERN = re.compile(
    r"\\begin\{thebibliography\}[\s\S]*?\\end\{thebibliography\}",
    re.IGNORECASE,
)

# Markdown and text blockquotes: lines starting with >
BLOCKQUOTE_PATTERN = re.compile(
    r"(?m)^(?:>[ \t]*[^\n]*\n?)+",
)

# LaTeX quote environments
LATEX_QUOTE_PATTERN = re.compile(
    r"\\begin\{(?:quote|quotation)\}[\s\S]*?\\end\{(?:quote|quotation)\}",
    re.IGNORECASE,
)

# Inline quotes: double quotes and curly quotes
# Matches "quote content" and “quote content”
INLINE_DOUBLE_QUOTES = re.compile(
    r'("[^"\n]{3,}"|[\u201c\u201f][^\u201d\u201e\n]{3,}[\u201d\u201e]|«[^»\n]{3,}»)',
)

# Inline single curly quotes ‘...’ containing at least 2 words (to prevent matching apostrophes)
INLINE_SINGLE_QUOTES = re.compile(
    r"([\u2018\u201b][^\u2019\u201a\n]{2,}\s+[^\u2019\u201a\n]{2,}[\u2019\u201a])",
)


def remove_references(text: str) -> Tuple[str, List[RemovedSection]]:
    """
    Identifies and removes the references/bibliography section from academic text.
    Records the removed text and character offsets.
    """
    removed: List[RemovedSection] = []

    # 1. Check for LaTeX thebibliography environment first
    latex_match = LATEX_BIB_PATTERN.search(text)
    if latex_match:
        content = latex_match.group(0)
        removed.append(
            RemovedSection(
                type="references",
                content=content,
                start=latex_match.start(),
                end=latex_match.end(),
            )
        )
        text = text[:latex_match.start()] + text[latex_match.end():]

    # 2. Check for textual References / Bibliography headers
    match = REFERENCES_HEADER_PATTERN.search(text)
    if match:
        ref_start = match.start()
        # Look for a subsequent Appendix section
        app_match = APPENDIX_HEADER_PATTERN.search(text, pos=match.end())
        if app_match:
            ref_end = app_match.start()
            content = text[ref_start:ref_end]
            removed.append(
                RemovedSection(
                    type="references",
                    content=content,
                    start=ref_start,
                    end=ref_end,
                )
            )
            text = text[:ref_start] + text[ref_end:]
        else:
            content = text[ref_start:]
            removed.append(
                RemovedSection(
                    type="references",
                    content=content,
                    start=ref_start,
                    end=len(text),
                )
            )
            text = text[:ref_start]

    return text, removed


def remove_quotes(text: str) -> Tuple[str, List[RemovedSection]]:
    """
    Identifies and removes blockquotes and inline quoted text.
    Records removed segments with offsets.
    """
    removed: List[RemovedSection] = []

    # 1. LaTeX quote environments
    for match in LATEX_QUOTE_PATTERN.finditer(text):
        removed.append(
            RemovedSection(
                type="quote",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = LATEX_QUOTE_PATTERN.sub(" ", text)

    # 2. Markdown / text blockquotes
    for match in BLOCKQUOTE_PATTERN.finditer(text):
        removed.append(
            RemovedSection(
                type="quote",
                content=match.group(0).strip(),
                start=match.start(),
                end=match.end(),
            )
        )
    text = BLOCKQUOTE_PATTERN.sub(" ", text)

    # 3. Inline double quotes
    for match in INLINE_DOUBLE_QUOTES.finditer(text):
        removed.append(
            RemovedSection(
                type="quote",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = INLINE_DOUBLE_QUOTES.sub(" ", text)

    # 4. Inline single curly quotes
    for match in INLINE_SINGLE_QUOTES.finditer(text):
        removed.append(
            RemovedSection(
                type="quote",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = INLINE_SINGLE_QUOTES.sub(" ", text)

    return text, removed


def normalize_clean_text(text: str) -> str:
    """Normalizes whitespace and blank lines in cleaned text."""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace lines containing only whitespace with empty lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse multiple consecutive blank lines into double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs into single space (preserving newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Re-normalize space around newlines
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> Tuple[str, List[RemovedSection]]:
    """
    Cleans raw text by removing references/bibliography and quoted text.
    Returns the normalized clean text and list of all removed sections.
    """
    all_removed: List[RemovedSection] = []

    # 1. Remove references section
    text_no_refs, ref_removed = remove_references(text)
    all_removed.extend(ref_removed)

    # 2. Remove quoted text
    text_no_quotes, quote_removed = remove_quotes(text_no_refs)
    all_removed.extend(quote_removed)

    # 3. Normalize whitespace
    cleaned = normalize_clean_text(text_no_quotes)

    return cleaned, all_removed
