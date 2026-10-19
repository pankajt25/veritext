"""LaTeX extraction and sanitization utilities."""

import re
from typing import List, Tuple
from veritext.ingest.models import RemovedSection


# Patterns for LaTeX math environments
MATH_ENVIRONMENT_NAMES = [
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "displaymath",
    "math",
    "eqnarray",
    "eqnarray*",
    "split",
    "matrix",
    "bmatrix",
    "pmatrix",
    "vmatrix",
    "Vmatrix",
]

# Patterns for code blocks in LaTeX
CODE_ENVIRONMENT_NAMES = [
    "verbatim",
    "lstlisting",
    "minted",
]

# Citation command names
CITE_COMMAND_NAMES = [
    "cite",
    "citep",
    "citet",
    "citeauthor",
    "citeyear",
    "nocite",
    "parencite",
    "textcite",
    "autocite",
    "footcite",
]


def remove_latex_math(text: str) -> Tuple[str, List[RemovedSection]]:
    """Removes display and inline math environments from LaTeX text."""
    removed: List[RemovedSection] = []

    # 1. Named math environments (\begin{equation}...\end{equation})
    for env in MATH_ENVIRONMENT_NAMES:
        escaped_env = re.escape(env)
        pattern = re.compile(
            rf"\\begin\{{{escaped_env}\}}[\s\S]*?\\end\{{{escaped_env}\}}",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            removed.append(
                RemovedSection(
                    type="latex_math",
                    content=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        text = pattern.sub(" ", text)

    # 2. Display math \[ ... \]
    display_bracket_pattern = re.compile(r"\\\[[\s\S]*?\\\]")
    for match in display_bracket_pattern.finditer(text):
        removed.append(
            RemovedSection(
                type="latex_math",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = display_bracket_pattern.sub(" ", text)

    # 3. Display math $$ ... $$
    display_dollar_pattern = re.compile(r"(?<!\\)\$\$[\s\S]*?\$\$")
    for match in display_dollar_pattern.finditer(text):
        removed.append(
            RemovedSection(
                type="latex_math",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = display_dollar_pattern.sub(" ", text)

    # 4. Inline math \( ... \)
    inline_bracket_pattern = re.compile(r"\\\([\s\S]*?\\\)")
    for match in inline_bracket_pattern.finditer(text):
        removed.append(
            RemovedSection(
                type="latex_math",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = inline_bracket_pattern.sub(" ", text)

    # 5. Inline math $ ... $ (excluding escaped \$)
    inline_dollar_pattern = re.compile(r"(?<!\\)\$(?!\$)(.*?)(?<!\\)\$")
    for match in inline_dollar_pattern.finditer(text):
        removed.append(
            RemovedSection(
                type="latex_math",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = inline_dollar_pattern.sub(" ", text)

    return text, removed


def remove_latex_code_blocks(text: str) -> Tuple[str, List[RemovedSection]]:
    """Removes code environments such as verbatim and lstlisting."""
    removed: List[RemovedSection] = []

    for env in CODE_ENVIRONMENT_NAMES:
        escaped_env = re.escape(env)
        pattern = re.compile(
            rf"\\begin\{{{escaped_env}\}}(?:\[.*?\])?[\s\S]*?\\end\{{{escaped_env}\}}",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            removed.append(
                RemovedSection(
                    type="code_block",
                    content=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        text = pattern.sub(" ", text)

    # Inline verb: \verb|...| or \verb+...*
    verb_pattern = re.compile(r"\\verb([^\w\s])[\s\S]*?\1")
    for match in verb_pattern.finditer(text):
        removed.append(
            RemovedSection(
                type="code_block",
                content=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    text = verb_pattern.sub(" ", text)

    return text, removed


def remove_latex_citations(text: str) -> Tuple[str, List[RemovedSection]]:
    """Removes \\cite and related citation commands."""
    removed: List[RemovedSection] = []

    for cmd in CITE_COMMAND_NAMES:
        # Matches \cite{keys}, \cite[opt]{keys}, \cite[opt1][opt2]{keys}
        escaped_cmd = re.escape(cmd)
        pattern = re.compile(
            rf"\\{escaped_cmd}(?:\[[^\]]*\])*(?:\{{[^}}]*\}})",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            removed.append(
                RemovedSection(
                    type="citation",
                    content=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
        text = pattern.sub(" ", text)

    return text, removed


def latex_to_plain_text(latex_code: str) -> str:
    """Converts sanitized LaTeX markup to plain text."""
    # Try pylatexenc if available
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        converter = LatexNodes2Text(keep_comments=False)
        converted = converter.latex_to_text(latex_code)
        # Normalize excessive whitespace
        return converted
    except ImportError:
        pass

    # Fallback basic regex cleanup
    # Remove comments (% ...)
    text = re.sub(r"(?<!\\)%.*$", "", latex_code, flags=re.MULTILINE)
    # Remove common preamble / structural macros
    text = re.sub(r"\\(documentclass|usepackage|title|author|date)(?:\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\b", "", text)
    text = text.replace(r"\%", "%").replace(r"\$", "$").replace(r"\_", "_")
    return text


def process_latex(text: str) -> Tuple[str, List[RemovedSection]]:
    """
    Main LaTeX processing function.
    Removes math environments, code blocks, and citations, then converts to plain text.
    """
    all_removed: List[RemovedSection] = []

    # 1. Remove comments before other structures
    # Preserve lines but strip comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # 2. Remove code blocks
    text, code_removed = remove_latex_code_blocks(text)
    all_removed.extend(code_removed)

    # 3. Remove math environments
    text, math_removed = remove_latex_math(text)
    all_removed.extend(math_removed)

    # 4. Remove citations
    text, cite_removed = remove_latex_citations(text)
    all_removed.extend(cite_removed)

    # 5. Convert remaining LaTeX markup to plain text
    plain_text = latex_to_plain_text(text)

    return plain_text, all_removed
