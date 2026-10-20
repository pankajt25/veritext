"""Stage 5: Extra integrity checks (hidden content, citations/quotes, self-plagiarism, code similarity)."""

from veritext.checks.citations import check_citations_and_quotes, detect_in_text_citations, parse_reference_section
from veritext.checks.code_similarity import (
    compare_code_directory,
    compare_code_files,
    check_code_similarity,
)
from veritext.checks.hidden import check_hidden_content
from veritext.checks.models import CheckResult, Finding, IntegrityReport
from veritext.checks.runner import run_all_checks
from veritext.checks.selfcheck import add_previous, check_self_plagiarism

__all__ = [
    "Finding",
    "CheckResult",
    "IntegrityReport",
    "check_hidden_content",
    "check_citations_and_quotes",
    "detect_in_text_citations",
    "parse_reference_section",
    "check_self_plagiarism",
    "add_previous",
    "check_code_similarity",
    "compare_code_files",
    "compare_code_directory",
    "run_all_checks",
]
