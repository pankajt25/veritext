"""Unified runner for document integrity checks.

Executes hidden content, citations/quotes, self-plagiarism, and code similarity checks,
returning an aggregated IntegrityReport.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

from veritext.checks.authorship import check_authorship_timeline
from veritext.checks.citations import check_citations_and_quotes
from veritext.checks.code_similarity import SUPPORTED_CODE_EXTENSIONS, check_code_similarity
from veritext.checks.hidden import check_hidden_content
from veritext.checks.models import CheckResult, IntegrityReport
from veritext.checks.selfcheck import check_self_plagiarism
from veritext.checks.spinner import check_spinner
from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document


def run_all_checks(
    document: Union[Document, str, Path],
    similarity_result: Optional[Any] = None,
    previous: bool = True,
    code_folder: Optional[Union[str, Path]] = None,
    template_file: Optional[Union[str, Path]] = None,
    custom_tortured_phrases: Optional[Union[str, Path]] = None,
) -> IntegrityReport:
    """Runs all applicable integrity checks on a document.

    Args:
        document: Document instance or file path.
        similarity_result: Optional Stage 3 SimilarityResult for verbatim quotation verification.
        previous: Whether to run self-plagiarism check against data/my_previous.
        code_folder: Optional folder of peer code files if document is a code file.
        template_file: Optional template boilerplate file for code checks.

    Returns:
        IntegrityReport containing all CheckResults and a plain-language summary.
    """
    doc_path: Optional[Path] = None
    if isinstance(document, (str, Path)):
        from veritext.ingest.pipeline import is_existing_file
        if is_existing_file(document):
            doc_path = Path(document)
            # If code file, do not load as academic document
            if doc_path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS:
                doc = Document(
                    raw_text=doc_path.read_text(errors="replace"),
                    clean_text=doc_path.read_text(errors="replace"),
                    sentences=[],
                    paragraphs=[],
                    removed_sections=[],
                    file_type=doc_path.suffix.lstrip("."),
                    path=str(doc_path.resolve()),
                )
            else:
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
        if doc.path:
            doc_path = Path(doc.path)

    results: List[CheckResult] = []

    # Check if target is a source code file
    is_code = (
        doc_path is not None
        and doc_path.suffix.lower() in SUPPORTED_CODE_EXTENSIONS
    ) or (doc.file_type.lower() in ("py", "java", "c", "cpp", "js"))

    if is_code:
        # Code similarity check
        res_code = check_code_similarity(
            doc,
            folder=code_folder,
            template_file=template_file,
        )
        results.append(res_code)
    else:
        # 1. Hidden content and text obfuscation check
        res_hidden = check_hidden_content(doc)
        results.append(res_hidden)

        # 2. Citation and quote consistency check
        res_citations = check_citations_and_quotes(
            doc,
            similarity_result=similarity_result,
        )
        results.append(res_citations)

        # 3. Self-plagiarism check against previous submissions
        if previous:
            res_self = check_self_plagiarism(doc)
            results.append(res_self)

        # 4. Text-spinner and tortured-phrase check
        res_spinner = check_spinner(
            doc,
            similarity_result=similarity_result,
            custom_file=custom_tortured_phrases,
        )
        results.append(res_spinner)

        # 5. Document authorship metadata timeline check
        res_authorship = check_authorship_timeline(doc)
        results.append(res_authorship)

    # Calculate overall status and summary
    total_findings = sum(len(r.findings) for r in results)
    high_count = sum(sum(1 for f in r.findings if f.severity == "high") for r in results)
    warning_count = sum(sum(1 for f in r.findings if f.severity == "warning") for r in results)
    info_count = sum(sum(1 for f in r.findings if f.severity == "info") for r in results)

    passed = all(r.passed for r in results)

    if passed:
        summary = (
            f"Integrity check PASSED across {len(results)} module(s). "
            f"No critical anomalies or policy violations detected ({info_count} informational notice(s))."
        )
    else:
        summary = (
            f"Integrity check FLAGGED {total_findings} item(s) for human review: "
            f"{high_count} high-severity issue(s), {warning_count} warning(s), and {info_count} informational note(s)."
        )

    return IntegrityReport(
        document_path=str(doc_path) if doc_path else None,
        results=results,
        summary=summary,
        passed=passed,
        metadata={
            "check_count": len(results),
            "total_findings": total_findings,
            "high_count": high_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "is_code": is_code,
        },
    )
