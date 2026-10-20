"""Service layer orchestrating the Veritext analysis pipeline for UI and external consumers."""

import gc
import io
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from veritext.ai_detect.detector import detect_ai
from veritext.checks.runner import run_all_checks
from veritext.ingest.loaders import LOADER_MAP
from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document
from veritext.report.builder import build_report
from veritext.report.models import FullReport
from veritext.similarity.engine import check_document

MAX_WORDS_LIMIT = 5000
UPLOADS_DIR = Path("data/uploads")


def run_pipeline(
    target: Union[str, Path, bytes, io.BytesIO],
    filename: str,
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> FullReport:
    """Executes the complete Veritext analysis pipeline with progress reporting and memory management.

    Args:
        target: File path, bytes, or BytesIO buffer containing uploaded document.
        filename: Original name of the document.
        settings: Configuration options (e.g. use_online, use_semantic, use_ai, use_integrity, min_words).
        progress_callback: Optional callback func(status_message, progress_fraction).

    Returns:
        FullReport containing unified findings and engine outputs.
    """
    opts = settings or {}
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    def report_progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    report_progress("Preparing and reading document...", 0.05)

    # 1. Resolve file path
    temp_file: Optional[Path] = None
    if isinstance(target, (bytes, io.BytesIO)):
        data_bytes = target.getvalue() if isinstance(target, io.BytesIO) else target
        temp_file = UPLOADS_DIR / f"upload_{filename}"
        temp_file.write_bytes(data_bytes)
        file_to_load = temp_file
    else:
        file_to_load = Path(target)

    # Validate file extension
    ext = Path(filename).suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(LOADER_MAP.keys())}")

    # 2. Ingest document
    report_progress("Parsing document structure...", 0.15)
    doc = load_document(file_to_load)

    # Check for empty content
    raw_text = doc.raw_text or ""
    if not raw_text.strip():
        raise ValueError(f"The document '{filename}' contains no readable text.")

    # 3. Check 5000-word limit and truncate if necessary
    words = raw_text.split()
    is_truncated = False
    original_word_count = len(words)
    if original_word_count > MAX_WORDS_LIMIT:
        is_truncated = True
        truncated_words = words[:MAX_WORDS_LIMIT]
        # Reconstruct truncated raw text
        raw_text = " ".join(truncated_words)
        doc.raw_text = raw_text
        doc.clean_text = raw_text
        report_progress(f"Document capped at {MAX_WORDS_LIMIT} words (original: {original_word_count} words)...", 0.20)

    # 4. Plagiarism / Similarity Check
    sim_result = None
    use_online = bool(opts.get("use_online", False))
    use_semantic = bool(opts.get("use_semantic", False))
    min_words = int(opts.get("min_words", 8))

    report_progress("Executing winnowing similarity analysis...", 0.35)
    try:
        sim_result = check_document(
            document=doc,
            use_online=use_online,
            use_local=True,
            options={
                "skip_semantic": not use_semantic,
                "min_words": min_words,
            },
        )
    except Exception as e:
        report_progress(f"Notice: Similarity engine encountered an issue: {e}", 0.45)

    # 5. AI Text Detection
    ai_result = None
    use_ai = bool(opts.get("use_ai", True))
    if use_ai:
        report_progress("Evaluating stylometry and causal language model...", 0.60)
        try:
            ai_result = detect_ai(doc, use_roberta=bool(opts.get("use_roberta", False)))
        except Exception as e:
            report_progress(f"Notice: AI detection skipped or failed: {e}", 0.70)
        finally:
            gc.collect()

    # 6. Integrity & Formatting Checks
    integ_result = None
    use_integrity = bool(opts.get("use_integrity", True))
    if use_integrity:
        report_progress("Verifying citations, quotes, and hidden content...", 0.80)
        try:
            integ_result = run_all_checks(doc, similarity_result=sim_result)
        except Exception as e:
            report_progress(f"Notice: Integrity checks encountered an issue: {e}", 0.90)

    # 7. Compile Full Report
    report_progress("Compiling consolidated report...", 0.95)
    pipeline_settings = {
        "filename": filename,
        "use_online": use_online,
        "use_semantic": use_semantic,
        "use_ai": use_ai,
        "use_integrity": use_integrity,
        "min_words": min_words,
        "truncated": is_truncated,
        "original_word_count": original_word_count,
        "analyzed_word_count": min(original_word_count, MAX_WORDS_LIMIT),
    }

    report = build_report(
        document=doc,
        similarity=sim_result,
        ai=ai_result,
        integrity=integ_result,
        settings=pipeline_settings,
    )
    report.metadata["truncated"] = is_truncated
    report.metadata["original_word_count"] = original_word_count

    # Cleanup temporary upload file
    if temp_file and temp_file.exists():
        try:
            temp_file.unlink()
        except OSError:
            pass

    gc.collect()
    report_progress("Analysis finished.", 1.0)
    return report
