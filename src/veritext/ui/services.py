"""Service layer orchestrating the Veritext adaptive analysis pipeline.

Profiles documents first, computes per-engine execution plans based on hardware calibration,
and executes memory-safe chunked processing with live progress reporting and clean cancellation.
No fixed word caps.
"""

from datetime import datetime, timezone
import gc
import io
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from veritext.ai_detect.detector import detect_ai
from veritext.checks.runner import run_all_checks
from veritext.ingest.loaders import LOADER_MAP
from veritext.ingest.models import Document
from veritext.ingest.pipeline import load_document
from veritext.planner import AnalysisPlan, load_limits_config, make_plan
from veritext.profile import DocumentProfile, profile_document
from veritext.report.builder import build_report
from veritext.report.models import FullReport
from veritext.similarity.engine import check_document
from veritext.timing import load_timing, update_timing

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path("data/uploads")


def prepare_document_and_plan(
    target: Union[str, Path, bytes, io.BytesIO],
    filename: str,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Document, DocumentProfile, AnalysisPlan]:
    """Ingests target document, computes statistical profile, and prepares per-engine plan.

    Fast, offline, and safe to call immediately upon file upload.
    """
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    opts = settings or {}

    # Resolve file path
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

    doc = load_document(file_to_load)
    raw_text = doc.raw_text or ""
    if not raw_text.strip():
        raise ValueError(f"The document '{filename}' contains no readable text.")

    profile = profile_document(doc)
    timing = load_timing()
    limits = load_limits_config()
    plan = make_plan(profile=profile, options=opts, timing=timing, settings=limits)

    return doc, profile, plan


def run_pipeline(
    target: Union[str, Path, bytes, io.BytesIO],
    filename: str,
    settings: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> FullReport:
    """Executes the adaptive Veritext analysis pipeline with per-engine planning.

    Args:
        target: File path, bytes, or BytesIO buffer containing uploaded document.
        filename: Original name of the document.
        settings: Configuration options (mode, time_budget_minutes, use_ai, use_semantic, etc.).
        progress_callback: Optional callback func(status_message, progress_fraction).
        cancel_check: Optional callable returning True when analysis should cancel cleanly.

    Returns:
        FullReport containing unified findings, document profile, and execution plan.
    """
    opts = settings or {}

    def report_progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)

    report_progress("Profiling document structure...", 0.05)
    doc, profile, plan = prepare_document_and_plan(target, filename, opts)

    total_words = profile.word_count
    observed_speeds: Dict[str, float] = {}

    sim_result = None
    ai_result = None
    integ_result = None

    # Check cancellation before starting
    if cancel_check and cancel_check():
        logger.info("Pipeline cancelled before engine execution.")
        return build_report(doc, settings=opts, profile=profile, plan=plan)

    # 1. Similarity / Plagiarism Check
    # Winnowing exact matching always processes full document (no cap)
    exact_plan = plan.engines.get("exact")
    semantic_plan = plan.engines.get("semantic")
    use_online = bool(opts.get("use_online", False))
    use_semantic = bool(semantic_plan and semantic_plan.enabled)
    use_fulltext = bool(opts.get("use_fulltext", use_online))
    min_words = int(opts.get("min_words", 8))

    if exact_plan and exact_plan.enabled:
        report_progress("Executing winnowing fingerprint analysis (full document)...", 0.20)
        t0_sem = time.time()
        try:
            sim_result = check_document(
                document=doc,
                use_online=use_online,
                use_local=True,
                options={
                    "skip_semantic": not use_semantic,
                    "min_words": min_words,
                    "fetch_fulltext": use_fulltext,
                    "class_name": opts.get("class_name"),
                    "show_names": opts.get("show_names", False),
                    "word_ranges": semantic_plan.word_ranges if (semantic_plan and semantic_plan.sampling_mode == "section_sampled") else None,
                    "progress_callback": lambda msg: report_progress(msg, 0.35),
                    "cancel_check": cancel_check,
                },
            )
            t_sem_elapsed = time.time() - t0_sem
            if use_semantic and semantic_plan and semantic_plan.words_to_analyze > 0:
                observed_speeds["semantic_sec_per_1000"] = round(
                    t_sem_elapsed * (1000.0 / semantic_plan.words_to_analyze), 2
                )
        except Exception as e:
            report_progress(f"Notice: Similarity engine issue: {e}", 0.45)
            logger.warning(f"Similarity engine failed: {e}")

    # Check cancellation after similarity
    if cancel_check and cancel_check():
        logger.info("Pipeline cancelled after similarity check.")
        return build_report(doc, similarity=sim_result, settings=opts, profile=profile, plan=plan)

    # 2. AI Text Detection
    ai_plan = plan.engines.get("ai_detect")
    if ai_plan and ai_plan.enabled:
        ai_desc = ai_plan.coverage_description
        report_progress(f"Evaluating AI detection ({ai_desc})...", 0.50)
        t0_ai = time.time()
        try:
            ai_result = detect_ai(
                document=doc,
                use_roberta=bool(opts.get("use_roberta", False)),
                word_ranges=ai_plan.word_ranges if ai_plan.sampling_mode == "section_sampled" else None,
                plan=plan,
                progress_callback=lambda msg: report_progress(msg, 0.65),
                cancel_check=cancel_check,
            )
            t_ai_elapsed = time.time() - t0_ai
            if ai_plan.words_to_analyze > 0:
                observed_speeds["ai_distilgpt2_sec_per_1000"] = round(
                    t_ai_elapsed * (1000.0 / ai_plan.words_to_analyze), 2
                )
        except Exception as e:
            report_progress(f"Notice: AI detection issue: {e}", 0.70)
            logger.warning(f"AI detection failed: {e}")
        finally:
            gc.collect()

    # Check cancellation after AI detection
    if cancel_check and cancel_check():
        logger.info("Pipeline cancelled after AI detection.")
        return build_report(doc, similarity=sim_result, ai=ai_result, settings=opts, profile=profile, plan=plan)

    # 3. Integrity & Formatting Checks (always full document, no cap)
    checks_plan = plan.engines.get("checks")
    if checks_plan and checks_plan.enabled:
        report_progress("Verifying integrity, white font, citations, and homoglyphs...", 0.85)
        try:
            integ_result = run_all_checks(doc, similarity_result=sim_result)
        except Exception as e:
            report_progress(f"Notice: Integrity checks issue: {e}", 0.90)
            logger.warning(f"Integrity checks failed: {e}")

    # 4. Update Hardware Timing Profile with Smoothed Average
    if observed_speeds:
        try:
            update_timing(observed_speeds)
        except Exception as e:
            logger.debug(f"Timing update omitted: {e}")

    # 5. Compile Full Report
    report_progress("Compiling consolidated report...", 0.95)
    pipeline_settings = {
        "filename": filename,
        "mode": plan.mode,
        "time_budget_minutes": plan.time_budget_minutes,
        "use_online": use_online,
        "use_semantic": use_semantic,
        "use_ai": bool(ai_plan and ai_plan.enabled),
        "use_integrity": bool(checks_plan and checks_plan.enabled),
        "min_words": min_words,
        "original_word_count": total_words,
        "analyzed_word_count": max(
            getattr(exact_plan, "words_to_analyze", 0),
            getattr(ai_plan, "words_to_analyze", 0),
        ),
    }

    report = build_report(
        document=doc,
        similarity=sim_result,
        ai=ai_result,
        integrity=integ_result,
        settings=pipeline_settings,
        profile=profile,
        plan=plan,
    )

    report_progress("Analysis complete!", 1.0)
    return report
