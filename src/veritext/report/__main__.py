"""CLI interface for generating comprehensive Veritext reports in JSON, HTML, and PDF."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional

from veritext.ai_detect.detector import detect_ai
from veritext.checks.runner import run_all_checks
from veritext.ingest.pipeline import load_document
from veritext.planner import load_limits_config, make_plan, print_profile_and_plan
from veritext.profile import profile_document
from veritext.report.builder import build_report
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.similarity.engine import check_document
from veritext.timing import load_timing


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veritext.report",
        description="Generates consolidated Veritext reports in JSON, HTML, and PDF formats.",
    )
    parser.add_argument("file", help="Path to document to analyze")
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Run plagiarism check offline using local library (default: True)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        default=False,
        help="Enable online source retrieval across academic APIs",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        default=False,
        help="Execute AI text detection engine",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        default=False,
        help="Execute dense neural vector paraphrase search",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "custom"],
        default="auto",
        help="Adaptive planning mode: 'auto' (budget-based), 'full' (no word caps), or 'custom'",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Analysis time budget per engine in minutes (default: 10.0)",
    )
    parser.add_argument(
        "--ai-max-words",
        type=int,
        default=0,
        help="Custom mode word limit for AI detection (0 for unlimited)",
    )
    parser.add_argument(
        "--semantic-max-words",
        type=int,
        default=0,
        help="Custom mode word limit for semantic matching (0 for unlimited)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        default=False,
        help="Print document profile and per-engine execution plan, then exit without running",
    )
    parser.add_argument(
        "--out-dir",
        default="data/reports",
        help="Directory to save generated report files (default: data/reports)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    target_path = Path(args.file)

    if not target_path.exists():
        print(f"Error: Target file not found: {target_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest document & Profile
    doc = load_document(target_path)
    profile = profile_document(doc)

    # 2. Build Analysis Plan
    plan_options = {
        "mode": args.mode,
        "time_budget_minutes": args.time_budget,
        "use_exact": True,
        "use_checks": True,
        "use_ai": bool(args.ai),
        "use_semantic": bool(args.semantic),
        "custom_ai_words": args.ai_max_words,
        "custom_semantic_words": args.semantic_max_words,
    }
    timing = load_timing()
    limits = load_limits_config()
    plan = make_plan(profile=profile, options=plan_options, timing=timing, settings=limits)

    # 3. Print Profile & Plan first
    print_profile_and_plan(profile, plan)

    if args.plan_only:
        print("[Plan-Only] Exiting without running analysis engines.")
        return 0

    print("=" * 65)
    print(f"  VERITEXT CONSOLIDATED REPORT GENERATOR")
    print("=" * 65)
    print(f"Document: {target_path}")
    print(f"Output Directory: {out_dir}")

    sim_result = None
    ai_result = None
    integ_result = None

    try:
        # Plagiarism / Similarity Analysis (always full document for exact winnowing)
        use_online = bool(args.online)
        skip_semantic = not bool(args.semantic)
        sem_plan = plan.engines.get("semantic")

        print(f"\n[1/3] Running similarity detection (online={use_online}, semantic={not skip_semantic})...")
        sim_result = check_document(
            document=doc,
            use_online=use_online,
            use_local=True,
            options={
                "skip_semantic": skip_semantic,
                "word_ranges": sem_plan.word_ranges if (sem_plan and sem_plan.sampling_mode == "section_sampled") else None,
            },
        )

        # AI Text Detection
        ai_plan = plan.engines.get("ai_detect")
        if args.ai and ai_plan and ai_plan.enabled:
            print(f"[2/3] Running AI detection ({ai_plan.coverage_description})...")
            try:
                ai_result = detect_ai(
                    doc,
                    plan=plan,
                    word_ranges=ai_plan.word_ranges if ai_plan.sampling_mode == "section_sampled" else None,
                )
            except Exception as e:
                print(f"Warning: AI detection failed ({e}). Proceeding without AI score.")
        else:
            print(f"[2/3] Skipping AI text detection ({ai_plan.reason if ai_plan else 'disabled'}).")

        # Integrity Checks (always full document)
        print("[3/3] Executing document integrity, citations, and evasion checks (full document)...")
        integ_result = run_all_checks(doc, similarity_result=sim_result)

    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user (Ctrl+C). Preserving partial results...")

    # Build and export Report (full or partial)
    print("\nConsolidating findings and compiling reports...")
    settings = {
        "offline": not bool(args.online),
        "semantic_enabled": bool(args.semantic),
        "ai_enabled": bool(args.ai),
        "mode": plan.mode,
        "time_budget_minutes": plan.time_budget_minutes,
        "target_file": str(target_path),
    }
    report = build_report(
        document=doc,
        similarity=sim_result,
        ai=ai_result,
        integrity=integ_result,
        settings=settings,
        profile=profile,
        plan=plan,
    )

    # Generate timestamped filenames
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = target_path.stem
    json_path = out_dir / f"{stem}_{ts}.json"
    html_path = out_dir / f"{stem}_{ts}.html"
    pdf_path = out_dir / f"{stem}_{ts}.pdf"

    # Export all three formats
    json_path.write_text(report.to_json(), encoding="utf-8")
    export_report_html(report, output_path=html_path)
    export_report_pdf(report, output_path=pdf_path)

    print("\n" + "=" * 65)
    print("REPORTS GENERATED SUCCESSFULLY:")
    print(f"  JSON: {json_path}")
    print(f"  HTML: {html_path}")
    print(f"  PDF:  {pdf_path}")
    print("=" * 65)
    print(f"Summary: {report.summary}")
    print("=" * 65)

    return 0


if __name__ == "__main__":
    sys.exit(main())
