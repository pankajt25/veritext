"""CLI interface for generating comprehensive Veritext reports in JSON, HTML, and PDF."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional

from veritext.ai_detect.detector import detect_ai
from veritext.checks.runner import run_all_checks
from veritext.ingest.pipeline import load_document
from veritext.report.builder import build_report
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.similarity.engine import check_document


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

    print("=" * 65)
    print(f"  VERITEXT CONSOLIDATED REPORT GENERATOR")
    print("=" * 65)
    print(f"Document: {target_path}")
    print(f"Output Directory: {out_dir}")

    # 1. Ingest document
    print("\n[1/4] Ingesting and canonicalizing document...")
    doc = load_document(target_path)

    # 2. Plagiarism / Similarity Analysis
    use_online = bool(args.online)
    skip_semantic = not bool(args.semantic)
    print(f"[2/4] Running similarity detection (online={use_online}, semantic={not skip_semantic})...")
    sim_result = check_document(
        document=doc,
        use_online=use_online,
        use_local=True,
        options={"skip_semantic": skip_semantic},
    )

    # 3. AI Text Detection
    ai_result = None
    if args.ai:
        print("[3/4] Running AI-text stylometric + LM detection...")
        try:
            ai_result = detect_ai(doc)
        except Exception as e:
            print(f"Warning: AI detection failed ({e}). Proceeding without AI score.")
    else:
        print("[3/4] Skipping AI text detection (pass --ai to enable).")

    # 4. Integrity Checks
    print("[4/4] Executing document integrity, citations, and evasion checks...")
    integ_result = run_all_checks(doc, similarity_result=sim_result)

    # 5. Build Unified Report
    print("\nConsolidating findings and compiling reports...")
    settings = {
        "offline": not use_online,
        "semantic_enabled": not skip_semantic,
        "ai_enabled": bool(args.ai),
        "target_file": str(target_path),
    }
    report = build_report(
        document=doc,
        similarity=sim_result,
        ai=ai_result,
        integrity=integ_result,
        settings=settings,
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
