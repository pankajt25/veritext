"""CLI for plagiarism detection and similarity analysis: python -m veritext.similarity <file>"""

import argparse
from pathlib import Path
import sys

from veritext.ingest.pipeline import load_document
from veritext.planner import load_limits_config, make_plan, print_profile_and_plan
from veritext.profile import profile_document
from veritext.similarity.engine import check_document
from veritext.timing import load_timing


def main():
    parser = argparse.ArgumentParser(
        prog="python -m veritext.similarity",
        description="Inspect document for plagiarism, verbatim copy, and semantic paraphrasing",
    )
    parser.add_argument("file", help="Path to document to check")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run analysis in offline mode (local reference library only)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        help="Optional file path to output complete results in JSON format",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Semantic similarity threshold (default: 0.75)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=30,
        help="Maximum candidate sources to retrieve (default: 30)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "custom"],
        default="auto",
        help="Planning mode ('auto', 'full', 'custom')",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=10.0,
        help="Time budget in minutes (default: 10.0)",
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
        help="Print profile and plan then exit without running",
    )

    args = parser.parse_args()

    doc_path = Path(args.file).resolve()
    if not doc_path.exists():
        print(f"Error: File not found: {doc_path}", file=sys.stderr)
        sys.exit(1)

    try:
        doc = load_document(doc_path)
    except Exception as e:
        print(f"Error loading document: {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Profile and Plan First
    profile = profile_document(doc)
    plan_options = {
        "mode": args.mode,
        "time_budget_minutes": args.time_budget,
        "use_exact": True,
        "use_checks": True,
        "use_ai": False,
        "use_semantic": True,
        "custom_ai_words": args.ai_max_words,
        "custom_semantic_words": args.semantic_max_words,
    }
    timing = load_timing()
    limits = load_limits_config()
    plan = make_plan(profile=profile, options=plan_options, timing=timing, settings=limits)

    # Print Profile and Plan first
    print_profile_and_plan(profile, plan)

    if args.plan_only:
        print("[Plan-Only] Exiting without running similarity check.")
        return

    sem_plan = plan.engines.get("semantic")

    print("=" * 70)
    print("Veritext Plagiarism & Integrity Analysis")
    print("=" * 70)
    print(f"Document: {doc_path.name}")
    print(f"Mode:     {'Offline (Local Library Only)' if args.offline else 'Online + Local'}")
    print(f"Semantic: {sem_plan.coverage_description if sem_plan else 'disabled'}")
    print("Analyzing document structure, fingerprints, and semantics...")

    result = None
    try:
        result = check_document(
            document=doc,
            use_online=not args.offline,
            use_local=True,
            max_sources=args.max_sources,
            similarity_threshold=args.threshold,
            options={
                "word_ranges": sem_plan.word_ranges if (sem_plan and sem_plan.sampling_mode == "section_sampled") else None,
            },
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Similarity check interrupted by user (Ctrl+C).")
        return

    print("\n" + "=" * 70)
    print(f"OVERALL SIMILARITY SCORE: {result.similarity_percentage:.1f}%")
    print(f"Characters Matched:       {result.matched_chars} / {result.total_clean_chars}")
    print(f"Total Match Spans:        {len(result.spans)}")
    if result.anomalies:
        print(f"Suspicious Anomalies:     {len(result.anomalies)} (homoglyphs / zero-width)")
    print("=" * 70)

    # Top Sources Breakdown
    print("\nTop Matching Sources:")
    if not result.sources:
        print("  (No significant matching sources found)")
    else:
        for idx, src in enumerate(result.sources[:5], 1):
            print(f"  {idx}. [{src.origin.upper()}] {src.source_title}")
            print(f"     Similarity: {src.percentage:.1f}% ({src.matched_chars} chars across {src.match_count} span(s))")
            if src.source_url:
                print(f"     URL:        {src.source_url}")

    # Top Matched Spans
    print("\nTop Matched Spans:")
    if not result.spans:
        print("  (No match spans detected)")
    else:
        for idx, span in enumerate(result.spans[:5], 1):
            tag = span.type.upper()
            snippet = span.matched_text.strip().replace("\n", " ")
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            print(f"  {idx}. [{tag} - Score: {span.score:.2f}] from '{span.source_title}'")
            print(f"     Offsets: {span.doc_start}:{span.doc_end} ({len(span.matched_text)} chars)")
            print(f"     Text:    \"{snippet}\"")

    # Anomalies Breakdown if present
    if result.anomalies:
        print("\nDetected Formatting & Character Anomalies:")
        for idx, anom in enumerate(result.anomalies[:5], 1):
            char_repr = repr(anom['character'])
            print(f"  {idx}. [{anom['type'].upper()}] {anom['description']} at char pos {anom['position']}")

    # Save JSON if requested
    if args.json_output:
        out_path = Path(args.json_output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.to_json(), encoding="utf-8")
        print(f"\nSaved complete JSON analysis report to: {out_path}")


if __name__ == "__main__":
    main()
