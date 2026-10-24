"""CLI for plagiarism detection and similarity analysis: python -m veritext.similarity <file>"""

import argparse
from pathlib import Path
import sys

from veritext.ingest.pipeline import load_document
from veritext.planner import load_limits_config, make_plan, print_profile_and_plan
from veritext.profile import profile_document
from veritext.similarity.crosslang import download_crosslang_model
from veritext.similarity.engine import check_document
from veritext.timing import load_timing


def main():
    parser = argparse.ArgumentParser(
        prog="python -m veritext.similarity",
        description="Inspect document for plagiarism, verbatim copy, and semantic paraphrasing",
    )
    parser.add_argument("file", nargs="?", default=None, help="Path to document to check")
    parser.add_argument(
        "--download-crosslang-model",
        action="store_true",
        help="Download multilingual model for cross-language matching",
    )
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
        default=0.78,
        help="Semantic similarity threshold (default: 0.78)",
    )
    parser.add_argument(
        "--include-semantic-in-score",
        action="store_true",
        default=False,
        help="Include semantic paraphrase matches in the headline score (label: 'score includes unverified semantic leads')",
    )
    parser.add_argument(
        "--include-single-window",
        action="store_true",
        default=False,
        help="Include isolated single-window semantic matches in headline similarity score (default: False, review only)",
    )
    parser.add_argument(
        "--min-consecutive-windows",
        type=int,
        default=2,
        help="Minimum consecutive matching sentence windows required for headline score (default: 2)",
    )
    parser.add_argument(
        "--fetch-fulltext",
        action="store_true",
        default=None,
        help="Fetch open-access full text for top matching candidates (on by default for online)",
    )
    parser.add_argument(
        "--no-fetch-fulltext",
        action="store_true",
        help="Disable open-access full text retrieval",
    )
    parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        default=None,
        help="Name of class repository to include in plagiarism check",
    )
    parser.add_argument(
        "--show-names",
        action="store_true",
        default=False,
        help="Show actual filenames instead of anonymised IDs for class repository submissions",
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

    if args.download_crosslang_model:
        success = download_crosslang_model()
        sys.exit(0 if success else 1)

    if not args.file:
        parser.print_help()
        sys.exit(1)

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

    # Determine full-text fetching policy
    fetch_ft = False if args.offline else True
    if args.fetch_fulltext:
        fetch_ft = True
    elif args.no_fetch_fulltext:
        fetch_ft = False

    result = None
    try:
        result = check_document(
            document=doc,
            use_online=not args.offline,
            use_local=True,
            max_sources=30,
            similarity_threshold=args.threshold,
            options={
                "fetch_fulltext": fetch_ft,
                "class_name": args.class_name,
                "show_names": args.show_names,
                "word_ranges": sem_plan.word_ranges if (sem_plan and sem_plan.sampling_mode == "section_sampled") else None,
                "include_single_window": args.include_single_window,
                "include_semantic_in_score": args.include_semantic_in_score,
                "min_consecutive_windows": args.min_consecutive_windows,
            },
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Similarity check interrupted by user (Ctrl+C).")
        return

    fulltext_cnt = result.metadata.get("fulltexts_fetched", 0)
    score_label = result.metadata.get("score_label", "exact & near-copy matches only")
    leads_cnt = result.metadata.get("paraphrase_leads_count", 0)
    leads_pct = result.metadata.get("paraphrase_leads_percentage", 0.0)
    strong_cnt = result.metadata.get("strong_leads_count", 0)
    weak_cnt = result.metadata.get("weak_leads_count", 0)

    print("\n" + "=" * 70)
    print(f"OVERALL SIMILARITY SCORE: {result.similarity_percentage:.1f}% ({score_label})")
    print(f"Characters Matched:       {result.matched_chars} / {result.total_clean_chars}")
    print(f"Total Match Spans:        {len(result.spans)}")
    if fulltext_cnt > 0:
        print(f"Full-Texts Fetched:       {fulltext_cnt} open-access paper(s)")
    if result.anomalies:
        print(f"Suspicious Anomalies:     {len(result.anomalies)} (homoglyphs / zero-width)")

    if leads_cnt > 0:
        print("-" * 70)
        print(f"Paraphrase leads (review): {leads_cnt} passages, {leads_pct:.1f}% of text")
        print(f"  {result.metadata.get('explanatory_line', '')}")
        print(f"  {result.metadata.get('interpretation_guide', '')}")
        print(f"  Breakdown: {strong_cnt} strong lead(s), {weak_cnt} weak lead(s)")
    print("=" * 70)

    # Top Sources Breakdown
    print("\nTop Matching Sources:")
    if not result.sources:
        print("  (No significant matching sources found)")
    else:
        for idx, src in enumerate(result.sources[:5], 1):
            print(f"  {idx}. [{src.origin.upper()}] {src.source_title}")
            print(f"     Headline Similarity: {src.percentage:.1f}% ({src.matched_chars} chars across {src.match_count} span(s))")
            if src.lead_count > 0:
                print(f"     Paraphrase leads:     {src.lead_count} lead(s) ({src.strong_lead_count} strong, {src.weak_lead_count} weak, ~{src.lead_percentage:.1f}% of text)")
            if src.source_url:
                print(f"     URL:                 {src.source_url}")

    # Top Matched Spans
    print("\nTop Matched Spans:")
    if not result.spans:
        print("  (No match spans detected)")
    else:
        for idx, span in enumerate(result.spans[:5], 1):
            tag = span.type.upper()
            lead_tag = f" [{span.metadata.get('lead_level', '').upper()}]" if span.metadata.get("lead_level") else ""
            snippet = span.matched_text.strip().replace("\n", " ")
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            print(f"  {idx}. [{tag}{lead_tag} - Score: {span.score:.2f}] from '{span.source_title}'")
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
