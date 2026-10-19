"""CLI for plagiarism detection and similarity analysis: python -m veritext.similarity <file>"""

import argparse
from pathlib import Path
import sys

from veritext.ingest.pipeline import load_document
from veritext.similarity.engine import check_document


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

    print("=" * 70)
    print("Veritext Plagiarism & Integrity Analysis")
    print("=" * 70)
    print(f"Document: {doc_path.name}")
    print(f"Mode:     {'Offline (Local Library Only)' if args.offline else 'Online + Local'}")
    print("Analyzing document structure, fingerprints, and semantics...")

    result = check_document(
        document=doc,
        use_online=not args.offline,
        use_local=True,
        max_sources=args.max_sources,
        similarity_threshold=args.threshold,
    )

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
