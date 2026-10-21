"""CLI for batch collusion analysis: python -m veritext.batch <folder>"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional

from veritext.batch import run_batch_analysis
from veritext.classrepo.repo import ClassRepositoryManager


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m veritext.batch",
        description="Compute pairwise winnowing similarity matrix, collusion clusters, and self-contained HTML heatmap",
    )
    parser.add_argument("folder", help="Path to folder containing submissions")
    parser.add_argument(
        "--class",
        dest="class_name",
        type=str,
        default=None,
        help="Optional class repository to cross-compare with",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30,
        help="Collusion threshold (default: 0.30 for 30%% similarity)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for similarity matrix CSV and HTML heatmap",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Use anonymised submission IDs instead of original filenames",
    )

    args = parser.parse_args(argv)

    folder_path = Path(args.folder).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Submissions directory not found: {folder_path}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("VERITEXT BATCH PAIRWISE COLLUSION & SIMILARITY ANALYSIS")
    print(f"Directory: {folder_path.name}")
    print(f"Threshold: {args.threshold * 100.0:.0f}%")
    print("=" * 70)

    try:
        result = run_batch_analysis(
            folder=folder_path,
            threshold=args.threshold,
            anonymize=args.anonymize,
            progress_callback=lambda msg, frac: print(f"[{frac*100.0:3.0f}%] {msg}"),
        )
    except Exception as e:
        print(f"Error during batch analysis: {e}", file=sys.stderr)
        return 1

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/reports") / f"batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "similarity_matrix.csv"
    csv_path.write_text(result.csv_content, encoding="utf-8")

    html_path = out_dir / "collusion_heatmap.html"
    html_path.write_text(result.html_report, encoding="utf-8")

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print(f"Total Submissions: {len(result.submissions)}")
    print(f"Pairs Flagged (&ge; {args.threshold * 100.0:.0f}%): {len(result.pairs_above_threshold)}")
    print(f"Clusters Detected: {len(result.clusters)}")
    print("=" * 70)

    if result.pairs_above_threshold:
        print("\nTop Collusion Pairs:")
        for idx, pair in enumerate(result.pairs_above_threshold[:5], 1):
            print(
                f"  {idx}. {pair.doc_a_name} <-> {pair.doc_b_name}: "
                f"{pair.similarity_percentage:.1f}% ({pair.shared_hashes_count} shared hashes)"
            )
            for sp in pair.shared_passages[:1]:
                print(f"     Passage: \"{sp.text_snippet[:100]}...\"")

    print(f"\nSaved CSV Similarity Matrix: {csv_path}")
    print(f"Saved HTML Collusion Heatmap: {html_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
