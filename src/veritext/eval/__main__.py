"""CLI interface for Veritext evaluation and benchmark suite.

Usage:
    python -m veritext.eval run [--semantic] [--ai] [--crosslang] [--out data/eval/results]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from veritext.eval.runner import BenchmarkRunner


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veritext.eval",
        description="Veritext Evaluation & Benchmark Suite",
    )
    subparsers = parser.add_subparsers(dest="command", help="Evaluation subcommands")

    run_parser = subparsers.add_parser("run", help="Execute the evaluation benchmark")
    run_parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic paraphrase matching ablation (requires sentence-transformers)",
    )
    run_parser.add_argument(
        "--ai",
        action="store_true",
        help="Enable deep AI detection models (DistilGPT2) in addition to stylometry",
    )
    run_parser.add_argument(
        "--crosslang",
        action="store_true",
        help="Enable multilingual cross-language matching evaluation (ES/FR)",
    )
    run_parser.add_argument(
        "--out",
        type=str,
        default="data/eval/results",
        help="Output directory for benchmark results (default: data/eval/results)",
    )

    args = parser.parse_args(argv)

    # If no subcommand provided or command is run, execute benchmark
    if args.command is None or args.command == "run":
        out_path = Path(getattr(args, "out", "data/eval/results"))
        semantic = getattr(args, "semantic", False)
        ai = getattr(args, "ai", False)
        crosslang = getattr(args, "crosslang", False)

        print("=" * 60)
        print("Starting Veritext Evaluation Benchmark")
        print(f"  Output directory: {out_path}")
        print(f"  Semantic enabled: {semantic}")
        if semantic:
            from veritext.similarity.semantic import (
                DEFAULT_SIMILARITY_THRESHOLD,
                DEFAULT_MIN_CONSECUTIVE_WINDOWS,
            )
            print("  Effective Semantic Settings:")
            print(f"    - Threshold (cosine):            {DEFAULT_SIMILARITY_THRESHOLD}")
            print(f"    - Min consecutive windows:       {DEFAULT_MIN_CONSECUTIVE_WINDOWS}")
            print(f"    - Include semantic in headline:  False (Exact & near-copy only by default)")
            print(f"    - Paraphrase leads tracked:      Separately for review")
            print(f"    - Lexical anchor rule:           Evaluated (N in {{1, 2, 3, 4}}, ratio in {{0.1, 0.2, 0.3}})")
        print(f"  Deep AI enabled:  {ai}")
        print(f"  Crosslang enabled:{crosslang}")
        print("=" * 60)

        runner = BenchmarkRunner(
            out_dir=out_path,
            semantic=semantic,
            ai=ai,
            crosslang=crosslang,
        )
        summary = runner.run()

        print("\nBenchmark completed successfully!")
        print(f"Evaluated {summary['total_documents_evaluated']} documents in {summary['total_runtime_seconds']}s")
        print(f"Peak memory: {summary['peak_memory_mb']} MB (~{summary['runtime_per_document_seconds']}s per doc)")
        print(f"Results written to: {out_path.resolve()}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
