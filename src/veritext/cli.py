"""Unified CLI dispatcher for the Veritext academic integrity suite.

Provides the console script 'veritext' with subcommands:
  - check: Plagiarism and similarity check
  - ai: AI-written text detector
  - report: Report builder and cryptographic seal verification
  - sources: Reference library and candidate source management
  - checks: Structural, authorship, and text-spinning integrity checks
  - batch: Batch pairwise collusion analysis
  - classrepo: Class repository indexer and cross-cohort matcher
  - api: REST API and SSE event server
  - eval: Evaluation benchmark and metric runner
  - timing: Hardware speed calibration
"""

import argparse
import sys
from typing import Optional, Sequence


def _dispatch(subcmd: str, main_func, sub_args: Sequence[str]) -> int:
    """Dispatches execution to a subpackage main function, preserving sys.argv."""
    old_argv = sys.argv
    try:
        sys.argv = [f"veritext {subcmd}"] + list(sub_args)
        try:
            res = main_func(sub_args)
        except TypeError:
            res = main_func()
        return res if isinstance(res, int) else 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.argv = old_argv


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point for 'veritext'."""
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list or args_list[0] in ["-h", "--help"]:
        print("Usage: veritext <subcommand> [options]\n")
        print("Veritext Academic Integrity & Plagiarism Suite\n")
        print("Subcommands:")
        print("  check       Inspect document for plagiarism, verbatim copy, and paraphrasing")
        print("  ai          Run AI-written text detection (Lite Mode & ensemble)")
        print("  report      Build comprehensive HTML/PDF reports or verify cryptographic seals")
        print("  sources     Manage local reference library, arXiv, and OpenAlex sources")
        print("  checks      Run integrity checks (tortured phrases, authorship, format)")
        print("  batch       Run batch pairwise collusion matrix across multiple submissions")
        print("  classrepo   Manage institutional class repositories and cross-cohort matching")
        print("  api         Start the local REST API server with SSE progress streaming")
        print("  eval        Execute the synthetic evaluation and benchmarking suite")
        print("  timing      Calibrate hardware execution speed on this machine")
        print("\nRun 'veritext <subcommand> --help' for detailed options on any subcommand.")
        return 0

    subcommand = args_list[0].lower()
    sub_args = args_list[1:]

    if subcommand == "check":
        from veritext.similarity.__main__ import main as sim_main
        return _dispatch("check", sim_main, sub_args)

    elif subcommand == "ai":
        from veritext.ai_detect.__main__ import main as ai_main
        return _dispatch("ai", ai_main, sub_args)

    elif subcommand == "report":
        from veritext.report.__main__ import main as report_main
        return _dispatch("report", report_main, sub_args)

    elif subcommand == "sources":
        from veritext.sources.__main__ import main as sources_main
        return _dispatch("sources", sources_main, sub_args)

    elif subcommand == "checks":
        from veritext.checks.__main__ import main as checks_main
        return _dispatch("checks", checks_main, sub_args)

    elif subcommand == "batch":
        from veritext.batch.__main__ import main as batch_main
        return _dispatch("batch", batch_main, sub_args)

    elif subcommand == "classrepo":
        from veritext.classrepo.__main__ import main as classrepo_main
        return _dispatch("classrepo", classrepo_main, sub_args)

    elif subcommand == "api":
        from veritext.api.__main__ import main as api_main
        return _dispatch("api", api_main, sub_args)

    elif subcommand == "eval":
        from veritext.eval.__main__ import main as eval_main
        return _dispatch("eval", eval_main, sub_args)

    elif subcommand == "timing":
        from veritext.timing import main as timing_main
        return _dispatch("timing", timing_main, sub_args)

    else:
        print(f"Error: Unknown subcommand '{subcommand}'", file=sys.stderr)
        print("Run 'veritext --help' for available subcommands.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
