"""CLI interface for Veritext Stage 5 integrity checks."""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from veritext.checks.code_similarity import compare_code_directory
from veritext.checks.runner import run_all_checks
from veritext.checks.selfcheck import add_previous


def print_banner(title: str):
    print("=" * 65)
    print(f"  VERITEXT INTEGRITY CHECK: {title}")
    print("=" * 65)


def cli_run_document_checks(args: argparse.Namespace) -> int:
    """Executes all integrity checks on a target document."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    print_banner(file_path.name)
    print(f"Inspecting file: {file_path}")

    report = run_all_checks(
        document=file_path,
        previous=not args.no_previous,
        code_folder=args.code_folder,
        template_file=args.template,
        custom_tortured_phrases=getattr(args, "custom_tortured_phrases", None),
    )

    print("\n--- RESULTS BY CHECK ---")
    for cr in report.results:
        status_symbol = "✓ PASS" if cr.passed else "⚠ FLAGGED"
        print(f"\n[{status_symbol}] {cr.check_name.upper()}: {cr.summary}")

        if cr.findings:
            print("  Top Findings:")
            for idx, f in enumerate(cr.findings[:5], 1):
                sev_label = f.severity.upper()
                print(f"    {idx}. [{sev_label}] (offset {f.start}:{f.end}) {f.message}")
            if len(cr.findings) > 5:
                print(f"    ... and {len(cr.findings) - 5} more finding(s)")

    print("\n" + "=" * 65)
    overall_status = "PASSED" if report.passed else "FLAGGED FOR REVIEW"
    print(f"OVERALL STATUS: {overall_status}")
    print(f"SUMMARY: {report.summary}")
    print("=" * 65)

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(report.to_json(), encoding="utf-8")
        print(f"\nDetailed report written to: {out_path}")

    return 0 if report.passed else 2


def cli_add_previous(args: argparse.Namespace) -> int:
    """Indexes previous work under origin='self' for self-plagiarism checking."""
    target = Path(args.target)
    if not target.exists():
        print(f"Error: Target path not found: {target}", file=sys.stderr)
        return 1

    print_banner("REGISTER PREVIOUS WORK")
    print(f"Indexing earlier work from: {target}...")

    indexed = add_previous(target)
    print(f"\nSuccessfully indexed {len(indexed)} document(s) as prior submissions (origin='self'):")
    for name in indexed:
        print(f"  - {name}")

    return 0


def cli_code_similarity(args: argparse.Namespace) -> int:
    """Executes pairwise code similarity on a folder of source files."""
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: Folder not found or not a directory: {folder}", file=sys.stderr)
        return 1

    print_banner(f"CODE SIMILARITY: {folder.name}")
    print(f"Analyzing source code in: {folder}")
    if args.template:
        print(f"Ignoring boilerplate template: {args.template}")

    comparisons = compare_code_directory(
        folder=folder,
        template_file=args.template,
    )

    if not comparisons:
        print("No comparable source code files found in directory.")
        return 0

    print("\nPairwise Similarity Results:")
    print(f"{'File 1':<24} {'File 2':<24} {'Similarity':<12} {'Matching Lines'}")
    print("-" * 75)

    high_matches = 0
    for comp in comparisons:
        lines_summary = []
        for mr in comp.matching_ranges[:2]:
            lines_summary.append(f"L{mr.lines_a[0]}-{mr.lines_a[1]} <-> L{mr.lines_b[0]}-{mr.lines_b[1]}")
        lines_str = ", ".join(lines_summary) if lines_summary else "none"

        sim_str = f"{comp.similarity_pct:.1f}%"
        print(f"{comp.file_a:<24} {comp.file_b:<24} {sim_str:<12} {lines_str}")
        if comp.similarity_pct >= 40.0:
            high_matches += 1

    print("-" * 75)
    print(f"Evaluated {len(comparisons)} pair(s). Flagged {high_matches} pair(s) with similarity >= 40.0%.")

    if args.json:
        out_path = Path(args.json)
        json_data = [c.to_dict() for c in comparisons]
        out_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
        print(f"Pairwise results saved to: {out_path}")

    return 0 if high_matches == 0 else 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="veritext.checks",
        description="Veritext Stage 5 Integrity Checks: Hidden content, citations, self-plagiarism, and code similarity.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # 1. Subcommand: add-previous
    parser_add = subparsers.add_parser("add-previous", help="Register previous work for self-plagiarism checks")
    parser_add.add_argument("target", help="File or folder of previous documents to index")

    # 2. Subcommand: code
    parser_code = subparsers.add_parser("code", help="Run pairwise code similarity on a directory")
    parser_code.add_argument("folder", help="Directory containing source code files")
    parser_code.add_argument("--template", help="Boilerplate template file to ignore")
    parser_code.add_argument("--json", help="Save pairwise results to JSON file")

    # Default command: check document
    parser_check = subparsers.add_parser("check", help="Run integrity checks on a document")
    parser_check.add_argument("file", help="Path to document to inspect")
    parser_check.add_argument("--json", help="Path to write JSON integrity report")
    parser_check.add_argument("--no-previous", action="store_true", help="Skip self-plagiarism check against previous work")
    parser_check.add_argument("--code-folder", help="Peer code folder if inspecting a single code file")
    parser_check.add_argument("--template", help="Boilerplate template file if inspecting code")
    parser_check.add_argument("--custom-tortured-phrases", help="Path to JSON file with additional tortured phrases")

    # Support default positional argument directly: python -m veritext.checks <file>
    if argv is None:
        raw_argv = sys.argv[1:]
    else:
        raw_argv = argv

    if raw_argv and raw_argv[0] not in ("add-previous", "code", "check", "-h", "--help"):
        # Insert "check" as the default subcommand
        raw_argv = ["check"] + raw_argv

    args = parser.parse_args(raw_argv)

    if args.subcommand == "add-previous":
        return cli_add_previous(args)
    elif args.subcommand == "code":
        return cli_code_similarity(args)
    elif args.subcommand == "check":
        return cli_run_document_checks(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
