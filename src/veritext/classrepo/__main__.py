"""CLI for Veritext class repository: python -m veritext.classrepo <command>"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from veritext.classrepo.repo import ClassRepositoryManager, OPT_IN_CONSENT_NOTICE


def print_consent_banner():
    print("=" * 70)
    print("VERITEXT LOCAL CLASS REPOSITORY")
    print(OPT_IN_CONSENT_NOTICE)
    print("=" * 70)


def cli_add(args: argparse.Namespace, mgr: ClassRepositoryManager) -> int:
    target = Path(args.file_or_folder).resolve()
    if not target.exists():
        print(f"Error: Path not found: {target}", file=sys.stderr)
        return 1

    print_consent_banner()
    mode_str = "FINGERPRINTS-ONLY (text discarded for privacy)" if args.fingerprints_only else "FULL PROSE + FINGERPRINTS"
    print(f"Adding to class repository '{args.class_name}' [Mode: {mode_str}]...")

    results = mgr.add_target(
        class_name=args.class_name,
        target_path=target,
        fingerprints_only=args.fingerprints_only,
    )

    if not results:
        print("No supported documents found to add.")
        return 1

    print(f"\nSuccessfully registered {len(results)} submission(s):")
    for fname, sub_id in results:
        print(f"  • {fname} -> Anonymised ID: {sub_id}")

    return 0


def cli_list(args: argparse.Namespace, mgr: ClassRepositoryManager) -> int:
    class_name = getattr(args, "class_name", None)

    if class_name:
        subs = mgr.list_submissions(class_name)
        if not subs:
            print(f"No submissions found in class repository '{class_name}'.")
            return 0

        print(f"\nSubmissions in class repository '{class_name}' ({len(subs)} total):")
        print("-" * 75)
        print(f"{'ID':<15} {'Mode':<18} {'Words':<8} {'Hashes':<8} {'Filename'}")
        print("-" * 75)
        for s in subs:
            mode = "Fingerprints-only" if s["fingerprints_only"] else "Stored text"
            print(f"{s['id']:<15} {mode:<18} {s['word_count']:<8} {s['fingerprint_count']:<8} {s['filename']}")
        print("-" * 75)
        return 0

    classes = mgr.list_classes()
    if not classes:
        print("No class repositories found in data/class_repository/.")
        return 0

    print(f"\nRegistered Class Repositories ({len(classes)} total):")
    print("-" * 75)
    print(f"{'Class Name':<25} {'Submissions':<14} {'Fingerprints-Only':<20} {'Total Hashes'}")
    print("-" * 75)
    for c in classes:
        print(
            f"{c['class_name']:<25} {c['submission_count']:<14} "
            f"{c['fingerprints_only_count']:<20} {c['total_fingerprints']}"
        )
    print("-" * 75)
    return 0


def cli_delete(args: argparse.Namespace, mgr: ClassRepositoryManager) -> int:
    deleted = mgr.delete_class(args.class_name)
    if deleted:
        print(f"Successfully deleted class repository '{args.class_name}'.")
        return 0
    else:
        print(f"Class repository '{args.class_name}' not found.", file=sys.stderr)
        return 1


def cli_purge(args: argparse.Namespace, mgr: ClassRepositoryManager) -> int:
    if not args.yes:
        confirm = input("Are you sure you want to delete ALL class repositories? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Purge operation cancelled.")
            return 0

    count = mgr.purge_all()
    print(f"Purged {count} class repository/repositories.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m veritext.classrepo",
        description="Local class repository manager for cross-submission and collusion matching",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Repository command")

    # add
    p_add = subparsers.add_parser("add", help="Add file or folder to a class repository")
    p_add.add_argument("class_name", help="Name of the target class")
    p_add.add_argument("file_or_folder", help="Path to document or directory of documents")
    p_add.add_argument(
        "--fingerprints-only",
        action="store_true",
        help="Store only winnowing hashes and offsets, discarding text for student privacy",
    )

    # list
    p_list = subparsers.add_parser("list", help="List registered classes or class submissions")
    p_list.add_argument("class_name", nargs="?", default=None, help="Optional class name to list submissions")

    # delete
    p_del = subparsers.add_parser("delete", help="Delete a specific class repository")
    p_del.add_argument("class_name", help="Name of the class repository to remove")

    # purge
    p_purge = subparsers.add_parser("purge", help="Purge all class repositories")
    p_purge.add_argument("--yes", "-y", action="store_true", help="Confirm deletion without interactive prompt")

    args = parser.parse_args(argv)
    mgr = ClassRepositoryManager()

    if args.subcommand == "add":
        return cli_add(args, mgr)
    elif args.subcommand == "list":
        return cli_list(args, mgr)
    elif args.subcommand == "delete":
        return cli_delete(args, mgr)
    elif args.subcommand == "purge":
        return cli_purge(args, mgr)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
