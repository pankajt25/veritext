"""CLI for managing the local reference library and candidate search: python -m veritext.sources <cmd>"""

import argparse
from pathlib import Path
import sys

from veritext.ingest.pipeline import load_document
from veritext.sources.http import SQLiteHttpCache
from veritext.sources.library import index_directory, index_file
from veritext.sources.retrieval import find_candidates
from veritext.sources.store import ReferenceStore


def handle_add(args):
    target = Path(args.target).resolve()
    if not target.exists():
        print(f"Error: Target path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    store = ReferenceStore()
    if target.is_dir():
        count = index_directory(target, store=store)
        print(f"Successfully indexed {count} new document(s) from directory: {target}")
    else:
        added = index_file(target, store=store)
        if added:
            print(f"Successfully indexed document: {target.name}")
        else:
            print(f"Document already indexed or unsupported: {target.name}")


def handle_search(args):
    doc_path = Path(args.document).resolve()
    if not doc_path.exists():
        print(f"Error: Document file not found: {doc_path}", file=sys.stderr)
        sys.exit(1)

    try:
        doc = load_document(doc_path)
    except Exception as e:
        print(f"Error loading document: {e}", file=sys.stderr)
        sys.exit(1)

    use_online = not args.offline
    print(f"Searching candidate sources for: {doc_path.name}")
    print(f"Mode: {'Offline (Local Library Only)' if args.offline else 'Online + Local'}")
    print("=" * 70)

    candidates = find_candidates(
        doc,
        use_online=use_online,
        use_local=True,
        max_sources=args.max,
    )

    if not candidates:
        print("No candidate sources found.")
        return

    print(f"Found {len(candidates)} candidate source(s):\n")
    for idx, c in enumerate(candidates, 1):
        print(f"{idx}. [{c.origin.upper()}] {c.title}")
        print(f"   URL:     {c.url or 'N/A'}")
        if c.authors:
            print(f"   Authors: {', '.join(c.authors[:3])}")
        if c.year:
            print(f"   Year:    {c.year}")
        if c.score:
            print(f"   Score:   {c.score}")
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        print(f"   Snippet: {snippet}")
        print("-" * 70)


def handle_stats(args):
    store = ReferenceStore()
    store_stats = store.stats()
    http_cache = SQLiteHttpCache()
    cache_stats = http_cache.stats()

    db_size_kb = store_stats["db_size_bytes"] / 1024
    cache_size_kb = cache_stats["size_bytes"] / 1024

    print("=" * 50)
    print("Veritext Reference Library & HTTP Cache Statistics")
    print("=" * 50)
    print(f"Indexed Sources:     {store_stats['sources_count']}")
    print(f"Indexed Passages:    {store_stats['passages_count']}")
    print(f"Database Size:       {db_size_kb:.2f} KB ({store.db_path})")
    print("-" * 50)
    print(f"HTTP Cache Entries:  {cache_stats['entry_count']}")
    print(f"HTTP Cache Size:     {cache_size_kb:.2f} KB ({http_cache.db_path})")
    print("=" * 50)


from typing import List, Optional


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="python -m veritext.sources",
        description="Manage local reference library and perform candidate source searches",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: add
    add_parser = subparsers.add_parser(
        "add", help="Add reference file or directory to local library"
    )
    add_parser.add_argument(
        "target", help="File or folder path to index into local library"
    )
    add_parser.set_defaults(func=handle_add)

    # Subcommand: search
    search_parser = subparsers.add_parser(
        "search", help="Search for candidate sources matching a document"
    )
    search_parser.add_argument("document", help="Path to document to analyze")
    search_parser.add_argument(
        "--offline",
        action="store_true",
        help="Run search in offline mode (use local library and cache only)",
    )
    search_parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="Maximum candidate sources to return (default: 10)",
    )
    search_parser.set_defaults(func=handle_search)

    # Subcommand: stats
    stats_parser = subparsers.add_parser(
        "stats", help="Display statistics for reference library and HTTP cache"
    )
    stats_parser.set_defaults(func=handle_stats)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
