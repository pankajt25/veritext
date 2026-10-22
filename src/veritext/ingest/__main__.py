"""CLI interface for document ingestion: python -m veritext.ingest <file>"""

import argparse
import sys
from pathlib import Path

from veritext.ingest.pipeline import load_document


from typing import List, Optional


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="python -m veritext.ingest",
        description="Ingest and inspect academic documents (.docx, .pdf, .tex, .txt, .md)",
    )
    parser.add_argument("file", help="Path to document to ingest")
    parser.add_argument(
        "--show-sentences",
        action="store_true",
        help="Print all extracted sentences with offsets",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        doc = load_document(file_path)
    except Exception as e:
        print(f"Error loading document: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print(f"Veritext Document Ingestion Summary")
    print("=" * 60)
    print(f"File Path:       {doc.path}")
    print(f"File Type:       {doc.file_type}")
    print(f"Raw Characters:  {len(doc.raw_text)}")
    print(f"Clean Chars:     {len(doc.clean_text)}")
    print(f"Paragraph Count: {len(doc.paragraphs)}")
    print(f"Sentence Count:  {len(doc.sentences)}")
    print("-" * 60)
    print(f"Removed Sections ({len(doc.removed_sections)} found):")
    if not doc.removed_sections:
        print("  (None)")
    else:
        for idx, sec in enumerate(doc.removed_sections, 1):
            snippet = sec.content.strip().replace("\n", " ")
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            print(f"  {idx}. [{sec.type.upper()}] {snippet} ({len(sec.content)} chars)")
    print("=" * 60)

    if args.show_sentences:
        print("Extracted Sentences:")
        for idx, sent in enumerate(doc.sentences, 1):
            print(f"  [{idx:02d}] ({sent.start}:{sent.end}) {sent.text}")


if __name__ == "__main__":
    main()
