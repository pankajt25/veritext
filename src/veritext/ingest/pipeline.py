"""Main pipeline for loading and processing documents into Document objects."""

from pathlib import Path
from typing import List, Union

from veritext.ingest.cleaner import clean_text
from veritext.ingest.latex import process_latex
from veritext.ingest.loaders import load_docx, load_pdf, read_text_file
from veritext.ingest.models import Document, RemovedSection
from veritext.ingest.splitter import split_paragraphs, split_sentences


def load_document(path: Union[str, Path]) -> Document:
    """
    Loads, cleans, and splits a document from the given file path.
    Supported extensions: .docx, .pdf, .tex, .txt, .md

    Returns:
        Document with raw_text, clean_text, sentences (text, start, end),
        paragraphs, removed_sections, and file_type.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = file_path.suffix.lower()
    file_type = ext.lstrip(".")

    removed_sections: List[RemovedSection] = []

    # 1. Load raw text according to file format
    if ext in (".txt", ".text"):
        raw_text = read_text_file(file_path)
        cleaned_text, clean_removed = clean_text(raw_text)
        removed_sections.extend(clean_removed)

    elif ext in (".md", ".markdown"):
        raw_text = read_text_file(file_path)
        cleaned_text, clean_removed = clean_text(raw_text)
        removed_sections.extend(clean_removed)

    elif ext in (".tex", ".latex"):
        raw_text = read_text_file(file_path)
        # Sanitize LaTeX (math, code blocks, citations)
        intermediate_text, latex_removed = process_latex(raw_text)
        removed_sections.extend(latex_removed)
        # Clean references and quotes from the extracted text
        cleaned_text, clean_removed = clean_text(intermediate_text)
        removed_sections.extend(clean_removed)

    elif ext == ".docx":
        raw_text = load_docx(file_path)
        cleaned_text, clean_removed = clean_text(raw_text)
        removed_sections.extend(clean_removed)

    elif ext == ".pdf":
        raw_text = load_pdf(file_path)
        cleaned_text, clean_removed = clean_text(raw_text)
        removed_sections.extend(clean_removed)

    elif ext in (".png", ".jpg", ".jpeg"):
        from veritext.ingest.loaders import load_image
        raw_text = load_image(file_path)
        cleaned_text, clean_removed = clean_text(raw_text)
        removed_sections.extend(clean_removed)

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: .docx, .pdf, .tex, .txt, .md, .png, .jpg, .jpeg"
        )

    # 2. Split paragraphs and sentences on clean_text
    paragraphs = split_paragraphs(cleaned_text)
    sentences = split_sentences(cleaned_text)

    return Document(
        raw_text=raw_text,
        clean_text=cleaned_text,
        sentences=sentences,
        paragraphs=paragraphs,
        removed_sections=removed_sections,
        file_type=file_type,
        path=str(file_path),
    )
