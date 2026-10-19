"""Document ingestion, loading, cleaning, and splitting module."""

from veritext.ingest.models import Document, Paragraph, RemovedSection, Sentence
from veritext.ingest.loaders import (
    load_docx,
    load_md,
    load_pdf,
    load_tex,
    load_txt,
    get_loader,
)
from veritext.ingest.latex import process_latex, remove_latex_math, remove_latex_code_blocks, remove_latex_citations
from veritext.ingest.cleaner import clean_text, remove_quotes, remove_references
from veritext.ingest.splitter import split_paragraphs, split_sentences
from veritext.ingest.pipeline import load_document

__all__ = [
    "Document",
    "Sentence",
    "Paragraph",
    "RemovedSection",
    "load_document",
    "load_docx",
    "load_md",
    "load_pdf",
    "load_tex",
    "load_txt",
    "get_loader",
    "process_latex",
    "remove_latex_math",
    "remove_latex_code_blocks",
    "remove_latex_citations",
    "clean_text",
    "remove_quotes",
    "remove_references",
    "split_paragraphs",
    "split_sentences",
]
