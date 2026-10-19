"""Search client implementations for open academic and knowledge sources."""

from veritext.sources.clients.arxiv import search as search_arxiv
from veritext.sources.clients.crossref import search as search_crossref
from veritext.sources.clients.openalex import search as search_openalex
from veritext.sources.clients.semantic_scholar import (
    search as search_semantic_scholar,
)
from veritext.sources.clients.wikipedia import search as search_wikipedia

ALL_CLIENTS = {
    "crossref": search_crossref,
    "openalex": search_openalex,
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "wikipedia": search_wikipedia,
}

__all__ = [
    "ALL_CLIENTS",
    "search_arxiv",
    "search_crossref",
    "search_openalex",
    "search_semantic_scholar",
    "search_wikipedia",
]
