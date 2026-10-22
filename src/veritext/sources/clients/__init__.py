"""Search client implementations for open academic and knowledge sources."""

from veritext.sources.clients.arxiv import search as search_arxiv
from veritext.sources.clients.core import (
    get_core_status,
    is_core_configured,
    search as search_core,
)
from veritext.sources.clients.crossref import search as search_crossref
from veritext.sources.clients.openalex import search as search_openalex
from veritext.sources.clients.semantic_scholar import (
    is_semantic_scholar_available,
    reset_semantic_scholar_session,
    search as search_semantic_scholar,
    set_semantic_scholar_disabled,
)
from veritext.sources.clients.wikipedia import search as search_wikipedia

ALL_CLIENTS = {
    "crossref": search_crossref,
    "openalex": search_openalex,
    "arxiv": search_arxiv,
    "semantic_scholar": search_semantic_scholar,
    "wikipedia": search_wikipedia,
    "core": search_core,
}

__all__ = [
    "ALL_CLIENTS",
    "search_arxiv",
    "search_core",
    "search_crossref",
    "search_openalex",
    "search_semantic_scholar",
    "search_wikipedia",
    "is_core_configured",
    "get_core_status",
    "is_semantic_scholar_available",
    "reset_semantic_scholar_session",
    "set_semantic_scholar_disabled",
]
