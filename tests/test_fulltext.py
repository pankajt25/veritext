"""Tests for full-text open-access retrieval, SSRF validation, size caps, and extraction."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

from fpdf import FPDF
import pytest
import requests

from veritext.sources.fulltext import (
    FullTextCache,
    FullTextRetriever,
    PoliteSecureDownloader,
    SecurityError,
    extract_text_from_europepmc_xml,
    extract_text_from_pdf,
    extract_text_from_wikipedia_json,
    split_text_into_passages,
)
from veritext.sources.models import SourceCandidate


def create_minimal_pdf_bytes(text: str = "Test document body text.") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text)
    return bytes(pdf.output())


SAMPLE_EPMC_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <article-meta>
      <title-group>
        <article-title>Open Access Biological Discoveries</article-title>
      </title-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <title>Introduction</title>
      <p>This research evaluates open-access genomic indexing and sequencing protocols in model organisms.</p>
      <p>Comprehensive trials confirmed substantial throughput gains across diverse biological assays.</p>
    </sec>
  </body>
  <back>
    <ref-list>
      <ref id="B1"><p>Author A. References section that must be stripped completely.</p></ref>
    </ref-list>
  </back>
</article>
"""

SAMPLE_WIKI_JSON = b"""{
  "query": {
    "pages": {
      "12345": {
        "pageid": 12345,
        "title": "Machine Learning",
        "extract": "Machine learning is a field of inquiry devoted to understanding and building methods that learn."
      }
    }
  }
}
"""


def test_pdf_text_extraction():
    sample_text = "Neural networks represent complex mathematical mapping functions."
    pdf_bytes = create_minimal_pdf_bytes(sample_text)
    extracted = extract_text_from_pdf(pdf_bytes)
    assert "Neural networks represent" in extracted


def test_europepmc_xml_extraction_and_ref_stripping():
    extracted = extract_text_from_europepmc_xml(SAMPLE_EPMC_XML)
    assert "Open Access Biological Discoveries" in extracted
    assert "genomic indexing and sequencing" in extracted
    # References must be stripped
    assert "References section that must be stripped" not in extracted


def test_wikipedia_json_extraction():
    extracted = extract_text_from_wikipedia_json(SAMPLE_WIKI_JSON)
    assert "field of inquiry devoted to understanding" in extracted


def test_passages_splitting():
    text = "Paragraph 1 with several informative sentences.\n\n" * 10
    passages = split_text_into_passages(text, target_word_count=50)
    assert len(passages) >= 1
    assert all(len(p.split()) > 0 for p in passages)


def test_security_non_https_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path))
    with pytest.raises(SecurityError, match="Non-HTTPS scheme refused"):
        downloader.fetch("http://example.com/paper.pdf")


def test_security_private_ip_direct_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path))
    with pytest.raises(SecurityError, match="private/loopback"):
        downloader.fetch("https://127.0.0.1/secret.pdf")

    with pytest.raises(SecurityError, match="private/loopback"):
        downloader.fetch("https://192.168.1.100/paper.pdf")


def test_security_hostname_resolving_to_loopback_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path))
    with pytest.raises(SecurityError, match="resolved to private/loopback"):
        downloader.fetch("https://localhost/paper.pdf")


def test_security_redirect_to_private_ip_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path))

    # Mock initial response redirecting to loopback
    mock_resp1 = MagicMock()
    mock_resp1.status_code = 302
    mock_resp1.headers = {"Location": "https://127.0.0.1/private_data.pdf"}

    with patch("requests.Session.get", return_value=mock_resp1), \
         patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]):
        with pytest.raises(SecurityError, match="private/loopback"):
            downloader.fetch("https://legit-academic-host.org/download")


def test_oversize_content_length_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path), max_bytes=1024)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "application/pdf",
        "Content-Length": "2048",  # > 1024
    }

    with patch.object(downloader, "_validate_url_security"), \
         patch.object(downloader, "_wait_rate_limit"), \
         patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(SecurityError, match="exceeds"):
            downloader.fetch("https://legit-academic-host.org/paper.pdf")


def test_oversize_streaming_bytes_refused(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path), max_bytes=100)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/pdf"}
    mock_resp.iter_content.return_value = [b"A" * 60, b"B" * 60]  # Total 120 > 100

    with patch.object(downloader, "_validate_url_security"), \
         patch.object(downloader, "_wait_rate_limit"), \
         patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(SecurityError, match="exceeded"):
            downloader.fetch("https://legit-academic-host.org/paper.pdf")


def test_content_type_mismatch_skipped(tmp_path):
    downloader = PoliteSecureDownloader(cache=FullTextCache(tmp_path))

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "application/zip"}

    with patch.object(downloader, "_validate_url_security"), \
         patch.object(downloader, "_wait_rate_limit"), \
         patch("requests.Session.get", return_value=mock_resp):
        with pytest.raises(ValueError, match="Unexpected Content-Type"):
            downloader.fetch("https://legit-academic-host.org/archive.zip", expected_types=("pdf",))


def test_fulltext_cache_persistence_and_expiry(tmp_path):
    cache = FullTextCache(tmp_path)
    url = "https://export.arxiv.org/pdf/2101.00001.pdf"
    cache.set(url, "Cached paper content", ttl_days=30)

    assert cache.get(url) == "Cached paper content"

    # Simulate expired entry
    with cache._get_connection() as conn:
        conn.execute("UPDATE fulltext_cache SET expires_at = 10.0 WHERE url = ?", (url,))

    assert cache.get(url) is None


def test_fulltext_retriever_arxiv_and_epmc_flow(tmp_path):
    cache = FullTextCache(tmp_path)
    retriever = FullTextRetriever(cache=cache)

    pdf_bytes = create_minimal_pdf_bytes("This is the full body text of the arXiv preprint.")

    cand_arxiv = SourceCandidate(
        id="arxiv_2101.00001",
        title="Attention is All You Need",
        authors=["Vaswani"],
        url="https://arxiv.org/abs/2101.00001",
        origin="arxiv",
        year=2021,
        text="Short abstract only",
        metadata={"arxiv_id": "2101.00001"},
    )

    cand_epmc = SourceCandidate(
        id="epmc_12345",
        title="Genomic Sequencing",
        authors=["Scientist"],
        url="https://europepmc.org/article/PMC/PMC3257376",
        origin="europepmc",
        year=2020,
        text="Short abstract only",
        metadata={"pmcid": "PMC3257376"},
    )

    # Mock downloads
    def mock_fetch(url, **kwargs):
        if "arxiv" in url:
            return pdf_bytes, "application/pdf"
        elif "europepmc" in url or "fullTextXML" in url:
            return SAMPLE_EPMC_XML, "text/xml"
        raise ValueError("Unknown URL")

    with patch.object(retriever.downloader, "fetch", side_effect=mock_fetch):
        updated, count = retriever.fetch_and_index_top_candidates([cand_arxiv, cand_epmc])
        assert count == 2
        assert "full body text of the arXiv preprint" in updated[0].text
        assert "genomic indexing and sequencing" in updated[1].text


def test_document_copied_from_body_of_arxiv_caught_through_fulltext_retrieval(tmp_path):
    from veritext.fingerprint.index import FingerprintIndex
    from veritext.ingest.models import Document
    from veritext.similarity.engine import check_document
    from veritext.sources.store import ReferenceStore

    body_text = (
        "In Section four we derive the multihead attention mechanism where queries keys and values "
        "are linearly projected multiple times with different learned linear projections to distinctive subspace dimensions."
    )
    pdf_bytes = create_minimal_pdf_bytes(
        f"Abstract: Overview of machine learning.\n\nSection 4:\n{body_text}\n\nReferences: [1] Vaswani et al."
    )

    cand_arxiv = SourceCandidate(
        id="arxiv_1706.03762",
        title="Attention is All You Need",
        authors=["Vaswani"],
        url="https://arxiv.org/abs/1706.03762",
        origin="arxiv",
        year=2017,
        text="Abstract: Overview of machine learning.",  # Note: abstract does NOT contain body_text
        metadata={"arxiv_id": "1706.03762"},
    )

    copied_doc = Document(
        raw_text=body_text,
        clean_text=body_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    store = ReferenceStore(tmp_path / "test_ref.sqlite")
    index = FingerprintIndex(tmp_path / "test_ref.sqlite")

    with patch("veritext.similarity.engine.find_candidates", return_value=[cand_arxiv]), \
         patch("veritext.sources.fulltext.PoliteSecureDownloader.fetch", return_value=(pdf_bytes, "application/pdf")):
        result = check_document(
            copied_doc,
            use_online=True,
            use_local=False,
            store=store,
            index=index,
            options={"fetch_fulltext": True},
        )

        assert result.similarity_percentage > 50.0
        assert result.metadata.get("fulltexts_fetched") == 1
        assert len(result.spans) >= 1
        assert "Attention is All You Need" in result.spans[0].source_title

