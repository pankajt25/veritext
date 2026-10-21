"""Full-text open-access retrieval, validation, extraction, and indexing.

Retrieves full text for top candidate sources (arXiv, Europe PMC, OpenAlex, Wikipedia)
under strict security rules:
- HTTPS only
- Redirect validation: refuses redirects to non-HTTPS, private, or loopback IP addresses (anti-SSRF)
- 8 MB download size cap with stream monitoring
- 15-second timeout
- Content-type validation
- Polite per-host rate limiting
- 30-day SQLite caching in data/cache/fulltext
- Graceful error handling (failures skip source without crashing)
"""

import hashlib
import io
import ipaddress
import json
import logging
from pathlib import Path
import re
import socket
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET

import pdfplumber
import requests

from veritext.fingerprint.index import FingerprintIndex
from veritext.ingest.cleaner import remove_references
from veritext.sources.models import SourceCandidate
from veritext.sources.store import ReferenceStore, compute_content_hash

logger = logging.getLogger(__name__)

USER_AGENT = "Veritext/1.0 (college project; github.com/pankajt25/veritext)"
DEFAULT_TIMEOUT = 15.0
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024  # 8 MB cap
DEFAULT_CACHE_DAYS = 30
DEFAULT_CACHE_DIR = Path("data/cache/fulltext")

# Per-host minimum delay in seconds
HOST_RATE_LIMITS: Dict[str, float] = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "www.ebi.ac.uk": 1.0,
    "api.openalex.org": 1.0,
    "en.wikipedia.org": 1.0,
}


class SecurityError(Exception):
    """Raised when an HTTP request violates security policies (SSRF, non-https, size limit)."""
    pass


class FullTextCache:
    """SQLite-backed cache for retrieved and extracted full-text documents with 30-day expiry."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "fulltext_cache.sqlite"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fulltext_cache (
                    url TEXT PRIMARY KEY,
                    content_type TEXT,
                    extracted_text TEXT,
                    created_at REAL,
                    expires_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_fulltext_expires ON fulltext_cache(expires_at)"
            )

    def get(self, url: str) -> Optional[str]:
        now = time.time()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT extracted_text, expires_at FROM fulltext_cache WHERE url = ?",
                (url,),
            ).fetchone()
            if not row:
                return None
            text, expires_at = row
            if expires_at and expires_at < now:
                conn.execute("DELETE FROM fulltext_cache WHERE url = ?", (url,))
                return None
            return text

    def set(
        self,
        url: str,
        extracted_text: str,
        content_type: str = "text/plain",
        ttl_days: int = DEFAULT_CACHE_DAYS,
    ):
        now = time.time()
        expires_at = now + (ttl_days * 86400)
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fulltext_cache
                (url, content_type, extracted_text, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, content_type, extracted_text, now, expires_at),
            )


class PoliteSecureDownloader:
    """Secure, polite HTTP downloader preventing SSRF, oversize files, and unsafe redirects."""

    def __init__(
        self,
        user_agent: str = USER_AGENT,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_DOWNLOAD_BYTES,
        cache: Optional[FullTextCache] = None,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.cache = cache or FullTextCache()
        self.last_host_time: Dict[str, float] = {}

    def _validate_url_security(self, url: str):
        """Validates that URL uses HTTPS and does not resolve to private/loopback/reserved IPs."""
        parsed = urlparse(url)
        if parsed.scheme.lower() != "https":
            raise SecurityError(f"Non-HTTPS scheme refused: {parsed.scheme} in {url}")

        hostname = parsed.hostname
        if not hostname:
            raise SecurityError(f"Missing hostname in URL: {url}")

        # Check for direct IP address in hostname
        try:
            ip = ipaddress.ip_address(hostname)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                raise SecurityError(f"Direct connection to private/loopback IP refused: {hostname}")
        except ValueError:
            # Not an IP literal, resolve domain via DNS
            try:
                addr_info = socket.getaddrinfo(hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
                for entry in addr_info:
                    ip_str = entry[4][0]
                    ip = ipaddress.ip_address(ip_str)
                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_multicast
                        or ip.is_reserved
                        or ip.is_unspecified
                    ):
                        raise SecurityError(
                            f"Hostname '{hostname}' resolved to private/loopback IP {ip_str} (refused)"
                        )
            except socket.gaierror as e:
                raise SecurityError(f"DNS resolution failed for {hostname}: {e}")

    def _wait_rate_limit(self, hostname: str):
        host = hostname.lower()
        interval = HOST_RATE_LIMITS.get(host, 1.0)
        last = self.last_host_time.get(host, 0.0)
        elapsed = time.time() - last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self.last_host_time[host] = time.time()

    def fetch(
        self,
        url: str,
        expected_types: Optional[Tuple[str, ...]] = ("pdf", "xml", "html", "json", "plain"),
        max_redirects: int = 5,
    ) -> Tuple[bytes, str]:
        """Downloads content with redirect verification, size capping, and content-type validation.

        Returns:
            Tuple of (content_bytes, content_type_header)
        """
        current_url = url
        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent})

        for redirect_count in range(max_redirects + 1):
            self._validate_url_security(current_url)
            parsed = urlparse(current_url)
            self._wait_rate_limit(parsed.hostname or "")

            resp = session.get(
                current_url,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )

            # Handle redirects manually to validate each target URL
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    resp.close()
                    raise SecurityError(f"HTTP redirect {resp.status_code} missing Location header")
                next_url = urljoin(current_url, location)
                resp.close()
                self._validate_url_security(next_url)
                current_url = next_url
                continue

            if resp.status_code != 200:
                resp.close()
                raise ValueError(f"HTTP error {resp.status_code} fetching {current_url}")

            # Validate Content-Type
            content_type = resp.headers.get("Content-Type", "").lower()
            if expected_types:
                matched = any(et in content_type for et in expected_types)
                if not matched:
                    resp.close()
                    raise ValueError(
                        f"Unexpected Content-Type '{content_type}' for {current_url} (expected {expected_types})"
                    )

            # Check Content-Length if present
            cl_header = resp.headers.get("Content-Length")
            if cl_header and cl_header.isdigit():
                cl_val = int(cl_header)
                if cl_val > self.max_bytes:
                    resp.close()
                    raise SecurityError(f"Content-Length {cl_val} exceeds {self.max_bytes} limit")

            # Stream chunks with cumulative byte counting
            chunks = []
            downloaded = 0
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > self.max_bytes:
                        resp.close()
                        raise SecurityError(
                            f"Downloaded content exceeded {self.max_bytes} byte limit"
                        )
                    chunks.append(chunk)

            resp.close()
            return b"".join(chunks), content_type

        raise SecurityError(f"Exceeded maximum redirects ({max_redirects})")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts plain text from PDF bytes using pdfplumber."""
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t and t.strip():
                pages_text.append(t.strip())
    return "\n\n".join(pages_text)


def extract_text_from_europepmc_xml(xml_bytes: bytes) -> str:
    """Extracts full body text from Europe PMC JATS / NLM XML while stripping references."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse Europe PMC XML: {e}")
        return ""

    # Find <body> tag under <article>
    body = root.find(".//body")
    if body is None:
        # Fallback to entire article if body tag not explicitly separated
        body = root

    paragraphs = []
    # Strip any reference list elements inside body if present
    for ref_elem in list(body.findall(".//ref-list")) + list(body.findall(".//back")):
        try:
            body.remove(ref_elem)
        except Exception:
            pass

    # Extract title
    title_elem = root.find(".//article-title")
    if title_elem is not None and title_elem.text:
        paragraphs.append(title_elem.text.strip())

    # Extract text from sections and paragraphs
    for p in body.findall(".//p"):
        p_text = "".join(p.itertext()).strip()
        if p_text and len(p_text.split()) >= 5:
            paragraphs.append(p_text)

    return "\n\n".join(paragraphs)


def extract_text_from_wikipedia_json(json_bytes: bytes) -> str:
    """Extracts plain-text full page extract from Wikipedia API JSON response."""
    try:
        data = json.loads(json_bytes.decode("utf-8", errors="replace"))
        pages = data.get("query", {}).get("pages", {})
        texts = []
        for pid, pdata in pages.items():
            ext = pdata.get("extract", "").strip()
            if ext:
                texts.append(ext)
        return "\n\n".join(texts)
    except Exception as e:
        logger.warning(f"Error parsing Wikipedia full text JSON: {e}")
        return ""


def split_text_into_passages(text: str, target_word_count: int = 120) -> List[str]:
    """Splits full document text into passages for indexing and comparison."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages: List[str] = []
    curr_words: List[str] = []

    for p in paragraphs:
        words = p.split()
        if not words:
            continue
        if len(curr_words) + len(words) < target_word_count:
            curr_words.extend(words)
        else:
            if curr_words:
                passages.append(" ".join(curr_words))
                curr_words = []
            if len(words) > target_word_count * 2:
                # Break large paragraph into chunks
                for i in range(0, len(words), target_word_count):
                    chunk = words[i : i + target_word_count]
                    if len(chunk) >= 20:
                        passages.append(" ".join(chunk))
            else:
                curr_words.extend(words)

    if curr_words and len(curr_words) >= 15:
        passages.append(" ".join(curr_words))

    return passages if passages else ([text.strip()] if text.strip() else [])


class FullTextRetriever:
    """Coordinator for open-access full-text retrieval, caching, extraction, and indexing."""

    def __init__(
        self,
        downloader: Optional[PoliteSecureDownloader] = None,
        cache: Optional[FullTextCache] = None,
        store: Optional[ReferenceStore] = None,
        index: Optional[FingerprintIndex] = None,
        max_fetch_per_doc: int = 10,
    ):
        self.cache = cache or FullTextCache()
        self.downloader = downloader or PoliteSecureDownloader(cache=self.cache)
        self.store = store or ReferenceStore()
        self.index = index or FingerprintIndex()
        self.max_fetch = max_fetch_per_doc

    def _resolve_download_target(
        self, candidate: SourceCandidate
    ) -> Optional[Tuple[str, str, str]]:
        """Resolves full-text download URL, format ('pdf', 'xml', 'wiki_json'), and cache key.

        Returns:
            Tuple of (url, doc_format, cache_key) or None if no open-access source available.
        """
        origin = candidate.origin.lower()
        url = candidate.url or ""
        meta = candidate.metadata or {}

        # 1. arXiv
        arxiv_id = meta.get("arxiv_id")
        if not arxiv_id and "arxiv.org" in url:
            arxiv_id = url.split("/abs/")[-1].split("/pdf/")[-1].replace(".pdf", "")

        if arxiv_id:
            arxiv_url = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
            return arxiv_url, "pdf", f"arxiv:{arxiv_id}"

        # 2. Europe PMC
        pmcid = meta.get("pmcid")
        if not pmcid and "PMC" in url:
            m = re.search(r"PMC\d+", url)
            if m:
                pmcid = m.group(0)

        if pmcid:
            epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            return epmc_url, "xml", f"pmc:{pmcid}"

        # 3. OpenAlex Best OA PDF
        oa_pdf = meta.get("best_oa_pdf_url") or meta.get("oa_pdf_url")
        if not oa_pdf and origin == "openalex":
            oa_loc = meta.get("best_oa_location")
            if isinstance(oa_loc, dict):
                oa_pdf = oa_loc.get("pdf_url")
        if oa_pdf and oa_pdf.startswith("https://"):
            return oa_pdf, "pdf", oa_pdf

        # 4. Wikipedia Full Page Text
        if origin == "wikipedia" or "wikipedia.org" in url:
            pageid = meta.get("pageid")
            title = candidate.title
            if pageid:
                wiki_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&explaintext=1&pageids={pageid}&format=json"
                )
                return wiki_url, "wiki_json", f"wiki:{pageid}"
            elif title:
                import urllib.parse
                safe_title = urllib.parse.quote(title.replace(" ", "_"))
                wiki_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&explaintext=1&titles={safe_title}&format=json"
                )
                return wiki_url, "wiki_json", f"wiki:{title}"

        return None

    def fetch_candidate_fulltext(self, candidate: SourceCandidate) -> Optional[str]:
        """Fetches and extracts full text for a single candidate.

        Returns:
            Extracted full text string or None if unavailable/failed.
        """
        target = self._resolve_download_target(candidate)
        if not target:
            return None

        url, doc_format, cache_key = target

        # Check 30-day cache first
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            if doc_format == "pdf":
                raw_bytes, ctype = self.downloader.fetch(url, expected_types=("pdf", "octet-stream"))
                raw_text = extract_text_from_pdf(raw_bytes)
            elif doc_format == "xml":
                raw_bytes, ctype = self.downloader.fetch(url, expected_types=("xml", "text"))
                raw_text = extract_text_from_europepmc_xml(raw_bytes)
            elif doc_format == "wiki_json":
                raw_bytes, ctype = self.downloader.fetch(url, expected_types=("json", "text"))
                raw_text = extract_text_from_wikipedia_json(raw_bytes)
            else:
                return None

            if not raw_text or not raw_text.strip():
                return None

            # Strip references section from academic papers
            clean_full_text, _ = remove_references(raw_text)
            final_text = clean_full_text.strip() if clean_full_text.strip() else raw_text.strip()

            # Save in 30-day cache
            self.cache.set(cache_key, final_text, content_type=doc_format)
            return final_text

        except Exception as e:
            logger.warning(f"Full-text retrieval skipped for candidate '{candidate.title}' ({url}): {e}")
            return None

    def fetch_and_index_top_candidates(
        self,
        candidates: List[SourceCandidate],
        max_fetch: Optional[int] = None,
    ) -> Tuple[List[SourceCandidate], int]:
        """Fetches full text for top N candidates, splits into passages, and updates indices.

        Args:
            candidates: List of candidate SourceCandidate items (ranked).
            max_fetch: Maximum number of full texts to download (default: self.max_fetch).

        Returns:
            Tuple of (updated_candidates_list, count_of_fulltexts_fetched).
        """
        limit = max_fetch if max_fetch is not None else self.max_fetch
        fetched_count = 0
        updated_candidates: List[SourceCandidate] = []

        for idx, cand in enumerate(candidates):
            if fetched_count < limit:
                full_text = self.fetch_candidate_fulltext(cand)
                if full_text and len(full_text) > len(cand.text):
                    cand.text = full_text
                    cand.metadata["has_fulltext"] = True
                    fetched_count += 1

                    # Split full text into passages and index into store and winnowing index
                    passages = split_text_into_passages(full_text)
                    try:
                        self.store.add_source(cand, passages=passages)
                    except Exception as e:
                        logger.debug(f"Store indexing note for '{cand.id}': {e}")

                    for p_idx, p_text in enumerate(passages):
                        p_id = f"{cand.id}_full_p{p_idx}"
                        try:
                            self.index.index_text(cand.id, p_id, p_text)
                        except Exception as e:
                            logger.debug(f"Fingerprint indexing note: {e}")

            updated_candidates.append(cand)

        return updated_candidates, fetched_count
