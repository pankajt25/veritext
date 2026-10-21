"""Text-spinner and tortured-phrase detection.

Detects odd synonym substitutions and text-spinning indicators based on:
1. Documented tortured phrases (Cabanac, Labbe and Magazinov, 2021, 'Tortured phrases')
2. Semantic spans with high similarity but low exact overlap and rare WordNet synonym substitutions.

All findings have severity at most 'warning' and the standardized message:
'possible automated paraphrasing; review manually'
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from veritext.checks.models import CheckResult, Finding
from veritext.ingest.models import Document
from veritext.similarity.models import SimilarityResult

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "config"
    / "tortured_phrases.json"
)

# Common academic stopwords to ignore when analyzing WordNet substitutions
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "won't", "would", "you", "your", "yours", "yourself", "yourselves"
}


def load_tortured_phrases(
    custom_file: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Loads well-documented tortured phrases from configuration and optional custom file."""
    phrases: Dict[str, str] = {}

    if DEFAULT_CONFIG_PATH.exists():
        try:
            data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
            phrases.update(data.get("phrases", {}))
        except Exception as e:
            logger.warning(f"Error loading built-in tortured phrases: {e}")

    if custom_file:
        custom_path = Path(custom_file)
        if custom_path.exists():
            try:
                data = json.loads(custom_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    custom_dict = data.get("phrases", data)
                    if isinstance(custom_dict, dict):
                        phrases.update(custom_dict)
            except Exception as e:
                logger.warning(f"Error loading custom tortured phrases from {custom_path}: {e}")

    return phrases


def _get_wordnet():
    """Safely attempts to import WordNet without raising uncaught exceptions."""
    try:
        from nltk.corpus import wordnet
        # Quick validation
        _ = wordnet.synsets("test")
        return wordnet
    except Exception:
        return None


def _get_synsets_and_similars(word: str, wn) -> Set[Any]:
    base = wn.morphy(word) or word
    syns = set(wn.synsets(base))
    if not syns and base != word:
        syns = set(wn.synsets(word))
    expanded = set(syns)
    for s in syns:
        try:
            expanded.update(s.similar_tos())
        except Exception:
            pass
        try:
            expanded.update(s.also_sees())
        except Exception:
            pass
    return expanded


def _find_wordnet_substitutions(
    doc_words: List[str], source_words: List[str], wn
) -> List[Dict[str, Any]]:
    """Identifies uncommon WordNet synonym substitutions between document words and source words."""
    if not wn:
        return []

    set_doc = {w.lower() for w in doc_words if len(w) >= 4 and w.lower() not in STOPWORDS and w.isalpha()}
    set_src = {w.lower() for w in source_words if len(w) >= 4 and w.lower() not in STOPWORDS and w.isalpha()}

    non_overlapping_doc = set_doc - set_src
    non_overlapping_src = set_src - set_doc

    substitutions: List[Dict[str, Any]] = []

    for w_doc in non_overlapping_doc:
        doc_synsets = _get_synsets_and_similars(w_doc, wn)
        if not doc_synsets:
            continue

        for w_src in non_overlapping_src:
            src_synsets = _get_synsets_and_similars(w_src, wn)
            shared = doc_synsets.intersection(src_synsets)
            if shared:
                substitutions.append({
                    "doc_word": w_doc,
                    "source_word": w_src,
                    "synset": list(shared)[0].name(),
                })

    return substitutions


def check_spinner(
    document: Document,
    similarity_result: Optional[SimilarityResult] = None,
    custom_file: Optional[Union[str, Path]] = None,
) -> CheckResult:
    """Checks document for text-spinning and tortured-phrase anomalies.

    Args:
        document: Ingested Document instance.
        similarity_result: Optional SimilarityResult from plagiarism check.
        custom_file: Optional path to JSON file with additional tortured phrases.

    Returns:
        CheckResult with severity 'warning' at most and message:
        'possible automated paraphrasing; review manually'
    """
    findings: List[Finding] = []
    text = document.clean_text or document.raw_text or ""
    phrases = load_tortured_phrases(custom_file=custom_file)

    # 1. Direct Tortured Phrases matching
    for phrase, standard_term in phrases.items():
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            findings.append(
                Finding(
                    check="spinner",
                    severity="warning",
                    message="possible automated paraphrasing; review manually",
                    start=m.start(),
                    end=m.end(),
                    details={
                        "type": "tortured_phrase",
                        "found_phrase": m.group(0),
                        "standard_term": standard_term,
                    },
                )
            )

    # 2. Semantic Paraphrase with rare WordNet synonym substitutions
    if similarity_result and similarity_result.spans:
        wn = _get_wordnet()
        for span in similarity_result.spans:
            # Check high semantic similarity (>= 0.70)
            if span.score < 0.70:
                continue

            doc_text = span.matched_text.strip()
            src_text = (span.source_text or "").strip()
            if not doc_text or not src_text:
                continue

            doc_words = re.findall(r"\b[a-zA-Z]{3,}\b", doc_text)
            src_words = re.findall(r"\b[a-zA-Z]{3,}\b", src_text)

            if not doc_words or not src_words:
                continue

            s_doc = set(w.lower() for w in doc_words)
            s_src = set(w.lower() for w in src_words)
            overlap = len(s_doc.intersection(s_src)) / max(1, len(s_doc.union(s_src)))

            # If exact lexical overlap is lower (< 0.75) despite high similarity
            if overlap < 0.75 and s_doc != s_src:
                subs = _find_wordnet_substitutions(doc_words, src_words, wn)
                if subs:
                    findings.append(
                        Finding(
                            check="spinner",
                            severity="warning",
                            message="possible automated paraphrasing; review manually",
                            start=span.doc_start,
                            end=span.doc_end,
                            details={
                                "type": "semantic_synonym_substitution",
                                "similarity_score": round(span.score, 3),
                                "lexical_overlap": round(overlap, 3),
                                "source_title": span.source_title,
                                "substitutions": subs[:5],
                            },
                        )
                    )

    # Sort findings by position
    findings.sort(key=lambda f: (f.start, f.end))

    passed = len(findings) == 0
    if passed:
        summary = "No tortured phrases or automated synonym substitutions detected."
    else:
        summary = (
            f"Flagged {len(findings)} possible automated paraphrasing / text-spinning indicator(s)."
        )

    return CheckResult(
        check_name="spinner",
        findings=findings,
        summary=summary,
        passed=passed,
        metadata={"findings_count": len(findings)},
    )
