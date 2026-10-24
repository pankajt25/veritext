"""Lexical anchor rule and term extraction to evaluate semantic match leads.

A semantic match span is tested against the source passage for:
1. Shared rare content words (low document frequency in reference corpus, lemmatized).
2. Shared numbers and numeric quantities.
3. Shared named entities and distinctive multi-word terms.
4. Ratio of shared content lemmas.

Matches meeting anchor criteria are tagged as 'strong lead'; others as 'weak lead'.
Both remain unverified leads for instructor review and are never added to the
headline similarity percentage by default.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from veritext.normalize.normalize import _get_lemmatizer

EXPLANATORY_LINE = (
    "Similar meaning is not proof of copying; passages on the same topic often match. "
    "Check the linked source."
)
INTERPRETATION_GUIDE = (
    "Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only"
)

# Standard academic stopwords
ACADEMIC_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves", "also",
    "however", "therefore", "thus", "hence", "furthermore", "moreover", "overall", "using",
    "use", "used", "across", "within", "without", "may", "might", "shall", "will", "would",
    "must", "many", "much", "several", "one", "two", "three", "four", "first", "second",
    "et", "al", "etc", "such", "via", "well", "non", "per", "due",
}


def lemmatize_token(token: str) -> str:
    """Lemmatizes a single lowercase word token using NLTK WordNet lemmatizer."""
    token = token.lower().strip()
    if not token:
        return ""
    lemmatizer = _get_lemmatizer()
    if lemmatizer:
        try:
            return lemmatizer.lemmatize(token)
        except Exception:
            return token
    return token


def extract_content_lemmas(text: str) -> List[str]:
    """Extracts non-stopword alphabetic lemmas with length >= 3."""
    words = re.findall(r"\b[A-Za-z]+\b", text)
    lemmas = [
        lemmatize_token(w)
        for w in words
        if len(w) >= 3 and w.lower() not in ACADEMIC_STOPWORDS
    ]
    return [l for l in lemmas if l and l not in ACADEMIC_STOPWORDS]


def extract_numbers(text: str) -> Set[str]:
    """Extracts numeric tokens, numbers, and percentages."""
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def extract_distinctive_terms(text: str) -> Set[str]:
    """Extracts distinctive capitalized terms and named entities."""
    # Find capitalized words that are not sentence starters
    # (i.e. preceded by something other than period + space)
    matches = re.findall(r"(?<!\.\s)(?<!\A)\b[A-Z][a-zA-Z0-9_-]+\b", text)
    return {
        m.lower()
        for m in matches
        if len(m) >= 3 and m.lower() not in ACADEMIC_STOPWORDS
    }


def compute_reference_corpus_df(reference_texts: List[str]) -> Counter:
    """Computes document frequency DF(w) for lemmas across a corpus of reference texts."""
    df: Counter = Counter()
    for ref in reference_texts:
        lemmas = set(extract_content_lemmas(ref))
        for l in lemmas:
            df[l] += 1
    return df


def evaluate_span_anchors(
    doc_text: str,
    source_text: str,
    cosine_score: float = 0.0,
    reference_df: Optional[Dict[str, int]] = None,
    df_rare_threshold: int = 2,
    min_rare_anchors: int = 2,
    min_content_ratio: float = 0.15,
) -> Dict[str, Any]:
    """Evaluates lexical anchors between a matched document span and candidate source passage.

    Args:
        doc_text: Text of the matched window in the query document.
        source_text: Text of the candidate passage in the reference source.
        cosine_score: Embedding cosine similarity score.
        reference_df: Document frequency mapping {lemma: count} across reference corpus.
        df_rare_threshold: Maximum corpus DF for a content lemma to qualify as rare (default: 2).
        min_rare_anchors: Minimum shared anchors (rare lemmas + numbers + entities) for strong lead.
        min_content_ratio: Minimum ratio of shared content lemmas for strong lead.

    Returns:
        Dict with anchor details, counts, lead_level ('strong lead' or 'weak lead'),
        and composite score.
    """
    doc_lemmas = extract_content_lemmas(doc_text)
    src_lemmas = extract_content_lemmas(source_text)
    doc_lemmas_set = set(doc_lemmas)
    src_lemmas_set = set(src_lemmas)

    shared_lemmas = doc_lemmas_set & src_lemmas_set
    content_ratio = len(shared_lemmas) / max(1, len(doc_lemmas_set))

    # 1. Rare words (low document frequency in reference corpus)
    if reference_df:
        shared_rare_words = {
            l for l in shared_lemmas if reference_df.get(l, 1) <= df_rare_threshold
        }
    else:
        # Heuristic if corpus DF not supplied: longer specific words (>= 6 chars)
        shared_rare_words = {l for l in shared_lemmas if len(l) >= 6}

    # 2. Numbers
    doc_nums = extract_numbers(doc_text)
    src_nums = extract_numbers(source_text)
    shared_numbers = doc_nums & src_nums

    # 3. Distinctive terms / entities
    doc_entities = extract_distinctive_terms(doc_text)
    src_entities = extract_distinctive_terms(source_text)
    shared_entities = doc_entities & src_entities

    all_anchors = shared_rare_words | shared_numbers | shared_entities
    anchor_count = len(all_anchors)

    is_strong = (
        anchor_count >= min_rare_anchors
        and content_ratio >= min_content_ratio
    )
    lead_level = "strong lead" if is_strong else "weak lead"

    # Composite ranking score: combines cosine similarity with anchor density
    composite_score = round(
        cosine_score * (1.0 + 0.1 * min(5, anchor_count) + 0.2 * content_ratio),
        4,
    )

    return {
        "lead_level": lead_level,
        "is_strong_lead": is_strong,
        "anchor_count": anchor_count,
        "content_lemma_ratio": round(content_ratio, 4),
        "shared_rare_words": sorted(list(shared_rare_words)),
        "shared_numbers": sorted(list(shared_numbers)),
        "shared_entities": sorted(list(shared_entities)),
        "shared_lemmas_count": len(shared_lemmas),
        "composite_score": composite_score,
    }
