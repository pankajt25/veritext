"""Query builder extracting key phrases using TF-IDF and n-grams from document paragraphs."""

import re
from typing import List, Optional, Set

from sklearn.feature_extraction.text import TfidfVectorizer

from veritext.ingest.models import Document, Paragraph
from veritext.sources.models import Query


DEFAULT_MIN_PARAGRAPH_CHARS = 50
DEFAULT_MIN_WORDS = 8
DEFAULT_MAX_QUERIES = 15


def extract_key_phrases(
    text: str, top_k: int = 5, min_n: int = 1, max_n: int = 3
) -> List[str]:
    """
    Extracts 3 to 5 salient key phrases from text using TF-IDF n-grams (unigrams, bigrams, trigrams).
    Stopwords are removed and alphabetic tokens with length >= 3 are prioritized.
    """
    if not text or len(text.strip()) < 20:
        return []

    # Split text into sentence/clause chunks first before removing punctuation
    raw_chunks = [c.strip() for c in re.split(r"[.\n;!?]+", text) if len(c.strip()) > 10]
    if not raw_chunks:
        raw_chunks = [text]
    chunks = [re.sub(r"[^\w\s-]", " ", c) for c in raw_chunks]

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(min_n, max_n),
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            max_features=100,
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across chunks
        scores = tfidf_matrix.sum(axis=0).A1
        scored_features = list(zip(feature_names, scores))

        # Prioritize 2-grams and 3-grams over unigrams for higher academic search specificity
        def specificity_boost(item):
            word, score = item
            num_words = len(word.split())
            boost = 1.0 + (0.6 * (num_words - 1))
            return score * boost

        scored_features.sort(key=specificity_boost, reverse=True)

        # De-duplicate overlapping phrases
        selected: List[str] = []
        for phrase, _ in scored_features:
            # Check if this phrase is a subset or superset of already selected phrases
            if any(phrase in s or s in phrase for s in selected):
                continue
            selected.append(phrase)
            if len(selected) >= top_k:
                break

        # Fallback if filtering was too aggressive
        if len(selected) < 3:
            for phrase, _ in scored_features:
                if phrase not in selected:
                    selected.append(phrase)
                if len(selected) >= top_k:
                    break

        return selected[:top_k]

    except (ValueError, ZeroDivisionError):
        # Fallback regex tokenization if vocabulary is empty
        tokens = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z]{4,}\b", text)
            if w.lower()
            not in {
                "this",
                "that",
                "with",
                "from",
                "were",
                "have",
                "their",
                "which",
                "there",
            }
        ]
        return list(dict.fromkeys(tokens))[:top_k]


def build_queries(
    document_or_paragraphs: Document | List[Paragraph] | List[str],
    max_queries: int = DEFAULT_MAX_QUERIES,
    min_paragraph_chars: int = DEFAULT_MIN_PARAGRAPH_CHARS,
    min_words: int = DEFAULT_MIN_WORDS,
) -> List[Query]:
    """
    Builds 1 to 3 search queries per substantial paragraph, capped at max_queries.
    Skips short or non-substantive paragraphs.
    """
    # Extract list of paragraph texts
    if isinstance(document_or_paragraphs, Document):
        paragraphs = [p.text for p in document_or_paragraphs.paragraphs]
    elif isinstance(document_or_paragraphs, list):
        paragraphs = [
            p.text if hasattr(p, "text") else str(p)
            for p in document_or_paragraphs
        ]
    else:
        paragraphs = [str(document_or_paragraphs)]

    queries: List[Query] = []
    seen_query_texts: Set[str] = set()

    for p_idx, p_text in enumerate(paragraphs):
        p_str = p_text.strip()
        words = p_str.split()

        # Skip very short paragraphs
        if len(p_str) < min_paragraph_chars or len(words) < min_words:
            continue

        key_phrases = extract_key_phrases(p_str, top_k=5)
        if not key_phrases:
            continue

        # Formulate 1 to 3 queries per paragraph
        # Query 1: Top 1-2 key phrases
        q1_text = key_phrases[0]
        if q1_text not in seen_query_texts:
            seen_query_texts.add(q1_text)
            queries.append(
                Query(
                    text=q1_text,
                    origin_paragraph_index=p_idx,
                    keywords=key_phrases[:2],
                    weight=1.0,
                )
            )

        # Query 2: Combined secondary phrases if available
        if len(key_phrases) >= 2 and len(queries) < max_queries:
            q2_text = key_phrases[1]
            if q2_text not in seen_query_texts:
                seen_query_texts.add(q2_text)
                queries.append(
                    Query(
                        text=q2_text,
                        origin_paragraph_index=p_idx,
                        keywords=[key_phrases[1]],
                        weight=0.8,
                    )
                )

        # Query 3: Multi-phrase combination for long paragraphs
        if len(words) >= 35 and len(key_phrases) >= 3 and len(queries) < max_queries:
            q3_text = f"{key_phrases[0]} {key_phrases[2]}"
            if q3_text not in seen_query_texts:
                seen_query_texts.add(q3_text)
                queries.append(
                    Query(
                        text=q3_text,
                        origin_paragraph_index=p_idx,
                        keywords=[key_phrases[0], key_phrases[2]],
                        weight=0.7,
                    )
                )

        if len(queries) >= max_queries:
            break

    return queries[:max_queries]
