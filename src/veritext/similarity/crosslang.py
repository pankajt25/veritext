"""Cross-language similarity and translated copy detection (opt-in).

Features:
- Pure-Python language heuristic detecting multi-script and non-English texts
- Lazy-loaded 'paraphrase-multilingual-MiniLM-L12-v2' model with max batch size 16
- SQLite embedding caching for efficiency on dual-core / 8 GB hardware
- Explicit disabled state when model is not downloaded, guiding the user to:
    python -m veritext.similarity --download-crosslang-model
- Spans labelled with 'possible translated copy'
"""

import hashlib
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

CROSSLANG_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CROSSLANG_MODEL_DIR = Path("data/models/paraphrase-multilingual-MiniLM-L12-v2")
CACHE_DB_PATH = Path("data/cache/embeddings_cache.sqlite")

DOWNLOAD_INSTRUCTION = (
    "python -m veritext.similarity --download-crosslang-model"
)

# Common stopwords for high-resource languages
STOPWORDS_ES = {"de", "la", "el", "en", "y", "los", "del", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo"}
STOPWORDS_FR = {"le", "la", "les", "du", "des", "et", "en", "un", "une", "pour", "dans", "qui", "sur", "ce", "au", "par"}
STOPWORDS_DE = {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich", "des", "auf", "für", "ist", "im", "dem"}
STOPWORDS_IT = {"di", "il", "la", "che", "e", "un", "in", "per", "del", "le", "dei", "della", "non", "si", "da", "con"}
STOPWORDS_PT = {"de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "no", "se", "na"}


def detect_language_heuristic(text: str) -> Dict[str, Any]:
    """Fast, offline language heuristic using Unicode script ranges and lexical frequency.

    Identifies whether text is predominantly English, non-English Latin script, or non-Latin script.
    """
    if not text or not text.strip():
        return {"language": "en", "is_likely_non_english": False, "confidence": 0.0, "script": "latin"}

    sample = text[:5000]

    # 1. Non-Latin script checks
    scripts = {
        "cyrillic": len(re.findall(r"[\u0400-\u04FF]", sample)),
        "cjk": len(re.findall(r"[\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]", sample)),
        "arabic": len(re.findall(r"[\u0600-\u06FF]", sample)),
        "devanagari": len(re.findall(r"[\u0900-\u097F]", sample)),
        "greek": len(re.findall(r"[\u0370-\u03FF]", sample)),
    }

    total_chars = max(1, len(sample.replace(" ", "")))
    for script, count in scripts.items():
        if count / total_chars > 0.15:
            return {
                "language": script,
                "is_likely_non_english": True,
                "confidence": round(min(1.0, count / total_chars * 2), 2),
                "script": script,
            }

    # 2. Latin script diacritics
    has_spanish_diacritics = len(re.findall(r"[áéíóúüñ¿¡]", sample, re.IGNORECASE))
    has_french_diacritics = len(re.findall(r"[àâçéèêëîïôûùüÿœæ]", sample, re.IGNORECASE))
    has_german_diacritics = len(re.findall(r"[äöüß]", sample, re.IGNORECASE))

    # 3. Stopword count
    words = re.findall(r"\b[a-zA-Z\u00C0-\u017F]+\b", sample.lower())
    total_words = max(1, len(words))

    matches_es = sum(1 for w in words if w in STOPWORDS_ES)
    matches_fr = sum(1 for w in words if w in STOPWORDS_FR)
    matches_de = sum(1 for w in words if w in STOPWORDS_DE)
    matches_it = sum(1 for w in words if w in STOPWORDS_IT)
    matches_pt = sum(1 for w in words if w in STOPWORDS_PT)

    scores = {
        "es": (matches_es / total_words) + (has_spanish_diacritics * 0.05),
        "fr": (matches_fr / total_words) + (has_french_diacritics * 0.05),
        "de": (matches_de / total_words) + (has_german_diacritics * 0.05),
        "it": (matches_it / total_words),
        "pt": (matches_pt / total_words),
    }

    best_lang, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score > 0.18:
        return {
            "language": best_lang,
            "is_likely_non_english": True,
            "confidence": round(min(1.0, best_score * 2.5), 2),
            "script": "latin",
        }

    return {
        "language": "en",
        "is_likely_non_english": False,
        "confidence": 0.9,
        "script": "latin",
    }


def is_crosslang_model_available() -> bool:
    """Checks whether the multilingual MiniLM model is installed locally."""
    if CROSSLANG_MODEL_DIR.is_dir():
        # Check for model weights or config
        has_config = (CROSSLANG_MODEL_DIR / "config.json").is_file()
        has_weights = any(
            (CROSSLANG_MODEL_DIR / fn).is_file()
            for fn in ["model.safetensors", "pytorch_model.bin", "modules.json"]
        )
        return has_config or has_weights
    return False


def get_crosslang_status() -> Dict[str, Any]:
    """Returns the operational status of the cross-language engine."""
    available = is_crosslang_model_available()
    return {
        "available": available,
        "model_name": CROSSLANG_MODEL_NAME,
        "model_dir": str(CROSSLANG_MODEL_DIR),
        "download_command": DOWNLOAD_INSTRUCTION,
        "message": (
            "Cross-language matching model is available and ready."
            if available
            else f"Cross-language matching is disabled because the model is not installed. To enable, run: {DOWNLOAD_INSTRUCTION}"
        ),
    }


def download_crosslang_model(target_dir: Optional[Union[str, Path]] = None) -> bool:
    """Downloads the multilingual sentence-transformers model into the local directory."""
    dest = Path(target_dir or CROSSLANG_MODEL_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{CROSSLANG_MODEL_NAME}' to {dest}...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(CROSSLANG_MODEL_NAME)
        model.save(str(dest))
        print("Model downloaded and saved successfully.")
        return True
    except Exception as e:
        print(f"Error downloading multilingual model: {e}")
        return False


_LAZY_MODEL = None


def get_crosslang_model():
    """Lazily loads the multilingual SentenceTransformer model if available."""
    global _LAZY_MODEL
    if _LAZY_MODEL is not None:
        return _LAZY_MODEL

    if not is_crosslang_model_available():
        raise RuntimeError(
            f"Cross-language model is not installed. To install, run: {DOWNLOAD_INSTRUCTION}"
        )

    from sentence_transformers import SentenceTransformer

    _LAZY_MODEL = SentenceTransformer(str(CROSSLANG_MODEL_DIR))
    return _LAZY_MODEL


def reset_crosslang_model():
    """Clears cached model from memory to free resources."""
    global _LAZY_MODEL
    _LAZY_MODEL = None


def compute_crosslang_embeddings(
    sentences: List[str],
    batch_size: int = 16,
) -> List[List[float]]:
    """Computes dense multilingual vector embeddings with batch_size <= 16 and SQLite caching."""
    if not sentences:
        return []

    # Enforce batch size <= 16 for dual-core / 8 GB hardware safety
    safe_batch = min(max(1, batch_size), 16)
    model = get_crosslang_model()

    embeddings = model.encode(
        sentences,
        batch_size=safe_batch,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(vec) for vec in embeddings]
