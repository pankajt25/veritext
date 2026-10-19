"""Word-level k-gram generator over normalized token sequences."""

from typing import Iterator, List, Sequence, Tuple


DEFAULT_K = 5


def generate_kgrams(
    tokens: Sequence[str], k: int = DEFAULT_K
) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Generates word-level k-grams over a sequence of normalized tokens.

    Args:
        tokens: Sequence of normalized tokens.
        k: Window size of tokens per k-gram (default: 5).

    Returns:
        List of (kgram_tuple, start_token_index) pairs.
    """
    if not tokens or k <= 0:
        return []

    n = len(tokens)
    if n < k:
        # If fewer than k tokens, emit the entire sequence as a single k-gram
        return [(tuple(tokens), 0)]

    kgrams: List[Tuple[Tuple[str, ...], int]] = []
    for i in range(n - k + 1):
        kgrams.append((tuple(tokens[i : i + k]), i))

    return kgrams


def generate_kgram_strings(
    tokens: Sequence[str], k: int = DEFAULT_K
) -> List[Tuple[str, int]]:
    """Generates space-separated k-gram strings and their starting token indices."""
    return [(" ".join(kg), pos) for kg, pos in generate_kgrams(tokens, k=k)]
