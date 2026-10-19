"""Winnowing algorithm for robust local document fingerprinting.

Mathematical Guarantee:
In any shared text run of length at least (w + k - 1) tokens, there are at least w consecutive
k-grams. A sliding window of size w is therefore completely contained within that shared run.
Because the winnowing algorithm selects the minimum hash in every window of size w, the two
documents are mathematically guaranteed to select at least one identical fingerprint!
"""

from typing import List, Sequence, Tuple

from veritext.fingerprint.hashing import hash_kgram
from veritext.fingerprint.kgrams import DEFAULT_K, generate_kgrams

DEFAULT_W = 4


def winnow_hashes(
    hash_positions: Sequence[Tuple[int, int]], w: int = DEFAULT_W
) -> List[Tuple[int, int]]:
    """
    Applies the winnowing algorithm to a sequence of (hash, token_position) pairs.

    In each sliding window of size w:
    - Selects the minimum hash value.
    - Tie-breaking: If multiple positions share the minimum hash, selects the rightmost occurrence.
    - Emits a fingerprint only when the selected minimum changes (or at the first window).

    Args:
        hash_positions: List of (hash_val, token_pos) pairs.
        w: Window size (default: 4).

    Returns:
        List of selected (hash, token_position) fingerprints.
    """
    if not hash_positions or w <= 0:
        return []

    n = len(hash_positions)
    fingerprints: List[Tuple[int, int]] = []

    # If sequence is shorter than or equal to window size, pick global minimum (rightmost)
    if n <= w:
        min_idx = 0
        for i in range(1, n):
            # Rightmost tie-break: <= selects the higher index if values are equal
            if hash_positions[i][0] <= hash_positions[min_idx][0]:
                min_idx = i
        return [hash_positions[min_idx]]

    prev_selected: Tuple[int, int] = (0, -1)

    # Slide window of size w across hash sequence
    for i in range(n - w + 1):
        window = hash_positions[i : i + w]

        # Find minimum in window with rightmost tie-break
        min_in_window = window[0]
        for item in window[1:]:
            # item[0] is hash, item[1] is token_position
            if item[0] <= min_in_window[0]:
                min_in_window = item

        if min_in_window != prev_selected:
            fingerprints.append(min_in_window)
            prev_selected = min_in_window

    return fingerprints


def winnow(
    tokens: Sequence[str], k: int = DEFAULT_K, w: int = DEFAULT_W
) -> List[Tuple[int, int]]:
    """
    High-level winnowing entrypoint: converts token sequence into word-level k-grams,
    hashes them with 64-bit BLAKE2b, and applies winnowing.

    Guarantees that any identical token sequence of length >= (w + k - 1) will share
    at least one fingerprint.

    Args:
        tokens: Sequence of normalized tokens.
        k: k-gram token window size (default: 5).
        w: winnowing window size (default: 4).

    Returns:
        List of (hash, token_position) fingerprints.
    """
    if not tokens:
        return []

    kgrams = generate_kgrams(tokens, k=k)
    hash_positions = [(hash_kgram(kg), pos) for kg, pos in kgrams]
    return winnow_hashes(hash_positions, w=w)
