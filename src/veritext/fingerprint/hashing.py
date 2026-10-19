"""Stable 64-bit hashing for text tokens and k-grams using BLAKE2b."""

import hashlib
from typing import Sequence, Union


def hash64(data: Union[str, bytes]) -> int:
    """
    Computes a deterministic 64-bit signed integer hash using BLAKE2b.
    Compatible with SQLite signed 64-bit INTEGER columns.
    """
    if isinstance(data, str):
        b = data.encode("utf-8")
    else:
        b = data

    digest = hashlib.blake2b(b, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def hash_kgram(tokens: Sequence[str]) -> int:
    """Computes a stable 64-bit hash for a sequence of tokens."""
    return hash64(" ".join(tokens))
