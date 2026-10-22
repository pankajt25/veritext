"""Pure-Python MinHash and Locality-Sensitive Hashing (LSH) for high-scale pair screening.

Features:
- 128 deterministic permutations generated via blake2b
- Locality-Sensitive Hashing (LSH) using 32 bands of 4 rows (32 * 4 = 128)
- Blazingly fast pure-Python implementation without third-party dependencies
- Winnowing serves as the final ground-truth judge for shortlisted candidate pairs
"""

from collections import defaultdict
import hashlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

NUM_PERMUTATIONS = 128
NUM_BANDS = 32
ROWS_PER_BAND = 4
MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1 Mersenne prime for Carter-Wegman hashing

# Deterministically derive 128 pairs of hash coefficients (a_i, b_i) using blake2b
_A_COEFFS: List[int] = []
_B_COEFFS: List[int] = []

for _i in range(NUM_PERMUTATIONS):
    _ha = hashlib.blake2b(f"veritext_minhash_coeff_a_{_i}".encode("ascii"), digest_size=8).digest()
    _hb = hashlib.blake2b(f"veritext_minhash_coeff_b_{_i}".encode("ascii"), digest_size=8).digest()
    _a = (int.from_bytes(_ha, "big") % (MERSENNE_PRIME - 1)) + 1
    _b = int.from_bytes(_hb, "big") % MERSENNE_PRIME
    _A_COEFFS.append(_a)
    _B_COEFFS.append(_b)


def hash_token_64(token: Union[str, int]) -> int:
    """Converts a string token or integer into a 64-bit integer hash using blake2b."""
    if isinstance(token, int):
        # Already integer; mix with blake2b for uniform distribution
        return token & 0xFFFFFFFFFFFFFFFF
    b = token.encode("utf-8") if isinstance(token, str) else str(token).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


def compute_minhash(
    tokens: Iterable[Union[str, int]],
    num_perm: int = NUM_PERMUTATIONS,
) -> List[int]:
    """Computes a 128-element MinHash signature over an iterable of tokens or hashes.

    Uses pure Python integer arithmetic modulo Mersenne prime 2^61 - 1.
    """
    token_hashes = [hash_token_64(t) for t in tokens]
    if not token_hashes:
        return [0] * num_perm

    sig = [MERSENNE_PRIME] * num_perm
    p = MERSENNE_PRIME
    a_coeffs = _A_COEFFS[:num_perm]
    b_coeffs = _B_COEFFS[:num_perm]

    # Vectorized loop over unique token hashes
    unique_hashes = set(token_hashes)
    for x in unique_hashes:
        for i in range(num_perm):
            h = (a_coeffs[i] * x + b_coeffs[i]) % p
            if h < sig[i]:
                sig[i] = h

    return sig


def minhash_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimates the Jaccard similarity between two MinHash signatures."""
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


class LSHIndex:
    """Locality-Sensitive Hashing (LSH) index for MinHash signatures.

    Configured with 32 bands of 4 rows (total 128 permutations).
    A collision in ANY band designates a candidate pair for subsequent winnowing verification.
    """

    def __init__(
        self,
        num_bands: int = NUM_BANDS,
        rows_per_band: int = ROWS_PER_BAND,
    ):
        if num_bands * rows_per_band > NUM_PERMUTATIONS:
            raise ValueError(
                f"num_bands * rows_per_band ({num_bands * rows_per_band}) exceeds NUM_PERMUTATIONS ({NUM_PERMUTATIONS})"
            )
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        # Table per band: band_idx -> bucket_hash -> set(doc_ids)
        self.tables: List[Dict[int, Set[str]]] = [defaultdict(set) for _ in range(num_bands)]
        self.signatures: Dict[str, List[int]] = {}

    def _bucket_hash(self, band_slice: Tuple[int, ...]) -> int:
        """Derives a fast 64-bit bucket key from a 4-row slice."""
        b = b"".join(x.to_bytes(8, "big") for x in band_slice)
        return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")

    def insert(self, doc_id: str, signature: List[int]) -> None:
        """Inserts a document's MinHash signature into the LSH index."""
        self.signatures[doc_id] = signature
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket = self._bucket_hash(band_slice)
            self.tables[b_idx][bucket].add(doc_id)

    def query(self, signature: List[int]) -> Set[str]:
        """Finds all indexed document IDs that collide in at least one band with signature."""
        candidates: Set[str] = set()
        for b_idx in range(self.num_bands):
            start = b_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket = self._bucket_hash(band_slice)
            bucket_docs = self.tables[b_idx].get(bucket)
            if bucket_docs:
                candidates.update(bucket_docs)
        return candidates

    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """Returns all shortlisted unique candidate pairs (doc_a, doc_b) with doc_a < doc_b."""
        candidate_pairs: Set[Tuple[str, str]] = set()
        for table in self.tables:
            for bucket_docs in table.values():
                if len(bucket_docs) > 1:
                    sorted_docs = sorted(bucket_docs)
                    for i in range(len(sorted_docs)):
                        for j in range(i + 1, len(sorted_docs)):
                            candidate_pairs.add((sorted_docs[i], sorted_docs[j]))
        return candidate_pairs


def screen_batch_candidates(
    documents: Dict[str, Iterable[Union[str, int]]],
) -> Tuple[Set[Tuple[str, str]], Dict[str, List[int]]]:
    """Generates MinHash signatures and extracts candidate pairs via LSH.

    Returns:
        (candidate_pairs, signatures_map)
    """
    index = LSHIndex()
    signatures: Dict[str, List[int]] = {}

    for doc_id, tokens in documents.items():
        sig = compute_minhash(tokens)
        signatures[doc_id] = sig
        index.insert(doc_id, sig)

    return index.get_candidate_pairs(), signatures
