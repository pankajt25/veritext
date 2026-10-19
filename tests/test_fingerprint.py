"""Tests for k-grams, 64-bit hashing, winnowing mathematical guarantee, and inverted index."""

import pytest
from veritext.fingerprint.hashing import hash64, hash_kgram
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.kgrams import generate_kgrams, generate_kgram_strings
from veritext.fingerprint.winnow import winnow, winnow_hashes


def test_kgram_generation():
    tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    kgrams = generate_kgrams(tokens, k=5)
    assert len(kgrams) == len(tokens) - 5 + 1
    assert kgrams[0] == (("the", "quick", "brown", "fox", "jumps"), 0)
    assert kgrams[-1] == (("jumps", "over", "the", "lazy", "dog"), 4)

    # Empty and sub-k length handling
    assert generate_kgrams([], k=5) == []
    short_kgrams = generate_kgrams(["a", "b", "c"], k=5)
    assert len(short_kgrams) == 1
    assert short_kgrams[0] == (("a", "b", "c"), 0)


def test_hashing_stability_and_bounds():
    h1 = hash64("test phrase for hashing")
    h2 = hash64("test phrase for hashing")
    assert h1 == h2

    # Check 64-bit signed integer limits for SQLite INTEGER compatibility
    min_int64 = -(2**63)
    max_int64 = 2**63 - 1
    assert min_int64 <= h1 <= max_int64

    # Hash k-gram helper
    tokens = ["deep", "learning", "attention", "mechanism"]
    hk = hash_kgram(tokens)
    assert min_int64 <= hk <= max_int64


def test_winnowing_mathematical_guarantee():
    """
    Mathematical Guarantee:
    Any two texts sharing an identical run of at least (w + k - 1) tokens
    MUST share at least one identical winnowing fingerprint, regardless of surrounding context.
    """
    k = 5
    w = 4
    min_shared_length = w + k - 1  # 4 + 5 - 1 = 8 tokens

    shared_run = [f"shared_term_{i}" for i in range(min_shared_length)]

    doc_a = ["alpha", "beta", "gamma", "delta"] + shared_run + ["epsilon", "zeta"]
    doc_b = ["omega", "psi", "chi"] + shared_run + ["upsilon", "tau", "sigma", "rho"]

    fps_a = winnow(doc_a, k=k, w=w)
    fps_b = winnow(doc_b, k=k, w=w)

    hashes_a = {h for h, pos in fps_a}
    hashes_b = {h for h, pos in fps_b}

    common_hashes = hashes_a.intersection(hashes_b)
    assert len(common_hashes) >= 1, "Winnowing guarantee failed to yield at least 1 shared fingerprint!"


def test_winnowing_rightmost_tie_break():
    # If consecutive hashes are equal, rightmost tie-break chooses the latter position
    hash_positions = [(100, 0), (50, 1), (50, 2), (200, 3)]
    fps = winnow_hashes(hash_positions, w=2)
    # Window 0: [(100, 0), (50, 1)] -> min is (50, 1)
    # Window 1: [(50, 1), (50, 2)] -> min is equal (50), rightmost tie-break picks (50, 2)
    # Window 2: [(50, 2), (200, 3)] -> min is (50, 2)
    # Filtered changes: (50, 1), (50, 2)
    assert len(fps) == 2
    assert fps[0] == (50, 1)
    assert fps[1] == (50, 2)


def test_fingerprint_index_operations(tmp_path):
    db_file = tmp_path / "test_fp.sqlite"
    index = FingerprintIndex(db_file)

    fps = [(123456789, 0), (987654321, 5), (-456789123, 10)]
    index.add_fingerprints(source_id="src_1", passage_id="p1", fingerprints=fps)

    assert index.is_source_indexed("src_1") is True
    assert index.is_source_indexed("src_nonexistent") is False

    # Lookup hashes
    lookup = index.lookup_hashes([123456789, 999999999])
    assert 123456789 in lookup
    assert len(lookup[123456789]) == 1
    assert lookup[123456789][0]["source_id"] == "src_1"
    assert lookup[123456789][0]["position"] == 0

    assert 999999999 in lookup
    assert len(lookup[999999999]) == 0

    stats = index.stats()
    assert stats["total_fingerprints"] == 3
    assert stats["distinct_sources"] == 1

    # Clear
    index.clear()
    assert index.stats()["total_fingerprints"] == 0
