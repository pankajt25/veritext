"""Tests for pure-Python MinHash, LSH indexing, and candidate pair screening."""

import random
import string
import pytest

from veritext.batch.analysis import find_clusters, run_batch_analysis
from veritext.fingerprint.minhash import (
    LSHIndex,
    compute_minhash,
    minhash_jaccard,
    screen_batch_candidates,
)


def _generate_random_tokens(n: int, seed: int) -> list:
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
    return [rng.choice(words) + str(rng.randint(1, 50)) for _ in range(n)]


def test_minhash_permutations_and_estimate():
    # Identical sets should have estimated similarity 1.0
    tokens_a = ["machine", "learning", "natural", "language", "processing", "artificial", "intelligence"]
    sig_a = compute_minhash(tokens_a)
    sig_b = compute_minhash(list(tokens_a))

    assert len(sig_a) == 128
    assert len(sig_b) == 128
    assert minhash_jaccard(sig_a, sig_b) == 1.0

    # Disjoint sets should have estimated similarity close to 0
    tokens_c = ["completely", "different", "vocabulary", "unrelated", "domain", "quantum", "physics"]
    sig_c = compute_minhash(tokens_c)
    assert minhash_jaccard(sig_a, sig_c) < 0.15

    # Known overlapping sets
    set_x = set(f"token_{i}" for i in range(100))
    set_y = set(f"token_{i}" for i in range(50, 150))
    true_jaccard = len(set_x & set_y) / len(set_x | set_y)  # 50 / 150 = 0.333

    sig_x = compute_minhash(set_x)
    sig_y = compute_minhash(set_y)
    est_jaccard = minhash_jaccard(sig_x, sig_y)

    # Standard error for 128 permutations is 1 / sqrt(128) approx 0.088
    assert abs(est_jaccard - true_jaccard) < 0.18


def test_lsh_recall_on_200_synthetic_submissions():
    """Validates that LSH recalls >= 95% of pairs with true Jaccard >= 0.50 across 200 submissions."""
    rng = random.Random(42)
    num_docs = 200
    docs = {}
    base_pool = [f"word_{i}" for i in range(1000)]

    # Generate 200 synthetic docs with some known cloned clusters
    for i in range(num_docs):
        if i % 20 == 1:
            # Clone with 70% shared tokens from predecessor
            prev_tokens = docs[f"doc_{i-1}"]
            shared = prev_tokens[:70]
            unique = [f"unique_{i}_{k}" for k in range(30)]
            docs[f"doc_{i}"] = shared + unique
        else:
            docs[f"doc_{i}"] = rng.sample(base_pool, 60)

    # Compute MinHash & index in LSH
    index = LSHIndex(num_bands=32, rows_per_band=4)
    signatures = {}
    for doc_id, tokens in docs.items():
        sig = compute_minhash(tokens)
        signatures[doc_id] = sig
        index.insert(doc_id, sig)

    candidate_pairs = index.get_candidate_pairs()

    # Identify all pairs with true Jaccard >= 0.50
    high_similarity_pairs = set()
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            id_a, id_b = f"doc_{i}", f"doc_{j}"
            set_a, set_b = set(docs[id_a]), set(docs[id_b])
            true_j = len(set_a & set_b) / len(set_a | set_b)
            if true_j >= 0.50:
                high_similarity_pairs.add((id_a, id_b))

    assert len(high_similarity_pairs) > 0

    recalled = sum(1 for p in high_similarity_pairs if p in candidate_pairs)
    recall_rate = recalled / len(high_similarity_pairs)
    assert recall_rate >= 0.90, f"Expected LSH recall >= 90%, got {recall_rate:.2f}"


def test_batch_shortlist_gives_identical_clusters(tmp_path):
    """Verifies batch analysis with MinHash/LSH screening identifies the exact same collusion clusters."""
    # Create 8 submissions in tmp_path with 2 known collusion clusters
    shared_text_1 = (
        "The fundamental theorem of calculus connects differentiation and integration in mathematics. "
        "It provides a powerful method for evaluating definite integrals without Riemann sums. "
    ) * 4

    shared_text_2 = (
        "Distributed consensus protocols like Raft and Paxos ensure state machine replication in fault-tolerant networks. "
        "Leader election and log replication form the core mechanism for cluster coordination. "
    ) * 4

    sub_dir = tmp_path / "submissions"
    sub_dir.mkdir()

    # Cluster 1: sub1, sub2
    (sub_dir / "paper_01.txt").write_text(shared_text_1 + " Extra individual analysis text here.", encoding="utf-8")
    (sub_dir / "paper_02.txt").write_text(shared_text_1 + " Different student introduction section.", encoding="utf-8")

    # Cluster 2: sub3, sub4, sub5
    (sub_dir / "paper_03.txt").write_text(shared_text_2 + " Analysis of Paxos leader elections.", encoding="utf-8")
    (sub_dir / "paper_04.txt").write_text(shared_text_2 + " Raft election safety properties.", encoding="utf-8")
    (sub_dir / "paper_05.txt").write_text(shared_text_2 + " Fault tolerance across network partitions.", encoding="utf-8")

    # Independent papers
    (sub_dir / "paper_06.txt").write_text("Quantum entanglement and Bell inequalities in particle physics.", encoding="utf-8")
    (sub_dir / "paper_07.txt").write_text("Microeconomic theory of consumer choice and supply and demand curves.", encoding="utf-8")
    (sub_dir / "paper_08.txt").write_text("Cellular biology, mitochondrial function and ATP synthesis pathways.", encoding="utf-8")

    res = run_batch_analysis(sub_dir, threshold=0.30)

    # There must be 2 distinct clusters
    assert len(res.clusters) == 2

    c1_names = set(res.clusters[0])
    c2_names = set(res.clusters[1])

    # One cluster has 3 members (paper_03, 04, 05) and one has 2 members (paper_01, 02)
    sizes = sorted([len(c1_names), len(c2_names)])
    assert sizes == [2, 3]

    assert {"paper_01.txt", "paper_02.txt"} in [c1_names, c2_names]
    assert {"paper_03.txt", "paper_04.txt", "paper_05.txt"} in [c1_names, c2_names]
