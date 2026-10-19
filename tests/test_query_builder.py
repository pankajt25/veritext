"""Unit tests for query extraction and formulation."""

from veritext.ingest.models import Document, Paragraph, Sentence
from veritext.sources.query_builder import build_queries, extract_key_phrases


def test_extract_key_phrases_academic_text():
    text = (
        "Adversarial perturbations against neural sequence models can degrade "
        "plagiarism detection accuracy. By introducing subtle token substitutions, "
        "adversaries evade detection systems without altering semantic intent."
    )
    phrases = extract_key_phrases(text, top_k=4)
    assert len(phrases) >= 2
    # Ensure phrases contain relevant domain tokens
    combined = " ".join(phrases)
    assert any(
        w in combined
        for w in [
            "adversarial",
            "neural",
            "plagiarism",
            "detection",
            "token",
            "models",
        ]
    )


def test_build_queries_from_document():
    p1 = (
        "Natural language processing has seen major advancements with transformer "
        "architectures. These models demonstrate state of the art results across benchmarks."
    )
    p2 = "Short text."  # Should be skipped because it's too short
    p3 = (
        "Algorithmic verification relies upon high dimensional vector embeddings. "
        "Cosine similarity measures the angular difference between sentence representations."
    )

    doc = Document(
        raw_text=f"{p1}\n\n{p2}\n\n{p3}",
        clean_text=f"{p1}\n\n{p2}\n\n{p3}",
        sentences=[Sentence(text=p1, start=0, end=len(p1))],
        paragraphs=[
            Paragraph(text=p1, start=0, end=len(p1)),
            Paragraph(text=p2, start=len(p1) + 2, end=len(p1) + 2 + len(p2)),
            Paragraph(text=p3, start=len(p1) + len(p2) + 4, end=len(p1) + len(p2) + 4 + len(p3)),
        ],
        removed_sections=[],
        file_type="txt",
    )

    queries = build_queries(doc, max_queries=10, min_paragraph_chars=30)
    assert len(queries) >= 2
    assert all(q.text for q in queries)
    # Check that short paragraph was ignored
    origin_indices = [q.origin_paragraph_index for q in queries]
    assert 1 not in origin_indices


def test_build_queries_capping():
    # Paragraphs across distinct topic areas
    paragraphs = [
        "Information retrieval and web search algorithms rely upon inverted index structures.",
        "Deep learning architectures optimize objective loss functions via stochastic gradient descent.",
        "Quantum computing utilizes qubits exhibiting superposition and quantum entanglement phenomena.",
        "Bioinformatics pipelines apply dynamic programming algorithms to sequence alignment challenges.",
        "Natural language processing constructs high dimensional vector representations from corpora.",
        "Computer vision systems extract convolutional features from digital image representations.",
        "Distributed consensus protocols guarantee transactional consistency across network partitions.",
    ]
    queries = build_queries(paragraphs, max_queries=5)
    assert len(queries) == 5
