"""Tests for local class repository, privacy modes, and anonymised ID matching."""

from pathlib import Path
from veritext.classrepo.repo import ClassRepositoryManager
from veritext.fingerprint.index import FingerprintIndex
from veritext.ingest.models import Document
from veritext.similarity.engine import check_document


def test_classrepo_add_full_and_fingerprints_only(tmp_path):
    mgr = ClassRepositoryManager(base_dir=tmp_path)

    # 1. Full text submission
    file1 = tmp_path / "student_alice_essay.txt"
    file1.write_text("This is Alice's original submission on machine learning and statistical models.", encoding="utf-8")

    sub_id1 = mgr.add_submission("cs101", file1, fingerprints_only=False)
    assert sub_id1.startswith("sub_")

    # 2. Fingerprints-only submission (privacy mode)
    file2 = tmp_path / "student_bob_essay.txt"
    file2.write_text("This is Bob's private submission with sensitive research details.", encoding="utf-8")

    sub_id2 = mgr.add_submission("cs101", file2, fingerprints_only=True)
    assert sub_id2.startswith("sub_")
    assert sub_id1 != sub_id2

    # Verify database contents
    subs = mgr.list_submissions("cs101")
    assert len(subs) == 2

    alice_sub = next(s for s in subs if s["id"] == sub_id1)
    bob_sub = next(s for s in subs if s["id"] == sub_id2)

    assert not alice_sub["fingerprints_only"]
    assert bob_sub["fingerprints_only"]
    assert bob_sub["fingerprint_count"] > 0

    # Verify Bob's text was NOT saved to disk
    with mgr._get_db_conn("cs101") as conn:
        bob_text = conn.execute("SELECT text FROM submissions WHERE id = ?", (sub_id2,)).fetchone()[0]
        assert bob_text is None

        alice_text = conn.execute("SELECT text FROM submissions WHERE id = ?", (sub_id1,)).fetchone()[0]
        assert alice_text is not None


def test_classrepo_list_and_delete(tmp_path):
    mgr = ClassRepositoryManager(base_dir=tmp_path)

    f1 = tmp_path / "test.txt"
    f1.write_text("Some sample content for testing deletion.", encoding="utf-8")

    mgr.add_submission("bio201", f1)
    mgr.add_submission("chem301", f1)

    classes = mgr.list_classes()
    assert len(classes) == 2
    cnames = [c["class_name"] for c in classes]
    assert "bio201" in cnames
    assert "chem301" in cnames

    # Delete single class
    assert mgr.delete_class("bio201")
    assert len(mgr.list_classes()) == 1

    # Purge all
    purged = mgr.purge_all()
    assert purged >= 1
    assert len(mgr.list_classes()) == 0


def test_classrepo_candidate_anonymisation(tmp_path):
    mgr = ClassRepositoryManager(base_dir=tmp_path)

    f1 = tmp_path / "john_doe_thesis.txt"
    f1.write_text("Academic paper discussing algorithms and data structures.", encoding="utf-8")

    sub_id = mgr.add_submission("eng400", f1)

    # Anonymised candidate (default)
    cands_anon, _ = mgr.get_candidates("eng400", show_names=False)
    assert len(cands_anon) == 1
    assert "john_doe" not in cands_anon[0].title
    assert sub_id in cands_anon[0].title
    assert cands_anon[0].origin == "class submission"

    # Non-anonymised candidate (--show-names)
    cands_named, _ = mgr.get_candidates("eng400", show_names=True)
    assert len(cands_named) == 1
    assert "john_doe_thesis.txt" in cands_named[0].title


def test_similarity_check_with_class_repo(tmp_path):
    mgr = ClassRepositoryManager(base_dir=tmp_path)

    shared_text = "The quick brown fox jumps over the lazy dog and explores the beautiful forest."
    f1 = tmp_path / "assignment_peer.txt"
    f1.write_text(shared_text, encoding="utf-8")

    mgr.add_submission("ai_class", f1, fingerprints_only=True)

    # Document copied from peer
    copied_doc = Document(
        raw_text=shared_text,
        clean_text=shared_text,
        sentences=[],
        paragraphs=[],
        removed_sections=[],
        file_type="txt",
    )

    # Run check document with class repository
    from unittest.mock import patch
    with patch("veritext.similarity.engine.ClassRepositoryManager", return_value=mgr):
        result = check_document(
            copied_doc,
            use_online=False,
            use_local=False,
            options={"class_name": "ai_class", "show_names": False},
        )

        assert result.similarity_percentage > 50.0
        assert any(s.origin == "class submission" for s in result.sources)
        class_src = next(s for s in result.sources if s.origin == "class submission")
        assert "Class Submission" in class_src.source_title
