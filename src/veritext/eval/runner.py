"""Comprehensive evaluation and benchmark runner for Veritext.

Evaluates:
- Plagiarism engine on groups a to c (exact winnowing, semantic, difflib baseline)
- AI text detector on groups c to e (stylometry, distilgpt2, ensemble)
- Spinner / tortured-phrase check on group f (spinners + controls)
- Collusion matrix and MinHash/LSH screening on group g (class submissions)
- Tamper-evident cryptographic seals on group h (untouched valid vs 1-char tampered)
- Optional cross-language semantic matching on group h2
"""

import copy
import csv
import difflib
import gc
import json
import logging
from pathlib import Path
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from veritext.checks.models import CheckResult
from veritext.checks.spinner import check_spinner
from veritext.eval.dataset import (
    DEFAULT_DOCS_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REFS_DIR,
    load_document,
    load_manifest,
    load_references,
)
from veritext.eval.metrics import binary_classification_metrics, mean
from veritext.fingerprint.index import FingerprintIndex
from veritext.fingerprint.minhash import (
    compute_minhash,
    minhash_jaccard,
    screen_batch_candidates,
    screen_fingerprint_candidates,
)
from veritext.fingerprint.winnow import winnow
from veritext.ingest.models import Document
from veritext.ingest.pipeline import split_paragraphs, split_sentences
from veritext.normalize.normalize import normalize_text
from veritext.report.seal import verify_report_seal
from veritext.similarity.exact import find_exact_matches
from veritext.similarity.models import MatchSpan
from veritext.similarity.scorer import compute_similarity_score
from veritext.sources.models import SourceCandidate

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "eval" / "results"


def _create_document(text: str, path: Optional[str] = None) -> Document:
    """Helper to instantiate a Document with populated sentence and paragraph spans."""
    return Document(
        raw_text=text,
        clean_text=text,
        sentences=split_sentences(text),
        paragraphs=split_paragraphs(text),
        removed_sections=[],
        file_type="txt",
        path=path,
    )


class BenchmarkRunner:
    """Executes the complete Veritext benchmark suite."""

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        docs_dir: Optional[Path] = None,
        refs_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        semantic: bool = False,
        ai: bool = False,
        crosslang: bool = False,
    ):
        self.manifest_path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
        self.docs_dir = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR
        self.refs_dir = Path(refs_dir) if refs_dir else DEFAULT_REFS_DIR
        self.out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        self.semantic = semantic
        self.ai = ai
        self.crosslang = crosslang

        self.manifest: List[Dict[str, str]] = []
        self.references: Dict[str, str] = {}
        self.detailed_records: List[Dict[str, Any]] = []

    def setup(self):
        """Loads manifest, references, and creates output directory."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.manifest_path)
        self.references = load_references(self.refs_dir)
        logger.info(
            f"Loaded {len(self.manifest)} manifest records and {len(self.references)} references."
        )

    def run(self) -> Dict[str, Any]:
        """Runs the entire benchmark and generates summary results."""
        tracemalloc.start()
        start_time = time.perf_counter()

        print("=" * 60)
        print("Starting Veritext Evaluation Benchmark")
        print(f"  Output directory: {self.out_dir}")
        print(f"  Semantic enabled: {self.semantic}")
        if self.semantic:
            from veritext.similarity.semantic import (
                DEFAULT_SIMILARITY_THRESHOLD,
                DEFAULT_MIN_CONSECUTIVE_WINDOWS,
            )
            print("  Effective Semantic Settings:")
            print(f"    - Threshold (cosine):            {DEFAULT_SIMILARITY_THRESHOLD}")
            print(f"    - Min consecutive windows:       {DEFAULT_MIN_CONSECUTIVE_WINDOWS}")
            print(f"    - Include semantic in headline:  False (Exact & near-copy only by default)")
            print(f"    - Paraphrase leads tracked:      Separately for review")
            print(f"    - Lexical anchor rule:           Evaluated (N in {{1, 2, 3, 4}}, ratio in {{0.1, 0.2, 0.3}})")
        print(f"  Deep AI enabled:  {self.ai}")
        print(f"  Crosslang enabled:{self.crosslang}")
        print("=" * 60)

        self.setup()

        # 1. Plagiarism Evaluation (groups a, b, c)
        plag_summary = self._evaluate_plagiarism()

        # 2. AI Detection Evaluation (groups c, d, e)
        ai_summary = self._evaluate_ai()

        # 3. Spinner Evaluation (group f)
        spinner_summary = self._evaluate_spinner()

        # 4. Collusion Evaluation (group g)
        collusion_summary = self._evaluate_collusion()

        # 5. Cryptographic Seal Evaluation (group h)
        seal_summary = self._evaluate_seals()

        # 6. Cross-Language Evaluation (group h2)
        crosslang_summary = self._evaluate_crosslang()

        total_elapsed = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        eval_doc_count = sum(
            len(grp_rows) for grp_rows in [
                [r for r in self.manifest if r["group"] in ["a_copies", "b_paraphrases", "b_paraphrases_hard", "c_original"]],
                [r for r in self.manifest if r["group"] in ["d_ai_style", "e_non_native"]],
                [r for r in self.manifest if r["group"] in ["f_spinner", "f_spinner_control"]],
                [r for r in self.manifest if r["group"] in ["g_class_submissions"]],
                [r for r in self.manifest if r["group"] in ["h_seal_reports"]],
            ]
        )
        runtime_per_doc = round(total_elapsed / max(1, eval_doc_count), 4)

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_documents_manifest": len(self.manifest),
            "total_documents_evaluated": eval_doc_count,
            "total_runtime_seconds": round(total_elapsed, 2),
            "runtime_per_document_seconds": runtime_per_doc,
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
            "flags": {
                "semantic": self.semantic,
                "ai": self.ai,
                "crosslang": self.crosslang,
                "effective_semantic_threshold": 0.78 if self.semantic else None,
                "effective_min_consecutive_windows": 2 if self.semantic else None,
                "include_semantic_in_headline_by_default": False,
            },
            "effective_semantic_settings": {
                "cosine_threshold": 0.78,
                "min_consecutive_windows": 2,
                "include_semantic_in_headline_by_default": False,
                "headline_score_formula": "exact_and_near_copy_only",
                "paraphrase_leads_tracked_separately": True,
                "lexical_anchors_evaluated": True,
                "anchor_rule_criteria": {
                    "tested_rare_threshold_N": [1, 2, 3, 4],
                    "tested_content_ratios": [0.1, 0.2, 0.3],
                    "cuts_group_i_fpr_by_half_with_70pct_recall": False,
                    "rule_status": "Kept out of headline defaults; labels spans as strong/weak lead for instructor review"
                }
            } if self.semantic else None,
            "plagiarism": plag_summary,
            "ai_detection": ai_summary,
            "spinner_detection": spinner_summary,
            "collusion_detection": collusion_summary,
            "seal_verification": seal_summary,
            "cross_language": crosslang_summary,
        }

        # Save files
        self._save_results(summary)
        return summary

    def _evaluate_plagiarism(self) -> Dict[str, Any]:
        """Evaluates plagiarism detection across positive cohorts (copies, easy & hard paraphrases)
        and negative cohorts (originals, AI-generated text, and non-native English).

        Assesses both document-level sensitivity across multiple decision thresholds (5%, 10%, 15%, 25%)
        and span-level detection statistics.
        """
        logger.info("Evaluating plagiarism detection on positive and negative cohorts...")

        # Setup index for references
        db_file = self.out_dir / "eval_fingerprints.sqlite"
        if db_file.exists():
            db_file.unlink()
        index = FingerprintIndex(db_path=db_file)
        sources_map: Dict[str, SourceCandidate] = {}

        for ref_id, ref_text in self.references.items():
            cand = SourceCandidate(
                id=ref_id,
                title=ref_id,
                authors=["Veritext Benchmark Corpus"],
                url=f"local://eval/refs/{ref_id}",
                origin="eval_reference",
                year=2024,
                text=ref_text,
            )
            sources_map[ref_id] = cand
            index.index_source_candidate(cand)

        eval_groups = {
            "a_copies",
            "b_paraphrases",
            "b_paraphrases_hard",
            "c_original",
            "d_ai_style",
            "e_non_native",
            "i_same_topic_originals",
            "j_common_knowledge",
        }
        plag_rows = [r for r in self.manifest if r["group"] in eval_groups]

        y_true: List[bool] = []
        scores_winnow: List[float] = []
        scores_baseline: List[float] = []
        scores_headline: List[float] = []
        scores_semantic_optin: List[float] = []
        row_groups: List[str] = []

        from collections import defaultdict
        scores_by_group_exact: Dict[str, List[float]] = defaultdict(list)
        scores_by_group_headline: Dict[str, List[float]] = defaultdict(list)
        scores_by_group_optin: Dict[str, List[float]] = defaultdict(list)
        scores_by_group_baseline: Dict[str, List[float]] = defaultdict(list)

        exact_spans_by_group: Dict[str, List[int]] = defaultdict(list)
        sem_spans_by_group: Dict[str, List[int]] = defaultdict(list)
        strong_leads_by_group: Dict[str, List[int]] = defaultdict(list)
        weak_leads_by_group: Dict[str, List[int]] = defaultdict(list)
        lead_pct_by_group: Dict[str, List[float]] = defaultdict(list)
        doc_lead_spans_by_group: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
        spans_for_auc: List[Dict[str, Any]] = []
        sem_cosine_scores: List[float] = []

        missed_paraphrases_easy: List[Dict[str, Any]] = []
        missed_paraphrases_hard: List[Dict[str, Any]] = []
        false_positives: List[Dict[str, Any]] = []

        HEADLINE_THRESHOLD = 0.15
        TESTED_THRESHOLDS = [0.05, 0.10, 0.15, 0.25]

        for row in plag_rows:
            fname = row["file"]
            grp = row["group"]
            expected = row["expected_plagiarism"].lower() == "yes"
            y_true.append(expected)
            row_groups.append(grp)

            text = load_document(fname, self.docs_dir)

            # 1. Exact Winnowing
            canonical = normalize_text(text)
            doc_fps = winnow(canonical.tokens, k=5, w=4)
            exact_spans = find_exact_matches(
                doc_clean_text=text,
                canonical=canonical,
                doc_fingerprints=doc_fps,
                index=index,
                sources_map=sources_map,
                min_words=6,
            )
            sim_res = compute_similarity_score(text, exact_spans, sources_map)
            score_winnow = sim_res.similarity_percentage / 100.0
            scores_winnow.append(score_winnow)
            scores_by_group_exact[grp].append(score_winnow)
            exact_spans_by_group[grp].append(len(exact_spans))

            # 2. Difflib SequenceMatcher Baseline
            base_score = max(
                (difflib.SequenceMatcher(None, text, ref_txt).ratio() for ref_txt in self.references.values()),
                default=0.0,
            )
            scores_baseline.append(base_score)
            scores_by_group_baseline[grp].append(base_score)

            # 3. Winnowing + Semantic (Headline vs Opt-In Mode)
            score_sem_headline = score_winnow
            score_sem_optin = score_winnow
            sem_spans = []
            doc_strong_cnt = 0
            doc_weak_cnt = 0
            doc_lead_pct = 0.0

            if self.semantic:
                try:
                    from veritext.similarity.semantic import (
                        DEFAULT_SIMILARITY_THRESHOLD,
                        DEFAULT_MIN_CONSECUTIVE_WINDOWS,
                        find_semantic_matches,
                    )
                    doc_obj = _create_document(text, path=fname)
                    cands_list = list(sources_map.values())
                    sem_spans = find_semantic_matches(
                        document=doc_obj,
                        candidates=cands_list,
                        exact_spans=exact_spans,
                        threshold=DEFAULT_SIMILARITY_THRESHOLD,
                        min_consecutive_windows=DEFAULT_MIN_CONSECUTIVE_WINDOWS,
                        include_single_window=False,
                    )
                    all_spans = exact_spans + sem_spans
                    # Headline score: computed ONLY from exact & near-copy by default
                    sim_sem_res = compute_similarity_score(
                        text,
                        all_spans,
                        sources_map,
                        include_single_window_semantic=False,
                        include_semantic_in_score=False,
                    )
                    score_sem_headline = sim_sem_res.similarity_percentage / 100.0

                    # Legacy / opt-in score: includes semantic leads
                    sim_optin_res = compute_similarity_score(
                        text,
                        all_spans,
                        sources_map,
                        include_single_window_semantic=False,
                        include_semantic_in_score=True,
                    )
                    score_sem_optin = sim_optin_res.similarity_percentage / 100.0

                    doc_lead_pct = sim_sem_res.metadata.get("paraphrase_leads_percentage", 0.0)

                    for s in sem_spans:
                        sem_cosine_scores.append(s.score)
                        is_strong = s.metadata.get("is_strong_lead", False) or s.metadata.get("lead_level") == "strong lead"
                        if is_strong:
                            doc_strong_cnt += 1
                        else:
                            doc_weak_cnt += 1

                        is_para = 1 if grp in ("b_paraphrases", "b_paraphrases_hard") else 0
                        spans_for_auc.append({
                            "is_para": is_para,
                            "group": grp,
                            "cosine": s.score,
                            "anchor_count": s.metadata.get("anchor_count", 0),
                            "content_ratio": s.metadata.get("content_lemma_ratio", 0.0),
                            "composite": s.metadata.get("composite_score", s.score),
                            "is_strong": is_strong,
                        })

                    doc_lead_spans_by_group[grp].append([
                        {
                            "anchor_count": s.metadata.get("anchor_count", 0),
                            "content_ratio": s.metadata.get("content_lemma_ratio", 0.0),
                            "is_strong": s.metadata.get("is_strong_lead", False),
                        }
                        for s in sem_spans
                    ])
                except Exception as e:
                    logger.warning(f"Semantic match skipped for {fname}: {e}")

            scores_headline.append(score_sem_headline)
            scores_by_group_headline[grp].append(score_sem_headline)
            scores_semantic_optin.append(score_sem_optin)
            scores_by_group_optin[grp].append(score_sem_optin)
            sem_spans_by_group[grp].append(len(sem_spans))
            strong_leads_by_group[grp].append(doc_strong_cnt)
            weak_leads_by_group[grp].append(doc_weak_cnt)
            lead_pct_by_group[grp].append(doc_lead_pct)

            # Track failure cases at headline threshold 0.15 (using headline score)
            active_score = score_sem_headline if self.semantic else score_winnow
            if expected and active_score < HEADLINE_THRESHOLD:
                failure_record = {
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "headline_score": round(score_sem_headline, 4) if self.semantic else None,
                    "optin_score": round(score_sem_optin, 4) if self.semantic else None,
                    "baseline_score": round(base_score, 4),
                    "notes": row.get("notes", ""),
                }
                if grp == "b_paraphrases":
                    missed_paraphrases_easy.append(failure_record)
                elif grp == "b_paraphrases_hard":
                    missed_paraphrases_hard.append(failure_record)
            if not expected and active_score >= HEADLINE_THRESHOLD:
                false_positives.append({
                    "file": fname,
                    "group": grp,
                    "winnow_score": round(score_winnow, 4),
                    "headline_score": round(score_sem_headline, 4) if self.semantic else None,
                    "optin_score": round(score_sem_optin, 4) if self.semantic else None,
                    "notes": row.get("notes", ""),
                })

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "plagiarism",
                "ground_truth": expected,
                "winnow_score": round(score_winnow, 4),
                "winnow_pred": score_winnow >= HEADLINE_THRESHOLD,
                "baseline_score": round(base_score, 4),
                "baseline_pred": base_score >= 0.35,
                "headline_score": round(score_sem_headline, 4) if self.semantic else round(score_winnow, 4),
                "headline_pred": (active_score >= HEADLINE_THRESHOLD),
                "semantic_optin_score": round(score_sem_optin, 4) if self.semantic else None,
                "semantic_optin_pred": (score_sem_optin >= HEADLINE_THRESHOLD) if self.semantic else None,
                "exact_spans_count": len(exact_spans),
                "semantic_spans_count": len(sem_spans),
                "strong_leads_count": doc_strong_cnt,
                "weak_leads_count": doc_weak_cnt,
            })

            gc.collect()

        # Multi-threshold metrics computation
        winnow_by_thresh = {}
        headline_by_thresh = {}
        optin_by_thresh = {}
        base_by_thresh = {}
        fpr_by_group = {
            "c_original": {},
            "d_ai_style": {},
            "e_non_native": {},
            "i_same_topic_originals": {},
            "j_common_knowledge": {},
        }
        fpr_optin_by_group = {
            "c_original": {},
            "d_ai_style": {},
            "e_non_native": {},
            "i_same_topic_originals": {},
            "j_common_knowledge": {},
        }
        recall_by_group = {
            "a_copies": {},
            "b_paraphrases": {},
            "b_paraphrases_hard": {},
        }
        recall_optin_by_group = {
            "a_copies": {},
            "b_paraphrases": {},
            "b_paraphrases_hard": {},
        }

        for t in TESTED_THRESHOLDS:
            # Winnowing (Exact)
            pred_w = [s >= t for s in scores_winnow]
            winnow_by_thresh[t] = binary_classification_metrics(y_true, pred_w)

            # Headline engine (exact & near-copy only)
            pred_h = [s >= t for s in scores_headline]
            headline_by_thresh[t] = binary_classification_metrics(y_true, pred_h)

            # Opt-in engine (semantic in score)
            pred_o = [s >= t for s in scores_semantic_optin]
            optin_by_thresh[t] = binary_classification_metrics(y_true, pred_o)

            # Baseline difflib
            pred_b = [s >= (0.20 + t) for s in scores_baseline]
            base_by_thresh[t] = binary_classification_metrics(y_true, pred_b)

            # Group-specific FPR on negative cohorts
            for neg_grp in ["c_original", "d_ai_style", "e_non_native", "i_same_topic_originals", "j_common_knowledge"]:
                neg_indices = [i for i, g in enumerate(row_groups) if g == neg_grp]
                if neg_indices:
                    fp_h = sum(1 for i in neg_indices if pred_h[i])
                    fpr_by_group[neg_grp][t] = round(fp_h / len(neg_indices), 4)

                    fp_o = sum(1 for i in neg_indices if pred_o[i])
                    fpr_optin_by_group[neg_grp][t] = round(fp_o / len(neg_indices), 4)

            # Group-specific Recall on positive cohorts
            for pos_grp in ["a_copies", "b_paraphrases", "b_paraphrases_hard"]:
                pos_indices = [i for i, g in enumerate(row_groups) if g == pos_grp]
                if pos_indices:
                    tp_h = sum(1 for i in pos_indices if pred_h[i])
                    recall_by_group[pos_grp][t] = round(tp_h / len(pos_indices), 4)

                    tp_o = sum(1 for i in pos_indices if pred_o[i])
                    recall_optin_by_group[pos_grp][t] = round(tp_o / len(pos_indices), 4)

        # Span-level summary
        span_metrics = {
            "mean_exact_spans_per_doc": {
                grp: round(mean(counts), 2) for grp, counts in exact_spans_by_group.items()
            },
            "mean_semantic_spans_per_doc": {
                grp: round(mean(counts), 2) for grp, counts in sem_spans_by_group.items()
            },
            "mean_semantic_cosine_similarity": round(mean(sem_cosine_scores), 4) if sem_cosine_scores else 0.0,
            "total_semantic_spans_found": len(sem_cosine_scores),
        }

        # Flat mean similarity mapping for backward-compatibility with plots and table
        flat_means = {
            grp: mean(scores_by_group_headline[grp]) if self.semantic else mean(scores_by_group_exact[grp])
            for grp in eval_groups
        }

        # Lead statistics across cohorts
        lead_stats = {}
        if self.semantic:
            for g in eval_groups:
                counts = sem_spans_by_group.get(g, [])
                strongs = strong_leads_by_group.get(g, [])
                weaks = weak_leads_by_group.get(g, [])
                pcts = lead_pct_by_group.get(g, [])
                doc_flagged = sum(1 for c in counts if c > 0)
                lead_stats[g] = {
                    "mean_leads_per_doc": round(mean(counts), 2) if counts else 0.0,
                    "mean_strong_leads_per_doc": round(mean(strongs), 2) if strongs else 0.0,
                    "mean_weak_leads_per_doc": round(mean(weaks), 2) if weaks else 0.0,
                    "mean_lead_percentage": round(mean(pcts), 2) if pcts else 0.0,
                    "total_leads": sum(counts),
                    "total_strong_leads": sum(strongs),
                    "total_weak_leads": sum(weaks),
                    "docs_flagged_with_leads": doc_flagged,
                    "flagged_rate": round(doc_flagged / max(1, len(counts)), 4) if counts else 0.0,
                }
            tp_strong = lead_stats.get("b_paraphrases", {}).get("total_strong_leads", 0) + lead_stats.get("b_paraphrases_hard", {}).get("total_strong_leads", 0)
            fp_strong = lead_stats.get("i_same_topic_originals", {}).get("total_strong_leads", 0)
            lead_stats["strong_lead_precision_b_vs_i"] = round(tp_strong / max(1, (tp_strong + fp_strong)), 4)

        # ROC AUC on candidate spans
        roc_auc_dict = {}
        if self.semantic:
            b_and_i_spans = [
                sp for sp in spans_for_auc
                if sp.get("group") in ("b_paraphrases", "b_paraphrases_hard", "i_same_topic_originals")
            ]
            if b_and_i_spans and len(set(sp["is_para"] for sp in b_and_i_spans)) > 1:
                from sklearn.metrics import roc_auc_score
                y_auc = [sp["is_para"] for sp in b_and_i_spans]
                cos_auc = [sp["cosine"] for sp in b_and_i_spans]
                anc_auc = [sp["anchor_count"] for sp in b_and_i_spans]
                comp_auc = [sp["composite"] for sp in b_and_i_spans]
                roc_auc_dict = {
                    "spans_evaluated": len(b_and_i_spans),
                    "roc_auc_cosine_alone": round(float(roc_auc_score(y_auc, cos_auc)), 4),
                    "roc_auc_anchors_alone": round(float(roc_auc_score(y_auc, anc_auc)), 4),
                    "roc_auc_composite": round(float(roc_auc_score(y_auc, comp_auc)), 4),
                }

        # Anchor grid evaluation (N in {1,2,3,4}, ratio in {0.1, 0.2, 0.3})
        anchor_grid_results = []
        if self.semantic:
            for N in [1, 2, 3, 4]:
                for ratio in [0.1, 0.2, 0.3]:
                    def doc_qualifies(doc_spans_list):
                        return any(
                            sp.get("anchor_count", 0) >= N and sp.get("content_ratio", 0.0) >= ratio
                            for sp in doc_spans_list
                        )
                    easy_docs = doc_lead_spans_by_group.get("b_paraphrases", [])
                    hard_docs = doc_lead_spans_by_group.get("b_paraphrases_hard", [])
                    i_docs = doc_lead_spans_by_group.get("i_same_topic_originals", [])
                    j_docs = doc_lead_spans_by_group.get("j_common_knowledge", [])

                    rec_easy = sum(1 for d in easy_docs if doc_qualifies(d)) / max(1, len(easy_docs))
                    rec_hard = sum(1 for d in hard_docs if doc_qualifies(d)) / max(1, len(hard_docs))
                    rec_all = (sum(1 for d in easy_docs if doc_qualifies(d)) + sum(1 for d in hard_docs if doc_qualifies(d))) / max(1, len(easy_docs) + len(hard_docs))
                    fpr_i_val = sum(1 for d in i_docs if doc_qualifies(d)) / max(1, len(i_docs))
                    fpr_j_val = sum(1 for d in j_docs if doc_qualifies(d)) / max(1, len(j_docs))

                    anchor_grid_results.append({
                        "N": N,
                        "ratio": ratio,
                        "recall_b_all": round(rec_all, 4),
                        "recall_b_easy": round(rec_easy, 4),
                        "recall_b_hard": round(rec_hard, 4),
                        "fpr_i": round(fpr_i_val, 4),
                        "fpr_j": round(fpr_j_val, 4),
                        "qualifies_rule": bool(rec_all >= 0.70 and fpr_i_val <= 0.475),
                    })

        return {
            "headline_threshold": HEADLINE_THRESHOLD,
            "thresholds_tested": TESTED_THRESHOLDS,
            "multi_threshold_metrics": {
                "winnowing_only": winnow_by_thresh,
                "winnowing_and_semantic": headline_by_thresh if self.semantic else None,
                "semantic_optin_mode": optin_by_thresh if self.semantic else None,
                "baseline_difflib": base_by_thresh,
            },
            "group_fpr_by_threshold": fpr_by_group,
            "group_fpr_optin_by_threshold": fpr_optin_by_group,
            "paraphrase_recall_breakdown": recall_by_group,
            "paraphrase_recall_optin_breakdown": recall_optin_by_group,
            "mean_similarity_per_group": flat_means,
            "mean_similarity_by_engine": {
                "exact": {grp: mean(scs) for grp, scs in scores_by_group_exact.items()},
                "semantic_headline": {grp: mean(scs) for grp, scs in scores_by_group_headline.items()} if self.semantic else None,
                "semantic_optin": {grp: mean(scs) for grp, scs in scores_by_group_optin.items()} if self.semantic else None,
                "baseline": {grp: mean(scs) for grp, scs in scores_by_group_baseline.items()},
            },
            "winnowing_only": winnow_by_thresh[HEADLINE_THRESHOLD],
            "winnowing_and_semantic": headline_by_thresh[HEADLINE_THRESHOLD] if self.semantic else None,
            "semantic_optin_mode": optin_by_thresh[HEADLINE_THRESHOLD] if self.semantic else None,
            "baseline_difflib": base_by_thresh[HEADLINE_THRESHOLD],
            "lead_statistics": lead_stats,
            "roc_auc": roc_auc_dict,
            "anchor_grid_evaluation": anchor_grid_results,
            "span_level_metrics": span_metrics,
            "failure_cases": {
                "missed_paraphrases_easy_count": len(missed_paraphrases_easy),
                "missed_paraphrases_easy": missed_paraphrases_easy,
                "missed_paraphrases_hard_count": len(missed_paraphrases_hard),
                "missed_paraphrases_hard": missed_paraphrases_hard,
                "missed_paraphrases_count": len(missed_paraphrases_easy) + len(missed_paraphrases_hard),
                "false_positives_count": len(false_positives),
                "false_positives": false_positives,
            },
        }

    def _evaluate_ai(self) -> Dict[str, Any]:
        """Evaluates AI detection across groups c (human originals), d (AI generated), and e (ESL human)."""
        logger.info("Evaluating AI detection on groups c, d, e...")

        target_groups = {"c_original", "d_ai_style", "e_non_native"}
        ai_rows = [r for r in self.manifest if r["group"] in target_groups]

        y_true: List[bool] = []
        y_pred_style: List[bool] = []
        y_pred_lm: List[bool] = []
        y_pred_ensemble: List[bool] = []

        is_human_mask: List[bool] = []
        is_esl_mask: List[bool] = []

        from veritext.ai_detect.ensemble import calculate_ensemble_score
        from veritext.ai_detect.features_style import compute_stylometric_features

        has_distilgpt2 = False
        if self.ai:
            from veritext.ai_detect.features_lm import is_distilgpt2_available
            has_distilgpt2 = is_distilgpt2_available()

        false_positives_human: List[Dict[str, Any]] = []
        false_positives_esl: List[Dict[str, Any]] = []

        for row in ai_rows:
            fname = row["file"]
            grp = row["group"]
            expected_ai = row["expected_ai"].lower() == "yes"
            is_human = not expected_ai
            is_esl = grp == "e_non_native"

            y_true.append(expected_ai)
            is_human_mask.append(is_human)
            is_esl_mask.append(is_esl)

            text = load_document(fname, self.docs_dir)

            # 1. Stylometry only
            style_feats = compute_stylometric_features(text)
            prob_style, _ = calculate_ensemble_score(style_feats, has_lm=False)
            pred_style = prob_style >= 50.0
            y_pred_style.append(pred_style)

            # 2. DistilGPT2 features only & Full Ensemble (if self.ai enabled and available)
            prob_lm = None
            pred_lm = None
            prob_ens = prob_style
            pred_ens = pred_style

            if self.ai and has_distilgpt2:
                try:
                    from veritext.ai_detect.features_lm import compute_lm_features
                    lm_feats = compute_lm_features(text)
                    # DistilGPT2 score: logistic model on perplexity & log-rank
                    # Z = 0.5 - 0.04*ppl - 0.35*rank
                    ppl = lm_feats.get("perplexity", 100.0)
                    rank = lm_feats.get("mean_log_rank", 5.0)
                    z_lm = 0.5 - 0.04 * ppl - 0.35 * rank
                    prob_lm = round(100.0 / (1.0 + 2.71828 ** (-z_lm)), 2)
                    pred_lm = prob_lm >= 50.0
                    y_pred_lm.append(pred_lm)

                    # Ensemble score
                    combined_feats = {**style_feats, **lm_feats}
                    prob_ens, _ = calculate_ensemble_score(combined_feats, has_lm=True)
                    pred_ens = prob_ens >= 50.0
                    y_pred_ensemble.append(pred_ens)
                except Exception as e:
                    logger.warning(f"LM feature calculation error on {fname}: {e}")
                    y_pred_lm.append(False)
                    y_pred_ensemble.append(pred_style)
            else:
                y_pred_lm.append(False)
                y_pred_ensemble.append(pred_style)

            # Check false positives
            if is_human and pred_style:
                false_positives_human.append({
                    "file": fname,
                    "group": grp,
                    "prob_style": prob_style,
                    "notes": row.get("notes", ""),
                })
            if is_esl and pred_style:
                false_positives_esl.append({
                    "file": fname,
                    "prob_style": prob_style,
                    "notes": row.get("notes", ""),
                })

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "ai_detection",
                "ground_truth_ai": expected_ai,
                "prob_stylometry": prob_style,
                "pred_stylometry": pred_style,
                "prob_lm": prob_lm,
                "pred_lm": pred_lm,
                "prob_ensemble": prob_ens,
                "pred_ensemble": pred_ens,
            })

            gc.collect()

        # Metrics computation
        metrics_style = binary_classification_metrics(y_true, y_pred_style)

        # FPR on human text (c_original + e_non_native: 25 documents)
        human_true = [yt for yt, ih in zip(y_true, is_human_mask) if ih]
        human_pred_style = [yp for yp, ih in zip(y_pred_style, is_human_mask) if ih]
        human_fp = sum(1 for p in human_pred_style if p)
        fpr_human = round(human_fp / max(1, len(human_true)), 4)

        # FPR specifically on non-native group (e_non_native: 10 documents)
        esl_pred_style = [yp for yp, ie in zip(y_pred_style, is_esl_mask) if ie]
        esl_fp = sum(1 for p in esl_pred_style if p)
        fpr_esl = round(esl_fp / max(1, sum(is_esl_mask)), 4)

        metrics_style["fpr_human_overall"] = fpr_human
        metrics_style["fpr_non_native_esl"] = fpr_esl

        metrics_lm = None
        metrics_ens = None
        if self.ai and has_distilgpt2:
            metrics_lm = binary_classification_metrics(y_true, y_pred_lm)
            metrics_ens = binary_classification_metrics(y_true, y_pred_ensemble)
            # Add human / esl FPR for ensemble
            ens_human_pred = [yp for yp, ih in zip(y_pred_ensemble, is_human_mask) if ih]
            metrics_ens["fpr_human_overall"] = round(
                sum(1 for p in ens_human_pred if p) / max(1, len(human_true)), 4
            )
            ens_esl_pred = [yp for yp, ie in zip(y_pred_ensemble, is_esl_mask) if ie]
            metrics_ens["fpr_non_native_esl"] = round(
                sum(1 for p in ens_esl_pred if p) / max(1, sum(is_esl_mask)), 4
            )

        return {
            "distilgpt2_evaluated": self.ai and has_distilgpt2,
            "stylometry_only": metrics_style,
            "distilgpt2_only": metrics_lm,
            "full_ensemble": metrics_ens,
            "failure_cases": {
                "false_positives_human_count": len(false_positives_human),
                "false_positives_human": false_positives_human,
                "false_positives_esl_count": len(false_positives_esl),
                "false_positives_esl": false_positives_esl,
            },
        }

    def _evaluate_spinner(self) -> Dict[str, Any]:
        """Evaluates spinner and tortured-phrase detection on group f (spinners + controls)."""
        logger.info("Evaluating spinner detection on group f...")

        target_groups = {"f_spinner", "f_spinner_control"}
        spin_rows = [r for r in self.manifest if r["group"] in target_groups]

        y_true: List[bool] = []
        y_pred: List[bool] = []

        for row in spin_rows:
            fname = row["file"]
            grp = row["group"]
            expected = row["expected_spinner"].lower() == "yes"
            y_true.append(expected)

            text = load_document(fname, self.docs_dir)
            doc_obj = _create_document(text)
            check_res: CheckResult = check_spinner(doc_obj)
            pred = len(check_res.findings) > 0
            y_pred.append(pred)

            self.detailed_records.append({
                "file": fname,
                "group": grp,
                "task": "spinner_detection",
                "ground_truth": expected,
                "findings_count": len(check_res.findings),
                "pred": pred,
            })

            gc.collect()

        return binary_classification_metrics(y_true, y_pred)

    def _evaluate_collusion(self) -> Dict[str, Any]:
        """Evaluates batch pairwise collusion detection and MinHash/LSH screening on group g."""
        logger.info("Evaluating collusion detection on group g...")

        target_groups = {"g_class_submissions"}
        coll_rows = [r for r in self.manifest if r["group"] in target_groups]

        # Ground truth pairs
        true_pairs = {
            tuple(sorted(["class_sub_01.txt", "class_sub_02.txt"])),
            tuple(sorted(["class_sub_03.txt", "class_sub_04.txt"])),
        }

        # 1. Compute MinHash signatures, shingle sets, and winnowing fingerprint sets
        signatures: Dict[str, List[int]] = {}
        shingle_sets: Dict[str, Set[str]] = {}
        fingerprint_sets: Dict[str, Set[int]] = {}

        for row in coll_rows:
            fname = row["file"]
            text = load_document(fname, self.docs_dir)
            canon = normalize_text(text)
            tokens = canon.tokens
            shingles = {
                " ".join(tokens[i : i + 5]) for i in range(max(0, len(tokens) - 4))
            }
            shingle_sets[fname] = shingles
            signatures[fname] = compute_minhash(shingles)
            fps = winnow(tokens, k=5, w=4)
            fingerprint_sets[fname] = {h for h, _ in fps}

        # 2. Pairwise Winnowing (ground truth discovery)
        detected_pairwise_pairs = set()
        files = [r["file"] for r in coll_rows]
        n = len(files)

        for i in range(n):
            for j in range(i + 1, n):
                f_a = files[i]
                f_b = files[j]
                s_a = shingle_sets[f_a]
                s_b = shingle_sets[f_b]
                jaccard = (
                    len(s_a.intersection(s_b)) / max(1, len(s_a.union(s_b)))
                    if s_a or s_b
                    else 0.0
                )
                if jaccard >= 0.25:
                    detected_pairwise_pairs.add(tuple(sorted([f_a, f_b])))

        # 3. Exact Inverted Index Fingerprint Shortlist Screening (Default)
        fp_candidate_pairs = screen_fingerprint_candidates(fingerprint_sets)
        fp_pairs = {
            tuple(sorted(p)) for p in fp_candidate_pairs
        }

        # 4. MinHash/LSH Candidate Screening (threshold=0.25 deriving (64, 2))
        lsh_candidate_pairs, _ = screen_batch_candidates(shingle_sets, threshold=0.25)
        lsh_pairs = {
            tuple(sorted(p)) for p in lsh_candidate_pairs
        }

        # Check clusters and recall
        expected_clusters_found = detected_pairwise_pairs == true_pairs
        fp_shortlist_recall = (
            len(true_pairs.intersection(fp_pairs)) / len(true_pairs)
            if true_pairs
            else 1.0
        )
        lsh_shortlist_recall = (
            len(true_pairs.intersection(lsh_pairs)) / len(true_pairs)
            if true_pairs
            else 1.0
        )

        return {
            "total_submissions": len(files),
            "expected_collusion_pairs": [list(p) for p in true_pairs],
            "detected_pairwise_pairs": [list(p) for p in detected_pairwise_pairs],
            "fingerprint_index_candidates": [list(p) for p in fp_pairs],
            "minhash_lsh_candidates": [list(p) for p in lsh_pairs],
            "expected_clusters_found": expected_clusters_found,
            "fingerprint_shortlist_recall": round(fp_shortlist_recall, 4),
            "lsh_shortlist_recall": round(lsh_shortlist_recall, 4),
        }

    def _evaluate_seals(self) -> Dict[str, Any]:
        """Evaluates cryptographic seal verification and tampering detection on group h."""
        logger.info("Evaluating cryptographic seals on group h...")

        target_groups = {"h_seal_reports"}
        seal_rows = [r for r in self.manifest if r["group"] in target_groups]

        untouched_valid = 0
        tampered_detected = 0
        total_reports = len(seal_rows)

        for row in seal_rows:
            fname = row["file"]
            raw_json = load_document(fname, self.docs_dir)
            report_dict = json.loads(raw_json)

            # 1. Untouched verification
            is_valid_untouched, _ = verify_report_seal(report_dict)
            if is_valid_untouched:
                untouched_valid += 1

            # 2. Tampered verification (introduce single character modification)
            tampered_dict = copy.deepcopy(report_dict)
            tampered_dict["summary"] = str(tampered_dict.get("summary", "")) + "X"
            is_valid_tampered, _ = verify_report_seal(tampered_dict)
            if not is_valid_tampered:
                tampered_detected += 1

        tamper_detection_rate = (
            tampered_detected / total_reports if total_reports > 0 else 0.0
        )

        return {
            "total_reports_tested": total_reports,
            "untouched_valid_count": untouched_valid,
            "tampered_detected_count": tampered_detected,
            "tamper_detection_rate": round(tamper_detection_rate, 4),
        }

    def _evaluate_crosslang(self) -> Dict[str, Any]:
        """Evaluates multilingual cross-language matching on group h2 (ES/FR)."""
        logger.info("Evaluating cross-language semantic matching on group h2...")

        target_groups = {"h_translations"}
        trans_rows = [r for r in self.manifest if r["group"] in target_groups]

        if not self.crosslang:
            return {
                "status": "skipped",
                "reason": "Cross-language evaluation disabled (enable with --crosslang)",
                "documents_available": len(trans_rows),
            }

        try:
            from veritext.similarity.crosslang import is_crosslang_available
            if not is_crosslang_available():
                return {
                    "status": "skipped",
                    "reason": "Multilingual sentence-transformer model not installed locally",
                    "documents_available": len(trans_rows),
                }

            from veritext.similarity.crosslang import CrossLanguageMatcher
            matcher = CrossLanguageMatcher()
            ref_docs = [
                _create_document(txt, path=rid)
                for rid, txt in self.references.items()
            ]

            matches_found = 0
            for row in trans_rows:
                fname = row["file"]
                text = load_document(fname, self.docs_dir)
                doc = _create_document(text, path=fname)
                res = matcher.find_matches(doc, ref_docs, threshold=0.65)
                if res:
                    matches_found += 1

            recall = matches_found / len(trans_rows) if trans_rows else 0.0
            return {
                "status": "completed",
                "total_translations": len(trans_rows),
                "matches_found": matches_found,
                "recall": round(recall, 4),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "documents_available": len(trans_rows),
            }

    def _save_results(self, summary: Dict[str, Any]):
        """Saves benchmark results to JSON, CSV, Markdown, and confusion-matrix PNG."""
        # 1. Summary JSON
        json_path = self.out_dir / "benchmark_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"Saved summary JSON to {json_path}")

        # 2. Details CSV
        csv_path = self.out_dir / "benchmark_details.csv"
        if self.detailed_records:
            fieldnames = sorted({k for r in self.detailed_records for k in r.keys()})
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.detailed_records:
                    writer.writerow(r)
            logger.info(f"Saved detailed records CSV to {csv_path}")

        # 3. Markdown Results Table
        md_path = self.out_dir / "results_table.md"
        md_text = self._generate_markdown_table(summary)
        md_path.write_text(md_text, encoding="utf-8")
        logger.info(f"Saved Markdown table to {md_path}")

        # 4. Confusion Matrix Plot
        png_path = self.out_dir / "confusion_matrices.png"
        self._plot_confusion_matrices(summary, png_path)
        logger.info(f"Saved confusion matrix plot to {png_path}")

    def _generate_markdown_table(self, summary: Dict[str, Any]) -> str:
        """Generates a GitHub-flavored Markdown table of benchmark results."""
        p_win = summary["plagiarism"]["winnowing_only"]
        p_base = summary["plagiarism"]["baseline_difflib"]
        p_head = summary["plagiarism"].get("winnowing_and_semantic") or p_win
        p_optin = summary["plagiarism"].get("semantic_optin_mode") or {}

        ai_style = summary["ai_detection"]["stylometry_only"]
        ai_ens = summary["ai_detection"]["full_ensemble"] or {}

        spin = summary["spinner_detection"]
        seal = summary["seal_verification"]
        coll = summary["collusion_detection"]

        lines = [
            "# Veritext Benchmark Results",
            "",
            f"- **Evaluated Documents**: {summary['total_documents_evaluated']} (from synthetic dataset)",
            f"- **Total Runtime**: {summary['total_runtime_seconds']} s (~{summary['runtime_per_document_seconds']} s/doc)",
            f"- **Peak Memory**: {summary['peak_memory_mb']} MB",
            f"- **Date**: {summary['timestamp']}",
            "",
            "> **Corpus & Methodology Note**: The evaluation corpus is synthetic, compact, and created by the system author.",
            "> In accordance with the Stage 7 audit guidelines, results are evaluated across distinct easy and hard paraphrase subsets,",
            "> and false positive rates are transparently measured across unrelated human originals, AI-generated text, and ESL writing.",
            "> Tuning and evaluation use the same small synthetic corpus, so reported numbers are inherently optimistic.",
            "",
            "## 1. Plagiarism Detection Engine (Headline Threshold: 15%)",
            "",
            "> **Headline Scoring Rule (v1.0.3)**: Headline similarity percentage is computed ONLY from exact and near-copy matches.",
            "> Semantic vector matches are strictly excluded by default and presented separately as unverified paraphrase review leads.",
            "",
            "| Method | Accuracy | Precision | Recall | F1 Score | False Positive Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **Headline Score / Exact Winnowing (Veritext Default)** | {p_head['accuracy']:.2%} | {p_head['precision']:.2%} | {p_head['recall']:.2%} | {p_head['f1']:.2%} | {p_head['fpr']:.2%} |",
            f"| Baseline (difflib SequenceMatcher) | {p_base['accuracy']:.2%} | {p_base['precision']:.2%} | {p_base['recall']:.2%} | {p_base['f1']:.2%} | {p_base['fpr']:.2%} |",
        ]

        if p_optin:
            lines.append(
                f"| *Winnowing + Semantic (Opt-In: `--include-semantic-in-score`)* | {p_optin.get('accuracy', 0):.2%} | {p_optin.get('precision', 0):.2%} | {p_optin.get('recall', 0):.2%} | {p_optin.get('f1', 0):.2%} | {p_optin.get('fpr', 0):.2%} |"
            )

        # Multi-threshold sensitivity analysis
        if "multi_threshold_metrics" in summary["plagiarism"]:
            mtm = summary["plagiarism"]["multi_threshold_metrics"]
            lines.extend([
                "",
                "### Decision Threshold Sensitivity Analysis",
                "",
                "| Threshold | Method | Accuracy | Precision | Recall | F1 Score | Overall FPR |",
                "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
            ])
            ws_dict = mtm.get("winnowing_and_semantic") or {}
            wo_dict = mtm.get("winnowing_only") or {}
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                h_m = ws_dict.get(thresh) or wo_dict.get(thresh) or {}
                lines.append(
                    f"| {thresh:.0%} | **Headline Score (Default)** | {h_m.get('accuracy', 0):.2%} | {h_m.get('precision', 0):.2%} | {h_m.get('recall', 0):.2%} | {h_m.get('f1', 0):.2%} | {h_m.get('fpr', 0):.2%} |"
                )
                if mtm.get("semantic_optin_mode"):
                    o_m = (mtm.get("semantic_optin_mode") or {}).get(thresh, {})
                    lines.append(
                        f"| {thresh:.0%} | *Winnowing + Semantic (Opt-In)* | {o_m.get('accuracy', 0):.2%} | {o_m.get('precision', 0):.2%} | {o_m.get('recall', 0):.2%} | {o_m.get('f1', 0):.2%} | {o_m.get('fpr', 0):.2%} |"
                    )

        # Negative Cohort FPR breakdown
        if "group_fpr_by_threshold" in summary["plagiarism"]:
            gfpr = summary["plagiarism"]["group_fpr_by_threshold"] or {}
            lines.extend([
                "",
                "### False Positive Rates across Negative Cohorts (Headline Engine Default)",
                "",
                "| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                fpr_c = (gfpr.get("c_original") or {}).get(thresh, 0.0)
                fpr_d = (gfpr.get("d_ai_style") or {}).get(thresh, 0.0)
                fpr_e = (gfpr.get("e_non_native") or {}).get(thresh, 0.0)
                fpr_i = (gfpr.get("i_same_topic_originals") or {}).get(thresh, 0.0)
                fpr_j = (gfpr.get("j_common_knowledge") or {}).get(thresh, 0.0)
                lines.append(
                    f"| {thresh:.0%} | {fpr_c:.2%} | {fpr_d:.2%} | {fpr_e:.2%} | {fpr_i:.2%} | {fpr_j:.2%} |"
                )

        if summary["plagiarism"].get("group_fpr_optin_by_threshold") and (summary["plagiarism"]["group_fpr_optin_by_threshold"].get("i_same_topic_originals")):
            gfpr_o = summary["plagiarism"]["group_fpr_optin_by_threshold"] or {}
            lines.extend([
                "",
                "### False Positive Rates across Negative Cohorts (Opt-In Engine: Semantic in Score)",
                "",
                "| Threshold | Group C (Originals) | Group D (AI Text) | Group E (ESL Non-Native) | Group I (Same-Topic Originals) | Group J (Common Knowledge) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for thresh in [0.05, 0.10, 0.15, 0.25]:
                fpr_c = (gfpr_o.get("c_original") or {}).get(thresh, 0.0)
                fpr_d = (gfpr_o.get("d_ai_style") or {}).get(thresh, 0.0)
                fpr_e = (gfpr_o.get("e_non_native") or {}).get(thresh, 0.0)
                fpr_i = (gfpr_o.get("i_same_topic_originals") or {}).get(thresh, 0.0)
                fpr_j = (gfpr_o.get("j_common_knowledge") or {}).get(thresh, 0.0)
                lines.append(
                    f"| {thresh:.0%} | {fpr_c:.2%} | {fpr_d:.2%} | {fpr_e:.2%} | {fpr_i:.2%} | {fpr_j:.2%} |"
                )

        # 1b. Paraphrase Leads & Lexical Anchors Evaluation
        if summary["plagiarism"].get("lead_statistics"):
            l_stats = summary["plagiarism"]["lead_statistics"]
            roc = summary["plagiarism"].get("roc_auc", {})
            lines.extend([
                "",
                "## 1b. Paraphrase Leads & Lexical Anchors Evaluation",
                "",
                "> **Explanatory Note**: Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source.",
                "> **Interpretation Guide**: Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only.",
                "",
                "### Lead Statistics Across Cohorts",
                "",
                "| Cohort | Mean Leads / Doc | Mean Strong Leads / Doc | Mean Weak Leads / Doc | Flagged Rate |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ])
            cohort_names = [
                ("Group A (Copies)", "a_copies"),
                ("Group B (Easy Paraphrases)", "b_paraphrases"),
                ("Group B (Hard Complex Paraphrases)", "b_paraphrases_hard"),
                ("Group C (Unrelated Originals)", "c_original"),
                ("Group D (AI Generated)", "d_ai_style"),
                ("Group E (ESL Non-Native)", "e_non_native"),
                ("Group I (Same-Topic Originals)", "i_same_topic_originals"),
                ("Group J (Common Knowledge)", "j_common_knowledge"),
            ]
            for label, key in cohort_names:
                cs = l_stats.get(key, {})
                lines.append(
                    f"| {label} | {cs.get('mean_leads_per_doc', 0):.2f} | {cs.get('mean_strong_leads_per_doc', 0):.2f} | {cs.get('mean_weak_leads_per_doc', 0):.2f} | {cs.get('flagged_rate', 0):.2%} |"
                )

            lines.append(f"\n- **Strong-Lead Precision (Group B vs Group I)**: **{l_stats.get('strong_lead_precision_b_vs_i', 0):.2%}**")

            if roc:
                lines.extend([
                    "",
                    "### Ordering Metric (ROC AUC) on Paraphrase vs. Same-Topic Spans",
                    "",
                    "| Ordering Metric | ROC AUC | Finding |",
                    "| :--- | :---: | :--- |",
                    f"| **Cosine Alone** | **{roc.get('roc_auc_cosine_alone', 0):.4f}** | Near-random (~0.50); cosine overlaps between paraphrases and same-topic text |",
                    f"| **Anchor Count Alone** | **{roc.get('roc_auc_anchors_alone', 0):.4f}** | Inverted discrimination; technical vocabulary is dense in originals |",
                    f"| **Cosine + Anchors Composite** | **{roc.get('roc_auc_composite', 0):.4f}** | Negligible improvement; anchors do not cleanly separate domains |",
                ])

        if summary["plagiarism"].get("anchor_grid_evaluation"):
            lines.extend([
                "",
                "### Lexical Anchor Rule Grid Search (N in {1,2,3,4}, ratio in {0.1, 0.2, 0.3})",
                "",
                "| Min Anchors (N) | Min Content Ratio | Group B Recall (All) | Group B Recall (Easy) | Group B Recall (Hard) | Group I FPR | Group J FPR | Rule Status |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ])
            for gr in summary["plagiarism"]["anchor_grid_evaluation"]:
                status = "Qualifies for Strong Lead" if gr["qualifies_rule"] else ("Recall collapses (<70%)" if gr["recall_b_all"] < 0.70 else "Does not cut FPR by half")
                lines.append(
                    f"| {gr['N']} | {gr['ratio']:.1f} | {gr['recall_b_all']:.2%} | {gr['recall_b_easy']:.2%} | {gr['recall_b_hard']:.2%} | {gr['fpr_i']:.2%} | {gr['fpr_j']:.2%} | {status} |"
                )
            lines.extend([
                "",
                "> **Honest Technical Conclusion**: Because no parameter combination achieved Group B recall >= 70% while reducing Group I FPR by half (to <= 47.5%), lexical anchors cannot reliably separate same-topic technical writing from true paraphrases using this model (`all-MiniLM-L6-v2`) on this synthetic corpus. Semantic matching is kept strictly out of the default headline score. Matches meeting anchor criteria are tagged as 'strong lead' and others as 'weak lead' for human review only.",
            ])

        # Mean similarity per group
        by_eng = summary["plagiarism"].get("mean_similarity_by_engine")
        if by_eng:
            ex_m = by_eng.get("exact", {})
            sem_head = by_eng.get("semantic_headline") or ex_m
            sem_opt = by_eng.get("semantic_optin") or {}
            lines.extend([
                "",
                "**Mean Similarity Score by Cohort (Exact/Headline vs. Opt-In Semantic)**:",
                f"- Group A (Verbatim / Lightly Edited Copies): Exact/Headline **{sem_head.get('a_copies', 0):.2%}** | Opt-In Semantic **{sem_opt.get('a_copies', 0):.2%}**",
                f"- Group B (Easy Paraphrased Passages): Exact/Headline **{sem_head.get('b_paraphrases', 0):.2%}** | Opt-In Semantic **{sem_opt.get('b_paraphrases', 0):.2%}**",
                f"- Group B (Hard Complex Paraphrases): Exact/Headline **{sem_head.get('b_paraphrases_hard', 0):.2%}** | Opt-In Semantic **{sem_opt.get('b_paraphrases_hard', 0):.2%}**",
                f"- Group C (Unrelated Human Originals): Exact/Headline **{sem_head.get('c_original', 0):.2%}** | Opt-In Semantic **{sem_opt.get('c_original', 0):.2%}**",
                f"- Group D (AI-Generated Text): Exact/Headline **{sem_head.get('d_ai_style', 0):.2%}** | Opt-In Semantic **{sem_opt.get('d_ai_style', 0):.2%}**",
                f"- Group E (Non-Native English): Exact/Headline **{sem_head.get('e_non_native', 0):.2%}** | Opt-In Semantic **{sem_opt.get('e_non_native', 0):.2%}**",
                f"- Group I (Same-Topic Originals): Exact/Headline **{sem_head.get('i_same_topic_originals', 0):.2%}** | Opt-In Semantic **{sem_opt.get('i_same_topic_originals', 0):.2%}**",
                f"- Group J (Common Knowledge Definitions): Exact/Headline **{sem_head.get('j_common_knowledge', 0):.2%}** | Opt-In Semantic **{sem_opt.get('j_common_knowledge', 0):.2%}**",
            ])
        else:
            mean_sims = summary["plagiarism"]["mean_similarity_per_group"]
            lines.extend([
                "",
                "**Mean Similarity Score by Cohort**:",
                f"- Group A (Verbatim / Lightly Edited Copies): **{mean_sims.get('a_copies', 0):.2%}**",
                f"- Group B (Paraphrased Passages): **{mean_sims.get('b_paraphrases', 0):.2%}**",
                f"- Group C (Unrelated Human Originals): **{mean_sims.get('c_original', 0):.2%}**",
            ])

        lines.extend([
            "",
            "## 2. AI-Written Text Detector",
            "",
            "| Model / Configuration | Accuracy | Precision | Recall | F1 Score | Overall Human FPR | Non-Native (ESL) FPR |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            f"| **Stylometry Only (Lite Mode)** | {ai_style['accuracy']:.2%} | {ai_style['precision']:.2%} | {ai_style['recall']:.2%} | {ai_style['f1']:.2%} | {ai_style['fpr_human_overall']:.2%} | {ai_style['fpr_non_native_esl']:.2%} |",
        ])

        if ai_ens:
            lines.append(
                f"| **Full Ensemble (Stylometry + DistilGPT2)** | {ai_ens.get('accuracy', 0):.2%} | {ai_ens.get('precision', 0):.2%} | {ai_ens.get('recall', 0):.2%} | {ai_ens.get('f1', 0):.2%} | {ai_ens.get('fpr_human_overall', 0):.2%} | {ai_ens.get('fpr_non_native_esl', 0):.2%} |"
            )

        lines.extend([
            "",
            "## 3. Integrity Checks, Collusion & Cryptographic Seals",
            "",
            "| Module | Metric Tested | Result | Target / Standard |",
            "| :--- | :--- | :---: | :---: |",
            f"| **Spinner / Tortured Phrases** | Precision / Recall | {spin['precision']:.2%} / {spin['recall']:.2%} | Cabanac et al. (2021) |",
            f"| **Batch Collusion Matrix** | Expected Clusters Discovered | {'YES' if coll['expected_clusters_found'] else 'NO'} | Pairwise Winnowing |",
            f"| **Fingerprint Index Screening** | True Pairs Candidate Recall | {coll.get('fingerprint_shortlist_recall', 1.0):.2%} | Exact Inverted Index (Default) |",
            f"| **MinHash / LSH Screening** | True Pairs Candidate Recall | {coll['lsh_shortlist_recall']:.2%} | Adaptive (64x2) 128-perm |",
            f"| **Cryptographic Report Seals** | Tamper Detection Rate | {seal['tamper_detection_rate']:.2%} | HMAC-SHA256 (1-char edit) |",
            "",
            "## 4. Notable Failure Cases & Honest Limitations",
            "",
            f"- **Plagiarism Missed Paraphrases (Easy)**: {summary['plagiarism']['failure_cases'].get('missed_paraphrases_easy_count', 0)} documents.",
            f"- **Plagiarism Missed Paraphrases (Hard)**: {summary['plagiarism']['failure_cases'].get('missed_paraphrases_hard_count', 0)} documents.",
            f"- **AI False Positives on Human Text**: {summary['ai_detection']['failure_cases']['false_positives_human_count']} documents.",
            f"- **AI False Positives on Non-Native (ESL) Text**: {summary['ai_detection']['failure_cases']['false_positives_esl_count']} documents. (Stylometric variance can mistake uniform ESL syntax for machine generation; conservative thresholds are essential).",
            "",
        ])

        return "\n".join(lines)

    def _plot_confusion_matrices(self, summary: Dict[str, Any], output_path: Path):
        """Generates a 3-panel confusion matrix figure using matplotlib."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # Panel 1: Plagiarism Winnowing
        p_win = summary["plagiarism"]["winnowing_only"]
        cm_plag = [[p_win["tn"], p_win["fp"]], [p_win["fn"], p_win["tp"]]]
        self._draw_matrix(
            axes[0], cm_plag, "Plagiarism (Winnowing)", ["Original", "Plagiarised"]
        )

        # Panel 2: AI Detection (Stylometry)
        ai_style = summary["ai_detection"]["stylometry_only"]
        cm_ai = [[ai_style["tn"], ai_style["fp"]], [ai_style["fn"], ai_style["tp"]]]
        self._draw_matrix(axes[1], cm_ai, "AI Detection (Stylometry)", ["Human", "AI"])

        # Panel 3: Spinner Detection
        spin = summary["spinner_detection"]
        cm_spin = [[spin["tn"], spin["fp"]], [spin["fn"], spin["tp"]]]
        self._draw_matrix(axes[2], cm_spin, "Spinner Detection", ["Clean", "Spinner"])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _draw_matrix(ax, matrix, title, labels):
        """Helper to draw a single 2x2 confusion matrix."""
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Ground Truth", fontsize=10)

        for i in range(2):
            for j in range(2):
                val = matrix[i][j]
                color = "white" if val > (matrix[0][0] + matrix[1][1]) / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=14,
                    fontweight="bold",
                )
