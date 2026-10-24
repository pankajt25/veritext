"""Streamlit web application for Veritext Stage 6B verification suite.

Features:
- Adaptive single document analysis (plagiarism, full-text retrieval, AI text, integrity)
- Extended exclusion filters with live recalculation without re-running engines
- Text-spinner and tortured-phrase detection (Cabanac et al. 2021)
- Local class repository integration and batch collusion heatmap matrix
"""

import io
from pathlib import Path
from typing import Any, Optional
import streamlit as st
import streamlit.components.v1 as components

from veritext.batch import run_batch_analysis
from veritext.classrepo.repo import ClassRepositoryManager
from veritext.ingest.pipeline import load_document
from veritext.planner import load_limits_config, make_plan
from veritext.report.dashboard import (
    generate_paragraph_heat_strip,
    generate_portfolio_csv,
    render_heat_strip_html,
    render_portfolio_html,
)
from veritext.report.diff import extract_report_diffs
from veritext.report.highlight import render_highlighted_html, render_legend_html
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.review import (
    FindingReview,
    ReviewSession,
    ReviewStatus,
    RubricAssessment,
    RubricScore,
    format_report_for_view,
    get_default_criteria,
    load_review_session,
    save_review_session,
)
from veritext.similarity.crosslang import is_crosslang_model_available
from veritext.similarity.filters import (
    ExclusionFilters,
    add_to_allowlist,
    load_allowlist,
    recalculate_report_filters,
    remove_from_allowlist,
)
from veritext.sources.clients.core import is_core_configured
from veritext.style import check_languagetool_support, lint_style
from veritext.ui.services import prepare_document_and_plan, run_pipeline

ROBERTA_MODEL_DIR = Path("data/models/roberta-chatgpt")


def init_session_state():
    """Initializes session state keys if not already present."""
    if "full_report" not in st.session_state:
        st.session_state["full_report"] = None
    if "ignored_spans" not in st.session_state:
        st.session_state["ignored_spans"] = set()
    if "excluded_sources" not in st.session_state:
        st.session_state["excluded_sources"] = set()
    if "template_text" not in st.session_state:
        st.session_state["template_text"] = ""
    if "last_analyzed_file" not in st.session_state:
        st.session_state["last_analyzed_file"] = ""
    if "cancel_requested" not in st.session_state:
        st.session_state["cancel_requested"] = False
    if "batch_result" not in st.session_state:
        st.session_state["batch_result"] = None
    if "review_session" not in st.session_state:
        st.session_state["review_session"] = None
    if "perspective_view" not in st.session_state:
        st.session_state["perspective_view"] = "Instructor View"


def main():
    st.set_page_config(
        page_title="Veritext - Academic Integrity Suite",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    limits_default = load_limits_config()

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.title("📝 Veritext")
        st.caption("Stage 6B Academic Integrity & Verification Suite")

        app_view = st.radio(
            "Navigation Mode",
            ["Single Document Verification", "Class Batch Collusion Matrix"],
            index=0,
        )
        st.markdown("---")

    # =========================================================================
    # VIEW 1: CLASS BATCH COLLUSION MATRIX
    # =========================================================================
    if app_view == "Class Batch Collusion Matrix":
        st.title("👥 Class Batch Collusion & Cross-Submission Matrix")
        st.markdown(
            "Compute pairwise winnowing similarity across an entire cohort, generate an interactive "
            "heatmap matrix, and detect collusion clusters with shared passage inspection. "
            "Engineered to be flat-memory safe for 100 submissions on 8 GB RAM."
        )

        with st.sidebar:
            st.subheader("Batch Settings")
            batch_threshold = st.slider(
                "Collusion Threshold (%)",
                min_value=10,
                max_value=90,
                value=30,
                step=5,
                help="Pairs with similarity at or above this threshold will be flagged and clustered.",
            )
            anonymize_batch = st.checkbox(
                "Anonymise Student Submissions",
                value=False,
                help="Display anonymised IDs (e.g. sub_001) instead of student filenames.",
            )

        st.subheader("1. Select Submissions")
        batch_source = st.radio(
            "Submissions Source",
            ["Upload Multiple Documents", "Local Class Repository", "Local Directory"],
            horizontal=True,
        )

        cohort_dir: Optional[Path] = None

        if batch_source == "Upload Multiple Documents":
            uploaded_batch_files = st.file_uploader(
                "Choose multiple documents (.docx, .pdf, .txt, .md, .tex)",
                type=["docx", "pdf", "txt", "md", "tex"],
                accept_multiple_files=True,
                help="Upload 2 or more student documents for pairwise collusion inspection.",
            )
            if uploaded_batch_files and len(uploaded_batch_files) >= 2:
                ts = Path("data/uploads/batch_temp")
                ts.mkdir(parents=True, exist_ok=True)
                for uf in uploaded_batch_files:
                    (ts / uf.name).write_bytes(uf.getvalue())
                cohort_dir = ts

        elif batch_source == "Local Class Repository":
            cmgr = ClassRepositoryManager()
            classes = cmgr.list_classes()
            if not classes:
                st.info("No class repositories found. Add one using `python -m veritext.classrepo add <class_name> <folder>`.")
            else:
                cnames = [c["class_name"] for c in classes]
                sel_class = st.selectbox("Select Class", options=cnames)
                st.caption(f"Repository path: `data/class_repository/{sel_class}`")
                cohort_dir = Path("data/class_repository") / sel_class

        else:
            dir_input = st.text_input("Directory path containing submission files", value="data/samples")
            if dir_input:
                cohort_dir = Path(dir_input)

        col_b_run, _ = st.columns([1, 3])
        run_batch_btn = col_b_run.button("🚀 Run Pairwise Analysis", type="primary")

        if run_batch_btn:
            if not cohort_dir or not cohort_dir.exists():
                st.error("Please provide a valid directory or upload at least 2 submissions.")
                return

            pbar = st.progress(0.0)
            pmsg = st.empty()

            try:
                with st.spinner("Computing pairwise winnowing fingerprints and collusion matrix..."):
                    result = run_batch_analysis(
                        folder=cohort_dir,
                        threshold=batch_threshold / 100.0,
                        anonymize=anonymize_batch,
                        progress_callback=lambda msg, frac: (pmsg.text(f"⏳ {msg}"), pbar.progress(frac)),
                    )
                st.session_state["batch_result"] = result
                pbar.empty()
                pmsg.success("✅ Batch analysis completed successfully!")
            except Exception as e:
                pbar.empty()
                pmsg.empty()
                st.error(f"❌ Batch analysis error: {e}")

        # Render batch results if available
        b_res = st.session_state.get("batch_result")
        if b_res:
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Submissions", len(b_res.submissions))
            m2.metric(f"Flagged Pairs (≥{b_res.threshold * 100.0:.0f}%)", len(b_res.pairs_above_threshold))
            m3.metric("Collusion Clusters", len(b_res.clusters))

            # HTML Heatmap display
            st.markdown("### 🗺️ Pairwise Similarity Heatmap")
            components.html(b_res.html_report, height=600, scrolling=True)

            # Clusters display
            st.markdown(f"### 🔗 Collusion Clusters (≥{b_res.threshold * 100.0:.0f}% Similarity)")
            if not b_res.clusters:
                st.info("No multi-submission collusion clusters detected.")
            else:
                for c_idx, cl in enumerate(b_res.clusters, 1):
                    with st.expander(f"Cluster #{c_idx} ({len(cl)} submissions): {', '.join(cl)}"):
                        st.write("Submissions in this cluster share significant verbatim or near-copy winnowing fingerprints.")

            # Flagged pairs with shared passages
            st.markdown("### 🔍 Flagged Pairs & Shared Passages")
            if not b_res.pairs_above_threshold:
                st.success("No submission pairs exceeded the collusion threshold.")
            else:
                for idx, p in enumerate(b_res.pairs_above_threshold, 1):
                    with st.expander(f"{idx}. {p.doc_a_name} ↔ {p.doc_b_name} ({p.similarity_percentage:.1f}% Similarity)"):
                        st.write(f"**Shared winnowing hashes:** {p.shared_hashes_count}")
                        if p.shared_passages:
                            st.markdown("**Shared Passages:**")
                            for sp in p.shared_passages:
                                st.markdown(f"> *\"{sp.text_snippet}\"*")

            # Downloads
            st.markdown("### 💾 Export Matrix")
            cd1, cd2 = st.columns(2)
            cd1.download_button(
                "📥 Download Similarity Matrix (CSV)",
                data=b_res.csv_content,
                file_name="similarity_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
            cd2.download_button(
                "📥 Download Standalone HTML Heatmap",
                data=b_res.html_report,
                file_name="collusion_heatmap.html",
                mime="text/html",
                use_container_width=True,
            )
        return

    # =========================================================================
    # VIEW 2: SINGLE DOCUMENT VERIFICATION
    # =========================================================================

    # Sidebar: Document upload
    with st.sidebar:
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document (.docx, .pdf, .txt, .md, .tex)",
            type=["docx", "pdf", "txt", "md", "tex"],
            help="Word DOCX, PDF, Text, Markdown, and LaTeX supported.",
        )

        st.markdown("---")
        st.subheader("2. Adaptive Planning Mode")

        mode_options = ["Auto", "Full document", "Custom"]
        default_mode_idx = 0
        cfg_mode = limits_default.get("mode", "auto").capitalize()
        if cfg_mode == "Full":
            default_mode_idx = 1
        elif cfg_mode == "Custom":
            default_mode_idx = 2

        mode_choice = st.selectbox(
            "Planning Mode",
            options=mode_options,
            index=default_mode_idx,
            help="Auto: Automatically fits engines to time budget with section sampling if needed.\n"
                 "Full document: Runs all enabled engines across the full document without caps.\n"
                 "Custom: Specify custom word limits or time budget.",
        )

        time_budget = st.slider(
            "Engine Time Budget (min)",
            min_value=1,
            max_value=60,
            value=int(limits_default.get("time_budget_minutes", 10)),
            step=1,
        )

        custom_ai_words = 0
        custom_semantic_words = 0
        if mode_choice == "Custom":
            st.caption("Custom Mode Overrides (0 = unlimited):")
            custom_ai_words = st.number_input("AI Detection Max Words", min_value=0, max_value=200000, value=0, step=500)
            custom_semantic_words = st.number_input("Semantic Matching Max Words", min_value=0, max_value=200000, value=0, step=500)

        st.markdown("---")
        st.subheader("3. Engines & Sources")

        use_ai = st.checkbox("AI-Text Detection", value=True, help="Stylometric features + DistilGPT-2 language model perplexity.")
        use_integrity = st.checkbox("Integrity Checks", value=True, help="Hidden formatting, homoglyphs, citations, and text spinners.")
        use_online = st.checkbox("Online Source Search", value=False, help="Search open repositories (arXiv, OpenAlex, Crossref, Wikipedia).")
        use_semantic = st.checkbox("Semantic Paraphrase Check", value=False, help="Dense vector embedding search (all-MiniLM-L6-v2).")
        st.caption("Turn this on to catch reworded text; it is slower on this machine.")

        use_fulltext = st.checkbox(
            "Fetch full text of top matches",
            value=use_online,
            disabled=not use_online,
            help="Fetch open-access full-text body (arXiv, Europe PMC, OpenAlex, Wikipedia) for top candidates.",
        )

        roberta_available = ROBERTA_MODEL_DIR.exists()
        use_roberta = st.checkbox(
            "Deep Classifier (RoBERTa)",
            value=False,
            disabled=not roberta_available,
            help="Pretrained classifier weights (Hello-SimpleAI/chatgpt-detector-roberta).",
        )

        crosslang_avail = is_crosslang_model_available()
        use_crosslang = st.checkbox(
            "Cross-Language Matching",
            value=False,
            disabled=not crosslang_avail,
            help="Multilingual dense embeddings (paraphrase-multilingual-MiniLM-L12-v2).",
        )
        if not crosslang_avail:
            st.caption("ℹ️ Cross-language model missing. To download: `python -m veritext.similarity --download-crosslang-model`")

        if is_core_configured():
            st.caption("🟢 CORE source client active (CORE_API_KEY detected)")
        else:
            st.caption("⚪ CORE source not configured (set CORE_API_KEY to enable)")

        min_words = st.slider(
            "Min Match Words",
            min_value=4,
            max_value=20,
            value=8,
            help="Minimum consecutive matching words to trigger a plagiarism span.",
        )

        st.markdown("---")
        st.subheader("4. Class Repository")
        cmgr = ClassRepositoryManager()
        avail_classes = cmgr.list_classes()
        class_options = ["None"] + [c["class_name"] for c in avail_classes]
        selected_class = st.selectbox(
            "Include Class Repository",
            options=class_options,
            index=0,
            help="Include submissions from a local class repository as comparison sources.",
        )
        show_class_names = st.checkbox(
            "Show Student Names (Instructional Staff Only)",
            value=False,
            help="Default: Anonymised submission IDs are shown for student confidentiality.",
        )

        st.markdown("---")
        st.subheader("5. Extended Exclusion Filters")
        tpl_uploader = st.file_uploader(
            "Template / Assignment Prompt File",
            type=["docx", "pdf", "txt", "md"],
            help="Exclude matches matching instructor prompt or boilerplate.",
        )
        if tpl_uploader is not None:
            try:
                tpl_bytes = tpl_uploader.getvalue()
                t_doc = load_document(tpl_bytes, filename=tpl_uploader.name)
                st.session_state["template_text"] = t_doc.clean_text or t_doc.raw_text or ""
            except Exception:
                st.session_state["template_text"] = tpl_uploader.getvalue().decode("utf-8", errors="replace")

        ex_small = st.checkbox("Exclude small matches (<25 chars)", value=True)
        ex_quotes = st.checkbox("Exclude quoted text", value=False)
        ex_template = st.checkbox("Exclude assignment template", value=bool(tpl_uploader))
        ex_academic = st.checkbox("Exclude common academic phrases", value=False)
        ex_allowlist = st.checkbox("Exclude allowlisted phrases", value=False)

        st.markdown("---")
        st.subheader("5. Perspective View")
        perspective_view = st.radio(
            "Perspective Mode",
            ["Instructor View", "Student View"],
            index=0 if st.session_state.get("perspective_view") == "Instructor View" else 1,
            help="Student View emphasizes developmental feedback and masks ranking/adversarial language.",
        )
        st.session_state["perspective_view"] = perspective_view

        st.markdown("---")

    # Disclaimer Banner
    st.info(
        "⚖️ **Responsible Use Notice:** Veritext provides statistical indicators and heuristic signals for self-review, "
        "not definitive proof of academic misconduct. Automated checks produce false positives; human pedagogical judgment is required.",
        icon="ℹ️",
    )

    # Document Profile & Live Plan Computation
    doc_obj = None
    profile_obj = None
    plan_obj = None

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            current_settings = {
                "mode": mode_choice.lower().replace(" document", ""),
                "time_budget_minutes": time_budget,
                "use_exact": True,
                "use_checks": use_integrity,
                "use_ai": use_ai,
                "use_semantic": use_semantic,
                "custom_ai_words": custom_ai_words if mode_choice == "Custom" else 0,
                "custom_semantic_words": custom_semantic_words if mode_choice == "Custom" else 0,
            }
            doc_obj, profile_obj, plan_obj = prepare_document_and_plan(
                target=file_bytes,
                filename=uploaded_file.name,
                settings=current_settings,
            )
        except Exception as e:
            st.error(f"Error parsing uploaded document: {e}")

    # Sidebar: Execution buttons
    with st.sidebar:
        if plan_obj is not None:
            total_sec = plan_obj.total_estimated_seconds
            total_m = total_sec / 60.0
            st.markdown(f"⏱️ **Total Estimated Time:** {total_sec:.1f}s (~{total_m:.1f} min)")

        col_run, col_stop = st.columns(2)
        run_btn = col_run.button("🚀 Run Analysis", type="primary", use_container_width=True)
        stop_btn = col_stop.button("⏹️ Stop Analysis", use_container_width=True)

        if stop_btn:
            st.session_state["cancel_requested"] = True
            st.warning("Cancellation requested. Stopping between chunks...")

    # Run Analysis
    if run_btn:
        if uploaded_file is None:
            st.error("Please upload a document to analyze.")
            return

        st.session_state["cancel_requested"] = False

        pipeline_settings = {
            "mode": mode_choice.lower().replace(" document", ""),
            "time_budget_minutes": time_budget,
            "use_online": use_online,
            "use_semantic": use_semantic,
            "use_fulltext": use_fulltext,
            "use_ai": use_ai,
            "use_integrity": use_integrity,
            "use_roberta": use_roberta,
            "min_words": min_words,
            "class_name": selected_class if selected_class != "None" else None,
            "show_names": show_class_names,
            "custom_ai_words": custom_ai_words if mode_choice == "Custom" else 0,
            "custom_semantic_words": custom_semantic_words if mode_choice == "Custom" else 0,
        }

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        live_badges = st.empty()
        live_status = {}

        def update_progress(msg: str, frac: float):
            status_text.text(f"⏳ {msg}")
            progress_bar.progress(frac)

        def on_live_engine_result(engine_name: str, res_obj: Any):
            live_status[engine_name] = res_obj
            chips = []
            if "similarity" in live_status:
                sim = live_status["similarity"]
                pct = f"{sim.similarity_percentage:.1f}%" if hasattr(sim, "similarity_percentage") else "Done"
                chips.append(f"🟢 Plagiarism: {pct}")
            if "ai_detect" in live_status:
                ai = live_status["ai_detect"]
                prob = f"{ai.overall_probability:.1f}%" if hasattr(ai, "overall_probability") and ai.overall_probability is not None else "Done"
                chips.append(f"🟣 AI Text: {prob}")
            if "integrity" in live_status:
                integ = live_status["integrity"]
                f_count = sum(len(cr.findings) for cr in getattr(integ, "results", []))
                chips.append(f"🔵 Integrity: {f_count} findings")
            if chips:
                live_badges.info(" | ".join(chips))

        def is_cancelled() -> bool:
            return bool(st.session_state.get("cancel_requested", False))

        try:
            with st.spinner("Executing adaptive analysis pipeline..."):
                file_bytes = uploaded_file.getvalue()
                report = run_pipeline(
                    target=file_bytes,
                    filename=uploaded_file.name,
                    settings=pipeline_settings,
                    progress_callback=update_progress,
                    cancel_check=is_cancelled,
                    live_result_callback=on_live_engine_result,
                )

            st.session_state["full_report"] = report
            st.session_state["ignored_spans"] = set()
            st.session_state["excluded_sources"] = set()
            st.session_state["last_analyzed_file"] = uploaded_file.name
            status_text.success("✅ Analysis completed successfully!")
            progress_bar.empty()
            live_badges.empty()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {e}")
            return

    # Check for live filter recalculation
    report = st.session_state["full_report"]

    if report is not None and report.similarity:
        # Build ExclusionFilters from current sidebar controls
        al_items = load_allowlist() if ex_allowlist else []
        live_filters = ExclusionFilters(
            exclude_small_matches=ex_small,
            min_match_chars=25,
            exclude_quotes=ex_quotes,
            exclude_sources=st.session_state.get("excluded_sources", set()),
            exclude_template=ex_template,
            template_text=st.session_state.get("template_text", ""),
            exclude_allowlist=ex_allowlist,
            allowlist=al_items,
            exclude_academic_phrases=ex_academic,
        )
        report = recalculate_report_filters(report, live_filters)
        st.session_state["full_report"] = report

    # MAIN VIEW
    if report is None:
        if profile_obj is not None and plan_obj is not None:
            st.markdown(f"### 📄 Document Profile: `{uploaded_file.name}`")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Word Count", f"{profile_obj.word_count:,}")
            p2.metric("Character Count", f"{profile_obj.character_count:,}")
            p3.metric("Estimated Pages", f"{profile_obj.page_estimate:,}", "500 words/page")
            p4.metric("Size Class", profile_obj.size_class.upper(), f"{profile_obj.language_guess} detected")

            st.markdown("### 📋 Analysis Plan")
            st.caption(f"Mode: **{plan_obj.mode.upper()}** | Per-Engine Budget: **{plan_obj.time_budget_minutes:.1f} min**")

            plan_table_data = []
            for eng_name, ep in plan_obj.engines.items():
                plan_table_data.append({
                    "Engine": eng_name,
                    "Status": "Run" if ep.enabled else "Skip",
                    "Words to Analyse": f"{ep.words_to_analyze:,}",
                    "Coverage": f"{ep.coverage_percent:.1f}%",
                    "Estimated Time": f"{ep.estimated_seconds:.1f}s",
                    "Sampling Mode": ep.sampling_mode,
                    "Reason": ep.reason,
                })
            st.dataframe(plan_table_data, use_container_width=True, hide_index=True)
            st.markdown("---")
            st.info("👈 Review the analysis plan above and click **🚀 Run Analysis** in the sidebar to begin.")
            return

        st.markdown("### Welcome to Veritext")
        st.write(
            "Veritext is an adaptive, privacy-respecting, open-source verification suite designed to help "
            "students, educators, and researchers evaluate academic integrity offline on modest hardware."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 🔍 Plagiarism & Full-Text")
            st.write("Winnowing fingerprinting with open-access full-text retrieval from arXiv, Europe PMC, OpenAlex, Wikipedia.")
        with c2:
            st.markdown("#### 🤖 AI-Text Stylometry")
            st.write("Chunked distilgpt2 perplexity, burstiness, and stylometric cadence with flat memory consumption.")
        with c3:
            st.markdown("#### 🛡️ Integrity & Text Spinners")
            st.write("Cabanac et al. (2021) tortured phrases, rare synonym substitution, Cyrillic homoglyphs, and white fonts.")

        st.markdown("---")
        st.info("👈 Upload a document in the left sidebar to inspect its statistical profile and generate an adaptive plan.")
        return

    # Tabs layout
    tab_summary, tab_dashboard, tab_doc, tab_review, tab_style, tab_sources, tab_ai, tab_integrity, tab_download = st.tabs([
        "📊 Summary & Plan",
        "📈 Multi-Signal Dashboard",
        "📄 Highlighted Document",
        "📝 Review & Rubric",
        "✍️ Style & Grammar",
        "📚 Sources & Exclusions",
        "🤖 AI Analysis",
        "🛡️ Integrity & Text Spinners",
        "💾 Export & Download",
    ])

    # 1. TAB: SUMMARY & PLAN
    with tab_summary:
        st.subheader(f"Analysis Summary: {report.document_name}")

        m1, m2, m3 = st.columns(3)
        with m1:
            sim_val = f"{report.similarity.similarity_percentage:.1f}%" if report.similarity else "N/A"
            score_label = report.similarity.metadata.get("score_label", "exact & near-copy only") if report.similarity else "Engine not executed"
            st.metric("Headline Similarity", sim_val, score_label)

        with m2:
            if not report.ai_detect:
                st.metric("AI Content Probability", "Not run", "Engine not executed")
            elif getattr(report.ai_detect, "status", "success") in ("insufficient text", "insufficient_text"):
                st.metric("AI Content Probability", "Insufficient text (under 150 words, not scored)", "Under 150 words")
            else:
                ai = report.ai_detect
                ai_prob = getattr(ai, "overall_probability", getattr(ai, "ai_probability", None))
                ai_val = f"{ai_prob:.1f}%" if ai_prob is not None else "Not run"
                ai_conf = getattr(ai, "confidence_band", getattr(ai, "confidence", "none")).lower()
                st.metric("AI Content Probability", ai_val, f"confidence in this estimate: {ai_conf}")

        with m3:
            high_cnt = sum(sum(1 for f in r.findings if f.severity == "high") for r in report.integrity.results) if report.integrity else 0
            warn_cnt = sum(sum(1 for f in r.findings if f.severity == "warning") for r in report.integrity.results) if report.integrity else 0
            info_cnt = sum(sum(1 for f in r.findings if f.severity == "info") for r in report.integrity.results) if report.integrity else 0
            total_integ = high_cnt + warn_cnt + info_cnt
            st.metric("Integrity Findings", total_integ, f"{high_cnt} High, {warn_cnt} Warn, {info_cnt} Info")

        if report.similarity:
            leads_cnt = report.similarity.metadata.get("paraphrase_leads_count", 0)
            leads_pct = report.similarity.metadata.get("paraphrase_leads_percentage", 0.0)
            strong_cnt = report.similarity.metadata.get("strong_leads_count", 0)
            weak_cnt = report.similarity.metadata.get("weak_leads_count", 0)
            if leads_cnt > 0:
                expl = report.similarity.metadata.get(
                    "explanatory_line",
                    "Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source.",
                )
                guide = report.similarity.metadata.get(
                    "interpretation_guide",
                    "Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only",
                )
                st.info(
                    f"**Paraphrase leads (review):** {leads_cnt} passages, {leads_pct:.1f}% of text "
                    f"({strong_cnt} strong lead(s), {weak_cnt} weak lead(s))\n\n"
                    f"*{expl}*\n\n"
                    f"**Guide:** *{guide}*"
                )

        st.markdown("---")
        st.markdown("#### Executive Overview")
        st.write(report.summary)

        # Active filters display
        active_filts = report.settings.get("active_filters", [])
        if active_filts:
            st.markdown(f"**Active Exclusion Filters:** {', '.join(active_filts)}")

        if report.profile:
            st.markdown("---")
            st.markdown("#### Document Profile & Engine Coverage")
            prof = report.profile
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Total Words:** {prof.get('word_count', 0):,}")
            c2.markdown(f"**Total Characters:** {prof.get('character_count', 0):,}")
            c3.markdown(f"**Estimated Pages:** {prof.get('page_estimate', 1):,}")
            c4.markdown(f"**Size Class:** {prof.get('size_class', 'medium').upper()}")

    # 1b. TAB: MULTI-SIGNAL DASHBOARD
    with tab_dashboard:
        st.subheader("Paragraph Multi-Signal Heat Strip")
        st.caption("Side-by-side visualization of similarity overlap, AI likelihood, and integrity flags per paragraph.")
        heat_strip_data = generate_paragraph_heat_strip(
            report.original_text,
            similarity=report.similarity,
            ai_detect=report.ai_detect,
            integrity=report.integrity,
            review_session=st.session_state.get("review_session"),
        )
        components.html(render_heat_strip_html(heat_strip_data), height=550, scrolling=True)

    # 1c. TAB: REVIEW & RUBRIC
    with tab_review:
        st.subheader("Instructor Review & Rubric Scoring")

        if st.session_state.get("review_session") is None:
            st.session_state["review_session"] = ReviewSession(
                report_id=report.document_name or "current_doc",
                document_name=report.document_name,
                author="Instructor",
            )
        rev_session: ReviewSession = st.session_state["review_session"]

        st.markdown("#### 1. Standard Academic Integrity Rubric")
        criteria = get_default_criteria()
        rubric_scores = {}
        r_cols = st.columns(2)
        for idx, crit in enumerate(criteria):
            with r_cols[idx % 2]:
                st.markdown(f"**{crit.name}** (0–4)")
                st.caption(crit.description)
                existing_s = rev_session.rubric.scores.get(crit.id).score if rev_session.rubric and crit.id in rev_session.rubric.scores else 3
                score_val = st.slider(f"Score: {crit.name}", 0, 4, int(existing_s), key=f"rubric_{crit.id}")
                comm_val = st.text_input(f"Notes for {crit.name}", key=f"rubric_comm_{crit.id}")
                rubric_scores[crit.id] = RubricScore(criterion_id=crit.id, score=score_val, comment=comm_val)

        overall_rubric_comm = st.text_area("Overall Instructor Rubric Feedback", value=rev_session.rubric.overall_comment if rev_session.rubric else "")
        rev_session.rubric = RubricAssessment(
            name="Standard Academic Integrity & Writing Rubric",
            scores=rubric_scores,
            overall_comment=overall_rubric_comm,
        )
        report.metadata["rubric"] = rev_session.rubric.to_dict()
        report.metadata["review_session"] = rev_session.to_dict()

        st.info(f"🏆 Total Rubric Score: **{rev_session.rubric.total_score} / {rev_session.rubric.max_possible} ({rev_session.rubric.percentage:.1f}%)**")

        st.markdown("---")
        st.markdown("#### 2. Item Review Decisions & Annotations")
        st.caption("Set status to 'Ignored' for false positives; ignored items are excluded from similarity calculation without re-running engines.")

        if report.highlights:
            for h in report.highlights[:15]:
                rev_c1, rev_c2, rev_c3 = st.columns([2, 2, 4])
                rev_c1.markdown(f"**{h.category.upper()}** (chars {h.start}:{h.end})")
                current_rev = rev_session.get_review(h.id)
                curr_status = current_rev.status if current_rev else ("ignored" if h.id in st.session_state["ignored_spans"] else "needs_discussion")

                new_status = rev_c2.selectbox(
                    "Decision",
                    ["needs_discussion", "accepted", "ignored"],
                    index=["needs_discussion", "accepted", "ignored"].index(curr_status),
                    key=f"rev_status_{h.id}",
                )
                new_comm = rev_c3.text_input("Instructor Comment", value=current_rev.comment if current_rev else "", key=f"rev_comm_{h.id}")

                rev_session.set_review(
                    target_id=h.id,
                    status=new_status,
                    comment=new_comm,
                    start=h.start,
                    end=h.end,
                )
                if new_status == "ignored" and h.id not in st.session_state["ignored_spans"]:
                    st.session_state["ignored_spans"].add(h.id)
                    st.rerun()
                elif new_status != "ignored" and h.id in st.session_state["ignored_spans"]:
                    st.session_state["ignored_spans"].discard(h.id)
                    st.rerun()

        if st.button("💾 Persist Review Session to Disk", key="btn_save_review_session"):
            p = save_review_session(rev_session)
            st.success(f"Review session successfully persisted to `{p}`!")

    # 1d. TAB: STYLE & GRAMMAR
    with tab_style:
        st.subheader("Grammar & Style Feedback (Informational)")
        st.caption("Style rules provide formative writing suggestions with info-level severity and never alter similarity or AI scores.")

        style_res = lint_style(report.clean_text or report.original_text)
        report.metadata["style"] = style_res.to_dict()

        if style_res.metrics:
            sm = style_res.metrics
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Flesch Reading Ease", f"{sm.flesch_reading_ease:.1f}", sm.reading_ease_label)
            sm2.metric("Gunning Fog Index", f"{sm.gunning_fog:.1f}", sm.reading_level)
            sm3.metric("Total Sentences", f"{sm.total_sentences:,}")
            sm4.metric("Complex Words", f"{sm.complex_words:,}", f"{(sm.complex_words / max(1, sm.total_words)) * 100:.1f}%")

        st.markdown("---")
        lt_stat = style_res.engine_status.get("languagetool", {})
        if not lt_stat.get("available"):
            st.info(f"ℹ️ **Optional Engine Status:** {lt_stat.get('reason', 'Built-in rules active.')}")

        st.markdown("#### Formative Writing Suggestions")
        if not style_res.findings:
            st.success("No stylistic issues or grammatical redundancies detected!")
        else:
            style_table = []
            for sf in style_res.findings[:25]:
                style_table.append({
                    "Rule": sf.check.replace("style_", "").replace("_", " ").title(),
                    "Offsets": f"{sf.start}:{sf.end}",
                    "Suggestion": sf.message,
                })
            st.dataframe(style_table, use_container_width=True, hide_index=True)

    # 2. TAB: HIGHLIGHTED DOCUMENT
    with tab_doc:
        st.subheader("Highlighted Document View")

        available_cats = sorted(list({h.category for h in report.highlights}))
        if available_cats:
            col_filter, col_manage = st.columns([2, 1])
            with col_filter:
                selected_cats = st.multiselect(
                    "Filter Categories:",
                    options=available_cats,
                    default=available_cats,
                )
            with col_manage:
                with st.expander("Dismiss / Accept Specific Findings"):
                    for h in report.highlights:
                        is_ignored = h.id in st.session_state["ignored_spans"]
                        checked = st.checkbox(
                            f"Dismiss: {h.category.upper()} (offset {h.start}:{h.end})",
                            value=is_ignored,
                            key=f"chk_ignore_{h.id}",
                        )
                        if checked and not is_ignored:
                            st.session_state["ignored_spans"].add(h.id)
                            st.rerun()
                        elif not checked and is_ignored:
                            st.session_state["ignored_spans"].discard(h.id)
                            st.rerun()

            active_highlights = [
                h for h in report.highlights
                if h.category in selected_cats and h.id not in st.session_state["ignored_spans"] and not h.ignored
            ]
        else:
            active_highlights = []

        st.markdown(render_legend_html(), unsafe_allow_html=True)

        html_view = render_highlighted_html(
            text=report.original_text,
            spans=active_highlights,
            ignored_ids=st.session_state["ignored_spans"],
        )
        st.markdown(
            f'<div style="max-height: 600px; overflow-y: auto; padding: 1.5rem; background-color: #ffffff; '
            f'border: 1px solid #cbd5e1; border-radius: 8px; font-family: Georgia, serif; line-height: 1.8; '
            f'color: #0f172a;">{html_view}</div>',
            unsafe_allow_html=True,
        )

        # Side-by-Side Source Diffs
        diff_items = extract_report_diffs(report, max_matches=10)
        if diff_items:
            st.markdown("---")
            st.subheader("Side-by-Side Source Comparisons (Top Matches)")
            st.caption("Word-level alignment of shared passages with bounded excerpts (max 1,500 characters). Click any match to view.")
            for d in diff_items:
                st.markdown(d.render_html_details(), unsafe_allow_html=True)

    # 3. TAB: SOURCES & EXCLUSIONS
    with tab_sources:
        st.subheader("Plagiarism & Matching Sources")

        with st.expander("Manage Persistent Allowlist of Harmless Phrases"):
            curr_allowlist = load_allowlist()
            new_al_phrase = st.text_input("Add harmless phrase to persistent allowlist", key="inp_add_al_tab")
            if st.button("➕ Add Phrase to Allowlist", key="btn_add_al_tab"):
                if new_al_phrase.strip():
                    add_to_allowlist(new_al_phrase.strip())
                    st.success(f"Added: '{new_al_phrase.strip()}'")
                    st.rerun()

            if curr_allowlist:
                st.caption(f"Currently active allowlisted phrases ({len(curr_allowlist)} total):")
                for p_idx, p_str in enumerate(curr_allowlist):
                    c_txt, c_del = st.columns([4, 1])
                    c_txt.code(p_str)
                    if c_del.button("❌", key=f"del_al_tab_{p_idx}"):
                        remove_from_allowlist(p_str)
                        st.rerun()

        st.markdown("---")

        if st.session_state.get("excluded_sources"):
            c_wrn, c_rst = st.columns([3, 1])
            c_wrn.warning(f"🚫 {len(st.session_state['excluded_sources'])} source(s) excluded from scoring.")
            if c_rst.button("🔄 Restore all excluded sources"):
                st.session_state["excluded_sources"] = set()
                st.rerun()

        if not report.similarity or not report.similarity.sources:
            st.info("No matching plagiarism sources detected in local reference library or online sources.")
        else:
            st.caption("Strong lead: several matching sentences and shared specific terms; weak lead: similar topic only")
            st.caption("Similar meaning is not proof of copying; passages on the same topic often match. Check the linked source.")
            for s in report.similarity.sources:
                lead_suffix = f" | {s.lead_count} lead(s)" if getattr(s, "lead_count", 0) > 0 else ""
                with st.expander(f"{s.source_title} - {s.percentage:.1f}% headline overlap{lead_suffix} ({s.origin.upper()})"):
                    col_info, col_ex = st.columns([3, 1])
                    with col_info:
                        st.write(f"**Source Origin:** `{s.origin}`")
                        if s.source_url:
                            st.write(f"**URL:** [{s.source_url}]({s.source_url})")
                        st.write(f"**Headline Matched Characters:** {s.matched_chars} chars across {s.match_count} distinct span(s)")
                        if getattr(s, "lead_count", 0) > 0:
                            st.write(
                                f"**Paraphrase Leads:** {s.lead_count} lead(s) "
                                f"({s.strong_lead_count} strong, {s.weak_lead_count} weak, ~{s.lead_percentage:.1f}% of text)"
                            )
                    with col_ex:
                        if st.button(f"🚫 Exclude Source", key=f"btn_ex_src_{s.source_id}"):
                            st.session_state["excluded_sources"].add(s.source_id)
                            st.rerun()

                    st.markdown("---")
                    st.markdown("**Sample Matched Passages:**")
                    for sp in s.spans[:3]:
                        st.markdown(f"> *\"{sp.matched_text[:200]}...\"*")

    # 4. TAB: AI ANALYSIS
    with tab_ai:
        st.subheader("AI Content Stylometric & Language Model Analysis")
        if not report.ai_detect:
            st.info("AI detection was not executed for this document (Not run).")
        elif getattr(report.ai_detect, "status", "success") in ("insufficient text", "insufficient_text"):
            st.warning("⚠️ Insufficient text (under 150 words, not scored)")
        else:
            ai = report.ai_detect
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Overall AI Probability", f"{ai.overall_probability:.1f}%")
                st.write(f"**Confidence in this estimate:** `{ai.confidence_band.lower()}`")
                st.write(f"**Mode:** `{ai.mode}`")
                if ai.reduced_confidence:
                    st.caption("ℹ️ Operating in stylometry fallback mode (reduced confidence).")
            with c2:
                st.markdown("**Signal Explanations:**")
                for exp in ai.explanations:
                    st.write(f"• {exp}")

            st.markdown("---")
            st.markdown("#### Sentence-Level AI Classification")
            sent_data = []
            for s in ai.sentences:
                sent_data.append({
                    "Sentence": s.text[:90] + ("..." if len(s.text) > 90 else ""),
                    "Label": s.label.upper(),
                    "Score": f"{s.score:.2f}",
                    "Perplexity": f"{s.perplexity:.1f}" if s.perplexity is not None else "N/A",
                })
            st.dataframe(sent_data, use_container_width=True)

    # 5. TAB: INTEGRITY & TEXT SPINNERS
    with tab_integrity:
        st.subheader("Integrity, Formatting Obfuscation & Text Spinners")
        if not report.integrity or not report.integrity.results:
            st.info("No integrity issues detected.")
        else:
            spinner_cr = next((r for r in report.integrity.results if r.check_name == "spinner"), None)
            other_crs = [r for r in report.integrity.results if r.check_name != "spinner"]

            if spinner_cr:
                st.markdown("### 🔄 Text-Spinner & Tortured-Phrase Findings")
                st.caption(
                    "Detection based on Cabanac, Labbe and Magazinov (2021) documented tortured phrases "
                    "and semantic synonym substitutions."
                )
                if not spinner_cr.findings:
                    st.success("✅ Passed - No text-spinning or tortured phrases detected.")
                else:
                    for f in spinner_cr.findings:
                        st.warning(f"⚠️ **[WARNING]** {f.message} (offset {f.start}:{f.end})")
                        if "found_phrase" in f.details:
                            st.write(f"• Tortured phrase: `{f.details['found_phrase']}` (Standard term: *{f.details.get('standard_term')}*)")
                        elif "substitutions" in f.details:
                            st.write(f"• Semantic synonym substitution (similarity: {f.details.get('similarity_score', 0):.2f}):")
                            for sub in f.details["substitutions"]:
                                st.write(f"   - WordNet substitution: `{sub.get('doc_word')}` instead of `{sub.get('source_word')}`")
                st.markdown("---")

            st.markdown("### 🛡️ Other Integrity & Formatting Checks")
            for cr in other_crs:
                st.markdown(f"#### Module: `{cr.check_name}`")
                st.caption(cr.summary)
                if not cr.findings:
                    st.success("✅ Passed - No issues found.")
                else:
                    for f in cr.findings:
                        icon = "🔴" if f.severity == "high" else ("🟡" if f.severity == "warning" else "🔵")
                        st.markdown(f"{icon} **[{f.severity.upper()}]** {f.message}")
                st.markdown("---")

    # 6. TAB: EXPORT & DOWNLOAD
    with tab_download:
        st.subheader("Export & Download Analysis Report")
        st.write("Generate and download comprehensive offline report files with active exclusion filters.")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "📥 Download JSON Data",
                data=report.to_json(),
                file_name=f"{Path(report.document_name).stem}_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_d2:
            html_content = export_report_html(report, ignored_ids=st.session_state["ignored_spans"])
            st.download_button(
                "📥 Download HTML Report",
                data=html_content,
                file_name=f"{Path(report.document_name).stem}_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_d3:
            pdf_bytes = export_report_pdf(report, ignored_ids=st.session_state["ignored_spans"])
            st.download_button(
                "📥 Download PDF Document",
                data=pdf_bytes,
                file_name=f"{Path(report.document_name).stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
