"""Streamlit web application for Veritext Stage 6 adaptive report generation and document inspection."""

import io
from pathlib import Path
import streamlit as st

from veritext.planner import load_limits_config, make_plan
from veritext.report.highlight import render_highlighted_html, render_legend_html
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.ui.services import prepare_document_and_plan, run_pipeline

ROBERTA_MODEL_DIR = Path("data/models/roberta-chatgpt")


def init_session_state():
    """Initializes session state keys if not already present."""
    if "full_report" not in st.session_state:
        st.session_state["full_report"] = None
    if "ignored_spans" not in st.session_state:
        st.session_state["ignored_spans"] = set()
    if "last_analyzed_file" not in st.session_state:
        st.session_state["last_analyzed_file"] = ""
    if "cancel_requested" not in st.session_state:
        st.session_state["cancel_requested"] = False


def main():
    st.set_page_config(
        page_title="Veritext - Adaptive Academic Integrity Suite",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    limits_default = load_limits_config()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.title("📝 Veritext")
        st.caption("Adaptive Academic Integrity Suite (Free & Open-Source)")

        st.markdown("---")
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document (.docx, .pdf, .txt, .md, .tex)",
            type=["docx", "pdf", "txt", "md", "tex"],
            help="Supported formats: Word DOCX, PDF, Text, Markdown, and LaTeX. Adaptive analysis profiles documents without fixed caps.",
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
            help="Maximum allocated execution time budget per engine in Auto mode.",
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
        use_integrity = st.checkbox("Integrity Checks", value=True, help="Hidden formatting, homoglyphs, citations, and self-plagiarism.")
        use_online = st.checkbox("Online Source Search", value=False, help="Search open academic repositories (Crossref, arXiv, OpenAlex, Semantic Scholar). Default: offline only.")
        use_semantic = st.checkbox("Semantic Paraphrase Check", value=False, help="Dense vector embedding search (all-MiniLM-L6-v2). Slower on CPU.")

        # RoBERTa opt-in (disabled if model weights absent)
        roberta_available = ROBERTA_MODEL_DIR.exists()
        use_roberta = st.checkbox(
            "Deep Classifier (RoBERTa)",
            value=False,
            disabled=not roberta_available,
            help="Pretrained classifier weights (Hello-SimpleAI/chatgpt-detector-roberta). Run 'python -m veritext.ai_detect --download-models' to install.",
        )
        if not roberta_available:
            st.caption("ℹ️ *RoBERTa not downloaded. Run `python -m veritext.ai_detect --download-models` to enable.*")

        min_words = st.slider(
            "Min Match Words",
            min_value=4,
            max_value=20,
            value=8,
            help="Minimum consecutive matching words to trigger a plagiarism span.",
        )

        st.markdown("---")

    # --- PERMANENT DISCLAIMER BANNER ---
    st.info(
        "⚖️ **Responsible Use Notice:** Veritext provides statistical indicators and heuristic signals for self-review, "
        "not definitive proof of academic misconduct. AI detectors are probabilistic and may misidentify human prose, "
        "especially from non-native English writers. Human pedagogical judgment is always required.",
        icon="ℹ️",
    )

    # --- PROFILE AND LIVE PLAN COMPUTATION ---
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

    # --- EXECUTION BUTTONS & ESTIMATES ---
    with st.sidebar:
        if plan_obj is not None:
            total_sec = plan_obj.total_estimated_seconds
            total_m = total_sec / 60.0
            st.markdown(f"⏱️ **Total Estimated Time:** {total_sec:.1f}s (~{total_m:.1f} min)")

            if total_sec > 15 * 60:
                st.warning(
                    "⚠️ Total estimated execution time exceeds 15 minutes. "
                    "Suggestions: untick the semantic paraphrase check or reduce the engine time budget.",
                    icon="⚠️",
                )

        col_run, col_stop = st.columns(2)
        run_btn = col_run.button("🚀 Run Analysis", type="primary", use_container_width=True)
        stop_btn = col_stop.button("⏹️ Stop Analysis", use_container_width=True)

        if stop_btn:
            st.session_state["cancel_requested"] = True
            st.warning("Cancellation requested. Stopping between engine chunks...")

        st.caption(
            "*(Note: In Streamlit, cancellation checks between processing chunks and engine steps, "
            "cleanly preserving completed engine results.)*"
        )

    # --- EXECUTE ANALYSIS ---
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
            "use_ai": use_ai,
            "use_integrity": use_integrity,
            "use_roberta": use_roberta,
            "min_words": min_words,
            "custom_ai_words": custom_ai_words if mode_choice == "Custom" else 0,
            "custom_semantic_words": custom_semantic_words if mode_choice == "Custom" else 0,
        }

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, frac: float):
            status_text.text(f"⏳ {msg}")
            progress_bar.progress(frac)

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
                )

            st.session_state["full_report"] = report
            st.session_state["ignored_spans"] = set()
            st.session_state["last_analyzed_file"] = uploaded_file.name
            status_text.success("✅ Analysis completed successfully!")
            progress_bar.empty()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {e}")
            return

    # --- MAIN VIEW ---
    report = st.session_state["full_report"]

    if report is None:
        if profile_obj is not None and plan_obj is not None:
            # Show Document Profile Card
            st.markdown(f"### 📄 Document Profile: `{uploaded_file.name}`")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Word Count", f"{profile_obj.word_count:,}")
            p2.metric("Character Count", f"{profile_obj.character_count:,}")
            p3.metric("Estimated Pages", f"{profile_obj.page_estimate:,}", "500 words/page")
            p4.metric("Size Class", profile_obj.size_class.upper(), f"{profile_obj.language_guess} detected")

            # Show Live Plan Table
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

            if plan_obj.warnings:
                for w in plan_obj.warnings:
                    st.warning(f"⚠️ {w}")

            st.markdown("---")
            st.info("👈 Review the analysis plan above and click **🚀 Run Analysis** in the sidebar to begin.")
            return

        st.markdown("### Welcome to Veritext")
        st.write(
            "Veritext is an adaptive, privacy-respecting, open-source verification suite designed to help "
            "students, educators, and researchers evaluate academic integrity offline on modest hardware."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 Plagiarism Detection")
            st.write("Fast winnowing fingerprinting and dense neural semantic search across open reference libraries.")
        with col2:
            st.markdown("#### 🤖 AI-Text Stylometry")
            st.write("Chunked distilgpt2 perplexity, burstiness, and stylometric cadence with flat memory consumption.")
        with col3:
            st.markdown("#### 🛡️ Formatting & Integrity")
            st.write("Cyrillic homoglyphs, genuine zero-width evasion, DOCX/PDF white font tricks, and citation gaps.")

        st.markdown("---")
        st.info("👈 Upload a document in the left sidebar to inspect its statistical profile and generate an adaptive plan.")
        return

    # Tabs layout
    tab_summary, tab_doc, tab_sources, tab_ai, tab_integrity, tab_download = st.tabs([
        "📊 Summary & Plan",
        "📄 Highlighted Document",
        "📚 Sources",
        "🤖 AI Analysis",
        "🛡️ Integrity Findings",
        "💾 Export & Download",
    ])

    # 1. TAB: SUMMARY & PLAN
    with tab_summary:
        st.subheader(f"Analysis Summary: {report.document_name}")

        m1, m2, m3 = st.columns(3)
        with m1:
            sim_val = f"{report.similarity.similarity_percentage:.1f}%" if report.similarity else "N/A"
            src_cnt = len(report.similarity.sources) if report.similarity else 0
            st.metric("Similarity Score", sim_val, f"{src_cnt} matched source(s)")

        with m2:
            ai_prob = getattr(report.ai_detect, "overall_probability", getattr(report.ai_detect, "ai_probability", None)) if report.ai_detect else None
            ai_val = f"{ai_prob:.1f}%" if ai_prob is not None else "N/A"
            ai_conf = getattr(report.ai_detect, "confidence_band", getattr(report.ai_detect, "confidence", "N/A")).upper() if report.ai_detect else "N/A"
            ai_status = getattr(report.ai_detect, "status", getattr(report.ai_detect, "label", "N/A")) if report.ai_detect else "N/A"
            st.metric("AI Content Probability", ai_val, f"{ai_conf} ({ai_status})")

        with m3:
            high_cnt = sum(sum(1 for f in r.findings if f.severity == "high") for r in report.integrity.results) if report.integrity else 0
            warn_cnt = sum(sum(1 for f in r.findings if f.severity == "warning") for r in report.integrity.results) if report.integrity else 0
            info_cnt = sum(sum(1 for f in r.findings if f.severity == "info") for r in report.integrity.results) if report.integrity else 0
            total_integ = high_cnt + warn_cnt + info_cnt
            st.metric("Integrity Findings", total_integ, f"{high_cnt} High, {warn_cnt} Warn, {info_cnt} Info")

        st.markdown("---")
        st.markdown("#### Executive Overview")
        st.write(report.summary)

        # Profile & Plan section in report
        if report.profile:
            st.markdown("---")
            st.markdown("#### Document Profile & Engine Coverage")
            prof = report.profile
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Total Words:** {prof.get('word_count', 0):,}")
            c2.markdown(f"**Total Characters:** {prof.get('character_count', 0):,}")
            c3.markdown(f"**Estimated Pages:** {prof.get('page_estimate', 1):,}")
            c4.markdown(f"**Size Class:** {prof.get('size_class', 'medium').upper()}")

        if report.plan:
            plan_data = report.plan.get("engines", {})
            st.markdown("**Per-Engine Analysis Coverage:**")
            for eng_k, eng_v in plan_data.items():
                cov_desc = eng_v.get("coverage_description", "full document analysed")
                st.write(f"• **{eng_k.replace('_', ' ').title()}**: {cov_desc}")

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
                if h.category in selected_cats and h.id not in st.session_state["ignored_spans"]
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

    # 3. TAB: SOURCES
    with tab_sources:
        st.subheader("Plagiarism & Matching Sources")
        if not report.similarity or not report.similarity.sources:
            st.info("No matching plagiarism sources detected in local reference library or online sources.")
        else:
            for s in report.similarity.sources:
                with st.expander(f"{s.source_title} - {s.percentage:.1f}% ({s.origin.upper()})"):
                    st.write(f"**Source Origin:** `{s.origin}`")
                    if s.source_url:
                        st.write(f"**URL:** [{s.source_url}]({s.source_url})")
                    st.write(f"**Matched Characters:** {s.matched_chars} chars across {s.match_count} distinct span(s)")
                    st.markdown("---")
                    st.markdown("**Sample Matched Passages:**")
                    for sp in s.spans[:3]:
                        st.markdown(f"> *\"{sp.matched_text[:200]}...\"*")

    # 4. TAB: AI ANALYSIS
    with tab_ai:
        st.subheader("AI Content Stylometric & Language Model Analysis")
        if not report.ai_detect:
            st.info("AI detection was not executed for this document.")
        elif report.ai_detect.status == "insufficient_text":
            st.warning("⚠️ Insufficient text: Document contains under 150 words.")
        else:
            ai = report.ai_detect
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Overall AI Probability", f"{ai.overall_probability:.1f}%")
                st.write(f"**Confidence Band:** `{ai.confidence_band.upper()}`")
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

    # 5. TAB: INTEGRITY FINDINGS
    with tab_integrity:
        st.subheader("Integrity, Formatting Obfuscation & Citations")
        if not report.integrity or not report.integrity.results:
            st.info("No integrity issues detected.")
        else:
            for cr in report.integrity.results:
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
        st.write("Generate and download comprehensive offline report files in JSON, HTML, and PDF formats.")

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
