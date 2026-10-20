"""Streamlit web application for Veritext Stage 6 report generation and document inspection."""

import io
from pathlib import Path
import streamlit as st

from veritext.report.highlight import render_highlighted_html, render_legend_html
from veritext.report.html_export import export_report_html
from veritext.report.pdf_export import export_report_pdf
from veritext.ui.services import MAX_WORDS_LIMIT, run_pipeline

ROBERTA_MODEL_DIR = Path("data/models/roberta-chatgpt")


def init_session_state():
    """Initializes session state keys if not already present."""
    if "full_report" not in st.session_state:
        st.session_state["full_report"] = None
    if "ignored_spans" not in st.session_state:
        st.session_state["ignored_spans"] = set()
    if "last_analyzed_file" not in st.session_state:
        st.session_state["last_analyzed_file"] = ""


def main():
    st.set_page_config(
        page_title="Veritext - Academic Integrity Suite",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.title("📝 Veritext")
        st.caption("Free & Open-Source Integrity Checker")

        st.markdown("---")
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document (.docx, .pdf, .txt, .md, .tex)",
            type=["docx", "pdf", "txt", "md", "tex"],
            help=f"Supported formats: Word DOCX, PDF, Text, Markdown, and LaTeX. Capped at {MAX_WORDS_LIMIT} words.",
        )

        st.markdown("---")
        st.subheader("2. Analysis Settings")

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
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("💻 **Machine Performance (WSL Dual-Core)**")
        st.caption("• Winnowing & Integrity: ~1s\n• DistilGPT-2 AI Model: ~30-40s\n• Vector Search: ~2-5s per source")

    # --- PERMANENT DISCLAIMER BANNER ---
    st.info(
        "⚖️ **Responsible Use Notice:** Veritext provides statistical indicators and heuristic signals for self-review, "
        "not definitive proof of academic misconduct. AI detectors are probabilistic and may misidentify human prose, "
        "especially from non-native English writers. Human pedagogical judgment is always required.",
        icon="ℹ️",
    )

    # --- EXECUTE ANALYSIS ---
    if run_btn:
        if uploaded_file is None:
            st.error("Please upload a document to analyze.")
            return

        settings = {
            "use_online": use_online,
            "use_semantic": use_semantic,
            "use_ai": use_ai,
            "use_integrity": use_integrity,
            "use_roberta": use_roberta,
            "min_words": min_words,
        }

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, frac: float):
            status_text.text(f"⏳ {msg}")
            progress_bar.progress(frac)

        try:
            with st.spinner("Analyzing document across engines..."):
                file_bytes = uploaded_file.getvalue()
                report = run_pipeline(
                    target=file_bytes,
                    filename=uploaded_file.name,
                    settings=settings,
                    progress_callback=update_progress,
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
        st.markdown("### Welcome to Veritext")
        st.write(
            "Veritext is a privacy-respecting, open-source document verification suite designed to help "
            "students, educators, and researchers evaluate academic integrity offline."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 🔍 Plagiarism Detection")
            st.write("Fast winnowing fingerprinting and dense neural semantic search across open reference libraries.")
        with col2:
            st.markdown("#### 🤖 AI-Text Stylometry")
            st.write("Perplexity, burstiness, and stylometric cadence detection calibrated for academic writing.")
        with col3:
            st.markdown("#### 🛡️ Formatting & Integrity")
            st.write("Detection of Cyrillic homoglyphs, zero-width characters, DOCX/PDF white font tricks, and citation gaps.")

        st.markdown("---")
        st.info("👈 Upload a document in the left sidebar and click **Run Analysis** to begin.")
        return

    # Show truncation warning if applicable
    if report.metadata.get("truncated"):
        st.warning(
            f"⚠️ Document length exceeded the {MAX_WORDS_LIMIT}-word analysis limit. "
            f"The document was automatically capped at the first {MAX_WORDS_LIMIT} words "
            f"(original length: {report.metadata.get('original_word_count')} words).",
            icon="⚠️",
        )

    # Tabs layout
    tab_summary, tab_doc, tab_sources, tab_ai, tab_integrity, tab_download = st.tabs([
        "📊 Summary",
        "📄 Highlighted Document",
        "📚 Sources",
        "🤖 AI Analysis",
        "🛡️ Integrity Findings",
        "💾 Export & Download",
    ])

    # 1. TAB: SUMMARY
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

        st.markdown("---")
        st.caption(f"Document Word Count: {report.metadata.get('word_count', 0)} words | Total Characters: {report.metadata.get('character_count', 0)}")

    # 2. TAB: HIGHLIGHTED DOCUMENT
    with tab_doc:
        st.subheader("Highlighted Document View")

        # Category Filter and Manage Findings
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

        # Render Legend and HTML Document
        st.markdown(render_legend_html(), unsafe_allow_html=True)

        html_view = render_highlighted_html(
            text=report.original_text,
            spans=active_highlights,
            ignored_ids=st.session_state["ignored_spans"],
        )
        st.markdown(html_view, unsafe_allow_html=True)

    # 3. TAB: SOURCES
    with tab_sources:
        st.subheader("Matched Sources & Repositories")
        if report.similarity and report.similarity.sources:
            for s in report.similarity.sources:
                with st.container():
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                    with c1:
                        st.markdown(f"**{s.source_title}**")
                        if s.source_url:
                            st.markdown(f"[{s.source_url}]({s.source_url})")
                        else:
                            st.caption("Local Reference Store")
                    with c2:
                        st.write(f"Origin: `{s.origin}`")
                    with c3:
                        st.write(f"Overlap: **{s.percentage:.1f}%**")
                    with c4:
                        st.write(f"{s.matched_chars} chars")
                    st.divider()
        else:
            st.info("No matching sources detected in local reference store or open academic repositories.")

    # 4. TAB: AI ANALYSIS
    with tab_ai:
        st.subheader("AI-Written Text Stylometric Analysis")
        if report.ai_detect:
            ai_prob = getattr(report.ai_detect, "overall_probability", getattr(report.ai_detect, "ai_probability", None))
            ai_prob_str = f"{ai_prob:.1f}%" if ai_prob is not None else "N/A"
            ai_conf = getattr(report.ai_detect, "confidence_band", getattr(report.ai_detect, "confidence", "N/A")).upper()
            ai_mode = getattr(report.ai_detect, "mode", getattr(report.ai_detect, "model_used", "full"))

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Overall AI Probability:** {ai_prob_str}")
                st.write(f"**Confidence Level:** {ai_conf}")
                st.write(f"**Status / Label:** `{getattr(report.ai_detect, 'status', getattr(report.ai_detect, 'label', 'N/A'))}`")
            with c2:
                st.write(f"**Detection Mode:** `{ai_mode}`")
                if report.ai_detect.reduced_confidence:
                    st.warning("Running on stylometry-only fallback mode (DistilGPT-2 weights not loaded).")

            drivers = getattr(report.ai_detect, "explanations", getattr(report.ai_detect, "drivers", []))
            if drivers:
                st.markdown("#### Primary Detection Drivers")
                for d in drivers:
                    st.markdown(f"- {d}")

            if report.ai_detect.sentences:
                st.markdown("#### Top Ranked AI-Like Sentences")
                top_sents = sorted(report.ai_detect.sentences, key=lambda s: s.score, reverse=True)[:10]
                for idx, sent in enumerate(top_sents, 1):
                    badge = "🔴 Likely AI" if sent.score >= 0.65 else ("🟡 Uncertain" if sent.score >= 0.35 else "🟢 Likely Human")
                    s_start = getattr(sent, "doc_start", getattr(sent, "start", 0))
                    s_end = getattr(sent, "doc_end", getattr(sent, "end", 0))
                    st.markdown(f"**{idx}. [{badge} - {sent.score:.1%}]** (offset `{s_start}:{s_end}`)")
                    st.markdown(f"> *\"{sent.text}\"*")
        else:
            st.info("AI detection was disabled for this analysis.")

    # 5. TAB: INTEGRITY FINDINGS
    with tab_integrity:
        st.subheader("Document Integrity & Policy Checks")
        if report.integrity and report.integrity.results:
            for cr in report.integrity.results:
                st.markdown(f"#### Check: `{cr.check_name.upper()}`")
                st.write(f"**Summary:** {cr.summary}")

                if cr.findings:
                    for f in cr.findings:
                        sev_badge = "🔴 HIGH" if f.severity == "high" else ("🟡 WARNING" if f.severity == "warning" else "🔵 INFO")
                        st.markdown(f"- **[{sev_badge}]** (offset `{f.start}:{f.end}`): {f.message}")
                else:
                    st.caption("No issues flagged.")
                st.divider()
        else:
            st.info("No integrity anomalies or policy issues recorded.")

    # 6. TAB: DOWNLOAD & EXPORT
    with tab_download:
        st.subheader("Export Consolidated Reports")
        st.write("Download reports reflecting your current session, excluding any dismissed findings.")

        # HTML Export
        html_bytes = export_report_html(
            report=report,
            ignored_ids=st.session_state["ignored_spans"],
        ).encode("utf-8")

        # PDF Export
        pdf_bytes = export_report_pdf(
            report=report,
            ignored_ids=st.session_state["ignored_spans"],
        )

        # JSON Export
        json_bytes = report.to_json().encode("utf-8")

        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_bytes,
                file_name=f"{Path(report.document_name).stem}_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                label="📑 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{Path(report.document_name).stem}_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with col_d3:
            st.download_button(
                label="💾 Download JSON Data",
                data=json_bytes,
                file_name=f"{Path(report.document_name).stem}_report.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
