"""Export unified FullReport to a self-contained, offline-compatible HTML file."""

import html
from pathlib import Path
from typing import Optional, Set, Union

from veritext.report.dashboard import generate_paragraph_heat_strip, render_heat_strip_html
from veritext.report.diff import extract_report_diffs
from veritext.report.highlight import render_highlighted_html, render_legend_html
from veritext.report.models import FullReport
from veritext.report.seal import compute_seal


def export_report_html(
    report: FullReport,
    output_path: Optional[Union[str, Path]] = None,
    ignored_ids: Optional[Set[str]] = None,
) -> str:
    """Generates a fully self-contained, responsive HTML report.

    Features:
    - 100% offline (no external fonts, CDNs, scripts, or remote styles)
    - Dark and light mode adaptive via CSS media queries
    - Comprehensive summary cards, tables, highlighted document, and disclaimer

    Args:
        report: FullReport object.
        output_path: Optional file path to write HTML output to disk.
        ignored_ids: Optional set of highlight IDs dismissed by user.

    Returns:
        Complete HTML string.
    """
    ignored = ignored_ids or set()
    doc_name = html.escape(report.document_name)
    timestamp = html.escape(report.timestamp)
    summary_text = html.escape(report.summary)
    disclaimer_text = html.escape(report.disclaimer)

    # 1. Metric values
    sim_pct = f"{report.similarity.similarity_percentage:.1f}%" if report.similarity else "N/A"
    if report.ai_detect:
        ai_st = getattr(report.ai_detect, "status", "success")
        if ai_st in ("insufficient text", "insufficient_text"):
            ai_pct = "Insufficient text (under 150 words, not scored)"
            ai_conf_display = "Under 150 words"
            ai_label = "Not scored"
        else:
            ai_prob = getattr(report.ai_detect, "overall_probability", getattr(report.ai_detect, "ai_probability", None))
            ai_pct = f"{ai_prob:.1f}%" if ai_prob is not None else "Not run"
            ai_conf = getattr(report.ai_detect, "confidence_band", getattr(report.ai_detect, "confidence", "none")).lower()
            ai_conf_display = f"confidence in this estimate: {ai_conf}"
            ai_label = getattr(report.ai_detect, "status", getattr(report.ai_detect, "label", ""))
    else:
        ai_pct = "Not run"
        ai_conf_display = "Engine not executed"
        ai_label = ""

    high_cnt = 0
    warn_cnt = 0
    info_cnt = 0
    if report.integrity and report.integrity.results:
        for cr in report.integrity.results:
            for f in cr.findings:
                if f.severity == "high":
                    high_cnt += 1
                elif f.severity == "warning":
                    warn_cnt += 1
                else:
                    info_cnt += 1

    # 2. Render highlighted document
    highlighted_doc_html = render_highlighted_html(
        report.original_text,
        report.highlights,
        ignored_ids=ignored,
    )
    legend_html = render_legend_html()

    # 3. Sources Table
    sources_rows = []
    if report.similarity and report.similarity.sources:
        for s in report.similarity.sources:
            s_title = html.escape(s.source_title)
            s_origin = html.escape(s.origin)
            s_url = s.source_url
            url_link = f'<a href="{html.escape(s_url)}" target="_blank" rel="noopener">{html.escape(s_url[:45])}...</a>' if s_url else "Local Library"
            pct_badge = f'<span style="font-weight:700; color:#dc2626;">{s.percentage:.1f}%</span>'
            sources_rows.append(
                f"<tr><td>{s_title}</td><td>{s_origin}</td><td>{url_link}</td><td>{pct_badge}</td><td>{s.matched_chars} chars</td></tr>"
            )
    sources_tbody = "".join(sources_rows) if sources_rows else '<tr><td colspan="5" style="text-align:center; color:#64748b;">No matching sources detected.</td></tr>'

    # 4. Top AI Sentences Table
    ai_rows = []
    if report.ai_detect and report.ai_detect.sentences:
        # Sort descending by score
        sorted_sents = sorted(report.ai_detect.sentences, key=lambda s: s.score, reverse=True)[:10]
        for s in sorted_sents:
            s_text = html.escape(s.text)
            s_score = f"{s.score:.1%}"
            s_label = html.escape(s.label)
            s_start = getattr(s, "doc_start", getattr(s, "start", 0))
            s_end = getattr(s, "doc_end", getattr(s, "end", 0))
            s_offsets = f"{s_start}:{s_end}"
            badge_color = "#dc2626" if s.score >= 0.65 else "#16a34a"
            ai_rows.append(
                f'<tr><td>{s_offsets}</td><td><span style="font-weight:600; color:{badge_color};">{s_score}</span></td><td>{s_label}</td><td style="font-size:0.9rem;">{s_text}</td></tr>'
            )
    ai_tbody = "".join(ai_rows) if ai_rows else '<tr><td colspan="4" style="text-align:center; color:#64748b;">No AI sentence analysis recorded.</td></tr>'

    # 5. Integrity Findings Table
    integ_rows = []
    if report.integrity and report.integrity.results:
        for cr in report.integrity.results:
            for f in cr.findings:
                sev_color = "#dc2626" if f.severity == "high" else ("#d97706" if f.severity == "warning" else "#2563eb")
                sev_badge = f'<span style="background:{sev_color}; color:#fff; padding:2px 6px; border-radius:3px; font-size:0.75rem; font-weight:700;">{f.severity.upper()}</span>'
                f_check = html.escape(f.check)
                f_msg = html.escape(f.message)
                f_offsets = f"{f.start}:{f.end}"
                integ_rows.append(
                    f"<tr><td>{f_check}</td><td>{sev_badge}</td><td>{f_offsets}</td><td>{f_msg}</td></tr>"
                )
    integ_tbody = "".join(integ_rows) if integ_rows else '<tr><td colspan="4" style="text-align:center; color:#64748b;">No integrity anomalies detected.</td></tr>'

    # 5b. Active Filters & Fulltext Count
    active_filters = (
        report.settings.get("active_filters")
        or (report.similarity.metadata.get("active_filters") if report.similarity and report.similarity.metadata else None)
        or ["Standard exclusions (small matches <25 chars)"]
    )
    filters_badges = "".join(
        f'<span style="display:inline-block; background:#f1f5f9; color:#334155; border:1px solid #cbd5e1; padding:3px 8px; border-radius:4px; margin-right:6px; margin-bottom:4px; font-size:0.85rem;">{html.escape(f)}</span>'
        for f in active_filters
    )
    filters_html = f'<div style="margin-top:14px; padding-top:10px; border-top:1px solid #e2e8f0; font-size:0.9rem;"><strong>Active Exclusion Filters:</strong><div style="margin-top:6px;">{filters_badges}</div></div>'

    ft_count = (
        report.metadata.get("fulltexts_fetched")
        or (report.similarity.metadata.get("fulltexts_fetched") if report.similarity and report.similarity.metadata else 0)
    )
    if ft_count > 0:
        filters_html += f'<div style="margin-top:6px; font-size:0.85rem; color:#0d9488;"><strong>Open-Access Retrieval:</strong> {ft_count} full-text paper(s) indexed for cross-body matching</div>'

    # 6. Profile & Plan HTML
    profile_html = ""
    if report.profile:
        p = report.profile
        profile_html = f"""
    <div class="section card">
      <h2>Document Profile & Sizing</h2>
      <div style="display:flex; flex-wrap:wrap; gap:16px; margin-bottom:16px;">
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Word Count</div>
          <div style="font-size:1.2rem; font-weight:700;">{p.get('word_count', 0):,}</div>
        </div>
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Character Count</div>
          <div style="font-size:1.2rem; font-weight:700;">{p.get('character_count', 0):,}</div>
        </div>
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Estimated Pages</div>
          <div style="font-size:1.2rem; font-weight:700;">{p.get('page_estimate', 1):,}</div>
        </div>
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Size Category</div>
          <div style="font-size:1.2rem; font-weight:700;">{str(p.get('size_class', 'medium')).upper()}</div>
        </div>
      </div>
    </div>
        """

    plan_html = ""
    if report.plan:
        pl = report.plan
        p_mode = html.escape(str(pl.get("mode", "auto")).upper())
        p_budget = f"{pl.get('time_budget_minutes', 10.0):.1f} min"
        plan_rows = []
        for eng_k, ep in pl.get("engines", {}).items():
            status = "Run" if ep.get("enabled", True) else "Skip"
            cov = f"{ep.get('coverage_percent', 100.0):.1f}%"
            est = f"{ep.get('estimated_seconds', 0.0):.1f}s"
            desc = html.escape(ep.get("coverage_description", ""))
            plan_rows.append(
                f"<tr><td><strong>{html.escape(eng_k)}</strong></td><td>{status}</td><td>{ep.get('words_to_analyze', 0):,}</td><td>{cov}</td><td>{est}</td><td>{desc}</td></tr>"
            )
        plan_tbody = "".join(plan_rows)
        plan_html = f"""
    <div class="section card">
      <h2>Adaptive Analysis Plan (Mode: {p_mode}, Budget: {p_budget})</h2>
      <table>
        <thead>
          <tr>
            <th>Engine</th>
            <th>Status</th>
            <th>Words Analysed</th>
            <th>Coverage</th>
            <th>Est. Time</th>
            <th>Coverage Description</th>
          </tr>
        </thead>
        <tbody>
          {plan_tbody}
        </tbody>
      </table>
    </div>
        """

    # 6b. Paragraph Multi-Signal Heat Strip
    heat_strip_data = generate_paragraph_heat_strip(
        report.original_text,
        similarity=report.similarity,
        ai_detect=report.ai_detect,
        integrity=report.integrity,
        review_session=report.metadata.get("review_session"),
    )
    heat_strip_section = f"""
    <div class="section card">
      <h2>Paragraph Multi-Signal Heat Strip</h2>
      <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:12px;">
        Side-by-side view of similarity overlap, AI probability, and integrity flags per paragraph.
      </p>
      {render_heat_strip_html(heat_strip_data)}
    </div>
    """

    # 6c. Instructor Review & Rubric Section
    rubric_section = ""
    rev_session = report.metadata.get("review_session")
    rubric_data = report.metadata.get("rubric") or (rev_session.get("rubric") if isinstance(rev_session, dict) else (getattr(rev_session, "rubric", None) if rev_session else None))
    if rubric_data:
        r_dict = rubric_data if isinstance(rubric_data, dict) else (rubric_data.to_dict() if hasattr(rubric_data, "to_dict") else {})
        r_name = html.escape(r_dict.get("name", "Academic Integrity Rubric"))
        r_score = r_dict.get("total_score", 0)
        r_max = r_dict.get("max_possible", 16)
        r_pct = r_dict.get("percentage", 0.0)
        r_comment = html.escape(r_dict.get("overall_comment", ""))

        score_rows = []
        for crit_id, sc in r_dict.get("scores", {}).items():
            s_val = sc.get("score", 0) if isinstance(sc, dict) else getattr(sc, "score", 0)
            s_comm = html.escape(sc.get("comment", "") if isinstance(sc, dict) else getattr(sc, "comment", ""))
            crit_name = html.escape(crit_id.replace("_", " ").title())
            score_rows.append(
                f"<tr><td><strong>{crit_name}</strong></td><td>{s_val} / 4</td><td>{s_comm or '—'}</td></tr>"
            )
        score_tbody = "".join(score_rows)

        rubric_section = f"""
    <div class="section card">
      <h2>Instructor Rubric Assessment: {r_name}</h2>
      <div style="display:flex; gap:16px; margin-bottom:14px;">
        <div style="background:rgba(0,0,0,0.02); padding:10px 16px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Total Score</div>
          <div style="font-size:1.3rem; font-weight:700;">{r_score} / {r_max} ({r_pct:.1f}%)</div>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Criterion</th>
            <th style="width: 100px;">Score</th>
            <th>Reviewer Comments</th>
          </tr>
        </thead>
        <tbody>
          {score_tbody}
        </tbody>
      </table>
      {f'<div style="margin-top:10px; font-size:0.9rem;"><strong>Overall Review Comments:</strong> {r_comment}</div>' if r_comment else ''}
    </div>
        """

    # 6d. Grammar & Style Feedback Section
    style_section = ""
    style_meta = report.metadata.get("style")
    if style_meta:
        s_dict = style_meta if isinstance(style_meta, dict) else (style_meta.to_dict() if hasattr(style_meta, "to_dict") else {})
        metrics = s_dict.get("metrics") or {}
        findings = s_dict.get("findings") or []
        flesch = metrics.get("flesch_reading_ease", 0.0)
        fog = metrics.get("gunning_fog", 0.0)
        ease_label = html.escape(metrics.get("reading_ease_label", "Standard"))
        level = html.escape(metrics.get("reading_level", "College"))
        lt_status = s_dict.get("engine_status", {}).get("languagetool", {})
        lt_note = html.escape(lt_status.get("reason", "Built-in rules active."))

        style_rows = []
        for sf in findings[:15]:
            sf_chk = html.escape(sf.get("check", "style").replace("style_", "").replace("_", " ").title())
            sf_msg = html.escape(sf.get("message", ""))
            sf_off = f"{sf.get('start', 0)}:{sf.get('end', 0)}"
            style_rows.append(f"<tr><td>{sf_chk}</td><td>{sf_off}</td><td>{sf_msg}</td></tr>")
        style_tbody = "".join(style_rows) if style_rows else '<tr><td colspan="3" style="text-align:center; color:#64748b;">No style or grammar suggestions.</td></tr>'

        style_section = f"""
    <div class="section card">
      <h2>Grammar & Style Feedback (Informational)</h2>
      <div style="display:flex; flex-wrap:wrap; gap:16px; margin-bottom:14px;">
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Flesch Reading Ease</div>
          <div style="font-size:1.3rem; font-weight:700;">{flesch:.1f} ({ease_label})</div>
        </div>
        <div style="flex:1; min-width:140px; background:rgba(0,0,0,0.02); padding:10px; border-radius:4px;">
          <div style="font-size:0.8rem; color:var(--text-muted);">Gunning Fog Index</div>
          <div style="font-size:1.3rem; font-weight:700;">{fog:.1f} ({level})</div>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th style="width: 140px;">Rule</th>
            <th style="width: 90px;">Offsets</th>
            <th>Suggestion</th>
          </tr>
        </thead>
        <tbody>
          {style_tbody}
        </tbody>
      </table>
      <div style="margin-top:8px; font-size:0.8rem; color:var(--text-muted);"><strong>Engine Status:</strong> {lt_note}</div>
    </div>
        """

    # 7. Settings list
    settings_items = [f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>" for k, v in report.settings.items()]
    settings_html = f"<ul>{''.join(settings_items)}</ul>" if settings_items else "<p>Default settings applied.</p>"

    # 8. Side-by-side diffs
    diff_items = extract_report_diffs(report, max_matches=10)
    if diff_items:
        diff_cards = "".join(item.render_html_details() for item in diff_items)
        diff_section = f"""
    <div class="section card">
      <h2>Side-by-Side Source Comparisons</h2>
      <p style="font-size:0.9rem; color:var(--text-muted); margin-bottom:14px;">
        Word-level alignment of shared passages (maximum 1,500 characters per excerpt). Click any source to expand.
      </p>
      {diff_cards}
    </div>
        """
    else:
        diff_section = ""

    # 9. Cryptographic seal
    seal_info = report.seal or compute_seal(report.to_dict())
    seal_section = f"""
    <div class="section card" style="border-left: 4px solid #059669; background: rgba(5, 150, 105, 0.05);">
      <h3 style="margin-top:0; color:#065f46; display:flex; align-items:center; gap:8px;">
        <span style="background:#059669; color:#ffffff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:700;">SEALED</span>
        Cryptographic Tamper-Evident Report Seal
      </h3>
      <p style="font-size:0.85rem; color:#047857; margin-bottom:8px;">
        {html.escape(seal_info.get('notice', ''))}
      </p>
      <div style="font-family:ui-monospace, monospace; font-size:0.8rem; background:var(--bg-body); padding:10px; border-radius:4px; border:1px solid var(--border-card); word-break:break-all;">
        <div><strong>Algorithm:</strong> {html.escape(seal_info.get('algorithm', 'HMAC-SHA256'))}</div>
        <div><strong>Canonical SHA-256:</strong> {html.escape(seal_info.get('canonical_sha256', ''))}</div>
        <div><strong>HMAC Signature:</strong> {html.escape(seal_info.get('signature', ''))}</div>
        <div><strong>Verify CLI:</strong> <code>python -m veritext.report verify &lt;report.json&gt;</code></div>
      </div>
    </div>
    """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Veritext Report - {doc_name}</title>
  <style>
    :root {{
      --bg-body: #f8fafc;
      --bg-card: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border-card: #e2e8f0;
      --accent: #2563eb;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg-body: #0b0f19;
        --bg-card: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border-card: #334155;
        --accent: #3b82f6;
      }}
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-body);
      color: var(--text-main);
      margin: 0;
      padding: 24px;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1040px;
      margin: 0 auto;
    }}
    .header {{
      border-bottom: 2px solid var(--border-card);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 1.8rem;
    }}
    .header .meta {{
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .grid-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background-color: var(--bg-card);
      border: 1px solid var(--border-card);
      border-radius: 8px;
      padding: 16px 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    .metric-val {{
      font-size: 1.8rem;
      font-weight: 800;
      margin: 4px 0;
    }}
    .metric-sub {{
      font-size: 0.85rem;
      color: var(--text-muted);
    }}
    .card h2 {{
      margin-top: 0;
      font-size: 1.25rem;
      border-bottom: 1px solid var(--border-card);
      padding-bottom: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      margin: 12px 0;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border-card);
    }}
    th {{
      background-color: rgba(0,0,0,0.02);
      font-weight: 700;
    }}
    .disclaimer-box {{
      background-color: #fffbeb;
      border: 1px solid #fde68a;
      border-left: 4px solid #f59e0b;
      padding: 14px 18px;
      border-radius: 4px;
      margin: 24px 0;
      font-size: 0.9rem;
      color: #92400e;
    }}
    .section {{
      margin-bottom: 32px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Veritext Verification Report</h1>
      <div class="meta">
        Document: <strong>{doc_name}</strong> &bull; Generated: {timestamp}
      </div>
    </div>

    <div class="grid-metrics">
      <div class="card">
        <div class="metric-sub">Plagiarism Similarity</div>
        <div class="metric-val" style="color: #dc2626;">{sim_pct}</div>
        <div class="metric-sub">Exact & paraphrase matches</div>
      </div>
      <div class="card">
        <div class="metric-sub">AI Content Probability</div>
        <div class="metric-val" style="color: #7c3aed; font-size: {'1.05rem' if len(ai_pct) > 10 else '1.8rem'};">{ai_pct}</div>
        <div class="metric-sub">{ai_conf_display}</div>
      </div>
      <div class="card">
        <div class="metric-sub">Integrity Findings</div>
        <div class="metric-val" style="color: #0284c7;">{high_cnt + warn_cnt + info_cnt}</div>
        <div class="metric-sub">{high_cnt} High, {warn_cnt} Warning, {info_cnt} Info</div>
      </div>
    </div>

    <div class="disclaimer-box">
      <strong>Responsible Use & Limitations Notice:</strong><br />
      {disclaimer_text}
    </div>

    {seal_section}

    <div class="section card">
      <h2>Executive Summary</h2>
      <p>{summary_text}</p>
      {filters_html}
    </div>

    {profile_html}
    {plan_html}
    {rubric_section}
    {heat_strip_section}
    {style_section}

    <div class="section card">
      <h2>Highlighted Document View</h2>
      {legend_html}
      {highlighted_doc_html}
    </div>

    {diff_section}

    <div class="section card">
      <h2>Matched Sources</h2>
      <table>
        <thead>
          <tr>
            <th>Source Title</th>
            <th>Origin</th>
            <th>URL / Repository</th>
            <th>Overlap %</th>
            <th>Matched Size</th>
          </tr>
        </thead>
        <tbody>
          {sources_tbody}
        </tbody>
      </table>
    </div>

    <div class="section card">
      <h2>Top AI-Like Sentences</h2>
      <table>
        <thead>
          <tr>
            <th>Offsets</th>
            <th>AI Score</th>
            <th>Classification</th>
            <th>Sentence Excerpt</th>
          </tr>
        </thead>
        <tbody>
          {ai_tbody}
        </tbody>
      </table>
    </div>

    <div class="section card">
      <h2>Integrity & Formatting Checks</h2>
      <table>
        <thead>
          <tr>
            <th>Check</th>
            <th>Severity</th>
            <th>Offsets</th>
            <th>Message / Anomaly</th>
          </tr>
        </thead>
        <tbody>
          {integ_tbody}
        </tbody>
      </table>
    </div>

    <div class="section card">
      <h2>Analysis Methodology & Settings</h2>
      {settings_html}
      <p style="font-size:0.85rem; color:var(--text-muted); margin-top:12px;">
        Generated completely offline by Veritext. No document contents were uploaded to remote servers or third-party APIs.
      </p>
    </div>
  </div>
</body>
</html>"""

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")

    return html_content
