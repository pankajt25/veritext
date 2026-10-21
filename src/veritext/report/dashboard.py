"""Combined dashboard: paragraph heat strip and cohort portfolio table.

Provides:
- Paragraph-level heat strip showing similarity, AI likelihood, and integrity flags side by side.
- Batch cohort portfolio table with file, similarity %, AI prob, integrity flags, max collusion similarity,
  sorting, filtering, and standalone CSV export (no external web resources).
"""

import csv
import html
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from veritext.similarity.scorer import calculate_covered_characters


def split_text_into_paragraphs(text: str) -> List[Tuple[int, int, int, str]]:
    """Splits text into paragraphs, returning (index, start_char, end_char, paragraph_text)."""
    if not text:
        return []

    paragraphs: List[Tuple[int, int, int, str]] = []
    # Match non-whitespace blocks separated by one or more blank lines
    pattern = re.compile(r"([^\r\n]+(?:\r?\n(?![ \t]*\r?\n)[^\r\n]+)*)", re.MULTILINE)
    idx = 1
    for m in pattern.finditer(text):
        p_str = m.group(1).strip()
        if p_str:
            paragraphs.append((idx, m.start(), m.end(), p_str))
            idx += 1

    # Fallback if no double newlines found
    if not paragraphs and text.strip():
        paragraphs.append((1, 0, len(text), text.strip()))

    return paragraphs


def generate_paragraph_heat_strip(
    text: str,
    similarity: Optional[Any] = None,
    ai_detect: Optional[Any] = None,
    integrity: Optional[Any] = None,
    style: Optional[Any] = None,
    review_session: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Computes side-by-side paragraph metrics for similarity, AI detection, and integrity flags."""
    raw_paras = split_text_into_paragraphs(text)
    heat_strip: List[Dict[str, Any]] = []

    # Pre-extract similarity spans
    sim_spans = []
    if similarity and hasattr(similarity, "spans"):
        sim_spans = similarity.spans
    elif isinstance(similarity, dict):
        sim_spans = similarity.get("spans", [])

    # Pre-extract AI sentences
    ai_sentences = []
    overall_ai_prob = 0.0
    if ai_detect:
        overall_ai_prob = getattr(ai_detect, "overall_probability", 0.0)
        ai_sentences = getattr(ai_detect, "sentences", [])

    # Pre-extract integrity findings
    integrity_findings = []
    if integrity:
        if hasattr(integrity, "results"):
            for res in integrity.results:
                integrity_findings.extend(getattr(res, "findings", []))
        elif isinstance(integrity, dict):
            for res in integrity.get("results", []):
                integrity_findings.extend(res.get("findings", []))

    # Pre-extract style findings
    style_findings = []
    if style:
        if hasattr(style, "findings"):
            style_findings = style.findings
        elif isinstance(style, dict):
            style_findings = style.get("findings", [])

    # Review lookup
    reviews_map = {}
    if review_session and hasattr(review_session, "reviews"):
        reviews_map = review_session.reviews

    for idx, p_start, p_end, p_text in raw_paras:
        p_len = max(1, p_end - p_start)

        # 1. Similarity overlap in paragraph
        p_intervals: List[Tuple[int, int]] = []
        for s in sim_spans:
            s_start = getattr(s, "doc_start", None) or s.get("doc_start", 0)
            s_end = getattr(s, "doc_end", None) or s.get("doc_end", 0)
            if s_start < p_end and s_end > p_start:
                inter_s = max(s_start, p_start)
                inter_e = min(s_end, p_end)
                p_intervals.append((inter_s, inter_e))

        covered_chars = calculate_covered_characters(p_intervals)
        sim_pct = min(100.0, (covered_chars / p_len) * 100.0)

        # 2. AI probability for paragraph
        matching_ai_probs = []
        for sent in ai_sentences:
            s_start = getattr(sent, "start", 0)
            s_end = getattr(sent, "end", 0)
            if s_start < p_end and s_end > p_start:
                matching_ai_probs.append(getattr(sent, "probability", 0.0))

        if matching_ai_probs:
            avg_ai_prob = sum(matching_ai_probs) / len(matching_ai_probs)
            # If in 0.0-1.0 format, scale to percentage
            if avg_ai_prob <= 1.0 and overall_ai_prob > 1.0:
                ai_pct = avg_ai_prob * 100.0
            else:
                ai_pct = avg_ai_prob
        else:
            ai_pct = overall_ai_prob

        # 3. Integrity flags in paragraph
        flags = []
        for f in integrity_findings:
            f_start = getattr(f, "start", 0) if hasattr(f, "start") else f.get("start", 0)
            f_end = getattr(f, "end", 0) if hasattr(f, "end") else f.get("end", 0)
            if f_start < p_end and f_end > p_start:
                check_name = getattr(f, "check", "") if hasattr(f, "check") else f.get("check", "")
                severity = getattr(f, "severity", "info") if hasattr(f, "severity") else f.get("severity", "info")
                message = getattr(f, "message", "") if hasattr(f, "message") else f.get("message", "")
                flags.append({"check": check_name, "severity": severity, "message": message})

        # 4. Style count
        s_count = 0
        for sf in style_findings:
            sf_start = getattr(sf, "start", 0) if hasattr(sf, "start") else sf.get("start", 0)
            sf_end = getattr(sf, "end", 0) if hasattr(sf, "end") else sf.get("end", 0)
            if sf_start < p_end and sf_end > p_start:
                s_count += 1

        # 5. Review status for paragraph
        para_review = "unreviewed"
        for rev_id, rev in reviews_map.items():
            r_start = getattr(rev, "start", 0)
            r_end = getattr(rev, "end", 0)
            if r_start < p_end and r_end > p_start:
                para_review = getattr(rev, "status", "unreviewed")
                break

        snippet = p_text[:120].strip() + ("..." if len(p_text) > 120 else "")

        heat_strip.append(
            {
                "paragraph_index": idx,
                "start": p_start,
                "end": p_end,
                "length": p_len,
                "snippet": snippet,
                "similarity_percentage": round(sim_pct, 1),
                "ai_probability": round(ai_pct, 1),
                "integrity_flags": flags,
                "style_issues_count": s_count,
                "review_status": para_review,
            }
        )

    return heat_strip


def render_heat_strip_html(heat_strip_data: List[Dict[str, Any]]) -> str:
    """Renders a self-contained HTML component for paragraph heat strip."""
    if not heat_strip_data:
        return "<p class='no-data'>No paragraph breakdown available.</p>"

    html_out = [
        "<div class='heat-strip-container'>",
        "  <div class='heat-strip-legend'>",
        "    <span class='legend-item'><span class='legend-box sim-low'></span> Sim &lt;15%</span>",
        "    <span class='legend-item'><span class='legend-box sim-mid'></span> Sim 15-40%</span>",
        "    <span class='legend-item'><span class='legend-box sim-high'></span> Sim &gt;40%</span>",
        "    <span class='legend-item' style='margin-left: 20px;'><span class='legend-box ai-low'></span> AI &lt;30%</span>",
        "    <span class='legend-item'><span class='legend-box ai-mid'></span> AI 30-70%</span>",
        "    <span class='legend-item'><span class='legend-box ai-high'></span> AI &gt;70%</span>",
        "  </div>",
        "  <div class='heat-strip-table-wrapper'>",
        "    <table class='heat-strip-table'>",
        "      <thead>",
        "        <tr>",
        "          <th style='width: 60px;'>Para</th>",
        "          <th>Text Preview</th>",
        "          <th style='width: 140px;'>Similarity %</th>",
        "          <th style='width: 140px;'>AI Likelihood</th>",
        "          <th style='width: 150px;'>Integrity Flags</th>",
        "          <th style='width: 90px;'>Review</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>",
    ]

    for p in heat_strip_data:
        sim = p["similarity_percentage"]
        ai = p["ai_probability"]
        sim_cls = "sim-high" if sim >= 40.0 else ("sim-mid" if sim >= 15.0 else "sim-low")
        ai_cls = "ai-high" if ai >= 70.0 else ("ai-mid" if ai >= 30.0 else "ai-low")

        flags_html = ""
        if p["integrity_flags"]:
            for f in p["integrity_flags"]:
                cname = html.escape(f.get("check", "Flag").replace("_", " ").title())
                sev = html.escape(f.get("severity", "info"))
                flags_html += f"<span class='flag-badge flag-{sev}' title='{cname}: {html.escape(f.get('message', ''))}'>{cname}</span> "
        else:
            flags_html = "<span class='text-muted'>—</span>"

        rev_badge = ""
        r_status = p.get("review_status", "unreviewed")
        if r_status != "unreviewed":
            rev_badge = f"<span class='rev-badge rev-{r_status}'>{html.escape(r_status.replace('_', ' ').capitalize())}</span>"
        else:
            rev_badge = "<span class='text-muted'>—</span>"

        html_out.append("        <tr>")
        html_out.append(f"          <td class='para-idx'><strong>#{p['paragraph_index']}</strong></td>")
        html_out.append(f"          <td class='para-snippet'>{html.escape(p['snippet'])}</td>")
        html_out.append(
            f"          <td>"
            f"            <div class='bar-container'><div class='bar-fill {sim_cls}' style='width: {min(100.0, sim)}%;'></div></div>"
            f"            <span class='bar-label'>{sim:.1f}%</span>"
            f"          </td>"
        )
        html_out.append(
            f"          <td>"
            f"            <div class='bar-container'><div class='bar-fill {ai_cls}' style='width: {min(100.0, ai)}%;'></div></div>"
            f"            <span class='bar-label'>{ai:.1f}%</span>"
            f"          </td>"
        )
        html_out.append(f"          <td>{flags_html}</td>")
        html_out.append(f"          <td>{rev_badge}</td>")
        html_out.append("        </tr>")

    html_out.append("      </tbody>")
    html_out.append("    </table>")
    html_out.append("  </div>")
    html_out.append("</div>")

    # Embedded CSS
    css = """
<style>
.heat-strip-container { margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.heat-strip-legend { display: flex; align-items: center; margin-bottom: 12px; font-size: 12px; color: #555; }
.legend-item { display: inline-flex; align-items: center; margin-right: 14px; }
.legend-box { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 5px; }
.heat-strip-table-wrapper { overflow-x: auto; border: 1px solid #e0e0e0; border-radius: 6px; }
.heat-strip-table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
.heat-strip-table th { background: #f8f9fa; padding: 10px 12px; border-bottom: 2px solid #e0e0e0; font-weight: 600; color: #333; }
.heat-strip-table td { padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: middle; }
.heat-strip-table tr:hover { background: #fbfbfb; }
.para-idx { color: #1a73e8; font-weight: bold; }
.para-snippet { color: #444; max-width: 320px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-container { background: #eee; border-radius: 4px; height: 10px; width: 85px; display: inline-block; vertical-align: middle; margin-right: 6px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-label { font-size: 11px; font-weight: 600; color: #555; }
.sim-low { background-color: #34a853; }
.sim-mid { background-color: #fbbc05; }
.sim-high { background-color: #ea4335; }
.ai-low { background-color: #34a853; }
.ai-mid { background-color: #ff9800; }
.ai-high { background-color: #673ab7; }
.flag-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: 500; display: inline-block; margin: 1px; }
.flag-info { background: #e8f0fe; color: #1967d2; }
.flag-warning { background: #fef7e0; color: #b06000; }
.flag-high { background: #fce8e6; color: #c5221f; }
.rev-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
.rev-accepted { background: #fce8e6; color: #c5221f; }
.rev-ignored { background: #e6f4ea; color: #137333; }
.rev-needs_discussion { background: #fef7e0; color: #b06000; }
.text-muted { color: #999; }
</style>
"""
    return css + "\n" + "\n".join(html_out)


def generate_portfolio_csv(items: List[Dict[str, Any]]) -> str:
    """Generates standard CSV output for cohort portfolio items."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "File Name",
        "Similarity %",
        "AI Probability %",
        "Integrity Findings",
        "Max Collusion %",
        "Status",
    ])
    for item in items:
        writer.writerow([
            item.get("file_name", ""),
            f"{item.get('similarity_percentage', 0.0):.1f}",
            f"{item.get('ai_probability', 0.0):.1f}",
            item.get("integrity_findings_count", 0),
            f"{item.get('max_collusion_similarity', 0.0):.1f}",
            item.get("status", "Completed"),
        ])
    return output.getvalue()


def render_portfolio_html(items: List[Dict[str, Any]]) -> str:
    """Renders a self-contained interactive portfolio table for batch runs.

    Includes client-side search, column sorting, and embedded CSV export data URI.
    """
    csv_data = generate_portfolio_csv(items)
    import base64
    b64_csv = base64.b64encode(csv_data.encode("utf-8")).decode("ascii")

    html_out = [
        "<div class='portfolio-container'>",
        "  <div class='portfolio-header'>",
        "    <div class='portfolio-search'>",
        "      <input type='text' id='portfolio-filter' onkeyup='filterPortfolioTable()' placeholder='Search submissions by filename or status...' />",
        "    </div>",
        f"    <a class='btn-download' href='data:text/csv;base64,{b64_csv}' download='cohort_portfolio.csv'>📥 Export Portfolio CSV</a>",
        "  </div>",
        "  <div class='portfolio-table-wrapper'>",
        "    <table class='portfolio-table' id='portfolioTable'>",
        "      <thead>",
        "        <tr>",
        "          <th onclick='sortPortfolioTable(0)'>File Name ⬍</th>",
        "          <th onclick='sortPortfolioTable(1)'>Similarity % ⬍</th>",
        "          <th onclick='sortPortfolioTable(2)'>AI Probability ⬍</th>",
        "          <th onclick='sortPortfolioTable(3)'>Integrity Findings ⬍</th>",
        "          <th onclick='sortPortfolioTable(4)'>Max Collusion % ⬍</th>",
        "          <th onclick='sortPortfolioTable(5)'>Status ⬍</th>",
        "        </tr>",
        "      </thead>",
        "      <tbody>",
    ]

    for it in items:
        fn = html.escape(it.get("file_name", ""))
        sim = it.get("similarity_percentage", 0.0)
        ai = it.get("ai_probability", 0.0)
        integ = it.get("integrity_findings_count", 0)
        coll = it.get("max_collusion_similarity", 0.0)
        st = html.escape(it.get("status", "Completed"))

        sim_color = "#c5221f" if sim >= 40 else ("#b06000" if sim >= 15 else "#137333")
        ai_color = "#673ab7" if ai >= 70 else ("#ff9800" if ai >= 30 else "#137333")
        coll_color = "#c5221f" if coll >= 30 else ("#b06000" if coll >= 15 else "#555")

        html_out.append("        <tr>")
        html_out.append(f"          <td><strong>{fn}</strong></td>")
        html_out.append(f"          <td style='color: {sim_color}; font-weight: 600;'>{sim:.1f}%</td>")
        html_out.append(f"          <td style='color: {ai_color}; font-weight: 600;'>{ai:.1f}%</td>")
        html_out.append(f"          <td>{integ}</td>")
        html_out.append(f"          <td style='color: {coll_color}; font-weight: 600;'>{coll:.1f}%</td>")
        html_out.append(f"          <td><span class='badge-status'>{st}</span></td>")
        html_out.append("        </tr>")

    html_out.append("      </tbody>")
    html_out.append("    </table>")
    html_out.append("  </div>")
    html_out.append("</div>")

    # Script and styles
    js_and_css = """
<style>
.portfolio-container { margin: 20px 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.portfolio-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
.portfolio-search input { width: 320px; padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
.btn-download { display: inline-block; padding: 8px 16px; background: #1a73e8; color: #fff; text-decoration: none; border-radius: 4px; font-size: 13px; font-weight: 500; }
.btn-download:hover { background: #1557b0; color: #fff; }
.portfolio-table-wrapper { border: 1px solid #e0e0e0; border-radius: 6px; overflow-x: auto; }
.portfolio-table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
.portfolio-table th { background: #f8f9fa; padding: 12px 14px; border-bottom: 2px solid #e0e0e0; cursor: pointer; user-select: none; font-weight: 600; color: #333; }
.portfolio-table th:hover { background: #eef1f5; }
.portfolio-table td { padding: 10px 14px; border-bottom: 1px solid #eee; }
.portfolio-table tr:hover { background: #fbfbfb; }
.badge-status { background: #e8f0fe; color: #1967d2; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
</style>
<script>
function filterPortfolioTable() {
  var input = document.getElementById("portfolio-filter");
  var filter = input.value.toLowerCase();
  var table = document.getElementById("portfolioTable");
  var trs = table.getElementsByTagName("tr");
  for (var i = 1; i < trs.length; i++) {
    var txt = trs[i].textContent || trs[i].innerText;
    trs[i].style.display = txt.toLowerCase().indexOf(filter) > -1 ? "" : "none";
  }
}
function sortPortfolioTable(colIndex) {
  var table = document.getElementById("portfolioTable");
  var rows = Array.from(table.rows).slice(1);
  var isAsc = table.getAttribute("data-sort-dir-" + colIndex) !== "asc";
  rows.sort(function(a, b) {
    var aText = a.cells[colIndex].innerText.replace('%', '').trim();
    var bText = b.cells[colIndex].innerText.replace('%', '').trim();
    var aNum = parseFloat(aText);
    var bNum = parseFloat(bText);
    if (!isNaN(aNum) && !isNaN(bNum)) {
      return isAsc ? aNum - bNum : bNum - aNum;
    }
    return isAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
  });
  rows.forEach(function(row) { table.tBodies[0].appendChild(row); });
  table.setAttribute("data-sort-dir-" + colIndex, isAsc ? "asc" : "desc");
}
</script>
"""
    return js_and_css + "\n" + "\n".join(html_out)
