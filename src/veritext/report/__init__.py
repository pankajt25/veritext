"""Stage 6: Report generation in JSON, HTML, and PDF formats."""

from veritext.report.builder import build_report
from veritext.report.highlight import render_highlighted_html, render_legend_html
from veritext.report.html_export import export_report_html
from veritext.report.models import DEFAULT_DISCLAIMER, FullReport, HighlightSpan
from veritext.report.pdf_export import export_report_pdf

__all__ = [
    "DEFAULT_DISCLAIMER",
    "FullReport",
    "HighlightSpan",
    "build_report",
    "export_report_html",
    "export_report_pdf",
    "render_highlighted_html",
    "render_legend_html",
]
