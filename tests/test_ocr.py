"""Tests for optional OCR engine and graceful degradation."""

from unittest.mock import MagicMock, patch
import pytest
from veritext.ingest.ocr import (
    get_ocr_engine,
    is_ocr_available,
    ocr_image,
    ocr_pdf_page,
    ocr_scanned_pdf,
)


def test_graceful_degradation_when_engine_missing():
    with patch("veritext.ingest.ocr.get_ocr_engine", return_value=None):
        assert is_ocr_available() is False
        assert ocr_image("nonexistent.png") == ""
        assert ocr_pdf_page("nonexistent.pdf", 0) == ""
        assert ocr_scanned_pdf("nonexistent.pdf") == ""


def test_ocr_image_with_mocked_engine():
    mock_engine = MagicMock()
    # RapidOCR returns tuple (list_of_items, elapsed_time)
    # where each item is [box, text_string, confidence]
    mock_engine.return_value = (
        [
            [[[0, 0], [10, 0], [10, 5], [0, 5]], "Scanned Document Header", 0.98],
            [[[0, 10], [10, 10], [10, 15], [0, 15]], "Second line of extracted text", 0.95],
        ],
        0.05,
    )

    with patch("veritext.ingest.ocr.get_ocr_engine", return_value=mock_engine):
        result = ocr_image(b"fake_image_bytes")
        assert "Scanned Document Header" in result
        assert "Second line of extracted text" in result
        assert result == "Scanned Document Header\nSecond line of extracted text"


def test_ocr_handles_exceptions_gracefully():
    mock_engine = MagicMock()
    mock_engine.side_effect = RuntimeError("Simulated OCR inference failure")

    with patch("veritext.ingest.ocr.get_ocr_engine", return_value=mock_engine):
        # Must catch the error, log a warning, and return empty string without raising
        result = ocr_image(b"broken_bytes")
        assert result == ""
