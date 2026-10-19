"""Optional OCR engine for scanned PDF pages and images using pypdfium2 and RapidOCR."""

import io
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

_ocr_engine = None
_ocr_checked = False


def get_ocr_engine():
    """Lazily initializes the RapidOCR engine if available."""
    global _ocr_engine, _ocr_checked
    if not _ocr_checked:
        _ocr_checked = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        except Exception as e:
            logger.info(f"RapidOCR engine not available: {e}")
            _ocr_engine = None
    return _ocr_engine


def is_ocr_available() -> bool:
    """Returns True if RapidOCR and pypdfium2 are installed and operational."""
    return get_ocr_engine() is not None


def ocr_image(image_input: Union[str, Path, bytes]) -> str:
    """
    Performs OCR on an image file (.png, .jpg, .jpeg) or image bytes.
    Returns extracted plain text string. Degrades gracefully if OCR is unavailable.
    """
    engine = get_ocr_engine()
    if engine is None:
        logger.warning("OCR requested for image, but rapidocr-onnxruntime is not installed.")
        return ""

    try:
        if isinstance(image_input, (str, Path)):
            img_path = str(Path(image_input).resolve())
            result, _ = engine(img_path)
        elif isinstance(image_input, bytes):
            result, _ = engine(image_input)
        else:
            result, _ = engine(image_input)

        if not result:
            return ""

        # RapidOCR returns list of [box, text, score]
        lines = [item[1].strip() for item in result if item and len(item) > 1 and item[1].strip()]
        return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Error during OCR image processing: {e}")
        return ""


def ocr_pdf_page(pdf_path: Union[str, Path], page_index: int) -> str:
    """
    Renders a specific PDF page using pypdfium2 and performs OCR.
    """
    engine = get_ocr_engine()
    if engine is None:
        return ""

    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(pdf_path))
        if page_index < 0 or page_index >= len(pdf):
            return ""

        page = pdf[page_index]
        # Render page at 2.0 scale (144 dpi) for OCR clarity
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()

        # Convert PIL image to PNG bytes for RapidOCR
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return ocr_image(png_bytes)

    except Exception as e:
        logger.warning(f"Error rendering and OCRing PDF page {page_index}: {e}")
        return ""


def ocr_scanned_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Renders and OCRs all pages of a PDF document where text could not be extracted natively.
    """
    engine = get_ocr_engine()
    if engine is None:
        return ""

    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(pdf_path))
        page_texts = []
        for i in range(len(pdf)):
            page_text = ocr_pdf_page(pdf_path, i)
            if page_text.strip():
                page_texts.append(page_text.strip())

        return "\n\n".join(page_texts)

    except Exception as e:
        logger.warning(f"Failed to perform fallback OCR on PDF: {e}")
        return ""
