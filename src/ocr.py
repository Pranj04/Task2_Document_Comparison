"""OCR extraction using EasyOCR with optional Tesseract fallback."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from preprocessing import preprocess_image
from utils import validate_file

LOGGER = logging.getLogger(__name__)


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class OCRResult:
    """OCR extraction result."""

    text: str
    confidence: float
    engine: str
    pages_processed: int


def _extract_with_easyocr(image: np.ndarray, languages: list[str]) -> tuple[str, float]:
    import easyocr

    reader = easyocr.Reader(languages, gpu=False)
    results = reader.readtext(image, detail=1, paragraph=False)
    if not results:
        raise RuntimeError("EasyOCR returned no text.")

    words: list[str] = []
    confidences: list[float] = []
    for result in results:
        if len(result) >= 3:
            _, text, confidence = result[:3]
            cleaned = str(text).strip()
            if cleaned:
                words.append(cleaned)
                confidences.append(float(confidence))

    if not words:
        raise RuntimeError("EasyOCR returned empty text.")
    return " ".join(words), float(np.mean(confidences)) if confidences else 0.0


def _extract_with_tesseract(image: np.ndarray) -> tuple[str, float]:
    import pytesseract

    text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confidences = [
        float(value)
        for value in data.get("conf", [])
        if str(value).strip() not in {"", "-1"} and float(value) >= 0
    ]
    if not text.strip():
        raise RuntimeError("Tesseract returned empty text.")
    return text.strip(), (float(np.mean(confidences)) / 100.0 if confidences else 0.0)


def _pdf_to_images(path: Path, dpi: int = 220) -> list[Path]:
    """Convert PDF pages to temporary PNG files using PyMuPDF."""
    import fitz

    temp_dir = Path(tempfile.mkdtemp(prefix="doc_compare_pdf_"))
    image_paths: list[Path] = []
    with fitz.open(path) as document:
        if document.page_count == 0:
            raise ValueError(f"PDF has no pages: {path}")
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = temp_dir / f"page_{index + 1}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)
    return image_paths


def _resolve_input_pages(path: Path) -> list[Path]:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return [path]
    if suffix == ".pdf":
        return _pdf_to_images(path)
    raise ValueError(f"Unsupported OCR input format: {path.suffix}")


def extract_text_from_image(
    image_path: Path,
    languages: list[str] | None = None,
    use_tesseract_fallback: bool = True,
) -> OCRResult:
    """Extract text from an image or PDF using EasyOCR, then Tesseract if enabled."""
    validate_file(image_path, "Image document")
    languages = languages or ["en"]
    pages = _resolve_input_pages(image_path)
    extracted_pages: list[str] = []
    confidences: list[float] = []
    engines: list[str] = []

    for page_path in pages:
        processed_image = preprocess_image(page_path)
        try:
            text, confidence = _extract_with_easyocr(processed_image, languages)
            engine = "easyocr"
        except Exception as easyocr_error:
            LOGGER.warning("EasyOCR failed for %s: %s", page_path, easyocr_error)
            if not use_tesseract_fallback:
                raise RuntimeError(f"OCR failed with EasyOCR: {easyocr_error}") from easyocr_error
            try:
                text, confidence = _extract_with_tesseract(processed_image)
                engine = "tesseract"
            except Exception as tesseract_error:
                raise RuntimeError(
                    f"OCR failed with EasyOCR and Tesseract for {page_path}: "
                    f"{easyocr_error}; {tesseract_error}"
                ) from tesseract_error

        extracted_pages.append(text)
        confidences.append(confidence)
        engines.append(engine)

    combined_text = "\n\n".join(extracted_pages).strip()
    if not combined_text:
        raise RuntimeError("OCR completed but extracted text is empty.")

    engine_label = engines[0] if len(set(engines)) == 1 else "+".join(sorted(set(engines)))
    return OCRResult(
        text=combined_text,
        confidence=float(np.mean(confidences)) if confidences else 0.0,
        engine=engine_label,
        pages_processed=len(pages),
    )

