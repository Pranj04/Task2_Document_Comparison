"""Command-line entry point for document comparison."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from comparator import (
    build_text_report,
    compare_documents,
    print_terminal_summary,
    save_html_report,
    save_pdf_report,
    save_similarity_chart,
)
from ocr import extract_text_from_image
from utils import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_TEXT_PATH,
    EXTRACTED_TEXT_PATH,
    JSON_REPORT_PATH,
    TEXT_REPORT_PATH,
    configure_logging,
    ensure_directories,
    read_text_file,
    write_json_file,
    write_text_file,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare a text document with OCR text from an image.")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT_PATH, help="Path to the source .txt file.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Path to the image/PDF file.")
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove English stopwords before comparison.")
    parser.add_argument("--no-tesseract", action="store_true", help="Disable Tesseract fallback when EasyOCR fails.")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def run() -> int:
    """Run OCR, comparison, and report generation."""
    args = parse_args()
    configure_logging(verbose=args.verbose)
    ensure_directories()

    try:
        source_text = read_text_file(args.text)
        ocr_result = extract_text_from_image(
            args.image,
            use_tesseract_fallback=not args.no_tesseract,
        )
        write_text_file(EXTRACTED_TEXT_PATH, ocr_result.text)

        print("OCR Completed.")
        print(f"Extracted {len(ocr_result.text.split())} words.")

        comparison = compare_documents(
            source_text,
            ocr_result.text,
            remove_stopwords=args.remove_stopwords,
        )
        report = comparison.to_report_dict(
            ocr_confidence=ocr_result.confidence,
            ocr_engine=ocr_result.engine,
            pages_processed=ocr_result.pages_processed,
        )

        write_json_file(JSON_REPORT_PATH, report)
        report_text = build_text_report(report)
        write_text_file(TEXT_REPORT_PATH, report_text)
        html_report = {
            **report,
            "source_text_lines": source_text.splitlines(),
            "ocr_text_lines": ocr_result.text.splitlines(),
        }
        save_html_report(html_report)
        save_pdf_report(report_text)
        save_similarity_chart(report)

        print_terminal_summary(report, use_color=not args.no_color)
        print("Comparison report saved successfully.")
        return 0
    except Exception as exc:
        LOGGER.exception("Document comparison failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
