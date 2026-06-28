"""Shared utilities for document comparison workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_TEXT_PATH = INPUT_DIR / "document.txt"
DEFAULT_IMAGE_PATH = INPUT_DIR / "document_image.png"
EXTRACTED_TEXT_PATH = OUTPUT_DIR / "extracted_text.txt"
JSON_REPORT_PATH = OUTPUT_DIR / "comparison_report.json"
TEXT_REPORT_PATH = OUTPUT_DIR / "comparison_report.txt"
HTML_REPORT_PATH = OUTPUT_DIR / "comparison_report.html"
PDF_REPORT_PATH = OUTPUT_DIR / "comparison_report.pdf"
CHART_PATH = OUTPUT_DIR / "similarity_chart.png"


def configure_logging(verbose: bool = False) -> None:
    """Configure application logging once."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_directories() -> None:
    """Create input and output directories if missing."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_file(path: Path, description: str) -> None:
    """Validate that a required file exists and is not empty."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {path}")


def read_text_file(path: Path) -> str:
    """Read a text file with robust encoding fallbacks."""
    validate_file(path, "Text document")
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Unable to decode {path} with supported encodings: {last_error}",
    )


def write_text_file(path: Path, content: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write pretty JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def truncate_items(items: list[str], limit: int = 50) -> list[str]:
    """Limit long report lists while keeping output readable."""
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... ({len(items) - limit} more)"]


def colorize(text: str, color: str, enabled: bool = True) -> str:
    """Apply ANSI color to terminal text when enabled."""
    if not enabled:
        return text
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "cyan": "\033[36m",
        "bold": "\033[1m",
    }
    reset = "\033[0m"
    return f"{colors.get(color, '')}{text}{reset}"

