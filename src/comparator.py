"""Text comparison metrics and report generation."""

from __future__ import annotations

import difflib
import html
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import normalize_text
from utils import (
    CHART_PATH,
    HTML_REPORT_PATH,
    PDF_REPORT_PATH,
    TEXT_REPORT_PATH,
    colorize,
    truncate_items,
    write_text_file,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    """Structured comparison result."""

    sequence_similarity: float
    cosine_similarity: float
    semantic_similarity: float
    total_words_text: int
    total_words_image: int
    missing_words: list[str]
    extra_words: list[str]
    modified_words: list[str]
    word_diff: list[str]
    character_diff: list[str]
    overall_result: str

    def to_report_dict(self, ocr_confidence: float, ocr_engine: str, pages_processed: int) -> dict[str, Any]:
        """Convert result to JSON-serializable report dictionary."""
        return {
            "ocr_confidence": round(ocr_confidence, 4),
            "ocr_engine": ocr_engine,
            "pages_processed": pages_processed,
            "sequence_similarity": round(self.sequence_similarity, 2),
            "cosine_similarity": round(self.cosine_similarity, 4),
            "semantic_similarity": round(self.semantic_similarity, 4),
            "total_words_text": self.total_words_text,
            "total_words_image": self.total_words_image,
            "missing_words": self.missing_words,
            "extra_words": self.extra_words,
            "modified_words": self.modified_words,
            "word_diff": self.word_diff,
            "character_diff": self.character_diff,
            "overall_result": self.overall_result,
        }


def sequence_similarity(text_a: str, text_b: str) -> float:
    """Return difflib similarity as a percentage."""
    return difflib.SequenceMatcher(None, text_a, text_b).ratio() * 100


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Return TF-IDF cosine similarity in the range 0..1."""
    if not text_a or not text_b:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        return 0.0
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def semantic_cosine_similarity(text_a: str, text_b: str) -> float:
    """Return sentence-transformer embedding cosine similarity in the range 0..1."""
    if not text_a or not text_b:
        return 0.0
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
    except Exception as exc:
        LOGGER.warning("Semantic similarity failed: %s", exc)
        return 0.0


def classify_similarity(sequence_score: float) -> str:
    """Classify an overall verdict using percentage thresholds."""
    if sequence_score >= 95:
        return "Nearly Identical"
    if sequence_score >= 85:
        return "Highly Similar"
    if sequence_score >= 70:
        return "Moderately Similar"
    if sequence_score >= 50:
        return "Somewhat Different"
    return "Completely Different"


def word_level_difference(text_words: list[str], image_words: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Compute missing, extra, modified, and highlighted word-level differences."""
    text_counter = Counter(text_words)
    image_counter = Counter(image_words)
    missing = sorted((text_counter - image_counter).elements())
    extra = sorted((image_counter - text_counter).elements())

    matcher = difflib.SequenceMatcher(None, text_words, image_words)
    modified: list[str] = []
    highlighted: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left = text_words[i1:i2]
        right = image_words[j1:j2]
        if tag == "equal":
            highlighted.extend(left)
        elif tag == "delete":
            highlighted.extend([f"[-{word}-]" for word in left])
        elif tag == "insert":
            highlighted.extend([f"[+{word}+]" for word in right])
        elif tag == "replace":
            paired_count = min(len(left), len(right))
            modified.extend(f"{left[index]} -> {right[index]}" for index in range(paired_count))
            if len(left) > paired_count:
                modified.extend(f"{word} -> " for word in left[paired_count:])
            if len(right) > paired_count:
                modified.extend(f" -> {word}" for word in right[paired_count:])
            highlighted.extend([f"[-{' '.join(left)}-]", f"[+{' '.join(right)}+]"])
    return missing, extra, modified, highlighted


def character_level_diff(text_a: str, text_b: str) -> list[str]:
    """Generate a unified character/text diff by line."""
    return list(
        difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile="document.txt",
            tofile="ocr_text",
            lineterm="",
        )
    )


def compare_documents(original_text: str, image_text: str, remove_stopwords: bool = False) -> ComparisonResult:
    """Normalize, compare, and return all required comparison metrics."""
    normalized_original = normalize_text(original_text, remove_stopwords=remove_stopwords)
    normalized_image = normalize_text(image_text, remove_stopwords=remove_stopwords)
    if not normalized_original:
        raise ValueError("Text document is empty after normalization.")
    if not normalized_image:
        raise ValueError("OCR text is empty after normalization.")

    original_words = normalized_original.split()
    image_words = normalized_image.split()
    missing, extra, modified, word_diff = word_level_difference(original_words, image_words)
    sequence_score = sequence_similarity(normalized_original, normalized_image)

    return ComparisonResult(
        sequence_similarity=sequence_score,
        cosine_similarity=tfidf_cosine_similarity(normalized_original, normalized_image),
        semantic_similarity=semantic_cosine_similarity(normalized_original, normalized_image),
        total_words_text=len(original_words),
        total_words_image=len(image_words),
        missing_words=missing,
        extra_words=extra,
        modified_words=modified,
        word_diff=word_diff,
        character_diff=character_level_diff(original_text, image_text),
        overall_result=classify_similarity(sequence_score),
    )


def build_text_report(report: dict[str, Any]) -> str:
    """Create a human-readable comparison report."""
    lines = [
        "===================================",
        "DOCUMENT COMPARISON REPORT",
        "===================================",
        "",
        f"OCR Engine            : {report['ocr_engine']}",
        f"OCR Confidence        : {report['ocr_confidence']:.2%}",
        f"Pages Processed       : {report['pages_processed']}",
        f"Sequence Similarity   : {report['sequence_similarity']:.2f}%",
        f"Cosine Similarity     : {report['cosine_similarity']:.4f}",
        f"Semantic Similarity   : {report['semantic_similarity']:.4f}",
        f"Total Words Text      : {report['total_words_text']}",
        f"Total Words Image     : {report['total_words_image']}",
        "",
        f"Missing Words ({len(report['missing_words'])})",
        ", ".join(truncate_items(report["missing_words"])) or "None",
        "",
        f"Extra Words ({len(report['extra_words'])})",
        ", ".join(truncate_items(report["extra_words"])) or "None",
        "",
        f"Modified Words ({len(report['modified_words'])})",
        ", ".join(truncate_items(report["modified_words"])) or "None",
        "",
        f"Overall Verdict       : {report['overall_result']}",
        "",
        "Character-Level Diff",
        "\n".join(truncate_items(report["character_diff"], 120)) or "No differences.",
    ]
    return "\n".join(lines) + "\n"


def save_html_report(report: dict[str, Any], path: Path = HTML_REPORT_PATH) -> None:
    """Save a compact HTML comparison report."""
    diff_html = difflib.HtmlDiff(wrapcolumn=90).make_file(
        report.get("source_text_lines", []),
        report.get("ocr_text_lines", []),
        "document.txt",
        "OCR text",
    )
    summary = f"""
    <h1>Document Comparison Report</h1>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>OCR Confidence</td><td>{report['ocr_confidence']:.2%}</td></tr>
      <tr><td>Sequence Similarity</td><td>{report['sequence_similarity']:.2f}%</td></tr>
      <tr><td>Cosine Similarity</td><td>{report['cosine_similarity']:.4f}</td></tr>
      <tr><td>Semantic Similarity</td><td>{report['semantic_similarity']:.4f}</td></tr>
      <tr><td>Overall Verdict</td><td>{html.escape(report['overall_result'])}</td></tr>
    </table>
    """
    content = diff_html.replace("<body>", f"<body>{summary}", 1)
    write_text_file(path, content)


def save_similarity_chart(report: dict[str, Any], path: Path = CHART_PATH) -> None:
    """Save a similarity bar chart if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt

        labels = ["Sequence", "TF-IDF", "Semantic"]
        values = [
            report["sequence_similarity"] / 100,
            report["cosine_similarity"],
            report["semantic_similarity"],
        ]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(labels, values, color=["#2e7d32", "#1565c0", "#6a1b9a"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Similarity")
        ax.set_title("Document Similarity Metrics")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("Unable to create similarity chart: %s", exc)


def save_pdf_report(report_text: str, path: Path = PDF_REPORT_PATH) -> None:
    """Save a PDF report if reportlab is available."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 50
        pdf.setFont("Helvetica", 10)
        for line in report_text.splitlines():
            if y < 50:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = height - 50
            pdf.drawString(40, y, line[:115])
            y -= 14
        pdf.save()
    except Exception as exc:
        LOGGER.warning("Unable to create PDF report: %s", exc)


def print_terminal_summary(report: dict[str, Any], use_color: bool = True) -> None:
    """Print the requested concise terminal summary."""
    verdict_color = "green" if report["sequence_similarity"] >= 85 else "yellow"
    print("Running comparisons...")
    print(f"Sequence Similarity : {report['sequence_similarity']:.2f}%")
    print(f"Cosine Similarity : {report['cosine_similarity']:.2f}")
    print(f"Semantic Similarity : {report['semantic_similarity']:.2f}")
    print(f"Missing Words : {len(report['missing_words'])}")
    print(f"Extra Words : {len(report['extra_words'])}")
    print(f"Modified Words : {len(report['modified_words'])}")
    print(f"Overall Verdict : {colorize(report['overall_result'], verdict_color, use_color)}")

