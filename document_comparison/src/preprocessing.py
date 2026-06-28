"""Image and text preprocessing utilities."""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover - sklearn is a declared dependency.
    ENGLISH_STOP_WORDS = frozenset()


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk and raise a clear error if it is invalid."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Invalid or unsupported image file: {path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(gray_image: np.ndarray) -> np.ndarray:
    """Reduce salt-and-pepper noise while preserving text edges."""
    return cv2.medianBlur(gray_image, 3)


def threshold_image(gray_image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for OCR-friendly contrast."""
    return cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )


def deskew_image(binary_image: np.ndarray) -> np.ndarray:
    """Deskew a binary text image when a meaningful skew angle is detected."""
    inverted = cv2.bitwise_not(binary_image)
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return binary_image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return binary_image

    height, width = binary_image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    LOGGER.debug("Deskewing image by %.2f degrees", angle)
    return cv2.warpAffine(
        binary_image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(path: Path) -> np.ndarray:
    """Run OCR-oriented image preprocessing."""
    image = load_image(path)
    gray = convert_to_grayscale(image)
    denoised = remove_noise(gray)
    thresholded = threshold_image(denoised)
    return deskew_image(thresholded)


def normalize_text(text: str, remove_stopwords: bool = False) -> str:
    """Normalize text for similarity comparison."""
    if text is None:
        return ""

    normalized = text.lower()
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if remove_stopwords:
        words = [word for word in normalized.split() if word not in ENGLISH_STOP_WORDS]
        normalized = " ".join(words)
    return normalized

