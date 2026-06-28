# Document Comparison

Production-ready Python project for comparing a plain text document with text extracted from an image or PDF using OCR.

## Features

- EasyOCR primary OCR engine
- Optional Tesseract fallback
- OCR-focused preprocessing: grayscale conversion, denoising, adaptive thresholding, and deskewing
- Text normalization with optional stopword removal
- Sequence similarity, TF-IDF cosine similarity, and semantic similarity
- Word-level and character-level differences
- JSON, text, HTML, PDF, and chart outputs
- Supports common image formats and PDF input

## Project Architecture

```text
document_comparison/
├── input/
│   ├── document.txt
│   └── document_image.png
├── output/
│   ├── extracted_text.txt
│   ├── comparison_report.json
│   └── comparison_report.txt
├── src/
│   ├── ocr.py
│   ├── preprocessing.py
│   ├── comparator.py
│   ├── utils.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

For Tesseract fallback, install the Tesseract OCR binary separately and ensure it is available on your `PATH`.

## How To Run

Place your files here:

- `input/document.txt`
- `input/document_image.png`

Then run from the project root:

```powershell
python src/main.py
```

Optional arguments:

```powershell
python src/main.py --text input/document.txt --image input/document_image.png
python src/main.py --remove-stopwords
python src/main.py --no-tesseract
python src/main.py --no-color
```

## Example Output

```text
OCR Completed.
Extracted 538 words.
Running comparisons...
Sequence Similarity : 94.82%
Cosine Similarity : 0.96
Semantic Similarity : 0.98
Missing Words : 4
Extra Words : 2
Modified Words : 3
Overall Verdict : Highly Similar
Comparison report saved successfully.
```

## Generated Files

- `output/extracted_text.txt`: raw OCR text
- `output/comparison_report.json`: structured machine-readable report
- `output/comparison_report.txt`: human-readable report
- `output/comparison_report.html`: visual HTML diff report
- `output/comparison_report.pdf`: PDF report
- `output/similarity_chart.png`: similarity bar chart

## JSON Report Example

```json
{
    "ocr_confidence": 0.9321,
    "ocr_engine": "easyocr",
    "pages_processed": 1,
    "sequence_similarity": 92.4,
    "cosine_similarity": 0.95,
    "semantic_similarity": 0.98,
    "total_words_text": 540,
    "total_words_image": 538,
    "missing_words": ["example"],
    "extra_words": ["sample"],
    "modified_words": ["organisation -> organization"],
    "overall_result": "Highly Similar"
}
```

## Similarity Thresholds

- `>= 95%`: Nearly Identical
- `85-95%`: Highly Similar
- `70-85%`: Moderately Similar
- `50-70%`: Somewhat Different
- `< 50%`: Completely Different

## Notes

EasyOCR and Sentence Transformers download model weights on first use. The first run may take longer and requires network access if the models are not already cached.
