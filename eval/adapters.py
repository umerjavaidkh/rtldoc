"""Adapters: one function per parser, all returning the same shape --

    {"blocks": [str, ...], "text": str}

`blocks` is that parser's own natural reading-order segmentation (paragraph,
line, or region -- whatever the tool produces); `text` is everything joined,
for the whole-page edit-distance metric. Reading-order scoring matches
`blocks` against gold by content, not by any shared ID scheme, precisely so
adapters don't need to agree on what a "block" is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtldoc import arabic, pipeline  # noqa: E402


def naive_pymupdf(pdf_path: str, page_no: int) -> dict:
    """Zero-effort baseline: page.get_text(), no processing at all."""
    doc = fitz.open(pdf_path)
    text = doc[page_no - 1].get_text()
    doc.close()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks and text.strip():
        blocks = [text.strip()]
    return {"blocks": blocks, "text": text}


def pdfplumber_adapter(pdf_path: str, page_no: int) -> dict:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no - 1]
        text = page.extract_text() or ""
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        if not blocks and text.strip():
            blocks = [text.strip()]
        tables = page.extract_tables()
    return {"blocks": blocks, "text": text, "tables": tables}


def rtldoc_adapter(pdf_path: str, page_no: int, style_map: dict | None = None) -> dict:
    doc = fitz.open(pdf_path)
    result = pipeline.parse_page(doc[page_no - 1], style_map, arabic.NormalizeOptions())
    doc.close()
    blocks = [b.text for b in result.blocks if b.text.strip()]
    tables = []
    for b in result.blocks:
        if b.role == "table" and "|" in b.text:
            rows = [r for r in b.text.split("\n") if r.strip().startswith("|")]
            rows = [r for r in rows if "---" not in r]
            tables.append([[c.strip() for c in row.strip("|").split("|")] for row in rows])
    return {"blocks": blocks, "text": "\n".join(blocks), "tables": tables}


ADAPTERS = {
    "naive_pymupdf": naive_pymupdf,
    "pdfplumber": pdfplumber_adapter,
    "rtldoc": rtldoc_adapter,
}
