"""
Layer 2c -- OCR fallback for scanned / no-text-layer pages.

Shells out to the `tesseract` CLI binary directly, not the pytesseract
Python wrapper: that wrapper pulls in pandas/pyarrow as a hard import-time
dependency for no benefit here, and a broken numpy/pandas ABI in that chain
is a common way for it to fail to import at all even when tesseract itself
works fine. The binary's own TSV output mode gives word-level text, bounding
boxes, and confidence directly -- no XML/hOCR parsing needed -- and pixel
coordinates convert back to PDF points with the same render scale used to
rasterize the page, so the result drops into the same (text, bbox) shape a
born-digital page's own paragraph blocks use.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path


def available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page(page: "object", dpi: int = 300, lang: str = "eng") -> list[tuple[str, tuple]]:
    """Render `page` and OCR it. Returns a list of (text, bbox) paragraphs
    in reading order (top-to-bottom). Empty if tesseract isn't installed,
    the page has no recognizable text, or the OCR call itself fails --
    never raises, so a scanned page without a working tesseract install
    just falls back to the previous behavior (no blocks) rather than
    breaking the page entirely.
    """
    if not available():
        return []

    import fitz
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    scale = 1.0 / zoom

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "page.png"
        pix.save(str(img_path))
        out_base = Path(tmp) / "out"
        try:
            subprocess.run(
                ["tesseract", str(img_path), str(out_base), "-l", lang, "tsv"],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return []
        tsv_path = out_base.with_suffix(".tsv")
        if not tsv_path.exists():
            return []
        rows = list(csv.DictReader(tsv_path.read_text(encoding="utf-8", errors="replace")
                                   .splitlines(), delimiter="\t"))

    words = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(r.get("conf", -1))
            x, y, w, h = (float(r["left"]), float(r["top"]), float(r["width"]), float(r["height"]))
        except (TypeError, ValueError, KeyError):
            continue
        if conf < 0:
            continue
        words.append({
            "text": text,
            "bbox": (x * scale, y * scale, (x + w) * scale, (y + h) * scale),
            # (block_num, par_num, line_num) -- tesseract's own layout
            # analysis already groups words into lines and paragraphs;
            # reusing that instead of re-deriving it from scratch.
            "line_key": (r.get("block_num"), r.get("par_num"), r.get("line_num")),
            "para_key": (r.get("block_num"), r.get("par_num")),
        })
    if not words:
        return []

    lines: dict[tuple, list[dict]] = {}
    for w in words:
        lines.setdefault(w["line_key"], []).append(w)

    paras: dict[tuple, list[tuple]] = {}
    line_rows = []
    for key, ws in lines.items():
        ws.sort(key=lambda w: w["bbox"][0])
        text = " ".join(w["text"] for w in ws)
        x0 = min(w["bbox"][0] for w in ws)
        y0 = min(w["bbox"][1] for w in ws)
        x1 = max(w["bbox"][2] for w in ws)
        y1 = max(w["bbox"][3] for w in ws)
        para_key = ws[0]["para_key"]
        line_rows.append((para_key, y0, text, (x0, y0, x1, y1)))

    for para_key, y0, text, bbox in line_rows:
        paras.setdefault(para_key, []).append((y0, text, bbox))

    out: list[tuple[str, tuple]] = []
    for _, para_lines in paras.items():
        para_lines.sort(key=lambda ln: ln[0])
        text = "\n".join(ln[1] for ln in para_lines)
        x0 = min(ln[2][0] for ln in para_lines)
        y0 = min(ln[2][1] for ln in para_lines)
        x1 = max(ln[2][2] for ln in para_lines)
        y1 = max(ln[2][3] for ln in para_lines)
        out.append((text, (x0, y0, x1, y1)))

    out.sort(key=lambda item: (item[1][1], item[1][0]))
    return out
