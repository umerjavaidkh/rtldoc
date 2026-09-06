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


def installed_langs() -> set:
    """Language packs tesseract can actually load."""
    if not available():
        return set()
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                             text=True, timeout=20).stdout
        return {ln.strip() for ln in out.splitlines()[1:] if ln.strip()}
    except Exception:
        return set()


def pick_lang(page: "object", default: str = "eng") -> str:
    """Choose the OCR language from the page's OWN glyphs where it has any.

    A scanned page has no text layer, but a document is rarely wholly
    scanned: its other pages, or this page's own stamps and headers,
    usually carry enough characters to name the script. Guessing "eng" on
    an Arabic page is not a small error -- tesseract will emit confident
    Latin nonsense rather than nothing, which is worse than no output.
    """
    langs = installed_langs()
    if not langs:
        return default
    try:
        doc = page.parent
        sample = ""
        for i in range(min(getattr(doc, "page_count", 0), 12)):
            sample += doc[i].get_text()
            if len(sample) > 4000:
                break
    except Exception:
        sample = ""
    arabic = sum(1 for c in sample if "\u0600" <= c <= "\u06ff")
    if arabic > 40 and arabic / max(1, len(sample)) > 0.15 and "ara" in langs:
        return "ara+eng" if "eng" in langs else "ara"

    # A wholly-scanned document has no text anywhere to sample, which is
    # exactly the case that matters: guessing English there makes
    # tesseract emit confident Latin nonsense for Arabic script rather
    # than nothing. Ask tesseract itself which script it sees.
    if not sample.strip() and "ara" in langs:
        if detect_script(page) == "Arabic":
            return "ara+eng" if "eng" in langs else "ara"
    return default if default in langs else next(iter(langs))


def detect_script(page: "object", dpi: int = 150) -> str | None:
    """Tesseract's own orientation-and-script detection for one page."""
    if not available():
        return None
    try:
        import fitz
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "osd.png"
            pix.save(str(img))
            out = subprocess.run(["tesseract", str(img), "stdout", "--psm", "0"],
                                 capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            if line.startswith("Script:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def ocr_page(page: "object", dpi: int = 300, lang: str | None = None) -> list[tuple[str, tuple]]:
    """Render `page` and OCR it. Returns a list of (text, bbox) paragraphs
    in reading order (top-to-bottom). Empty if tesseract isn't installed,
    the page has no recognizable text, or the OCR call itself fails --
    never raises, so a scanned page without a working tesseract install
    just falls back to the previous behavior (no blocks) rather than
    breaking the page entirely.
    """
    if not available():
        return []
    if lang is None:
        lang = pick_lang(page)

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
                # --psm 6: treat the page as one uniform block of text.
                #
                # The default (psm 3) runs full page segmentation, and on a
                # scan that means tesseract tries to READ the decorations: a
                # ruled line comes back as "ااا ااا اا", a page border as
                # "|||", and decorative marks as English -- 90 characters of
                # "enone nena enenenenenenene" on p3 of the scanned Saudi
                # labour regulation.
                #
                # This is the one thing EasyOCR does better architecturally:
                # it detects text regions first and recognises only those, so
                # it never reads a rule. psm 6 gets most of that for free.
                # Measured over three scanned pages: Latin junk 65 -> 42,
                # single-character repeat junk 7 -> 1, real Arabic words
                # 1684 -> 1729, and 24.2s -> 21.1s. Less noise, MORE text,
                # and faster -- EasyOCR itself is 4x too slow at 30s/page.
                ["tesseract", str(img_path), str(out_base), "-l", lang,
                 "--psm", "6", "tsv"],
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
            "word_num": int(r.get("word_num") or 0),
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
        # Tesseract already emits a line's words in READING order, so keep
        # it. Re-sorting by x is right-to-left backwards for Arabic and
        # reversed every line ("وزارة التعليم العالي" came out
        # "العالي التعليم وزارة"), which is worse than not running OCR at
        # all -- the words are all present and all in the wrong order.
        ws.sort(key=lambda w: w["word_num"])
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
