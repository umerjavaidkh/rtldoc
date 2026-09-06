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
import re
import shutil
import statistics
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
    latin = sum(1 for c in sample if c.isascii() and c.isalpha())
    if arabic > 40 and arabic / max(1, len(sample)) > 0.15 and "ara" in langs:
        return _with_english(latin, arabic, langs)

    # A wholly-scanned document has no text anywhere to sample, which is
    # exactly the case that matters: guessing English there makes
    # tesseract emit confident Latin nonsense for Arabic script rather
    # than nothing. Ask tesseract itself which script it sees.
    if not sample.strip() and "ara" in langs:
        if detect_script(page) == "Arabic":
            return "ara"
    return default if default in langs else next(iter(langs))


# A document is "bilingual" only when a fifth of its letters are Latin.
# Below that, adding "eng" to the language set costs more than it buys.
LATIN_BILINGUAL_SHARE = 0.20


def _with_english(latin: int, arabic: int, langs: set) -> str:
    """Add English to the language set only when the document is really
    bilingual.

    Tesseract's multi-language mode does not read each script with the
    matching model; it lets the recogniser choose, per word, whichever
    language scores higher. On an Arabic page that trade is always bad:
    an Arabic word it half-recognises comes back as a confident English
    one. Measured over four scanned pages of the Saudi labour
    regulation, "ara+eng" against "ara" alone:

        Arabic words   1929  vs  2008
        Latin tokens     70  vs     0      (every one junk: QUALI,
                                            ables, digall, digo, aloe,
                                            par, pall, ope, IN, lb)
        time          25.6s  vs  18.7s

    So English cost 79 real Arabic words, invented 70 fake English ones,
    and was 27% slower. Filtering the junk afterwards does not work
    either -- it arrives with high confidence ("Lo" at 96, "of" at 94,
    against a real "Accrual" at 97), so no confidence threshold and no
    per-page confidence comparison can tell the two apart.

    The one thing that does separate them is whether the document has
    Latin text at all. Across the 18-document Gulf corpus the Arabic
    documents sit at 0.0-5.3% Latin letters, far below this bar, so they
    all read as Arabic -- while a genuinely mixed document still gets
    both models.
    """
    if "eng" not in langs:
        return "ara"
    letters = arabic + latin
    if letters and latin / letters >= LATIN_BILINGUAL_SHARE:
        return "ara+eng"
    return "ara"


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


# A line is a read of a printed rule, not of text, when it is far taller
# than the page's own writing AND far less certain than it. Both bars are
# read off the page itself, so they travel to any scan resolution or type
# size.
RULE_READ_HEIGHT_MULT = 2.0
RULE_READ_CONF_FRAC = 0.6
# A line the recogniser itself barely believes -- under a third of the
# confidence it gave the rest of this page.
DISBELIEVED_CONF_FRAC = 0.35
# Arabic does not write the same letter three times running. Tatweel
# (U+0640, the kashida) does elongate, and is excluded here.
_IMPOSSIBLE_REPEAT = re.compile(r"([\u0621-\u063f\u0641-\u064a])\1{2,}")


def _drop_rule_reads(lines: dict) -> dict:
    """Drop OCR lines that are tesseract reading the page's decorations.

    A scan's rules, borders and banner edges are ink, so tesseract tries
    to read them, and it answers with letters: `ال اس لللبللللللللللنتنناتم`
    for one rule on p7 of the Saudi labour regulation. That is not a
    recognition this parser can repair -- there is no text under it.

    Two measurements separate those reads from real ones, and both are
    taken against the page's own median rather than a fixed number:

      * height. The box tesseract fits over a rule spans the decoration,
        so it comes out around 95px where the page's writing is 38.
      * confidence. The rule reads score 6-38 where the page's text
        scores 69-96.

    Over 20 scanned pages (750 lines) this drops 9 lines, every one junk,
    and keeps the three lines that are tall but confident -- real text in
    a larger face. Requiring BOTH is what protects a heading: a genuine
    large line is read confidently and stays.

    Height alone misses the flat decorations -- a dotted leader or a thin
    frame produces a normal-height line at 0.00-0.20 of the page's
    confidence. So a second, independent test drops any line the
    recogniser barely believes. Over 30 scanned pages (1,091 lines) the
    two populations do not overlap in the middle: lines carrying an
    impossible letter-repeat sit at a median 0.21 of the page's
    confidence, every other line at 0.97, with a tenth percentile of
    0.85.

    The impossible repeat itself is applied per WORD, never per line.
    Arabic does not write one letter three times running -- 22
    occurrences in 341,534 born-digital tokens across 30 documents
    (0.006%), and each of those 22 is itself corrupt -- but a rule read
    can be merged into the same OCR line as real text, and the highest
    confidence line carrying one (0.80) is a real sentence with junk
    appended. Dropping that line would lose the sentence.
    """
    if not lines:
        return lines
    all_words = [w for ws in lines.values() for w in ws]
    med_h = statistics.median(w["bbox"][3] - w["bbox"][1] for w in all_words)
    med_c = statistics.median(w["conf"] for w in all_words)
    if med_h <= 0 or med_c <= 0:
        return lines
    kept = {}
    for key, ws in lines.items():
        h = statistics.median(w["bbox"][3] - w["bbox"][1] for w in ws)
        c = sum(w["conf"] for w in ws) / len(ws)
        if h > RULE_READ_HEIGHT_MULT * med_h and c < RULE_READ_CONF_FRAC * med_c:
            continue
        if c < DISBELIEVED_CONF_FRAC * med_c:
            continue
        ws = [w for w in ws if not _IMPOSSIBLE_REPEAT.search(w["text"])]
        if ws:
            kept[key] = ws
    return kept


def parse_tsv(text: str) -> list[dict]:
    """Read tesseract's TSV output.

    quoting=QUOTE_NONE is load-bearing, not tidiness. Tesseract's TSV is
    not quoted CSV: it never escapes anything, it just writes the
    recognised characters into the text column. When OCR reads one glyph
    as a bare double quote -- which a scan's tick marks, hamza and dotted
    form rules routinely produce -- csv's default quotechar swallows
    every following row into that one field until the next quote appears.
    One stray quote on p92 of the scanned Saudi labour regulation ate 191
    of the page's 327 word rows and re-emitted them as an 8,148-character
    wall of raw TSV rendered as body text.
    """
    return list(csv.DictReader(text.splitlines(), delimiter="\t",
                               quoting=csv.QUOTE_NONE))


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
        rows = parse_tsv(tsv_path.read_text(encoding="utf-8", errors="replace"))

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
            "conf": conf,
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
    lines = _drop_rule_reads(lines)
    if not lines:
        return []

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
