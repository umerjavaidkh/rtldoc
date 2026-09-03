"""
Layer 0 -- producer identification.

Every PDF is a recording of what one specific program did, and each program
leaves a signature in the content stream that is far more reliable than the
/Producer string it writes into /Info (arXiv, for one, rewrites /Producer to
"pikepdf" on every paper it serves, erasing the real generator).

Knowing the generator matters because the failure modes are generator-shaped,
not document-shaped:

  * Skia (Chrome print-to-PDF, Google Docs export) emits one Tj per *glyph*,
    each preceded by its own Td, and wraps roughly one word per BT/ET. There
    is no word or run grouping anywhere in the file -- word breaks exist only
    as geometry. It also uses a y-flipped text matrix (1 0 0 -1).
  * pdfTeX/LuaTeX emit TJ only, never Tj, position with Td, and ship Type1
    subsets where barely half the fonts carry a /ToUnicode -- so glyph->text
    is lossy for maths and ligatures no matter how good the layout logic is.
  * Word, Adobe PDFMaker and Google Docs all emit a real /StructTreeRoot with
    /Table, /TR, /TD, /TH -- the table structure is stated, not inferrable.
  * ReportLab, pdf-lib and Apache FOP emit no tags at all and often no
    /ToUnicode, so geometry is genuinely the only evidence available.

This module answers one question -- "what wrote this file, and what can I
therefore trust?" -- so the layers above can pick an evidence strategy instead
of applying one set of geometric heuristics to every document alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class ProducerProfile:
    """What the generating tool is, and what evidence it makes available."""

    name: str = "unknown"          # canonical family, e.g. "skia", "pdftex"
    detail: str = ""               # raw /Producer or /Creator that identified it
    tagged: bool = False           # a usable /StructTreeRoot is present
    per_glyph_text: bool = False   # one text-showing op per glyph (Skia)
    y_flipped: bool = False        # text matrix has negative d
    tounicode_frac: float = 1.0    # fraction of fonts carrying /ToUnicode
    evidence: str = "geometry"     # best available: "tags" | "marked" | "geometry"
    notes: list[str] = field(default_factory=list)


# Ordered most-specific-first: "Skia/PDF m152 Google Docs Renderer" must match
# the Google Docs rule before the generic Skia one.
_SIGNATURES: tuple[tuple[str, str], ...] = (
    (r"Google Docs Renderer", "google-docs"),
    (r"Skia/PDF", "skia"),
    (r"Acrobat PDFMaker", "pdfmaker"),
    (r"Acrobat Distiller", "distiller"),
    (r"Adobe PDF Library", "adobe-lib"),
    (r"Microsoft.{0,3} Word", "word"),
    (r"Microsoft: Print To PDF", "mspdf"),
    (r"LibreOffice|OpenOffice", "libreoffice"),
    (r"WeasyPrint", "weasyprint"),
    (r"wkhtmltopdf", "wkhtmltopdf"),
    (r"Prince", "princexml"),
    (r"Apache FOP", "fop"),
    (r"arXiv GenPDF|tex2pdf", "pdftex"),
    (r"pdfTeX", "pdftex"),
    (r"LaTeX with hyperref", "pdftex"),
    (r"LuaTeX", "luatex"),
    (r"XeTeX", "xetex"),
    (r"ReportLab", "reportlab"),
    (r"pdf-lib", "pdflib-js"),
    (r"iText", "itext"),
    (r"Ghostscript", "ghostscript"),
    (r"Quartz PDFContext", "quartz"),
    (r"Canva", "canva"),
    (r"Designer \d", "livecycle"),
    (r"pikepdf|qpdf", "rewrapped"),
)

# Producers that only ever rewrite an existing file. Seeing one means the real
# generator is upstream, so /Creator (and the operator mix) decide instead.
_REWRAPPERS = {"rewrapped", "ghostscript"}

_TM = re.compile(rb"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
                 rb"(-?[\d.]+)\s+(-?[\d.]+)\s+Tm")


def _match(text: str) -> str | None:
    for pat, name in _SIGNATURES:
        if re.search(pat, text, re.I):
            return name
    return None


def _strip_strings(b: bytes) -> bytes:
    """Blank out ( ) and < > strings so operator counting isn't fooled by text."""
    out = bytearray()
    i, n = 0, len(b)
    while i < n:
        c = b[i]
        if c == 0x28:  # (
            depth = 1
            i += 1
            while i < n and depth:
                if b[i] == 0x5C:
                    i += 2
                    continue
                if b[i] == 0x28:
                    depth += 1
                elif b[i] == 0x29:
                    depth -= 1
                i += 1
        elif c == 0x3C and i + 1 < n and b[i + 1] != 0x3C:  # < but not <<
            j = b.find(b">", i)
            i = n if j < 0 else j + 1
        else:
            out.append(c)
            i += 1
    return bytes(out)


def _count(pat: bytes, data: bytes) -> int:
    return len(re.findall(pat, data))


def identify(doc: "fitz.Document", sample_pages: int = 4) -> ProducerProfile:
    """Identify the generating tool and what evidence it left behind.

    Metadata names the tool when it is honest; the operator mix is what
    catches it when it is not. Both are cheap -- a few sampled pages.
    """
    md = doc.metadata or {}
    producer = (md.get("producer") or "").strip()
    creator = (md.get("creator") or "").strip()

    name = _match(producer)
    if name is None or name in _REWRAPPERS:
        # /Producer was a rewrapper (or unrecognised) -- believe /Creator.
        name = _match(creator) or name or "unknown"

    prof = ProducerProfile(name=name, detail=producer or creator)

    cat = doc.pdf_catalog()
    try:
        prof.tagged = doc.xref_get_key(cat, "StructTreeRoot")[0] != "null"
    except Exception:
        prof.tagged = False

    # --- operator-level fingerprint over a few sampled pages -----------------
    if doc.page_count:
        step = max(1, doc.page_count // max(1, sample_pages))
        idx = list(range(0, doc.page_count, step))[:sample_pages]
        tj = tj_ops = tm = flip = bdc = 0
        fonts: dict[str, bool] = {}
        for pno in idx:
            try:
                page = doc[pno]
                raw = page.read_contents()
            except Exception:
                continue
            stripped = _strip_strings(raw)
            tj_ops += _count(rb"(?<![A-Za-z0-9])Tj(?![A-Za-z0-9])", stripped)
            bdc += _count(rb"(?<![A-Za-z0-9])BDC(?![A-Za-z0-9])", stripped)
            ms = _TM.findall(raw)
            tm += len(ms)
            flip += sum(1 for m in ms if float(m[3]) < 0)
            # single-glyph shows are the Skia tell
            for m in re.finditer(rb"<([0-9A-Fa-f\s]{1,8})>\s*Tj", raw):
                tj += 1 if len(re.sub(rb"\s", b"", m.group(1))) <= 4 else 0
            for m in re.finditer(rb"\(((?:\\.|[^\\()]){1,2})\)\s*Tj", raw):
                tj += 1
            for f in page.get_fonts(full=True):
                xref = f[0]
                if xref:
                    try:
                        fonts[f[3]] = doc.xref_get_key(xref, "ToUnicode")[0] != "null"
                    except Exception:
                        fonts[f[3]] = False

        prof.per_glyph_text = tj_ops > 0 and tj / max(1, tj_ops) > 0.8
        prof.y_flipped = tm > 0 and flip / tm > 0.8
        if fonts:
            prof.tounicode_frac = round(sum(fonts.values()) / len(fonts), 3)
        if prof.tagged:
            prof.evidence = "tags"
        elif bdc > 0:
            prof.evidence = "marked"
        else:
            prof.evidence = "geometry"

    # --- notes the layers above act on --------------------------------------
    if prof.per_glyph_text:
        prof.notes.append(
            "one text op per glyph: word breaks exist only as geometry, "
            "never as emitted spacing"
        )
    if prof.y_flipped:
        prof.notes.append("y-flipped text matrix (1 0 0 -1)")
    if prof.tounicode_frac < 0.9:
        prof.notes.append(
            f"only {prof.tounicode_frac:.0%} of fonts carry /ToUnicode: "
            "glyph->text is lossy, expect dropped ligatures and maths"
        )
    if prof.name == "rewrapped":
        prof.notes.append("/Producer was rewritten by a repackager; real generator unknown")
    return prof
