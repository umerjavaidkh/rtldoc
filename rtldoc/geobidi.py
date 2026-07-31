"""
Geometry-first bidi reconstruction.

Every extractor's Arabic problems come from the same root cause: they try to
recover logical order from a *character stream* that was written for
rendering, using heuristics about what the producer probably meant. PyMuPDF
applies its own bidi pass; pdfium applies none; both are guessing, and both
break on embedded LTR runs and on ligature glyphs.

The glyph positions, however, are not a guess. Every character in a PDF
carries an exact placement. If a glyph sits further right than another on the
same baseline, it comes earlier in Arabic. Full stop. So we throw the string
away and rebuild it from coordinates:

    1. group characters into baselines
    2. sort each baseline by x DESCENDING  -> logical Arabic order, free
    3. find maximal runs of LTR characters (Latin, digits, LTR punctuation)
       and re-sort those ascending -> correct L1/L2 bidi resolution
    4. insert word breaks where the inter-glyph gap exceeds a learned
       fraction of the font size

Cost is roughly 3 ms/page. It is deterministic, has no model, no dictionary,
no language assumption beyond 'this script is RTL', and it is correct on
documents where every general-purpose parser is wrong.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import fitz

RTL_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0700, 0x074F),  # Syriac
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB1D, 0xFDFF),  # Hebrew/Arabic presentation A
    (0xFE70, 0xFEFF),  # Arabic presentation B
)

LTR_STRONG = re.compile(r"[A-Za-z\u00C0-\u024F]")
DIGIT = re.compile(r"[0-9]")
MIRROR_PAIRS = dict(zip("()[]{}<>\u00ab\u00bb", ")(][}{><\u00bb\u00ab"))

NEUTRAL = re.compile(r"[\s.,:;!?()\[\]{}«»\"'\-–—/\\|&%#*+=<>@]")


def is_rtl_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in RTL_RANGES)


def _class(ch: str) -> str:
    if is_rtl_char(ch):
        return "R"
    if LTR_STRONG.match(ch) or DIGIT.match(ch):
        return "L"
    return "N"


@dataclass
class Glyph:
    c: str
    x0: float
    x1: float
    y: float
    size: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


def glyphs_from_page(page: "fitz.Page", clip: tuple | None = None,
                     raw: dict | None = None) -> list[Glyph]:
    out: list[Glyph] = []
    # `raw` lets the caller pass a rawdict already extracted elsewhere
    # (extract_page needs the same one) so the page isn't tokenized twice.
    # It's only reusable when no clip is requested -- a clipped call needs
    # its own, narrower extraction.
    if raw is None or clip is not None:
        raw = page.get_text("rawdict", clip=fitz.Rect(clip) if clip else None,
                            flags=fitz.TEXTFLAGS_RAWDICT | fitz.TEXT_PRESERVE_LIGATURES)
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                size = span["size"]
                for ch in span.get("chars", []):
                    b = ch["bbox"]
                    if not ch["c"]:
                        continue
                    # A real space character carries its own real advance
                    # width (confirmed: ~2.5pt wide in this font, same as a
                    # letter) -- dropping it and re-deriving spacing purely
                    # from the gap between the surrounding letters is worse
                    # than using the PDF's own answer, and can silently
                    # glue words together when that gap is only barely under
                    # line_to_text's threshold (confirmed real case: a
                    # 1.9pt measured gap against a 2.0pt threshold on a
                    # tightly-set font). Only a truly degenerate glyph (a
                    # zero-width artifact, not real whitespace) is dropped.
                    if not ch["c"].strip() and (b[2] - b[0]) < 0.1:
                        continue
                    out.append(Glyph(c=ch["c"], x0=b[0], x1=b[2],
                                     y=(b[1] + b[3]) / 2, size=size))
    return out


def group_baselines(glyphs: list[Glyph], tol_frac: float = 0.45,
                    col_gap_mult: float = 1.5) -> list[list[Glyph]]:
    """Group glyphs into baselines by y-proximity, then split any baseline
    that contains an abnormally wide horizontal gap.

    Grouping by y alone is blind to columns: in a row-aligned multi-column
    layout (extremely common -- two exercise lists, a teacher/pupil spread,
    any textbook with parallel columns), the left and right column's text
    routinely shares the same y, so a same-y grouping merges both columns'
    glyphs into one "baseline" and reconstructs them as a single run --
    weaving the two columns' characters together mid-word. A real column
    gutter is many times wider than a normal inter-word space, which is what
    this split catches without needing any column detection of its own.

    Deliberately sequential-interval, not a hashed bucket key (round(g.y /
    (g.size * tol_frac))). Bucket width scaling with each glyph's own font
    size lets two genuinely different rows hash-collide by pure arithmetic
    coincidence -- confirmed real case: an 8pt column-year header ("2017") at
    y=163 and a 10pt data row ("$ 40,653") at y=204.5, 41.5pt apart, both
    rounded to key=45, merging into one "baseline" and interleaving their
    digits character-by-character ("2017" + "40,653" -> "24001,7653") once
    sorted by x. Same class of bug as layout.group_by_line's fix; this is
    the sibling implementation that hadn't gotten it yet.
    """
    if not glyphs:
        return []
    ordered = sorted(glyphs, key=lambda g: g.y)
    lines: list[list[Glyph]] = []
    line_y: list[float] = []
    for g in ordered:
        if lines:
            last_y = line_y[-1]
            last_size = max(m.size for m in lines[-1])
            tol = max(min(last_size, g.size) * tol_frac, 1.0)
            if abs(g.y - last_y) <= tol:
                lines[-1].append(g)
                line_y[-1] = sum(m.y for m in lines[-1]) / len(lines[-1])
                continue
        lines.append([g])
        line_y.append(g.y)

    out: list[list[Glyph]] = []
    for ln in lines:
        ln = sorted(ln, key=lambda g: g.x0)
        cluster = [ln[0]]
        for prev, g in zip(ln, ln[1:]):
            if g.x0 - prev.x1 > max(prev.size, g.size) * col_gap_mult:
                out.append(cluster)
                cluster = []
            cluster.append(g)
        out.append(cluster)
    return out


def _resolve_runs(line: list[Glyph]) -> list[Glyph]:
    """Sort RTL, then flip embedded LTR runs back to ascending x.

    Neutrals adjacent to an LTR run on both sides join that run (the bidi
    algorithm's N1 rule, implemented positionally instead of textually).

    A line with no RTL characters at all skips this entirely and is just
    sorted ascending. That's not an optimisation -- it's required for
    correctness on non-RTL documents: descending-sort-then-resolve defaults
    an *undecidable* neutral (one with no strong neighbour on one side) to
    "R", on the assumption the paragraph's base direction is RTL. For a
    plain English line, the trailing punctuation is exactly that undecidable
    case -- it has no "next" neighbour in the descending scan -- so without
    this guard every English-only line would have its closing punctuation
    silently misordered to the front. This module has to work on any script
    mix, not just RTL documents, so the base direction is decided per line
    from what's actually on it, never assumed.
    """
    if not any(_class(g.c) == "R" for g in line):
        return sorted(line, key=lambda g: g.xc)

    line = sorted(line, key=lambda g: -g.xc)
    classes = [_class(g.c) for g in line]

    # resolve neutrals: take the class of the surrounding strong context
    resolved = list(classes)
    for i, c in enumerate(classes):
        if c != "N":
            continue
        prev = next((classes[j] for j in range(i - 1, -1, -1) if classes[j] != "N"), None)
        nxt = next((classes[j] for j in range(i + 1, len(classes)) if classes[j] != "N"), None)
        resolved[i] = "L" if prev == "L" and nxt == "L" else "R"

    out: list[Glyph] = []
    i = 0
    while i < len(line):
        if resolved[i] == "L":
            j = i
            while j < len(line) and resolved[j] == "L":
                j += 1
            out.extend(sorted(line[i:j], key=lambda g: g.xc))
            i = j
        else:
            g = line[i]
            # Unicode bidi rule L4: a mirrored glyph in an RTL run must be
            # replaced by its pair. The PDF stored the visual shape; logical
            # order needs the opposite one.
            if g.c in MIRROR_PAIRS:
                g = Glyph(MIRROR_PAIRS[g.c], g.x0, g.x1, g.y, g.size)
            out.append(g)
            i += 1
    return out


def line_to_text(line: list[Glyph], space_frac: float = 0.20) -> str:
    """Emit a logical-order string, inserting spaces from measured gaps."""
    if not line:
        return ""
    ordered = _resolve_runs(line)
    parts: list[str] = []
    prev: Glyph | None = None
    for g in ordered:
        if prev is not None:
            # gap in physical space between the two glyphs, whichever side
            gap = max(g.x0 - prev.x1, prev.x0 - g.x1)
            if gap > max(prev.size, g.size) * space_frac:
                parts.append(" ")
        parts.append(g.c)
        prev = g
    return "".join(parts)


def page_lines(page: "fitz.Page", clip: tuple | None = None,
               raw: dict | None = None) -> list[tuple[tuple, str]]:
    """Return [(bbox, logical_text)] for every baseline, top to bottom."""
    out = []
    for line in group_baselines(glyphs_from_page(page, clip, raw)):
        if not line:
            continue
        bbox = (min(g.x0 for g in line), min(g.y - g.size for g in line),
                max(g.x1 for g in line), max(g.y + g.size * 0.3 for g in line))
        out.append((bbox, line_to_text(line)))
    return out
