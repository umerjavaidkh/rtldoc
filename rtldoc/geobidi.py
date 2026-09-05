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
    # Writing direction of the line this glyph came from, as PyMuPDF reports
    # it: (1, 0) for ordinary horizontal text, (0, -1) / (0, 1) for text
    # rotated 90 degrees. Needed because every step below reasons in page
    # coordinates and silently assumes horizontal text.
    dir: tuple = (1.0, 0.0)

    @property
    def rotated(self) -> bool:
        return abs(self.dir[1]) > abs(self.dir[0])

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
        # Through primitives.rawdict, not page.get_text: a clipped call
        # needs its own extraction, but it needs the SAME repairs -- going
        # direct silently re-imported every defect they fix (confirmed:
        # the '1'-for-space corruption reappeared inside clipped regions).
        from .primitives import rawdict as _rawdict
        raw = _rawdict(page, clip=clip)
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            ldir = tuple(line.get("dir", (1.0, 0.0)))
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
                                     y=(b[1] + b[3]) / 2, size=size, dir=ldir))
    return out


STACK_TOL = 0.1          # an advance below this does not move the pen
STACK_MIN = 4            # this many glyphs on one spot is a collapsed run


def _drop_stacked_glyphs(glyphs: list["Glyph"]) -> list["Glyph"]:
    """Remove runs of glyphs emitted on top of each other with no advance.

    Chrome's print-to-PDF, and the Wikipedia print stylesheet through it, emits
    a hidden copy of a line's letters stacked on a single point -- every glyph
    at the same x, none advancing. They draw nothing, but they sort into the
    line by position and weld a run of glued letters onto the front of it:

        علملوثةلعرقيحدُيطلق اسم علم الوراثة العرقي  ->  ُيطلق اسم علم الوراثة العرقي

    Between 42% and 55% of lines carry one in the worst files of the Arabic
    Wikipedia stratum (biology 170/308, iraq 172/405, agriculture 135/291).

    The test is STACKING, not zero width. Testing width alone destroys real
    text: many fonts render the lam-alef ligature as one glyph whose alef
    component carries no advance, and dropping those turned الأول into الول
    and, on arwiki_geography, الُجْغَرافّية into الَُْافّة. Those components sit
    at their own x; a collapsed run puts four or more on the same point, which
    no real typesetting does.
    """
    if not glyphs:
        return glyphs
    spots: dict[tuple, int] = {}
    for g in glyphs:
        if (g.x1 - g.x0) < STACK_TOL:
            key = (round(g.y, 1), round(g.x0, 1))
            spots[key] = spots.get(key, 0) + 1
    if not spots:
        return glyphs
    out = [g for g in glyphs
           if (g.x1 - g.x0) >= STACK_TOL
           or spots.get((round(g.y, 1), round(g.x0, 1)), 0) < STACK_MIN]
    return out or glyphs

def _drop_shadow_glyphs(glyphs: list["Glyph"]) -> list["Glyph"]:
    """Remove a duplicated text layer drawn as a drop shadow / double strike.

    Some producers fake a bold or shadowed heading by drawing the same run
    twice, offset by a fraction of the font size. Both copies land inside
    group_baselines' y tolerance (0.45 x font size), so they merge into one
    baseline and, once sorted by x, interleave character by character:
    "من المحكي" comes out "ممننااللممححككيي". Confirmed real case: an Arabic
    teacher's guide whose section headings are all set this way -- offset
    measured at dx 0.23pt, dy 1.05pt on a 12.8pt font.

    A duplicate is the SAME character occupying essentially the SAME box:
    x-overlap above 60% of the narrower glyph, and a y difference under a
    quarter of the font size. That is far tighter than real adjacent text --
    a genuine doubled letter (Arabic "اللغة" has two real lams) sits at
    clearly separate x positions and is untouched.
    """
    if not glyphs:
        return []
    order = sorted(range(len(glyphs)), key=lambda i: (glyphs[i].x0, glyphs[i].y))
    drop = set()
    for k, i in enumerate(order):
        if i in drop:
            continue
        a = glyphs[i]
        wa = a.x1 - a.x0
        if wa <= 0 or not a.c.strip():
            continue
        for j in order[k + 1:]:
            b = glyphs[j]
            if b.x0 - a.x0 > wa:
                break
            if j in drop or b.c != a.c:
                continue
            wb = b.x1 - b.x0
            if wb <= 0:
                continue
            ov = min(a.x1, b.x1) - max(a.x0, b.x0)
            if ov > 0.6 * min(wa, wb) and abs(a.y - b.y) < max(a.size, b.size) * 0.25:
                drop.add(j)
    return [g for i, g in enumerate(glyphs) if i not in drop]


def _group_rotated(glyphs: list[Glyph], tol_frac: float = 0.45) -> list[list[Glyph]]:
    """Group 90-degree-rotated glyphs into their own lines.

    Same idea as the horizontal case with the axes swapped: a rotated line's
    glyphs share an x centre and advance along y. Order within the line
    follows the writing direction -- dir (0, -1) reads bottom-to-top on the
    page, (0, 1) top-to-bottom -- so the result comes out in reading order
    rather than page order.
    """
    out: list[list[Glyph]] = []
    for g in sorted(glyphs, key=lambda g: g.xc):
        if out:
            last = out[-1]
            tol = max(min(max(m.size for m in last), g.size) * tol_frac, 1.0)
            if abs(g.xc - sum(m.xc for m in last) / len(last)) <= tol:
                last.append(g)
                continue
        out.append([g])
    for col in out:
        # dir[1] < 0 means the text runs up the page, so later glyphs sit at
        # smaller y and reading order is descending y.
        col.sort(key=lambda g: g.y, reverse=col[0].dir[1] < 0)
    return out


def group_baselines(glyphs: list[Glyph], tol_frac: float = 0.45,
                    col_gap_mult: float = 1.3,
                    bands: list[tuple[float, float]] | None = None,
                    gutters: list[tuple[float, float, float]] | None = None) -> list[list[Glyph]]:
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

    `bands` supersedes the gap heuristic when the page's real column ranges
    are known. The gap rule guesses at a gutter from one line's spacing, so it
    is bounded by exactly the problem it is trying to solve: it must pick a
    multiple of the font size big enough not to fire on a wide word gap, and
    that lower bound (1.3x = 13pt on 10pt text) sits ABOVE the commonest real
    gutter there is -- LaTeX's 10pt columnsep. No value of col_gap_mult
    separates the two populations, because they overlap. Splitting at known
    band boundaries instead is not a guess at all: the split point is a gutter
    the page-level detector already proved empty over the full content height,
    so it cannot be confused by wide word spacing, and it works for any number
    of columns and any gutter width. The gap rule stays as the fallback for
    callers with no band information (a single table cell, a test).

    col_gap_mult=1.3, not the original 1.5: a real 2-column WHO report's
    actual gutter measured 14.9947pt on 10pt body text -- 0.0053pt under
    the old 15.0pt cutoff (1.5x), so the split silently failed by a
    rounding hair on this page's first couple of lines (later lines'
    slightly wider gaps happened to clear it, which is why only some
    lines merged). A normal inter-word gap is ~0.2x font size (see
    line_to_text's own space_frac) -- 1.3x still leaves a 6x+ margin
    above real word-spacing, comfortable room to lower it without risking
    an ordinary wide gap (after punctuation, justified text) being
    mistaken for a column gutter.
    """
    if not glyphs:
        return []
    glyphs = _drop_stacked_glyphs(_drop_shadow_glyphs(glyphs))

    # Rotated (90-degree) text has to be grouped on its OWN axis. Every step
    # here reasons in page coordinates and assumes horizontal text: a line is
    # glyphs sharing a y, ordered by x. For text turned on its side that is
    # exactly wrong -- its glyphs share an x and advance down the y axis, so
    # y-grouping makes every single glyph its own one-character "line", and
    # those then interleave with whatever real horizontal lines occupy the
    # same y band. Confirmed real case: the vertical "arXiv:1810.04805v2
    # [cs.CL] 24 May 2019" stamp down the left margin of an arXiv paper --
    # present on essentially every arXiv PDF -- shredded into single
    # characters that wove themselves through the body text ("...ELMo
    # (Peters / 2 / y / a / M / 4 / 2 / ]"). Grouping rotated glyphs by
    # their shared x instead, and ordering them along y, reconstructs the
    # stamp as the one line it actually is and keeps it out of the body.
    rotated = [g for g in glyphs if g.rotated]
    if rotated:
        horizontal = [g for g in glyphs if not g.rotated]
        out = group_baselines(horizontal, tol_frac, col_gap_mult, bands, gutters) if horizontal else []
        for col in _group_rotated(rotated, tol_frac):
            out.append(col)
        return out

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

    multi = bool(gutters) or (bands is not None and len(bands) > 1)

    def _band_index(g: Glyph) -> int:
        cx = (g.x0 + g.x1) / 2
        if gutters:
            # count the gutters active at this glyph's own y that lie left of
            # it -- a gutter outside its rows does not separate anything here
            return sum(1 for gx, gy0, gy1 in gutters
                       if gx <= cx and gy0 <= g.y <= gy1)
        for i, (a, b) in enumerate(bands):
            if a <= cx < b:
                return i
        return min(range(len(bands)),
                   key=lambda i: abs(cx - (bands[i][0] + bands[i][1]) / 2))

    out: list[list[Glyph]] = []
    for ln in lines:
        ln = sorted(ln, key=lambda g: g.x0)
        cluster = [ln[0]]
        for prev, g in zip(ln, ln[1:]):
            if multi:
                split = _band_index(g) != _band_index(prev)
            else:
                split = g.x0 - prev.x1 > max(prev.size, g.size) * col_gap_mult
            if split:
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
    # A rotated line was already put in reading order along its own axis by
    # _group_rotated; every sort here is by x, which for a line running down
    # the page is a near-constant and would scramble it back into an
    # arbitrary order.
    if line and line[0].rotated:
        return line
    if not any(_class(g.c) == "R" for g in line):
        return sorted(line, key=lambda g: g.xc)

    line = sorted(line, key=lambda g: -g.xc)
    classes = [_class(g.c) for g in line]

    # resolve neutrals: take the class of the surrounding strong context
    resolved = list(classes)
    for i, c in enumerate(classes):
        if c != "N":
            continue
        pj = next((j for j in range(i - 1, -1, -1) if classes[j] != "N"), None)
        nj = next((j for j in range(i + 1, len(classes)) if classes[j] != "N"), None)
        prev = classes[pj] if pj is not None else None
        nxt = classes[nj] if nj is not None else None
        joins = prev == "L" and nxt == "L"
        if joins and pj is not None and nj is not None:
            # N1 bridges a neutral between two LTR runs -- right when they
            # are one number ("10 of 12"), wrong when they are separate
            # elements that merely sit side by side. Type size says which:
            # an activity chip set at 17.5pt beside a 12.0pt duration is
            # not one run, and merging them flipped "11 (10 دقائق)" into
            # "10( 11 دقائق)". Same size still merges, so ordinary mixed
            # numbers are untouched.
            a, b = line[pj].size, line[nj].size
            if max(a, b) > 0 and abs(a - b) > max(a, b) * 0.15:
                joins = False
        resolved[i] = "L" if joins else "R"

    out: list[Glyph] = []
    mirrorable: list[int] = []
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
            if g.c in MIRROR_PAIRS:
                mirrorable.append(len(out))
            out.append(g)
            i += 1
    return _mirror_if_it_helps(out, mirrorable)


_OPENERS = "([{\u00ab"
_CLOSERS = {")": "(", "]": "[", "}": "{", "\u00bb": "\u00ab"}


def _bracket_score(chars) -> int:
    """How many bracket pairs nest correctly, reading in logical order."""
    stack, ok = [], 0
    for c in chars:
        if c in _OPENERS:
            stack.append(c)
        elif c in _CLOSERS and stack and stack[-1] == _CLOSERS[c]:
            stack.pop()
            ok += 1
    return ok


def _mirror_if_it_helps(out: list["Glyph"], mirrorable: list[int]) -> list["Glyph"]:
    """Apply Unicode bidi rule L4 only when the producer actually needs it.

    L4 says a mirrored character inside an RTL run must be swapped for its
    pair, because the PDF stored the VISUAL shape and logical order needs the
    opposite one. That is true of producers that mirror at render time -- but
    plenty store the LOGICAL character already, and mirroring those corrupts
    correct text. Applying L4 unconditionally turned a correctly stored
    "(طه، 2018)." into ")طه، 2018 (." -- both brackets inverted (confirmed
    real case, Arabic teacher's guide p8).

    Which convention a file uses is decidable from the text itself, with no
    producer sniffing: brackets have to nest. Score the line both ways and
    keep the reading where more pairs close correctly; on a tie, leave the
    stored characters alone, since inventing a swap is the riskier of the
    two. Deliberately per line -- a single document can mix producers via
    embedded or pasted content.
    """
    if not mirrorable:
        return out
    as_is = [g.c for g in out]
    flipped = list(as_is)
    for i in mirrorable:
        flipped[i] = MIRROR_PAIRS[flipped[i]]
    if _bracket_score(flipped) <= _bracket_score(as_is):
        return out
    swapped = list(out)
    for i in mirrorable:
        g = swapped[i]
        swapped[i] = Glyph(MIRROR_PAIRS[g.c], g.x0, g.x1, g.y, g.size, g.dir)
    return swapped


_ALEF_VARIANTS = frozenset("\u0627\u0623\u0625\u0622")   # ا أ إ آ
_LAM = "\u0644"


def _fix_lam_alef_order(ordered: list["Glyph"]) -> list["Glyph"]:
    """Repair a lam-alef ligature emitted in visual order.

    When a PDF draws the lam-alef ligature (لا) it emits ONE glyph carrying
    both letters, but its ToUnicode map still yields two codepoints. Some
    producers emit those two in the order they appear on the page -- alef
    first, then lam -- which is the reverse of logical order, so every word
    containing lam-alef is silently corrupted: الاصطناعي becomes االصطناعي,
    الأول becomes األ...

    arabic.normalize already guards the case where the ligature survives as
    a single presentation-form codepoint (U+FEFB), by reordering before
    deshaping. It cannot help here: these arrive already split into two
    ordinary base letters, with no presentation form anywhere on the page.

    The signal is geometric, not linguistic, which is what makes it safe.
    The ligature's alef is emitted with ZERO advance width -- it is not
    drawn, the lam glyph already contains it -- and that lam is drawn at
    roughly double a normal lam's width because it carries both letters.
    Measured over a 143-page Arabic teacher's guide:

        zero-width alef followed by lam : 4218   <- the corruption
        zero-width alef followed by other:   6
        normal-width alef followed by lam: 16897 <- ordinary "ال" article,
                                                    left untouched
        lam width after zero-width alef  : 6.08 (median)
        lam width after normal alef      : 3.05 (median)

    So the ordinary definite article -- which is a genuine alef-then-lam and
    must NOT be touched -- is separated from the corruption by a property of
    the glyphs themselves, not by guessing at the word. Purely textual rules
    cannot do this: قال is a legitimate alef-before-lam too.
    """
    out = list(ordered)
    i = 0
    while i < len(out) - 1:
        a, b = out[i], out[i + 1]
        if (a.c in _ALEF_VARIANTS and (a.x1 - a.x0) < 0.5
                and b.c == _LAM and (b.x1 - b.x0) > max(a.size, b.size) * 0.4):
            # Reposition the alef to the LEFT edge of the lam as well as
            # reordering it. The ligature's alef is stored at the lam's RIGHT
            # edge (it is a zero-width mark on a glyph that is drawn as one
            # piece), so simply swapping the two leaves it sitting a whole
            # lam-width away from the letter that follows -- and line_to_text
            # measures word gaps from exactly that distance, so it inserted a
            # space mid-word: "الأنواع" came out "الأ نواع". In the ligature
            # the alef IS the left half, so the left edge is also its true
            # position.
            out[i] = b
            out[i + 1] = Glyph(a.c, b.x0, b.x0, a.y, a.size, a.dir)
            i += 2
            continue
        i += 1
    return out


def line_to_text(line: list[Glyph], space_frac: float = 0.20) -> str:
    """Emit a logical-order string, inserting spaces from measured gaps."""
    if not line:
        return ""
    ordered = _fix_lam_alef_order(_resolve_runs(line))
    parts: list[str] = []
    prev: Glyph | None = None
    prev_ink: Glyph | None = None      # last glyph that actually draws
    for gi, g in enumerate(ordered):
        # A space GLYPH that leaves no visual gap between the inked glyphs
        # on either side is padding, not a word break: a combining mark is
        # zero-width and drawn over its base letter, so the base letter's
        # advance surfaces as a space inside one word ("القصّ ة"). Same ink
        # test the span path applies to the same defect.
        if g.c.isspace() and prev_ink is not None:
            nxt = next((h for h in ordered[gi + 1:] if (h.x1 - h.x0) > 0.01), None)
            if nxt is not None and not nxt.rotated and not g.rotated:
                if max(nxt.x0 - prev_ink.x1, prev_ink.x0 - nxt.x1) < 0.5:
                    continue
        if prev is not None:
            # gap in physical space between the two glyphs, whichever side.
            # A rotated line advances along y, so its word gaps are there --
            # measuring x would report ~0 for every pair and run the whole
            # line together into one word.
            if g.rotated:
                gap = abs(g.y - prev.y) - max(prev.size, g.size) * 0.6
            else:
                # Measure from the last glyph that draws ink. A combining
                # mark is zero-width and sits ON its base letter, so
                # measuring from the mark reports the base letter's own
                # advance as a gap and splits one word in two ("القصّ ة"
                # for "القصّة"). Same rule the span path uses.
                ref = prev_ink if prev_ink is not None else prev
                gap = max(g.x0 - ref.x1, ref.x0 - g.x1)
            if gap > max(prev.size, g.size) * space_frac:
                parts.append(" ")
        parts.append(g.c)
        prev = g
        if (g.x1 - g.x0) > 0.01:
            prev_ink = g
    return "".join(parts)


def page_lines(page: "fitz.Page", clip: tuple | None = None,
               raw: dict | None = None,
               bands: list[tuple[float, float]] | None = None,
               gutters: list[tuple[float, float, float]] | None = None) -> list[tuple[tuple, str]]:
    """Return [(bbox, logical_text)] for every baseline, top to bottom.

    Pass `bands` (the page's column x-ranges) whenever they are known, so a
    baseline is never allowed to span two columns."""
    out = []
    for line in group_baselines(glyphs_from_page(page, clip, raw),
                                bands=bands, gutters=gutters):
        if not line:
            continue
        bbox = (min(g.x0 for g in line), min(g.y - g.size for g in line),
                max(g.x1 for g in line), max(g.y + g.size * 0.3 for g in line))
        out.append((bbox, line_to_text(line)))
    return out
