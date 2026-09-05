"""Detectors for the defects a reader sees when they open the output.

The problem this solves
-----------------------
The existing invariant checks (eval/invariants.py) measure character coverage
and excess. Those catch *lost* and *duplicated* text, and they are worth
keeping -- but they are blind to the failures that actually get noticed,
because a broken table contains exactly the right characters in the wrong
structure. Coverage 0.98 and a destroyed table are entirely compatible.

Every detector here is written to a single rule:

    A detector may only fire on something a person would point at
    and call broken.

That rule is what makes the suite verifiable. Each detector returns page
numbers and a short piece of evidence, so eval/review.py can put the rendered
page beside the emitted text and let you check the call by eye -- and
eval/calibrate.py can measure how often each detector is right.

Detectors are label-free: they compare the output against the source PDF's own
geometry, never against a hand-made answer key. That is what lets them run
over 43,000 pages instead of the 20 pages someone had time to transcribe.

Severity:
  HARD -- a violation is always a bug, no judgement needed.
  SOFT -- a strong smell; expect a real but nonzero false-positive rate,
          which eval/calibrate.py measures rather than assumes.
"""

from __future__ import annotations

import collections
import math
import re
import statistics
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# thresholds -- every one of these is a judgement call, so they live together
# where they can be seen, argued with, and re-tuned against calibration data.
# --------------------------------------------------------------------------
EMPTY_CELL_FRAC = 0.40      # a grid this sparse was invented, not found
HEADING_MAX_CHARS = 200     # longer than this and it is a paragraph
HEADING_FLOOD_FRAC = 0.35   # more headings than this and roles are noise
BODY_SIZE_RATIO = 1.25      # a span this much larger than body text is a title
MIN_LATTICE_LINES = (3, 2)  # (horizontal, vertical) rules to call it a table
LATTICE_IOU = 0.10          # overlap that counts as "a table was emitted here"
ORDER_BACKJUMP_PT = 12.0    # reading upward by more than this is an inversion
REPEAT_NGRAM = 8            # words; a repeated run this long is duplication
GUTTER_MIN_PT = 10.0        # an empty vertical strip this wide is a real gutter

_AR_LETTER = re.compile(r"[ء-يٱ-ۓۺ-ۿ]")
_LEADERS = re.compile(r"[.·•_\-]{6,}")
_SENT_END = re.compile(r"[.!?؟۔](\s|$)")
_DIGIT_IN_WORD = re.compile(r"(?<=[^\W\d_])\d(?=[^\W\d_])", re.UNICODE)
_DIGIT_IN_ARABIC = re.compile(r"(?<=[\u0621-\u064a])[0-9\u0660-\u0669](?=[\u0621-\u064a])")
_NUMERIC_CELL = re.compile(r"^[\s(]*[-+$£€]?[\d,.٫٬٠-٩]+\)?%?\s*$")


@dataclass
class Finding:
    """One defect, on one page, with the evidence that triggered it."""
    code: str
    severity: str          # "HARD" | "SOFT"
    page: int
    detail: str            # human-readable, short enough for a table cell
    evidence: str = ""     # the offending text, for eyeballing

    def as_row(self) -> dict:
        return {"code": self.code, "severity": self.severity, "page": self.page,
                "detail": self.detail, "evidence": self.evidence[:300]}


@dataclass
class PageProbe:
    """Everything the detectors need about one page, gathered once.

    Built from the source page (geometry, font sizes, drawn rules) and the
    parsed result (blocks, roles, order). Sharing this keeps the detectors
    cheap enough to run over the whole corpus."""
    number: int
    blocks: list                     # rtldoc Block
    born_digital: bool
    source_text: str
    words: list                      # (x0, y0, x1, y1, word, block, line, word_no)
    spans: list                      # (size, font, text, bbox)
    lattices: list                   # bboxes of ruled table regions in the source
    page_rect: tuple


# --------------------------------------------------------------------------
# source-side geometry
# --------------------------------------------------------------------------

def find_lattices(drawings: list, page_rect) -> list[tuple]:
    """Bounding boxes of ruled grids drawn on the page.

    A ruled table is the one table type whose existence is not a matter of
    opinion: someone drew the lines. If the source has a lattice and the
    output has no table there, a table was definitively lost -- no answer key
    needed. That makes this the strongest table check available label-free.

    Rules are collected from both stroked line segments and the thin filled
    rectangles that many generators use instead of strokes."""
    h_lines, v_lines = [], []
    for d in drawings:
        for item in d.get("items", []):
            kind = item[0]
            if kind == "l":
                (x0, y0), (x1, y1) = item[1], item[2]
                if abs(y1 - y0) <= 1.0 and abs(x1 - x0) > 8:
                    h_lines.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2))
                elif abs(x1 - x0) <= 1.0 and abs(y1 - y0) > 8:
                    v_lines.append((min(y0, y1), max(y0, y1), (x0 + x1) / 2))
            elif kind == "re":
                r = item[1]
                w, h = r.x1 - r.x0, r.y1 - r.y0
                # a rectangle thinner than 2pt in one axis is a drawn rule
                if h <= 2.0 and w > 8:
                    h_lines.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
                elif w <= 2.0 and h > 8:
                    v_lines.append((r.y0, r.y1, (r.x0 + r.x1) / 2))

    if len(h_lines) < MIN_LATTICE_LINES[0] or len(v_lines) < MIN_LATTICE_LINES[1]:
        return []

    # Cluster horizontal rules into vertically-contiguous bands; a table's
    # rules sit close together, while a page's header/footer rules do not.
    h_lines.sort(key=lambda t: t[2])
    bands, current = [], [h_lines[0]]
    for line in h_lines[1:]:
        if line[2] - current[-1][2] <= 120:
            current.append(line)
        else:
            bands.append(current)
            current = [line]
    bands.append(current)

    out = []
    for band in bands:
        if len(band) < MIN_LATTICE_LINES[0]:
            continue
        y0, y1 = band[0][2], band[-1][2]
        x0 = min(b[0] for b in band)
        x1 = max(b[1] for b in band)
        # verticals must actually cross this band, or the "rules" are just
        # stacked underlines (a form, a list of totals) and not a grid
        crossing = [v for v in v_lines
                    if v[0] < y1 - 2 and v[1] > y0 + 2 and x0 - 4 <= v[2] <= x1 + 4]
        if len(crossing) >= MIN_LATTICE_LINES[1] and (y1 - y0) > 12:
            out.append((x0, y0, x1, y1))
    return out


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


# --------------------------------------------------------------------------
# TABLES
# --------------------------------------------------------------------------

def _grid_of(block) -> list[list[str]] | None:
    """The block's cell grid, from the stored grid when present and from the
    rendered markdown pipes otherwise."""
    if getattr(block, "table_grid", None):
        return block.table_grid
    rows = []
    for line in block.text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows or None


def table_defects(probe: PageProbe) -> list[Finding]:
    out: list[Finding] = []
    tables = [b for b in probe.blocks if b.role == "table"]

    for b in tables:
        grid = _grid_of(b)
        if not grid:
            continue
        widths = {len(r) for r in grid}
        if len(widths) > 1:
            out.append(Finding("table_ragged", "HARD", probe.number,
                               f"rows have {sorted(widths)} columns",
                               b.text[:200]))
        ncols = max(widths) if widths else 0
        cells = [c for r in grid for c in r]
        if cells:
            empty = sum(1 for c in cells if not c.strip()) / len(cells)
            if empty > EMPTY_CELL_FRAC and len(cells) >= 6:
                out.append(Finding("table_mostly_empty", "SOFT", probe.number,
                                   f"{empty:.0%} of {len(cells)} cells are empty",
                                   b.text[:200]))
        if ncols <= 1 and len(grid) > 1:
            out.append(Finding("table_single_column", "SOFT", probe.number,
                               f"{len(grid)} rows but only 1 column",
                               b.text[:200]))
        # A cell holding several sentence-ends is several rows glued together:
        # the row splitter failed and the reader sees a wall of text in one box.
        for ri, row in enumerate(grid):
            for ci, cell in enumerate(row):
                if len(cell) > 160 and len(_SENT_END.findall(cell)) >= 3:
                    out.append(Finding("table_row_collapse", "SOFT", probe.number,
                                       f"cell r{ri}c{ci} holds {len(cell)} chars, "
                                       f"{len(_SENT_END.findall(cell))} sentences",
                                       cell[:200]))
                    break

    # A lattice drawn in the source with no table emitted over it: the clearest
    # possible "the table is gone" signal, and one of the two failures people
    # notice first.
    for lat in probe.lattices:
        if not any(_iou(lat, tuple(b.bbox)) > LATTICE_IOU for b in tables):
            covering = [b.role for b in probe.blocks if _iou(lat, tuple(b.bbox)) > LATTICE_IOU]
            out.append(Finding("table_missed", "SOFT", probe.number,
                               f"ruled grid at {tuple(round(v) for v in lat)} "
                               f"emitted as {sorted(set(covering)) or ['nothing']}",
                               ""))

    # The mirror image: a table claimed where the source drew no rules and the
    # text underneath shows no repeated column alignment.
    for b in tables:
        if any(_iou(tuple(b.bbox), lat) > LATTICE_IOU for lat in probe.lattices):
            continue
        inside = [w for w in probe.words
                  if w[0] >= b.bbox[0] - 2 and w[2] <= b.bbox[2] + 2
                  and w[1] >= b.bbox[1] - 2 and w[3] <= b.bbox[3] + 2]
        if len(inside) < 8:
            continue
        if _column_alignment(inside) < 0.35:
            out.append(Finding("table_false", "SOFT", probe.number,
                               "table with no drawn rules and no aligned columns",
                               b.text[:200]))
    return out


def _column_alignment(words: list) -> float:
    """Fraction of words whose left edge is shared by a word on another line.

    Real columns line up; prose does not. This is the evidence a borderless
    table actually exists, and its absence is the evidence one was invented."""
    by_line = collections.defaultdict(list)
    for w in words:
        by_line[round(w[1] / 3)].append(w)
    if len(by_line) < 3:
        return 1.0
    starts = collections.Counter()
    for line in by_line.values():
        for x in {round(w[0] / 4) for w in line}:
            starts[x] += 1
    shared = sum(c for x, c in starts.items() if c >= max(2, len(by_line) // 3))
    return shared / max(1, sum(starts.values()))


# --------------------------------------------------------------------------
# HEADINGS
# --------------------------------------------------------------------------

def heading_defects(probe: PageProbe) -> list[Finding]:
    out: list[Finding] = []
    headings = [b for b in probe.blocks if b.role and b.role.startswith("heading")]
    body = [b for b in probe.blocks if b.role in ("paragraph", "list_item")]

    for b in headings:
        text = b.text.strip()
        if len(text) > HEADING_MAX_CHARS:
            out.append(Finding("heading_too_long", "SOFT", probe.number,
                               f"{len(text)} chars marked as a heading", text[:200]))
        elif len(_SENT_END.findall(text)) >= 2:
            out.append(Finding("heading_is_prose", "SOFT", probe.number,
                               "heading contains multiple sentences", text[:200]))
        if text and len(text) <= 2 and not text.isdigit():
            out.append(Finding("heading_fragment", "SOFT", probe.number,
                               f"heading is {len(text)} character(s)", text))

    if probe.blocks:
        frac = len(headings) / len(probe.blocks)
        if frac > HEADING_FLOOD_FRAC and len(probe.blocks) >= 6:
            out.append(Finding("heading_flood", "SOFT", probe.number,
                               f"{frac:.0%} of {len(probe.blocks)} blocks are headings",
                               " | ".join(b.text.strip()[:40] for b in headings[:4])))

    # A page whose source has a clear size hierarchy but whose output has no
    # heading at all: the structure was there to be found and was not found.
    if not headings and body:
        sizes = [round(s[0], 1) for s in probe.spans if s[2].strip()]
        if len(sizes) >= 20:
            body_size = statistics.median(sizes)
            big = [s for s in sizes if s >= body_size * BODY_SIZE_RATIO]
            # a handful of large spans, not a whole large-print page
            if body_size > 0 and 2 <= len(big) <= len(sizes) * 0.25:
                out.append(Finding("heading_starved", "SOFT", probe.number,
                                   f"{len(big)} spans >= {BODY_SIZE_RATIO}x body "
                                   f"({body_size}pt) but no heading emitted", ""))

    # Levels that skip a rank read as a broken outline in any downstream use
    # (a table of contents, a chunker, a screen reader).
    levels = [int(b.role[-1]) for b in headings
              if b.role and b.role[-1].isdigit()]
    for a, b_ in zip(levels, levels[1:]):
        if b_ - a >= 2:
            out.append(Finding("heading_level_jump", "SOFT", probe.number,
                               f"h{a} followed by h{b_}", ""))
            break
    return out


# --------------------------------------------------------------------------
# READING ORDER
# --------------------------------------------------------------------------

def order_defects(probe: PageProbe) -> list[Finding]:
    """Does the emitted sequence agree with the geometry it claims?

    No answer key is needed: rtldoc gives every block a bbox and an order, and
    those two must be consistent. Within one column text runs downward, and a
    reader moves through columns one at a time. Violations of either are
    exactly what "the text is jumbled" looks like."""
    out: list[Finding] = []
    flow = [b for b in probe.blocks
            if b.role in ("paragraph", "list_item") or (b.role or "").startswith("heading")]
    if len(flow) < 3:
        return out
    flow = sorted(flow, key=lambda b: b.order)

    # within-column backward jumps
    backjumps = 0
    for a, b_ in zip(flow, flow[1:]):
        ax0, _, ax1, _ = a.bbox[0], a.bbox[1], a.bbox[2], a.bbox[3]
        bx0, bx1 = b_.bbox[0], b_.bbox[2]
        overlap = min(ax1, bx1) - max(ax0, bx0)
        same_column = overlap > 0.5 * min(ax1 - ax0, bx1 - bx0)
        if same_column and b_.bbox[1] < a.bbox[1] - ORDER_BACKJUMP_PT:
            backjumps += 1
    if backjumps:
        out.append(Finding("order_backjump", "SOFT", probe.number,
                           f"{backjumps} block(s) read above the block before them",
                           ""))

    # Column ping-pong. The gutter is FOUND, not assumed to be the page
    # midpoint: a real two-column page has a vertical strip with no words in
    # it. Without that evidence this cannot distinguish a two-column page from
    # a wide table or a figure pair, and would invent defects on both.
    gutter = _find_gutter(probe)
    if gutter is not None:
        side = [0 if (b.bbox[0] + b.bbox[2]) / 2 < gutter else 1 for b in flow]
        # blocks spanning the gutter (a full-width title, a wide figure) belong
        # to neither column and must not count as a crossing
        side = [s for s, b in zip(side, flow)
                if not (b.bbox[0] < gutter - 4 and b.bbox[2] > gutter + 4)]
        switches = sum(1 for a, b_ in zip(side, side[1:]) if a != b_)
        if switches >= 4:
            out.append(Finding("column_interleave", "SOFT", probe.number,
                               f"reading order crosses the gutter {switches} times "
                               f"({side.count(0)} left / {side.count(1)} right blocks)",
                               ""))
    return out


def _find_gutter(probe: PageProbe) -> float | None:
    """The x of a vertical strip that no word crosses, near the middle of the
    text area. Returns None when the page is not actually two-column."""
    words = [w for w in probe.words if w[4].strip()]
    if len(words) < 40:
        return None
    x0 = min(w[0] for w in words)
    x1 = max(w[2] for w in words)
    width = x1 - x0
    if width < 100:
        return None
    # occupancy histogram across the text area, 2pt buckets
    nb = max(8, int(width / 2))
    occupied = [False] * (nb + 1)
    for w in words:
        a = int((w[0] - x0) / width * nb)
        b = int((w[2] - x0) / width * nb)
        for i in range(max(0, a), min(nb, b) + 1):
            occupied[i] = True
    # widest empty run inside the middle half of the text area
    best = (0, None)
    i = int(nb * 0.30)
    end = int(nb * 0.70)
    while i < end:
        if occupied[i]:
            i += 1
            continue
        j = i
        while j < end and not occupied[j]:
            j += 1
        if j - i > best[0]:
            best = (j - i, (i + j) / 2)
        i = j + 1
    run, centre = best
    if centre is None or run * (width / nb) < GUTTER_MIN_PT:
        return None
    return x0 + centre / nb * width


# --------------------------------------------------------------------------
# TEXT INTEGRITY
# --------------------------------------------------------------------------

def text_defects(probe: PageProbe) -> list[Finding]:
    out: list[Finding] = []
    text = "\n".join(b.text for b in probe.blocks)
    if not text.strip():
        if len(probe.source_text.strip()) > 60:
            out.append(Finding("page_empty", "HARD", probe.number,
                               f"source has {len(probe.source_text.strip())} chars, "
                               f"output has none", probe.source_text[:200]))
        return out

    if "�" in text:
        out.append(Finding("replacement_char", "HARD", probe.number,
                           f"{text.count(chr(0xfffd))} U+FFFD in output", ""))

    for m in _LEADERS.finditer(text):
        out.append(Finding("leader_run", "SOFT", probe.number,
                           f"{len(m.group())}-char leader run survived",
                           text[max(0, m.start() - 40):m.end() + 20]))
        break

    # Deliberately NOT applied to Latin text. "5G", "Simu5G", "OMNeT++" and
    # "H2O" are ordinary words, and a detector that calls them broken teaches
    # you to ignore it. In Arabic a Latin digit inside a word is unambiguous:
    # no Arabic word contains one, so every hit is a decoding fault.
    for m in _DIGIT_IN_ARABIC.finditer(text):
        out.append(Finding("digit_in_word", "SOFT", probe.number,
                           "a digit decoded inside an Arabic word",
                           text[max(0, m.start() - 40):m.start() + 40]))
        break

    # A run of words repeated verbatim is duplicated text -- the failure where
    # a passage appears twice on one page.
    # A repeat only counts when the OUTPUT has more of it than the SOURCE did.
    # Documents legitimately repeat themselves -- a shared affiliation printed
    # under two authors, a running header, a repeated table stub -- and
    # flagging those makes the detector a liar. The defect is duplication that
    # rtldoc introduced, so the source is the baseline.
    # Tables are excluded on purpose. A table legitimately repeats itself --
    # "| P | R | F1 |" under three model names is correct output -- and the
    # markdown pipes themselves tokenize into repeating runs. Running this on
    # prose is where a repeated run actually means duplicated text.
    prose = "\n".join(b.text for b in probe.blocks if b.role != "table")
    words = re.findall(r"\S+", prose)
    if len(words) > REPEAT_NGRAM * 3:
        src_words = re.findall(r"\S+", probe.source_text)
        src_counts = collections.Counter(
            tuple(src_words[i:i + REPEAT_NGRAM])
            for i in range(max(0, len(src_words) - REPEAT_NGRAM)))
        out_counts = collections.Counter(
            tuple(words[i:i + REPEAT_NGRAM])
            for i in range(len(words) - REPEAT_NGRAM))
        for key, n in out_counts.items():
            if n > max(1, src_counts.get(key, 0)):
                out.append(Finding("repeated_run", "SOFT", probe.number,
                                   f"{REPEAT_NGRAM}-word run appears {n}x in output, "
                                   f"{src_counts.get(key, 0)}x in source",
                                   " ".join(key)[:200]))
                break
    return out


def arabic_defects(probe: PageProbe) -> list[Finding]:
    """Checks that only apply to Arabic text, kept separate so they never
    dilute the Latin numbers (and vice versa)."""
    out: list[Finding] = []
    text = "\n".join(b.text for b in probe.blocks)
    if not _AR_LETTER.search(text):
        return out
    leaks = [c for c in text if _is_presentation_form(c)]
    if leaks:
        out.append(Finding("presentation_forms", "HARD", probe.number,
                           f"{len(leaks)} presentation-form codepoint(s) in output",
                           "".join(leaks[:20])))
    # A space inside an Arabic word: the reader sees the word cut in half.
    splits = re.findall(r"[ء-ي]{1,2} [ء-ي]{1,2}(?=\s|$)", text)
    if len(splits) >= 3:
        out.append(Finding("arabic_word_split", "SOFT", probe.number,
                           f"{len(splits)} suspected false spaces inside words",
                           " | ".join(splits[:6])))
    return out


def _is_presentation_form(ch: str) -> bool:
    if not ("ﭐ" <= ch <= "﷿" or "ﹰ" <= ch <= "﻿"):
        return False
    return bool(unicodedata.decomposition(ch))


ALL_DETECTORS = (table_defects, heading_defects, order_defects,
                 text_defects, arabic_defects)

# Every code this module can emit, with a one-line description of the defect a
# reader would see. Used by the scorecard and the review bundle, and the place
# to look when a code appears in a report.
CODES = {
    "table_ragged":       "table rows have different column counts",
    "table_mostly_empty": "table grid is mostly empty cells",
    "table_single_column":"table collapsed to a single column",
    "table_row_collapse": "several table rows glued into one cell",
    "table_missed":       "source draws a ruled grid, no table was emitted",
    "table_false":        "table emitted where no columns exist",
    "heading_too_long":   "a paragraph was promoted to a heading",
    "heading_is_prose":   "heading is prose, not a title",
    "heading_fragment":   "heading is a stray character",
    "heading_flood":      "most blocks on the page are headings",
    "heading_starved":    "clear size hierarchy in source, no heading found",
    "heading_level_jump": "heading levels skip a rank",
    "order_backjump":     "text reads upward within a column",
    "column_interleave":  "reading order ping-pongs between columns",
    "page_empty":         "source has text, output has none",
    "replacement_char":   "U+FFFD in the output",
    "leader_run":         "dot-leader run left in the text",
    "digit_in_word":      "a digit decoded inside a word",
    "repeated_run":       "a run of words appears twice",
    "presentation_forms": "Arabic presentation forms survived deshaping",
    "arabic_word_split":  "false space inside an Arabic word",
}
