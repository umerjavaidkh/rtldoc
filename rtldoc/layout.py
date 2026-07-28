"""
Layer 2 -- layout: regions and reading order.

Two ideas do the heavy lifting here.

(1) VECTOR-FIRST REGION PROPOSAL. In a professionally typeset textbook the
    semantic containers are *drawn*: a tinted panel around a reading passage,
    a coloured chip carrying the exercise number, a rule separating the
    teacher column from the pupil column. Those shapes are exact vector data.
    A layout CNN re-detects them from rasterised pixels at ~85-95% IoU; we
    read them off at 100%. The CNN is kept only as a fallback for pages where
    the author drew nothing.

(2) RTL RECURSIVE XY-CUT. Reading order is not a learned property here, it is
    a geometric one, and the geometry is inverted relative to every reading-
    order model trained on arXiv: columns run right-to-left. We cut on
    whitespace valleys and order the children by x-descending on vertical
    cuts, y-ascending on horizontal ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .primitives import Fill, PagePrimitives, Rect, Span, containment

# A "cell-like" span: short, so a wrapped prose line never votes for a column.
_CELL_MAX_CHARS = 18
# A numeric cell: digits + the punctuation that decorates them ($ , . % ( ) - +).
# The aligned columns of a genuine borderless data table are overwhelmingly
# these; aligned *words* (Arabic MCQ options, an answer key, a two-column list)
# are not -- which is the signal that keeps this off text that merely lines up.
_NUM_CELL = re.compile(r"^[\s$€£¥%()+\-–.,0-9٠-٩۰-۹]+$")


def _is_numeric_cell(t: str) -> bool:
    t = t.strip()
    if not t or not _NUM_CELL.match(t):
        return False
    return any(ch.isdigit() for ch in t) or t in "$€£¥%"

RegionKind = Literal["panel", "chip", "figure", "flow", "rule", "table"]


@dataclass
class Region:
    bbox: Rect
    kind: RegionKind
    spans: list[Span] = field(default_factory=list)
    fill_color: tuple[float, float, float] | None = None
    activity: int | None = None
    column: int | None = None
    order: int | None = None
    role: str | None = None
    # populated only for kind == "table": one flat list of cell sub-regions,
    # each carrying its (row, col) position in the grid.
    cells: list["Region"] = field(default_factory=list)
    table_row: int | None = None
    table_col: int | None = None
    # populated only for kind == "figure": the xref needed to pull the image
    # back out of the PDF later (see pipeline.save_images).
    image_xref: int | None = None

    @property
    def x_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2


# ---------------------------------------------------------------------------
# region proposal
# ---------------------------------------------------------------------------

def _merge_nested(fills: list[Fill]) -> list[Fill]:
    """InDesign emits stacked identical rects; keep the outermost."""
    keep: list[Fill] = []
    for f in sorted(fills, key=lambda x: -x.area):
        if not any(containment(f.bbox, k.bbox) > 0.92 for k in keep):
            keep.append(f)
    return keep


def _merge_collinear(rules: list[Fill], horizontal: bool, tol: float = 2.0) -> list[tuple]:
    """Merge rule segments that share a position on the perpendicular axis
    (same y for horizontal rules, same x for vertical) into one logical
    border, unioning their extent along the rule's own axis.

    Publishers frequently draw one table border as several adjoining
    stroked segments rather than a single continuous line -- this book's
    row separators are each split into 3-4 pieces at the same y.
    """
    groups: dict[float, list[Fill]] = {}
    for f in rules:
        pos = f.bbox[1] if horizontal else f.bbox[0]
        key = round(pos / tol) * tol
        groups.setdefault(key, []).append(f)
    out = []
    for key, members in groups.items():
        if horizontal:
            out.append((key, min(m.bbox[0] for m in members), max(m.bbox[2] for m in members)))
        else:
            out.append((key, min(m.bbox[1] for m in members), max(m.bbox[3] for m in members)))
    return out


def detect_tables(prim: PagePrimitives, min_rows: int = 2, min_cols: int = 2,
                  coverage: float = 0.5) -> list[Region]:
    """Recover table grids from stroked border lines (Fill.is_rule).

    Table structure drawn as vector rules is exact, free, and immune to the
    usual OCR table-structure problem entirely -- it only needs reading, not
    inferring. We require at least one real internal row split *and* one
    internal column split before calling something a table, so an ordinary
    ruled list (row separators only, e.g. a bibliography) doesn't get
    misdetected as a grid.
    """
    horiz_rules = [f for f in prim.fills if f.is_rule and (f.bbox[2] - f.bbox[0]) > (f.bbox[3] - f.bbox[1])]
    vert_rules = [f for f in prim.fills if f.is_rule and (f.bbox[3] - f.bbox[1]) > (f.bbox[2] - f.bbox[0])]
    if not horiz_rules or not vert_rules:
        return []

    hlines = _merge_collinear(horiz_rules, horizontal=True)
    vlines = _merge_collinear(vert_rules, horizontal=False)

    span_x = max(x1 for _, _, x1 in hlines) - min(x0 for _, x0, _ in hlines)
    span_y = max(y1 for _, y0, y1 in vlines) - min(y0 for _, y0, _ in vlines)
    if span_x <= 0 or span_y <= 0:
        return []

    row_lines = [(y, x0, x1) for y, x0, x1 in hlines if (x1 - x0) >= coverage * span_x]
    col_lines = [(x, y0, y1) for x, y0, y1 in vlines if (y1 - y0) >= coverage * span_y]

    # Consistency guard: a real table's rows all span roughly the same
    # left/right extent (they're borders of the SAME table), and its columns
    # all span roughly the same top/bottom extent. Two unrelated decorative
    # rules -- an "Example" sidebar bar and a different equation's fraction
    # underline, say -- can each individually pass the coverage check above
    # while sharing no real relationship; requiring them to actually line up
    # is what tells a genuine grid apart from that kind of coincidence.
    def _consistent(lines: list[tuple[float, float, float]], span: float, tol_frac: float = 0.15) -> bool:
        if len(lines) <= 1:
            return True
        los = [a for _, a, _ in lines]
        his = [b for _, _, b in lines]
        tol = tol_frac * span
        return (max(los) - min(los)) <= tol and (max(his) - min(his)) <= tol

    if not _consistent(row_lines, span_x) or not _consistent(col_lines, span_y):
        return []

    row_ys = {round(y) for y, _, _ in row_lines}
    col_xs = {round(x) for x, _, _ in col_lines}
    # a table's outer frame is sometimes implied only by the perpendicular
    # rules (no drawn top/bottom border, as on this book's tables) -- fold
    # those extremes in as the missing boundary.
    row_ys |= {round(min(y0 for _, y0, _ in vlines)), round(max(y1 for _, _, y1 in vlines))}
    col_xs |= {round(min(x0 for _, x0, _ in hlines)), round(max(x1 for _, _, x1 in hlines))}
    row_ys, col_xs = sorted(row_ys), sorted(col_xs)

    if len(row_ys) - 1 < min_rows or len(col_xs) - 1 < min_cols:
        return []

    x0, x1 = max(0.0, col_xs[0]), col_xs[-1]
    y0, y1 = row_ys[0], row_ys[-1]
    table = Region(bbox=(x0, y0, x1, y1), kind="table")
    for ri in range(len(row_ys) - 1):
        for ci in range(len(col_xs) - 1):
            cx0 = max(0.0, col_xs[ci])
            cell = Region(bbox=(cx0, row_ys[ri], col_xs[ci + 1], row_ys[ri + 1]),
                         kind="flow", table_row=ri, table_col=ci)
            table.cells.append(cell)
    return [table]


def group_by_line(spans: list[Span], tol_frac: float = 0.5) -> list[list[Span]]:
    """Group spans into visual lines by baseline proximity, top-to-bottom.

    Deliberately sequential-interval, not a hashed bucket key. A bucket key
    of the form round(y / (own_size * factor)) makes the bucket WIDTH scale
    with each span's own font size -- which means a large-font line and a
    small-font line far apart on the page can round to the exact same
    integer key by pure arithmetic coincidence (a 17pt heading at y=220 and
    a 9.5pt body line at y=126 hashing to the same bucket is a real case
    this caught, not a hypothetical one). That silently merges two unrelated
    lines into one before any paragraph-level logic even runs, and no
    downstream size/gap check can undo it because the merge already
    happened. Proximity to the line's own running position, gated by a
    tolerance no wider than the smaller of the two font sizes involved,
    can't alias that way.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.bbox[1] + s.bbox[3]) / 2)
    lines: list[list[Span]] = []
    line_yc: list[float] = []
    for s in ordered:
        yc = (s.bbox[1] + s.bbox[3]) / 2
        if lines:
            last_yc = line_yc[-1]
            last_size = max(m.size for m in lines[-1])
            tol = max(min(last_size, s.size) * tol_frac, 1.0)
            if abs(yc - last_yc) <= tol:
                lines[-1].append(s)
                line_yc[-1] = sum((m.bbox[1] + m.bbox[3]) / 2 for m in lines[-1]) / len(lines[-1])
                continue
        lines.append([s])
        line_yc.append(yc)
    return lines


def detect_borderless_tables(prim: PagePrimitives, min_rows: int = 3, min_cols: int = 3,
                             tol: float = 6.0, pad: float = 2.0) -> list[Region]:
    """Recover tables that have no drawn rules at all -- financial statements
    and data tables that separate columns with shading or whitespace only.

    The signal is column *alignment recurring across rows*: in a table, the
    same x-positions carry a cell row after row; in prose they don't. So we
    only ever call something a table where that alignment actually exists,
    which is what keeps this from shredding ordinary paragraphs the way a
    page-wide text-alignment table finder does.

    Numeric/short cells are right-aligned almost universally in real tables,
    so columns are found by clustering the *right edges* of short spans and
    keeping only those an alignment supports across >= min_rows rows. Long
    (prose) spans never vote, so a wrapped sentence can't invent a column.
    """
    lines = group_by_line(prim.spans)
    if len(lines) < min_rows:
        return []

    # 1. column votes: right edges of short, cell-like spans, tagged by row
    votes: list[tuple[float, int]] = []
    for li, ln in enumerate(lines):
        for s in ln:
            if len(s.text.strip()) <= _CELL_MAX_CHARS:
                votes.append((s.bbox[2], li))
    if len(votes) < min_rows * min_cols:
        return []
    votes.sort()

    # 2. greedy-cluster right edges into candidate columns
    clusters: list[dict] = []
    cur: dict | None = None
    for x, li in votes:
        if cur is not None and x - cur["last"] <= tol:
            cur["xs"].append(x)
            cur["lines"].add(li)
            cur["last"] = x
        else:
            cur = {"xs": [x], "lines": {li}, "last": x}
            clusters.append(cur)

    # 3. a real column is one an alignment supports across enough rows
    real = [c for c in clusters if len(c["lines"]) >= min_rows]
    if len(real) < min_cols:
        return []
    real_lines = [c["lines"] for c in real]

    # 4. strongly-tabular rows hit >= min_cols of those columns; split them
    #    into contiguous bands (a page can hold two stacked tables)
    def hits(li: int) -> int:
        return sum(1 for s in real_lines if li in s)

    tab = [li for li in range(len(lines)) if hits(li) >= min_cols]
    if len(tab) < min_rows:
        return []

    bands: list[list[int]] = []
    run = [tab[0]]
    for prev, cur_li in zip(tab, tab[1:]):
        # allow up to 2 non-tabular lines between (wrapped labels, blank rows)
        if cur_li - prev <= 3:
            run.append(cur_li)
        else:
            bands.append(run)
            run = [cur_li]
    bands.append(run)

    out: list[Region] = []
    for band in bands:
        if len(band) < min_rows:
            continue
        lo, hi = band[0], band[-1]
        row_lines = lines[lo:hi + 1]                # include wrapped-label rows
        band_spans = [s for ln in row_lines for s in ln]
        # which columns actually appear in this band
        band_cols = [c for c in real if c["lines"] & set(range(lo, hi + 1))]
        if len(band_cols) < min_cols:
            continue
        centers = sorted(sum(c["xs"]) / len(c["xs"]) for c in band_cols)

        # Guard: the aligned cells must be predominantly numeric. This is what
        # separates a real data table from Arabic MCQ options / an answer key /
        # a two-column list that merely happens to line up. Without it, aligned
        # *words* get shredded into a garbage grid.
        aligned = [s for ln in row_lines for s in ln
                   if len(s.text.strip()) <= _CELL_MAX_CHARS
                   and any(abs(s.bbox[2] - cx) <= tol for cx in centers)]
        if not aligned:
            continue
        numeric_frac = sum(_is_numeric_cell(s.text) for s in aligned) / len(aligned)
        if numeric_frac < 0.6:
            continue

        # Guard: the grid must actually be *filled*. A real data table puts a
        # value at most row/column intersections; scattered list numbers or
        # page-credit numbers that merely happen to align leave the grid mostly
        # empty. This rejects a numbered exercise or an image-credits page whose
        # numbers coincidentally line up in >= min_cols places.
        filled = sum(
            1
            for ln in row_lines
            for cx in centers
            if any(abs(s.bbox[2] - cx) <= tol and len(s.text.strip()) <= _CELL_MAX_CHARS for s in ln)
        )
        if filled / (len(row_lines) * len(centers)) < 0.5:
            continue

        # each column's left/right extent, from the spans that align to it
        extents = []
        for cx in centers:
            members = [sp for ln in row_lines for sp in ln if abs(sp.bbox[2] - cx) <= tol]
            if not members:
                continue
            extents.append((min(sp.bbox[0] for sp in members), max(sp.bbox[2] for sp in members)))
        if len(extents) < min_cols:
            continue

        table_x0 = min(s.bbox[0] for s in band_spans)
        # vertical column splits: label | col0 | col1 | ...
        splits = [table_x0, extents[0][0] - pad]
        for j in range(len(extents) - 1):
            splits.append((extents[j][1] + extents[j + 1][0]) / 2)
        splits.append(extents[-1][1] + pad)

        row_ys = [ (min(s.bbox[1] for s in ln) + max(s.bbox[3] for s in ln)) / 2 for ln in row_lines ]
        rbounds = [min(s.bbox[1] for s in band_spans) - pad]
        for a, b in zip(row_ys, row_ys[1:]):
            rbounds.append((a + b) / 2)
        rbounds.append(max(s.bbox[3] for s in band_spans) + pad)

        table = Region(bbox=(splits[0], rbounds[0], splits[-1], rbounds[-1]), kind="table")
        for ri in range(len(rbounds) - 1):
            for ci in range(len(splits) - 1):
                table.cells.append(Region(
                    bbox=(splits[ci], rbounds[ri], splits[ci + 1], rbounds[ri + 1]),
                    kind="flow", table_row=ri, table_col=ci))
        out.append(table)
    return out


def propose_regions(prim: PagePrimitives, min_panel_area: float = 2000.0) -> list[Region]:
    regions: list[Region] = []

    tables = detect_tables(prim)
    # borderless detection only where a drawn-rule table doesn't already cover
    # the area, so the two never fight over the same grid
    for bt in detect_borderless_tables(prim):
        if not any(containment(bt.bbox, t.bbox) > 0.3 or containment(t.bbox, bt.bbox) > 0.3
                   for t in tables):
            tables.append(bt)
    regions.extend(tables)

    panels = _merge_nested([f for f in prim.fills if f.is_panel and f.area >= min_panel_area])
    for f in panels:
        regions.append(Region(bbox=f.bbox, kind="panel", fill_color=f.color))

    for f in prim.fills:
        if f.is_chip:
            regions.append(Region(bbox=f.bbox, kind="chip", fill_color=f.color))

    for img in prim.images:
        regions.append(Region(bbox=img.bbox, kind="figure", image_xref=img.xref))

    return regions


def assign_spans(prim: PagePrimitives, regions: list[Region], thresh: float = 0.6) -> list[Region]:
    """Drop every span into its tightest containing region; leftovers become
    free-flow regions clustered by line proximity, column by column."""
    tables = [r for r in regions if r.kind == "table"]
    chips = [r for r in regions if r.kind == "chip"]
    containers = [r for r in regions if r.kind in ("panel", "figure")]
    orphans: list[Span] = []

    for s in prim.spans:
        placed = False
        for t in tables:
            if containment(s.bbox, t.bbox) > thresh:
                best_cell, best_score = None, thresh
                for cell in t.cells:
                    c = containment(s.bbox, cell.bbox)
                    if c > best_score:
                        best_cell, best_score = cell, c
                if best_cell is not None:
                    best_cell.spans.append(s)
                    t.spans.append(s)
                placed = True
                break
        if placed:
            continue
        # chips win: they are tiny and unambiguous
        for r in chips:
            if containment(s.bbox, r.bbox) > thresh:
                r.spans.append(s)
                placed = True
                break
        if placed:
            continue
        best, best_score = None, thresh
        for r in containers:
            c = containment(s.bbox, r.bbox)
            if c > best_score:
                best, best_score = r, c
        if best is not None and best.kind == "panel":
            best.spans.append(s)
        else:
            orphans.append(s)

    # Cluster leftover text into paragraphs *within* each column first. Doing
    # this page-wide (the old behaviour) let a same-height line from one
    # column merge with a line from the other whenever their x-ranges
    # happened to overlap -- exactly what scrambled the two-column teacher /
    # pupil pages, since paragraph clustering ran before columns existed.
    boundaries = _column_boundaries([s.bbox for s in prim.spans], prim.width, prim.height)
    by_col: dict[int, list[Span]] = {}
    for s in orphans:
        col = _column_of((s.bbox[0] + s.bbox[2]) / 2, boundaries)
        by_col.setdefault(col, []).append(s)
    for col, spans in by_col.items():
        for r in _cluster_flow(spans):
            r.column = col
            regions.append(r)

    return [r for r in regions if r.spans or r.kind in ("figure", "table")]


def _cluster_flow(spans: list[Span], gap_mult: float = 1.6, size_ratio: float = 1.3) -> list[Region]:
    """Greedy line-then-paragraph clustering for text outside any drawn box."""
    if not spans:
        return []
    ordered = group_by_line(spans)
    ordered.sort(key=lambda ls: min(s.bbox[1] for s in ls))
    heights = [np.median([s.bbox[3] - s.bbox[1] for s in ls]) for ls in ordered]
    lead = float(np.median(heights)) if heights else 10.0
    sizes = [float(np.median([s.size for s in ls])) for ls in ordered]

    groups: list[list[Span]] = []
    prev_bottom = None
    prev_xrange = None
    prev_size = None
    for ls, size in zip(ordered, sizes):
        top = min(s.bbox[1] for s in ls)
        x0, x1 = min(s.bbox[0] for s in ls), max(s.bbox[2] for s in ls)
        overlaps = prev_xrange is not None and not (x1 < prev_xrange[0] or x0 > prev_xrange[1])
        # A font-size jump is a semantic break in its own right, independent of
        # whitespace: a heading immediately following body text with only
        # ordinary paragraph spacing above it would otherwise merge into one
        # blob purely because the gap check passed, and the whole thing then
        # gets mis-typed by whatever the biggest span in it happens to be.
        same_size = prev_size is None or max(size, prev_size) / min(size, prev_size) <= size_ratio
        if groups and prev_bottom is not None and (top - prev_bottom) < lead * gap_mult and overlaps and same_size:
            groups[-1].extend(ls)
        else:
            groups.append(list(ls))
        prev_bottom = max(s.bbox[3] for s in ls)
        prev_xrange = (x0, x1)
        prev_size = size

    out = []
    for g in groups:
        bbox = (min(s.bbox[0] for s in g), min(s.bbox[1] for s in g),
                max(s.bbox[2] for s in g), max(s.bbox[3] for s in g))
        out.append(Region(bbox=bbox, kind="flow", spans=g))
    return out


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------

def _column_boundaries(bboxes: list[Rect], page_width: float, page_height: float,
                       min_gap: float = 10.0, max_width_frac: float = 0.6,
                       empty_thresh: float = 0.04) -> list[float]:
    """2D-aware whitespace-gutter finder.

    min_gap=10 (not the wider value this used to have): a real column gutter
    measured directly on a physics textbook came out at 18pt, and 2pt x-axis
    binning quantizes that down further -- a width floor much above 10
    starts rejecting genuine, if narrow, gutters. Width is the weaker of the
    two guards anyway; `empty_thresh` (persistence across the page's full
    content height) is what actually tells a real gutter apart from an
    indent or list marker, since those are never empty for anywhere near the
    full column height the way a genuine gutter is.

    A 1D x-projection (ink present/absent per x, collapsing all y) is fooled
    two different ways: a single full-width element (a header, a footer, a
    page-wide rule) fills in a real column gutter everywhere it crosses, and
    a recurring narrow indent (an MCQ option column, a list marker) can look
    exactly like a gutter even though no two columns of running text sit
    either side of it -- both defeat a simple "is there any ink here"
    profile. Both are fixed by requiring a candidate gap to stay empty
    across almost the *entire* vertical extent of the page's content, not
    merely somewhere in it: that persistence is what a real inter-column
    gutter has and a local indent does not.

    Bboxes wider than `max_width_frac` of the page are additionally excluded
    from the emptiness measurement (though still assigned a column
    afterwards) -- otherwise one full-width line still poisons every column
    it happens to cross, persistence check or not.
    """
    narrow = [b for b in bboxes if (b[2] - b[0]) <= page_width * max_width_frac]
    use = narrow if narrow else bboxes
    if not use:
        return [0.0, page_width]

    xres, yres = 2.0, 4.0
    xbins, ybins = int(page_width / xres) + 1, int(page_height / yres) + 1
    grid = np.zeros((ybins, xbins), dtype=bool)
    for x0, y0, x1, y1 in use:
        xa, xb = max(int(x0 / xres), 0), min(int(x1 / xres) + 1, xbins)
        ya, yb = max(int(y0 / yres), 0), min(int(y1 / yres) + 1, ybins)
        if xb > xa and yb > ya:
            grid[ya:yb, xa:xb] = True

    inked_rows = np.nonzero(grid.any(axis=1))[0]
    inked_cols = np.nonzero(grid.any(axis=0))[0]
    if inked_rows.size == 0 or inked_cols.size == 0:
        return [0.0, page_width]

    content = grid[inked_rows[0]:inked_rows[-1] + 1, :]
    fill_frac = content.mean(axis=0)
    left, right = inked_cols[0] * xres, inked_cols[-1] * xres

    gaps, run = [], None
    for i in range(inked_cols[0], inked_cols[-1] + 1):
        if fill_frac[i] <= empty_thresh:
            run = i if run is None else run
        elif run is not None:
            if (i - run) * xres >= min_gap:
                gaps.append((run * xres, i * xres))
            run = None
    if run is not None and (inked_cols[-1] + 1 - run) * xres >= min_gap:
        gaps.append((run * xres, (inked_cols[-1] + 1) * xres))

    return [left] + [((a + b) / 2) for a, b in gaps] + [right + xres]


def _column_of(x_center: float, boundaries: list[float], rtl: bool = True) -> int:
    ncols = len(boundaries) - 1
    for i in range(ncols):
        if boundaries[i] <= x_center < boundaries[i + 1]:
            # column 0 is always "read first": rightmost for RTL, leftmost for LTR
            return ncols - 1 - i if rtl else i
    return 0


def detect_columns(regions: list[Region], page_width: float, page_height: float,
                   min_gap: float = 10.0, rtl: bool = True) -> int:
    """Whitespace-projection column finder. Returns number of columns and
    tags each region with its column index (0 = read first)."""
    if not regions:
        return 0
    boundaries = _column_boundaries([r.bbox for r in regions], page_width, page_height, min_gap)
    ncols = len(boundaries) - 1
    for r in regions:
        r.column = _column_of(r.x_center, boundaries, rtl)
    return ncols


def rtl_xy_cut(regions: list[Region], min_gap: float = 14.0, depth: int = 0, rtl: bool = True) -> list[Region]:
    """Recursive XY-cut, horizontal order set by `rtl` -- right-to-left for
    Arabic/Hebrew, left-to-right for everything else. Direction is a property
    of the page's own text, decided once by the caller (parse_page checks
    the page's script), never assumed: hardcoding "right first" reads an
    English two-column page's columns in the wrong order, which is exactly
    as wrong as reading an Arabic page left-to-right."""
    if len(regions) <= 1 or depth > 12:
        return regions

    def _cut(axis: int) -> list[list[Region]] | None:
        lo, hi = (0, 2) if axis == 0 else (1, 3)
        intervals = sorted(((r.bbox[lo], r.bbox[hi], r) for r in regions), key=lambda t: t[0])
        groups, cur_end, bucket = [], None, []
        for a, b, r in intervals:
            if cur_end is not None and a - cur_end >= min_gap:
                groups.append(bucket)
                bucket = []
            bucket.append(r)
            cur_end = b if cur_end is None else max(cur_end, b)
        groups.append(bucket)
        return groups if len(groups) > 1 else None

    vgroups = _cut(0)
    if vgroups:
        if rtl:
            vgroups.sort(key=lambda g: -max(r.bbox[2] for r in g))   # RIGHT first
        else:
            vgroups.sort(key=lambda g: min(r.bbox[0] for r in g))    # LEFT first
        out = []
        for g in vgroups:
            out.extend(rtl_xy_cut(g, min_gap, depth + 1, rtl))
        return out

    hgroups = _cut(1)
    if hgroups:
        hgroups.sort(key=lambda g: min(r.bbox[1] for r in g))    # TOP first
        out = []
        for g in hgroups:
            out.extend(rtl_xy_cut(g, min_gap, depth + 1, rtl))
        return out

    # unsplittable: fall back to top-then-(right|left)
    key = (lambda r: (round(r.bbox[1] / 6), -r.bbox[2])) if rtl else (lambda r: (round(r.bbox[1] / 6), r.bbox[0]))
    return sorted(regions, key=key)


def order_regions(regions: list[Region], page_width: float, page_height: float, rtl: bool = True) -> list[Region]:
    detect_columns(regions, page_width, page_height, rtl=rtl)
    ordered = rtl_xy_cut(regions, rtl=rtl)
    for i, r in enumerate(ordered):
        r.order = i
    return ordered
