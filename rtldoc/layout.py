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

from .arabic import is_arabic
from .primitives import Fill, ImageRef, PagePrimitives, Rect, Span, containment

# A "cell-like" span: short, so a wrapped prose line never votes for a column.
_CELL_MAX_CHARS = 18
# A numeric cell: digits + the punctuation that decorates them ($ , . % ( ) - +).
# The aligned columns of a genuine borderless data table are overwhelmingly
# these; aligned *words* (Arabic MCQ options, an answer key, a two-column list)
# are not -- which is the signal that keeps this off text that merely lines up.
# An em dash on its own (--, U+2014) is included: it's the standard accounting
# convention for a zero/nil cell in a financial statement, not prose -- confirmed
# real case, a diluted-EPS table where every zero cell is a lone "--" character.
_NUM_CELL = re.compile(r"^[\s$€£¥%()+\-–—.,0-9٠-٩۰-۹]+$")


def _is_numeric_cell(t: str) -> bool:
    t = t.strip()
    if not t or not _NUM_CELL.match(t):
        return False
    return any(ch.isdigit() for ch in t) or t in "$€£¥%—–-"

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
    # back out of the PDF later (see pipeline.save_images). None for a
    # composite figure (see below) -- there's no single xref that represents
    # a merged cluster of overlapping image fragments.
    image_xref: int | None = None
    # kind == "figure" only: True when this region is a merged cluster of
    # multiple overlapping/nested raster fragments (see _cluster_images) --
    # save_images rasterizes the bbox directly from the page instead of
    # extracting any one xref, since no single fragment is "the" image.
    composite: bool = False

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


def _cluster_rules(fills: list[Fill], pad: float | None = None) -> list[list[Fill]]:
    """Group rule fills by spatial proximity (union-find on padded bboxes).

    A page can carry more than one table, plus assorted unrelated short
    rules elsewhere (a decorative underline, a divider in a page header).
    Computing one table's geometry from ALL rule fills on the page is
    fragile: a single stray mark far from the real table can badly distort
    a page-wide span calculation and silently break detection for a table
    that has nothing to do with it (confirmed case: a ~470pt page-wide span
    from two unrelated 1.6pt marks near the page header suppressed
    detection of a genuine table whose real dividers only spanned 125pt).
    Scoping every table's own geometry to its own spatially-local cluster
    of rules removes that cross-contamination and also lets a page contain
    more than one independently-detected table.

    `pad` defaults to the page's OWN typical row/column spacing (1.3x the
    median gap between distinct rule edge positions), not a fixed constant.
    A fixed 8pt pad is too small for a spaciously-set table and fragments
    it into one isolated cluster per row divider -- confirmed real case: a
    ruled table with a uniform 16.5pt row pitch had every horizontal rule
    line hash into its own tiny cluster, each too small to individually
    pass min_rows/min_cols, so all but the table's last couple of rows
    silently vanished from detection. Deriving pad from the page's actual
    spacing bridges one table's own rows while still leaving a much larger
    gap to a genuinely separate table untouched; bounded to [8, 30]pt so a
    sparse page with few rules can't blow the pad up arbitrarily.
    """
    if pad is None:
        edges = sorted({round(f.bbox[1], 1) for f in fills} | {round(f.bbox[3], 1) for f in fills})
        gaps = [b - a for a, b in zip(edges, edges[1:]) if b - a > 0.5]
        pad = min(max(8.0, float(np.median(gaps)) * 1.3), 30.0) if gaps else 8.0

    n = len(fills)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def close(a: Rect, b: Rect) -> bool:
        return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])

    for i in range(n):
        for j in range(i + 1, n):
            if close(fills[i].bbox, fills[j].bbox):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[Fill]] = {}
    for i, f in enumerate(fills):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def detect_tables(prim: PagePrimitives, min_rows: int = 2, min_cols: int = 2,
                  coverage: float = 0.5) -> list[Region]:
    """Recover table grids from stroked border lines (Fill.is_rule).

    Table structure drawn as vector rules is exact, free, and immune to the
    usual OCR table-structure problem entirely -- it only needs reading, not
    inferring. We require at least one real internal row split *and* one
    internal column split before calling something a table, so an ordinary
    ruled list (row separators only, e.g. a bibliography) doesn't get
    misdetected as a grid.

    Rules are clustered spatially first (see _cluster_rules) and each
    cluster is evaluated independently, so a page can yield more than one
    table and a stray rule elsewhere on the page can't corrupt one it has
    nothing to do with.
    """
    all_rules = [f for f in prim.fills if f.is_rule]
    if not all_rules:
        return []

    tables: list[Region] = []
    for cluster in _cluster_rules(all_rules):
        tables.extend(_detect_table_in_cluster(cluster, min_rows, min_cols, coverage))
    return tables


def _detect_table_in_cluster(cluster: list[Fill], min_rows: int, min_cols: int,
                             coverage: float) -> list[Region]:
    horiz_rules = [f for f in cluster if (f.bbox[2] - f.bbox[0]) > (f.bbox[3] - f.bbox[1])]
    vert_rules = [f for f in cluster if (f.bbox[3] - f.bbox[1]) > (f.bbox[2] - f.bbox[0])]
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

    # An INTERNAL row boundary (a header/first-row divider, say) can also be
    # marked with no horizontal rule at all -- only by where the column
    # dividers' own segments break. Trust a break only where a majority of
    # the real column dividers agree on nearly the same y; a single
    # divider's own rendering quirk can't invent a row boundary alone.
    seg_break_votes: dict[int, int] = {}
    for f in vert_rules:
        if not any(abs(f.bbox[0] - x) <= 2.0 for x in col_xs):
            continue
        for y in (round(f.bbox[1]), round(f.bbox[3])):
            seg_break_votes[y] = seg_break_votes.get(y, 0) + 1
    min_agree = max(2, (len(col_xs) + 1) // 2)
    row_ys |= {y for y, votes in seg_break_votes.items() if votes >= min_agree}

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


def _merge_wrapped_label_rows(row_lines: list[list[Span]], centers: list[float],
                              tol: float, indent_ratio: float = 1.8) -> list[list[list[Span]]]:
    """Merge a row label's wrapped continuation lines into its data row.

    A borderless table's row_lines are one physical text line each, but a
    long row label routinely wraps across 2-3 lines while its numbers sit on
    only one of them -- left unmerged, each wrapped line becomes its own
    spurious output row with the label disconnected from its data
    (confirmed real case: "Interest and other income" / "(expense), net"
    came out as two separate table rows instead of one).

    Two signals distinguish a genuine wrap from a new row starting (a
    section header like "Costs and expenses:" immediately followed by its
    first real sub-item, which must NOT merge):

    1. Indentation step. A wrapped continuation line sits at roughly the
       same left margin as the label's first line (a small hanging-indent
       bump); a genuine new sub-item is indented a full outline level
       deeper. Measured relative to the following line's own font size
       (not a fixed point value) so it holds across documents with
       different type sizes -- confirmed real case: ~11pt continuation
       bump vs ~24pt sub-item indent on the same page, a >2x difference.
    2. Trailing colon. A label ending in ":" is a complete, self-terminating
       header by typographic convention, never a fragment awaiting more
       text -- checked even when the indentation alone would look like a
       continuation (confirmed real case: a 3-line header ending in
       "common stockholders:" sits at the same indent as the unrelated data
       row right after it, which must NOT merge in).

    Merging stops the moment a line actually carries an aligned number: that
    line's data belongs to the row being built, and the row is complete.
    """
    def has_data(ln: list[Span]) -> bool:
        return any(any(abs(s.bbox[2] - cx) <= tol for cx in centers) for s in ln)

    def indent(ln: list[Span]) -> float:
        return min(s.bbox[0] for s in ln)

    def ends_with_colon(ln: list[Span]) -> bool:
        return max(ln, key=lambda s: s.bbox[2]).text.strip().endswith(":")

    groups: list[list[list[Span]]] = []
    i, n = 0, len(row_lines)
    while i < n:
        group = [row_lines[i]]
        base_indent = indent(row_lines[i])
        j = i
        while not has_data(row_lines[j]) and not ends_with_colon(row_lines[j]) and j + 1 < n:
            nxt = row_lines[j + 1]
            step = indent(nxt) - base_indent
            avg_size = sum(s.size for s in nxt) / len(nxt)
            if step > avg_size * indent_ratio:
                break
            group.append(nxt)
            j += 1
            if has_data(nxt):
                break
        groups.append(group)
        i = j + 1
    return groups


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

    def _has_section_break(prev: int, cur_li: int) -> bool:
        """A colon-terminated PROSE SENTENCE between two tabular rows is a
        new section's own intro (a new table's caption, not part of this
        one) and must hard-break a band even within the small gap otherwise
        tolerated for wrapped labels. A short colon-terminated LABEL
        ("Revenue:", "Costs and expenses:") is not this -- it's a
        legitimate divider row *within* the same table (confirmed against
        gold: it appears as its own row, blank-valued, inside one
        continuous table) and must not split anything. Word count is what
        tells them apart: a divider label is a few words; a real section
        intro is a full sentence (confirmed real case: "Share-based
        compensation expense included in costs and expenses:" at 9 words,
        introducing a wholly separate table two lines later, vs "Costs and
        expenses:" at 3 words, a divider inside the same table)."""
        for li in range(prev + 1, cur_li):
            ln = lines[li]
            if not ln:
                continue
            text = " ".join(s.text for s in sorted(ln, key=lambda s: s.bbox[0])).strip()
            if text.endswith(":") and len(text.split()) >= 6:
                return True
        return False

    bands: list[list[int]] = []
    run = [tab[0]]
    for prev, cur_li in zip(tab, tab[1:]):
        # allow up to 2 non-tabular lines between (wrapped labels, blank rows)
        if cur_li - prev <= 3 and not _has_section_break(prev, cur_li):
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
        band_range = set(range(lo, hi + 1))
        band_cols = [c for c in real if c["lines"] & band_range]
        # Drop columns whose support *within this band* is weak relative to
        # the band's strongest column. A wide financial table's real data
        # columns are hit by nearly every row (confirmed case: 22-24 of 24
        # rows); a row LABEL's own word can coincidentally right-align across
        # a handful of rows too (different line items happen to have same-
        # length wrapped words) and clear the bare min_rows floor without
        # being a real column at all -- confirmed case: two such noise
        # columns (support 4 and 6) sat alongside eight genuine columns
        # (support 22-24) on the same page, inflating a 9-column table to 12
        # and shifting every row's label into the wrong cell. Relative to
        # the band's OWN best column, not an absolute count, so this scales
        # correctly for a small table too.
        band_support = {id(c): len(c["lines"] & band_range) for c in band_cols}
        max_support = max(band_support.values(), default=0)
        band_cols = [c for c in band_cols if band_support[id(c)] >= max(min_rows, max_support * 0.4)]
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
        #
        # The denominator counts only *tabular* lines (hits >= min_cols), not
        # every line in row_lines -- row_lines deliberately also includes
        # wrapped-label filler lines (the band-forming step above allows up
        # to 3 non-tabular lines through so a 2-line row label doesn't split
        # the table), and those filler lines are never supposed to have any
        # aligned numbers. Counting them against the grid double-penalizes
        # exactly the rows the band logic already agreed to tolerate --
        # confirmed real case: a diluted-EPS table with many two-line row
        # labels came out at 0.45 "filled" and was rejected outright, though
        # every actual data row was completely filled (0.78 once filler
        # lines are excluded from the count).
        data_lines = [ln for li, ln in zip(range(lo, hi + 1), row_lines) if hits(li) >= min_cols]
        filled = sum(
            1
            for ln in data_lines
            for cx in centers
            if any(abs(s.bbox[2] - cx) <= tol and len(s.text.strip()) <= _CELL_MAX_CHARS for s in ln)
        )
        if not data_lines or filled / (len(data_lines) * len(centers)) < 0.5:
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

        row_groups = _merge_wrapped_label_rows(row_lines, centers, tol)
        row_ys = [(min(s.bbox[1] for ln in g for s in ln) + max(s.bbox[3] for ln in g for s in ln)) / 2
                 for g in row_groups]
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

    for cluster in _cluster_images(prim.images):
        if len(cluster) == 1:
            regions.append(Region(bbox=cluster[0].bbox, kind="figure", image_xref=cluster[0].xref))
        else:
            x0 = min(im.bbox[0] for im in cluster); y0 = min(im.bbox[1] for im in cluster)
            x1 = max(im.bbox[2] for im in cluster); y1 = max(im.bbox[3] for im in cluster)
            regions.append(Region(bbox=(x0, y0, x1, y1), kind="figure", composite=True))

    return regions


def _cluster_images(images: list[ImageRef]) -> list[list[ImageRef]]:
    """Group images whose boxes overlap or nest, transitively.

    A complex illustration is very often exported as many small overlapping
    raster layers -- a hand-drawn icon and a set of colored sticky-note
    shapes, say, each its own placed image, tiled and stacked to form one
    picture. Left as separate figure regions, each fragment independently
    grabs the same nearby caption text, duplicating it once per fragment
    (24 times, on the page that surfaced this). Merging overlapping images
    into one composite region first means one caption assignment, and
    save_images renders the merged region as a single rasterized image
    instead of trying to reassemble N separately-encoded layers.
    """
    def intersects(a: Rect, b: Rect) -> bool:
        return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

    # Union-find: a complex illustration can be tiled from a hundred-plus
    # tiny raster pieces (one real case hit 152 on a single page), and
    # restarting an O(n^2) pairwise scan from the top after every single
    # merge -- the previous approach -- degrades badly at that size. This
    # does the same O(n^2) intersection test but merges in near-constant
    # time per union, so a pathological page doesn't slow down parsing.
    n = len(images)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if intersects(images[i].bbox, images[j].bbox):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[ImageRef]] = {}
    for i, im in enumerate(images):
        groups.setdefault(find(i), []).append(im)
    return list(groups.values())


def _repeat_unit(words: list[str]) -> list[str] | None:
    """If `words` is some shorter sequence repeated >=2 times end to end,
    return that shortest repeating unit; otherwise None."""
    n = len(words)
    for p in range(1, n // 2 + 1):
        if n % p:
            continue
        unit = words[:p]
        if unit * (n // p) == words:
            return unit
    return None


def _split_repeated_span(s: Span, tables: list[Region]) -> list[Span]:
    """Split a span whose text is N adjacent table cells' identical value
    drawn as one run, e.g. "non-reserved non-reserved non-reserved" for
    three columns that happen to share a value.

    Some PDF generators emit consecutive same-styled table cells as a single
    text-showing run when their content happens to be identical, rather than
    one run per cell. The combined span's bbox then straddles the boundary
    between adjacent columns, so its containment with any single cell falls
    under assign_spans's threshold and the whole value is silently dropped
    from all of them -- confirmed real case: a wide reference table where a
    large fraction of rows have this pattern (2-way and 3-way both occur on
    the same page) lost entire columns' worth of "non-reserved" / "reserved"
    values. The signature is narrow and safe: the text must be some shorter
    word sequence repeated end to end with no leftover, which ordinary
    prose essentially never produces, and the span must actually sit inside
    a table. Splitting the bbox into N equal-width parts is safe here
    specifically because all N copies are, by construction, identical text
    in the same font/size -- they occupy equal width.
    """
    if not any(containment(s.bbox, t.bbox) > 0.6 for t in tables):
        return [s]
    words = s.text.split()
    unit = _repeat_unit(words)
    if unit is None:
        return [s]
    n_copies = len(words) // len(unit)
    x0, y0, x1, y1 = s.bbox
    width = (x1 - x0) / n_copies
    text = " ".join(unit)
    return [
        Span(text=text, bbox=(x0 + i * width, y0, x0 + (i + 1) * width, y1),
             font=s.font, size=s.size, color=s.color, flags=s.flags, dir=s.dir)
        for i in range(n_copies)
    ]


def _merge_marker_columns(by_col: dict[int, list[Span]], min_rows: int = 3, short_frac: float = 0.8) -> None:
    """Merge a narrow marker/number column's spans into its adjacent wide
    content column when their entries share the same row.

    A Table of Contents' "5.1" section numbers, or a category list's "1."
    markers, routinely land in their own detected column right beside the
    titles/content they introduce -- column detection is purely geometric
    and has no way to know these should stay row-paired rather than be
    read as two independent columns (every marker, then every title, in
    two separate passes). Confirmed real cases: a Table of Contents whose
    numbers came out entirely separated from their own titles, and a
    category list whose headers came out separated from their own items.

    Reassigning the marker column's spans into the content column lets the
    existing same-line grouping (group_by_line) recombine each marker with
    its row automatically, since the two already share almost exactly the
    same y-position by construction -- no new line-matching logic needed.
    A column only qualifies as a marker column when almost every one of
    its own lines is short (a real paragraph column will have long-wrapped
    lines mixed in), so an ordinary two-column page of running prose is
    never mistaken for this pattern.
    """
    cols = sorted(by_col)
    for i in range(len(cols) - 1):
        a, b = cols[i], cols[i + 1]
        for src, dst in ((a, b), (b, a)):
            src_lines = group_by_line(by_col[src])
            if len(src_lines) < min_rows:
                continue
            short = sum(1 for ln in src_lines
                       if len(" ".join(s.text for s in ln).strip()) <= _CELL_MAX_CHARS)
            if short / len(src_lines) < short_frac:
                continue
            dst_lines = group_by_line(by_col[dst])
            if len(dst_lines) < min_rows:
                continue
            by_col[dst] = by_col[dst] + by_col[src]
            by_col[src] = []
            break


def assign_spans(prim: PagePrimitives, regions: list[Region], thresh: float = 0.6) -> list[Region]:
    """Drop every span into its tightest containing region; leftovers become
    free-flow regions clustered by line proximity, column by column."""
    tables = [r for r in regions if r.kind == "table"]
    chips = [r for r in regions if r.kind == "chip"]
    containers = [r for r in regions if r.kind in ("panel", "figure")]
    orphans: list[Span] = []

    spans = [sub for s in prim.spans for sub in _split_repeated_span(s, tables)]
    for s in spans:
        placed = False
        for t in tables:
            if containment(s.bbox, t.bbox) > thresh:
                # A rowspan-merged label cell is vertically centered across
                # its N sub-rows, so its own bbox straddles a row boundary
                # near its center. That center falls squarely inside a
                # single row when N is odd (the middle row), but sits
                # almost exactly ON the boundary between two rows when N is
                # even -- neither individual cell then reaches the general
                # `thresh` containment, and the whole label was silently
                # dropped (confirmed real case: a graphic-organizer table
                # with alternating 2-row and 3-row label groups kept every
                # 3-row label, "تفاصيل مهمّة", but lost both 2-row ones,
                # "المقدّمة" and "النهاية", entirely). Once a span is already
                # confirmed to belong to THIS table, there is no other
                # reasonable destination for it, so the best-matching cell
                # wins outright rather than needing to also clear the
                # stricter region-vs-region threshold.
                best_cell, best_score = None, 0.0
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
    _merge_marker_columns(by_col)
    for col, spans in by_col.items():
        for r in _cluster_flow(spans):
            r.column = col
            regions.append(r)

    return [r for r in regions if r.spans or r.kind in ("figure", "table")]


_LIST_MARKER_RE = re.compile(r"^[•‣◦●○▪▫∙·oO*\-–—]$|^\(?[0-9]{1,3}[.)]?$|^\(?[a-zA-Z][.)]$")


def _starts_with_list_marker(ls: list[Span], gap_ratio: float = 2.0) -> bool:
    """Does this line open with an isolated bullet/number marker?

    Direction-aware: a line's marker sits at its LEFTMOST span for LTR text
    but its RIGHTMOST span for RTL text -- reading order runs the opposite
    way. Checking only the leftmost span (as if every document were LTR)
    silently never matches a single RTL numbered item; confirmed real case,
    an Arabic textbook where a numbered marker ("N ." at the line's
    rightmost position) never registered as a marker at all, so consecutive
    numbered category headers with only a normal-sized gap between them (no
    hanging indent) kept merging into one paragraph.

    Two independent signals, either one sufficient:
    1. A hanging-indent list item's marker sits far enough from the body
       text that the gap is much wider than an ordinary inter-word space
       (confirmed: ~35pt marker-to-text gap vs ~2.5pt normal word spacing).
       Checked relative to font size, not an absolute distance, so it holds
       across font sizes/documents.
    2. The line's own label ends in a colon (checked at whichever end of
       the raw string that logical end lands on, since RTL text is often
       stored in visual left-to-right order -- the colon can appear as the
       first character of the string, not the last). A colon-terminated
       label is a standalone header by convention (see
       _merge_wrapped_label_rows's identical reasoning) and always starts a
       new item even sitting directly against its marker with no gap at
       all -- confirmed real case: RTL numbered category headers, each
       ending in ':', packed tight against their own marker digit.
    """
    if len(ls) < 2:
        return False
    rtl = is_arabic(" ".join(s.text for s in ls))
    by_x = sorted(ls, key=lambda s: s.bbox[0])
    first, second = (by_x[-1], by_x[-2]) if rtl else (by_x[0], by_x[1])
    if not _LIST_MARKER_RE.match(first.text.strip()):
        return False
    gap = (first.bbox[0] - second.bbox[2]) if rtl else (second.bbox[0] - first.bbox[2])
    if gap > first.size * gap_ratio:
        return True
    label = (by_x[0] if rtl else by_x[-1]).text.strip()
    return bool(label) and (label[0] == ":" or label[-1] == ":")


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
        # A hanging-indent list marker (bullet, "1.", "a)") starting a line is
        # a new-item signal in its own right, independent of vertical gap: a
        # list's inter-item spacing is often barely larger than its intra-
        # paragraph line spacing (confirmed: 17.5pt actual gap vs a 17.6pt
        # threshold from line-height alone), so the gap check by itself can
        # merge two distinct items by a hair. A leading marker always starts
        # a new item regardless of how tight that gap happens to be.
        new_item = _starts_with_list_marker(ls)
        if groups and prev_bottom is not None and (top - prev_bottom) < lead * gap_mult and overlaps and same_size and not new_item:
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
                       min_gap: float = 10.0, max_width_frac: float = 0.92,
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

    Bboxes wider than `max_width_frac` of the *detected content width* (not
    the raw page width -- confirmed bug: an ordinary single-column business
    filing has body lines occupying ~90% of the raw page width once normal
    margins are accounted for, so a page-width-relative threshold excluded
    literally every real text line, leaving only a bullet column and a
    stray page-number to define "columns" from noise) are excluded from the
    emptiness measurement (though still assigned a column afterwards) --
    otherwise one genuinely full-bleed element (a header, footer, rule
    spanning the whole content region) still poisons every column it
    happens to cross, persistence check or not.

    A candidate gap is further rejected unless BOTH sides have at least one
    contiguous multi-line run of ink (see `max_run` below) -- confirmed bug:
    a hanging-indent bullet list produces a "column" on the marker side
    that is persistently empty in the gutter-check sense (same mechanism as
    a real gutter) since each marker is a single isolated line followed by
    a paragraph-height gap before the next one, no matter how many list
    items there are. A genuine second column is running text: it always has
    at least a few lines stacked back-to-back somewhere. That contiguous
    multi-line evidence, not mere gap-emptiness, is what actually tells two
    parallel columns apart from a marker/indent gap.
    """
    if not bboxes:
        return [0.0, page_width]
    content_left = min(b[0] for b in bboxes)
    content_right = max(b[2] for b in bboxes)
    content_width = max(content_right - content_left, 1.0)
    narrow = [b for b in bboxes if (b[2] - b[0]) <= content_width * max_width_frac]
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

    # A single text line's height in row-bins, used below as the yardstick
    # for "is there more than one line's worth of stacked content here".
    line_bins = max(1, round(float(np.median([b[3] - b[1] for b in bboxes])) / yres))

    def max_run(col_lo: int, col_hi: int) -> int:
        """Longest run of consecutive rows with any ink in [col_lo, col_hi)."""
        if col_hi <= col_lo:
            return 0
        has_ink = content[:, col_lo:col_hi].any(axis=1)
        best = cur = 0
        for v in has_ink:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        return best

    # A real second column is running text: multiple lines stack with no
    # full blank-paragraph gap between them, so somewhere it has a
    # contiguous ink run spanning several lines. A list marker or indent
    # column (bullets, item numbers) only ever produces isolated one-line
    # bursts -- each marker sits alone, followed by the paragraph-height
    # gap until the next one -- so its longest run tops out around a
    # single line no matter how many items there are. Requiring >1 line's
    # worth of contiguous run on *both* sides of a candidate gap is what
    # actually distinguishes two parallel columns from a marker/indent gap;
    # emptiness of the gap itself is necessary but not sufficient (a
    # marker's gap-to-text is empty in exactly the same way a real gutter
    # is).
    min_run = max(2, round(line_bins * 1.8))

    gaps, run = [], None
    for i in range(inked_cols[0], inked_cols[-1] + 1):
        if fill_frac[i] <= empty_thresh:
            run = i if run is None else run
        elif run is not None:
            if (i - run) * xres >= min_gap:
                if max_run(inked_cols[0], run) >= min_run and max_run(i, inked_cols[-1] + 1) >= min_run:
                    gaps.append((run * xres, i * xres))
            run = None
    if run is not None and (inked_cols[-1] + 1 - run) * xres >= min_gap:
        if max_run(inked_cols[0], run) >= min_run and max_run(inked_cols[-1] + 1, inked_cols[-1] + 1) >= min_run:
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
