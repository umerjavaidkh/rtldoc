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
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import statistics

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


def _evidence_score(signals: dict[str, float], weights: dict[str, float]) -> float:
    """Combine named table-ness signals (each caller-normalized to [0,1])
    into one confidence via a fixed weighted sum, so a table candidate is
    accepted when its evidence clears a threshold rather than surviving a
    chain of independently AND'd vetoes. A veto chain means guard #10,
    tuned for the document that motivated it, can silently kill a true
    positive shaped like guard #3's case -- a strongly-framed non-numeric
    table (real rules, no digits) should survive on framing+fill alone, and
    a coincidental alignment that's merely mediocre on every axis should
    fail without needing its own bespoke rejection rule. Weights are tuned
    against rtldoc/eval/golden's fixtures, each of which used to be a
    `# confirmed real case:` comment guarding one specific veto -- see
    _detect_borderless_in_lines and _detect_row_wrapped_in_frame for the
    signals actually fed in."""
    return sum(weights[k] * signals[k] for k in weights)

RegionKind = Literal["panel", "chip", "figure", "flow", "rule", "table", "rotated"]


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
    # Cluster by proximity rather than rounding to a grid: two rules 2pt apart
    # can straddle a bin boundary and stay separate, which is how a separator
    # drawn twice (y=344 and y=346 on UAE manual p21) survived as two row lines
    # of different lengths and dragged the consistency vote below its bar.
    ordered = sorted(rules, key=lambda f: f.bbox[1] if horizontal else f.bbox[0])
    groups: dict[float, list[Fill]] = {}
    key = None
    for f in ordered:
        pos = f.bbox[1] if horizontal else f.bbox[0]
        if key is None or pos - key > tol:
            key = pos
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
    all_rules = all_rules + _synth_row_rules(prim, all_rules)

    def _detect(rules: list["Fill"]) -> list[Region]:
        found: list[Region] = []
        for cluster in _cluster_rules(rules):
            for band in _split_by_column_profile(cluster):
                found.extend(_detect_table_in_cluster(band, min_rows, min_cols,
                                                      coverage))
        return found

    tables = _detect(all_rules)

    # Last resort only. Inferring columns from where row rules break is right
    # for a cell-ruled table (see _synth_col_rules) but wrong wherever real
    # rules already describe a grid: a 10-K underlines its numeric cells, and
    # those underline endpoints recur row to row convincingly enough to pass
    # every test the synthesis can apply, yet they are not the column edges --
    # letting them run there turned a correct 18x7 into a welded 9x3. Running
    # it only when the rules yielded nothing at all keeps it from ever
    # overriding a detection that already works.
    # What the rules have already explained: a detected table, and any y a
    # vertical divider covers. Whatever horizontal rules are left over may
    # still be a cell-ruled table that draws no verticals at all -- p19 of the
    # UAE manual is exactly this, a synthesised-column table stacked above a
    # normally ruled one, so the test has to be per REGION and not per page.
    covered = [(t.bbox[1], t.bbox[3]) for t in tables]
    covered += [(f.bbox[1], f.bbox[3]) for f in all_rules
                if (f.bbox[3] - f.bbox[1]) > (f.bbox[2] - f.bbox[0])]
    free = [f for f in all_rules
            if (f.bbox[2] - f.bbox[0]) > (f.bbox[3] - f.bbox[1])
            and not any(lo - 1.0 <= f.bbox[1] <= hi + 1.0 for lo, hi in covered)]
    synth = _synth_col_rules(prim, free)
    if synth:
        tables.extend(_detect(free + synth))
    return tables


def _synth_row_rules(prim: PagePrimitives, all_rules: list["Fill"]) -> list["Fill"]:
    """Horizontal rules a table draws only as breaks in its vertical ones.

    A ruled grid needs both directions (_detect_table_in_cluster bails
    without them), but a common publishing style draws ONLY the column
    dividers and delimits rows with colored bands instead. The row
    boundaries are still stated exactly: each column divider is emitted as
    several collinear segments, and where one segment stops and the next
    begins IS a row edge. A divider that is absent from one band is a
    merged cell in that row -- so the segments carry the spans too.

    Confirmed real case (Arabic teacher's guide p36, InDesign): three
    dividers at x=201.3/360.0/518.7, each drawn in segments breaking at
    y=156.1/215.4/402.5, and only x=360.0 continuing on to y=445.0 --
    which is precisely the 4-column table whose bottom row is two merged
    cells. Zero horizontal rules on the page, so the grid was discarded
    whole.

    The outer left/right edge is not in the rules (the dividers are all
    interior), so it comes from the colored band the dividers cross.
    """
    verts = [f for f in all_rules
             if (f.bbox[3] - f.bbox[1]) > (f.bbox[2] - f.bbox[0])]
    horiz = [f for f in all_rules
             if (f.bbox[2] - f.bbox[0]) > (f.bbox[3] - f.bbox[1])]
    if horiz or len(verts) < 2:
        return []

    xs = sorted({round(f.bbox[0], 1) for f in verts})
    if len(xs) < 2:                       # need a real internal column split
        return []
    ys = sorted({round(v, 1) for f in verts for v in (f.bbox[1], f.bbox[3])})
    if len(ys) < 3:                       # need >= 2 rows
        return []

    vx0, vx1 = min(xs), max(xs)
    vy0, vy1 = ys[0], ys[-1]
    # Outer extent: the widest band the dividers actually cross. Without one
    # the dividers alone bound only the interior columns, losing the outer
    # two entirely.
    bands = [f for f in prim.fills
             if not f.is_rule
             and f.bbox[0] <= vx0 + 1.0 and f.bbox[2] >= vx1 - 1.0
             and f.bbox[1] >= vy0 - 1.0 and f.bbox[3] <= vy1 + 1.0]
    if not bands:
        return []
    x0 = min(f.bbox[0] for f in bands)
    x1 = max(f.bbox[2] for f in bands)
    if x1 - x0 <= 0:
        return []
    return [Fill(bbox=(x0, y, x1, y), color=(0.0, 0.0, 0.0),
                 is_rule=True, is_stroke=True) for y in ys]


# Two rules this close in x are one divider drawn twice (a stroked line and
# its fill edge, or a 0.2pt rendering artefact), not two columns.
COL_RULE_TOL = 1.5
# A band must be at least this tall, and carry this many row rules, to be a
# table in its own right; anything smaller is a sliver and folds into its
# neighbour rather than becoming a one-row fragment.
BAND_MIN_HEIGHT = 12.0
BAND_MIN_ROWS = 2
# Abutting per-cell divider segments join across a gap this small; a
# divider genuinely missing for a section leaves a wider hole than this.
DIVIDER_JOIN = 3.0
# Interior dividers must be present over this much of the cluster height
# before a change between them is read as a change of table.
PROFILE_INTERIOR_COVER = 0.5
# A divider running this much of the height is part of the frame, not interior.
PROFILE_FULL_RUN = 0.9


def _divider_runs(verts: list[Fill]) -> dict:
    """Where each column divider actually runs, x -> [(y0, y1), ...].

    Grouped by x (to COL_RULE_TOL, so a line drawn twice counts once) and
    joined along y across DIVIDER_JOIN, because these files draw a divider as
    one short segment per cell rather than one long line. Joining abutting
    segments is what keeps a per-cell-drawn divider from reading as a change
    of layout at every single row; a real gap -- the divider genuinely not
    being there for a section -- is wider than a rule is thick and survives.
    """
    groups: dict[float, list[tuple[float, float]]] = {}
    for f in verts:
        key = round(f.bbox[0] / COL_RULE_TOL) * COL_RULE_TOL
        groups.setdefault(key, []).append((f.bbox[1], f.bbox[3]))
    out: dict[float, list[list[float]]] = {}
    for key, ivs in groups.items():
        runs: list[list[float]] = []
        for y0, y1 in sorted(ivs):
            if runs and y0 - runs[-1][1] <= DIVIDER_JOIN:
                runs[-1][1] = max(runs[-1][1], y1)
            else:
                runs.append([y0, y1])
        out[key] = runs
    return out


def _split_by_column_profile(cluster: list[Fill]) -> list[list[Fill]]:
    """Cut one rule cluster into bands that each have a single column layout.

    Why this exists
    ---------------
    _detect_table_in_cluster keeps a vertical rule as a column only if it
    spans `coverage` (0.5) of the CLUSTER's full height. That is right for
    one table and wrong for several stacked in one frame, which is a very
    common shape: a spec sheet or a bilingual contract draws an outer box and
    changes its internal column count section by section. Every divider that
    serves only one section then measures short against the whole stack and
    is dropped, so that section loses its columns and its text falls out of
    the grid as loose prose.

    Two confirmed cases, previously believed unrelated: the UAE service
    manual (600bb49d p19-23), where one lattice y149-489 carries 2-, 3- and
    4-column sections; and the Saudi labour contract (f9b88f18 p81), three
    stacked tables at x-edges [71,262,525], [71,252,525] then
    [71,216,305,525], which collapsed to a single 8x1 column of mush. Page 80
    of that same file has ONE profile throughout and parsed correctly --
    which is why fixing 80 never fixed 81.

    Splitting on the profile makes each band internally consistent, so every
    divider spans its own band fully and the coverage test passes on its own
    terms. A cluster with one profile comes back unchanged, so ordinary
    single-layout tables take exactly the path they did before.
    """
    verts = [f for f in cluster if (f.bbox[3] - f.bbox[1]) > (f.bbox[2] - f.bbox[0])]
    horiz = [f for f in cluster if (f.bbox[2] - f.bbox[0]) > (f.bbox[3] - f.bbox[1])]
    if len(verts) < 2 or not horiz:
        return [cluster]

    runs = _divider_runs(verts)
    # Stacked tables versus a table with subdivided rows. Both show interior
    # dividers at differing x down the page, and no threshold on an individual
    # divider's length can tell them apart -- measured, the labour contract p81
    # NEEDS a divider running 0.175 of the height while the UAE manual p25 must
    # IGNORE one running 0.204. What separates them is whether the interior
    # dividers TILE: stacked tables each supply their own, so some interior
    # divider is present at nearly every y, whereas a row split leaves most of
    # the height with no interior divider at all (p81 covers ~1.0, p25 ~0.3).
    top = min(f.bbox[1] for f in verts)
    height = max(f.bbox[3] for f in verts) - top
    if height <= 0:
        return [cluster]
    # "Interior" means not full-height, NOT merely not-outermost: p25 is a
    # two-column table whose real divider at x=472.5 runs the whole way, and
    # counting that as interior made its sparse row splits look like tiling.
    inner = [iv for x, rs in runs.items()
             if sum(b - a for a, b in rs) < PROFILE_FULL_RUN * height
             for iv in rs]
    covered, merged = 0.0, []
    for a, b in sorted(inner):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    covered = sum(b - a for a, b in merged)
    if covered / height < PROFILE_INTERIOR_COVER:
        return [cluster]
    edges = sorted({round(v, 1) for rs in runs.values() for r in rs for v in r})
    if len(edges) < 3:
        return [cluster]

    def profile(lo: float, hi: float) -> tuple:
        mid = (lo + hi) / 2.0
        return tuple(sorted(x for x, rs in runs.items()
                            if any(a <= mid <= b for a, b in rs)))

    # A divider that DISAPPEARS for a stretch is a merged cell -- the single
    # commonest thing in a real table -- so a profile that is a subset of its
    # neighbour continues the same table. A divider that MOVES (an x present
    # in neither direction's set) is a different table underneath, and only
    # that splits. Without this distinction every merged cell starts a new
    # one-row "table": confirmed on the labour contract p80, whose header row
    # drops the middle divider and was being cut off as its own 1x2 band.
    bands: list[list] = []
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo <= 0.5:
            continue
        prof = profile(lo, hi)
        if bands:
            prev = bands[-1][2]
            a, b = set(prev), set(prof)
            if a <= b or b <= a:
                bands[-1][1] = hi
                bands[-1][2] = tuple(sorted(a | b))
                continue
        bands.append([lo, hi, prof])

    # A band with fewer than two dividers is not a grid of its own; fold it
    # into whichever neighbour it touches so its rules are not simply lost.
    merged: list[list] = []
    for b in bands:
        thin = len(b[2]) < 2 or (b[1] - b[0]) < BAND_MIN_HEIGHT
        if thin and merged:
            merged[-1][1] = b[1]
        elif not thin:
            merged.append(b)
    if len(merged) <= 1:
        return [cluster]

    out: list[list[Fill]] = []
    for lo, hi, _prof in merged:
        rows = [f for f in horiz if lo - 1.0 <= f.bbox[1] <= hi + 1.0]
        if len(rows) < BAND_MIN_ROWS:
            continue
        cols = []
        for f in verts:
            y0, y1 = max(f.bbox[1], lo), min(f.bbox[3], hi)
            if y1 - y0 > 1.0:
                cols.append(Fill(bbox=(f.bbox[0], y0, f.bbox[2], y1),
                                 color=f.color, is_rule=True,
                                 is_stroke=f.is_stroke))
        if len(cols) >= 2:
            out.append(rows + cols)
    return out or [cluster]


# A row-ruled table needs at least this many separators before its columns are
# worth inferring; two lines are an underline and a rule, not a grid.
SYNTH_COL_MIN_ROWS = 3
SYNTH_COL_EXTENT_TOL = 6.0
# Fraction of rows that must break at an x before it counts as a column edge.
SYNTH_COL_EDGE_AGREE = 0.6
# How much of the row width the rules must actually cover to be a cell grid.
SYNTH_COL_MIN_COVERAGE = 0.6
# Two row rules closer than this are one separator drawn twice.
SYNTH_COL_ROW_TOL = 3.0


def _synth_col_rules(prim: PagePrimitives, horiz: list["Fill"]) -> list["Fill"]:
    """Column dividers a table states only by where its row rules stop.

    The mirror image of _synth_row_rules. That one covers a table that draws
    only its column dividers and breaks them at the rows; this covers the
    opposite, a table that draws only its row rules and breaks THEM at the
    columns. _detect_table_in_cluster requires both directions and returns
    nothing for these, so the table falls through to prose; worse,
    _cluster_rules has no vertical to bridge the rows with and shatters one
    table into a cluster per row.

    Confirmed real cases, both in the UAE service manual (600bb49d): p16, a
    seven-row service list with 35 horizontal rules, ZERO vertical rules, and
    every row drawn as five segments at x 68-172, 172-228, 232-292, 292-428
    and 444-524 -- precisely its five columns; and p19, where a table of this
    kind sits directly ABOVE a conventionally ruled one, which is why the
    caller passes only the rules nothing else has explained rather than
    testing the page as a whole.

    Columns are read from the rule geometry itself, never inferred from text:
    each segment endpoint is a column edge. Rules are grouped into tables by
    vertical proximity first, because two stacked tables of this kind have
    different columns and pooling their endpoints yields edges belonging to
    neither.
    """
    if len(horiz) < SYNTH_COL_MIN_ROWS:
        return []

    # distinct rows, merging the near-duplicates left by a rule drawn twice
    ys: list[float] = []
    for y in sorted(round(f.bbox[1], 1) for f in horiz):
        if not ys or y - ys[-1] > SYNTH_COL_ROW_TOL:
            ys.append(y)
    if len(ys) < SYNTH_COL_MIN_ROWS:
        return []

    # split into tables where the row pitch jumps: a gap several times the
    # usual one is the space between two tables, not a tall row
    gaps = [b_ - a_ for a_, b_ in zip(ys, ys[1:])]
    limit = max(3.0 * statistics.median(gaps), 30.0) if gaps else 30.0
    groups: list[list[float]] = [[ys[0]]]
    for prev, y in zip(ys, ys[1:]):
        (groups[-1] if y - prev <= limit else groups.append([]) or groups[-1]).append(y)

    out: list[Fill] = []
    for grp in groups:
        if len(grp) < SYNTH_COL_MIN_ROWS:
            continue
        y0, y1 = grp[0], grp[-1]
        rows: dict[float, list[Fill]] = {}
        for f in horiz:
            for y in grp:
                if abs(f.bbox[1] - y) <= SYNTH_COL_ROW_TOL:
                    rows.setdefault(y, []).append(f)
                    break
        if len(rows) < SYNTH_COL_MIN_ROWS:
            continue
        # A cell-ruled table rules EVERY cell, so its segments tile the row
        # almost end to end. A financial statement underlines only its numeric
        # cells and leaves the label column bare -- the same segment count over
        # a fraction of the width, and those endpoints are NOT column edges.
        # Measured: the UAE service list tiles 0.79 of its width, a GOOGL 10-K
        # page 0.27, and letting the latter run welded a correct 18x7 to 9x3.
        if statistics.median(len(v) for v in rows.values()) < 2:
            continue
        lo_x = min(f.bbox[0] for v in rows.values() for f in v)
        hi_x = max(f.bbox[2] for v in rows.values() for f in v)
        if hi_x - lo_x <= 0:
            continue
        covs = []
        for segs in rows.values():
            merged: list[list[float]] = []
            for a_, b_ in sorted((f.bbox[0], f.bbox[2]) for f in segs):
                if merged and a_ <= merged[-1][1] + 2.0:
                    merged[-1][1] = max(merged[-1][1], b_)
                else:
                    merged.append([a_, b_])
            covs.append(sum(b_ - a_ for a_, b_ in merged) / (hi_x - lo_x))
        if statistics.median(covs) < SYNTH_COL_MIN_COVERAGE:
            continue

        edges: list[float] = []
        for x in sorted(v for segs in rows.values() for f in segs
                        for v in (f.bbox[0], f.bbox[2])):
            if not edges or x - edges[-1] > SYNTH_COL_EXTENT_TOL:
                edges.append(x)
            else:
                edges[-1] = (edges[-1] + x) / 2.0
        if len(edges) < 3:
            continue
        # A real column edge recurs: the rows break at the SAME x. An edge seen
        # in one row alone is that row's own quirk and would slice the others.
        keep = [edges[0]]
        for x in edges[1:-1]:
            seen = sum(1 for segs in rows.values()
                       if any(abs(v - x) <= SYNTH_COL_EXTENT_TOL
                              for f in segs for v in (f.bbox[0], f.bbox[2])))
            if seen >= max(2, len(rows) * SYNTH_COL_EDGE_AGREE):
                keep.append(x)
        if len(keep) < 2:
            continue
        keep.append(edges[-1])
        out.extend(Fill(bbox=(x, y0, x, y1), color=(0.0, 0.0, 0.0),
                        is_rule=True, is_stroke=True) for x in keep)
    return out


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
    def _consistent(lines: list[tuple[float, float, float]], span: float,
                    tol_frac: float = 0.15, agree: float = 0.7) -> bool:
        """Do these rules share an extent -- allowing for a few that don't?

        Measured against the MEDIAN extent with a majority vote, not against
        max-minus-min. A real table routinely has a rule or two that stop
        short: a merged cell, a column ruled only in the body, a header band
        drawn narrower than the grid. Under max-minus-min a single such rule
        fails the whole table -- confirmed case, a 13x8 NIST table whose rules
        clustered perfectly (the cluster bbox matched the table to the point)
        and was thrown away because its right edges spread 48% against a 15%
        bar. Two genuinely unrelated decorative rules still fail this, because
        they have no dominant extent for a majority to agree on.
        """
        if len(lines) <= 1:
            return True
        los = [a for _, a, _ in lines]
        his = [b for _, _, b in lines]
        tol = tol_frac * span
        mid_lo = statistics.median(los)
        mid_hi = statistics.median(his)
        # A rule that stops short at ONE end, consistently, is a merged cell
        # above or below it -- not a stray mark. The commonest shape in these
        # documents is a merged header cell spanning the rate columns, under
        # which those columns' dividers legitimately begin one row lower.
        # Requiring both endpoints to match scored the Saudi penalties table
        # (7 column rules, 3 starting 35pt lower) at 0.571 against a 0.7 bar
        # and discarded a fully-ruled 6-column grid, which then fell to the
        # geometric path and came out as 2 columns of interleaved text.
        # Containment within the majority extent, covering most of it, keeps
        # that table while still rejecting two unrelated decorative rules --
        # they share no extent to be contained in.
        span_lo, span_hi = mid_lo - tol, mid_hi + tol
        need = CONSISTENT_CONTAIN_FRAC * (mid_hi - mid_lo)
        ok = 0
        for a, b in zip(los, his):
            if abs(a - mid_lo) <= tol and abs(b - mid_hi) <= tol:
                ok += 1
            elif span_lo <= a and b <= span_hi and (b - a) >= need:
                ok += 1
        return ok >= max(2, len(lines) * agree)

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


def column_bands(bboxes: list[Rect], page_width: float, page_height: float,
                 min_gap: float | None = None) -> list[tuple[float, float]]:
    """The page's column x-ranges as (x0, x1) pairs, ordered left to right.

    One band means a single-column page, and every caller then behaves exactly
    as it did before columns existed."""
    b = _column_boundaries(bboxes, page_width, page_height, min_gap)
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def _block_split_gap(gaps: list[float], scale: float) -> float:
    """How large a horizontal gap has to be to end one block and start another.

    Not a constant multiple of the type size. Within a block the gaps between
    lines cluster tightly around the leading; a block boundary is a gap that
    stands OUT of that cluster. So the threshold is a robust outlier test on
    the page's own gap distribution -- median + 3 MAD -- which adapts to the
    document's leading instead of assuming one. Confirmed case: a page whose
    block boundary was 10.6pt against within-block gaps of 5-8pt on 9pt type;
    every fixed multiple either missed it or split the prose apart.

    The floor is a full line height of clear space, and it is doing real work:
    on a table-heavy page the gaps between table rows are tight AND uniform, so
    the MAD is tiny and the outlier test alone would split every row into its
    own band -- after which each row's X-projection reads the table's own
    column gaps as page columns. Requiring at least one blank line before
    calling something a block boundary is what keeps a table one block; the
    outlier test only ever raises the bar above that, for documents set with
    looser leading.
    """
    if len(gaps) < 4:
        return scale * BLOCK_SPLIT_FRAC
    med = statistics.median(gaps)
    mad = statistics.median([abs(g - med) for g in gaps]) or 0.5
    return max(scale * 1.0, med + 3.0 * mad)


def _horizontal_bands(bboxes: list[Rect], scale: float
                      ) -> list[tuple[list[Rect], float, float]]:
    """Split a set of boxes on horizontal whitespace into stacked bands.

    The Y half of an XY-cut. A band is a run of content with no full-width
    horizontal gap inside it, which is exactly the unit within which a column
    structure is constant: a banner, a full-width table, a block of prose.
    """
    if not bboxes:
        return []
    events = sorted(((b[1], b[3]) for b in bboxes))
    seen = []
    hi_scan = events[0][1]
    for y0, y1 in events[1:]:
        if y0 > hi_scan:
            seen.append(y0 - hi_scan)
        hi_scan = max(hi_scan, y1)
    min_gap = _block_split_gap(seen, scale)
    bands: list[tuple[float, float]] = []
    lo, hi = events[0]
    for y0, y1 in events[1:]:
        if y0 - hi > min_gap:
            bands.append((lo, hi))
            lo, hi = y0, y1
        else:
            hi = max(hi, y1)
    bands.append((lo, hi))
    out = []
    for y0, y1 in bands:
        inside = [b for b in bboxes if (b[1] + b[3]) / 2 >= y0 - 1
                  and (b[1] + b[3]) / 2 <= y1 + 1]
        if inside:
            out.append((inside, y0, y1))
    return out


def page_gutters(bboxes: list[Rect], page_width: float, page_height: float,
                 min_gap: float | None = None, _depth: int = 0,
                 exclude: list[Rect] | None = None
                 ) -> list[tuple[float, float, float]]:
    """Each column gutter as (x, y_top, y_bottom) -- WHERE it separates columns,
    not merely that it does.

    A page is rarely one layout all the way down. A paper puts a full-width
    table or figure above two columns of prose; a report puts a banner over a
    three-column body; a form alternates. A gutter found from the page as a
    whole is real only over the rows where it is actually clear, and applying
    it outside them cuts full-width content in half -- confirmed case: a
    three-table arXiv page whose full-width Table 1 was sliced down the middle
    by the gutter belonging to the prose two-thirds further down the page.

    Carrying the vertical extent is what makes mixed layouts work without
    special-casing them: a line is split at a gutter only when the line lies
    inside that gutter's own rows.

    Derived by RECURSIVE XY-CUT rather than by one projection over the whole
    page. The page is first cut on horizontal whitespace into stacked bands --
    banner, full-width table, prose -- and columns are then sought inside each
    band independently. A gutter is therefore a property of the band that owns
    it and has no authority outside it, which is what makes a full-width table
    immune to the gutter of the prose beneath it. Each column is recursed into,
    so a column containing its own sub-columns is handled by the same rule
    rather than by a special case.

    A global projection cannot express this: it has to answer "is there a
    gutter on this page" with one number per x, and every mixed layout then
    forces a choice between missing the gutter and cutting the full-width
    content. Segmenting first removes the question.
    """
    if _depth > 3 or not bboxes:
        return []
    scale = _body_size(bboxes)
    out: list[tuple[float, float, float]] = []

    # Blocks are cut from ALL the content, so a full-width table forms its own
    # block and the layout below it cannot reach across.
    bands = _horizontal_bands(bboxes, scale)
    for band, y0, y1 in bands:
        # Columns are projected from PROSE only. A table's own column gaps are
        # empty over its full height and are otherwise indistinguishable from a
        # gutter -- width, coverage, line count, spacing regularity and fill
        # ratio were each measured on both populations and each overlapped. So
        # the table's words are withheld from the projection rather than
        # separated from it by a threshold.
        proj = band if not exclude else [
            b for b in band
            if not any(containment(b, ex) > 0.5 for ex in exclude)]
        # A block that is mostly table gets no vote. Its handful of non-table
        # words is usually just the caption, and projecting columns from a
        # caption then applying them to the whole block slices the table it
        # describes -- confirmed on a page whose two full-width tables were cut
        # in half by a gutter found in "Table 1. Performance on video depth
        # estimation. We follow the protocols of...".
        if len(proj) < 8 or len(proj) < 0.5 * len(band):
            continue
        bounds = _column_boundaries(proj, page_width, page_height, min_gap)
        if len(bounds) <= 2:
            continue
        # ...but the gutter is scoped to the WHOLE band, tables included. A
        # column gutter is a property of the block it divides, so it applies
        # throughout that block -- which is what lets a table wrongly spanning
        # two columns be cut at the same gutter as the prose around it, even
        # though the table contributed nothing to finding it.
        # A column is a run of RUNNING TEXT, and running text takes many lines.
        # Without this, any row-structured block whose fields happen to align --
        # a table the detector missed, a table of contents, a definition list --
        # is read as columns, and reading it column-major tears every row apart:
        # the entry text ends up separated from its page number, the model name
        # from its parameter count.
        #
        # Measured on 49 pages: blocks that were genuinely two-column prose had
        # a median of 56 lines in their smaller column (p10 = 15); blocks that
        # were row-structured had a median of 4 (p10 = 2). The populations
        # separate cleanly, which is why this guard is a line count and not
        # another geometric ratio.
        rows = {}
        for bx in band:
            rows.setdefault(round(bx[1]), []).append(bx)
        def _lines_between(lo: float, hi: float) -> int:
            return sum(1 for _, bs in rows.items()
                       if any(lo <= (b[0] + b[2]) / 2 < hi for b in bs))
        if any(_lines_between(a, b) < MIN_COLUMN_LINES
               for a, b in zip(bounds, bounds[1:])):
            continue
        for x in bounds[1:-1]:
            out.append((x, y0, y1))
        for a, b in zip(bounds, bounds[1:]):
            sub = [bx for bx in band if a <= (bx[0] + bx[2]) / 2 < b]
            if len(sub) < len(band):
                out.extend(page_gutters(sub, page_width, page_height,
                                        min_gap, _depth + 1, exclude))
    return out


def _split_index(x_center: float, gutters: list[tuple[float, float, float]],
                 y: float) -> int:
    """How many gutters, active at this y, lie left of x -- the piece of the
    line this glyph belongs to."""
    return sum(1 for gx, gy0, gy1 in gutters if gx <= x_center and gy0 <= y <= gy1)


def _band_of(x_center: float, bands: list[tuple[float, float]]) -> int:
    for i, (a, b) in enumerate(bands):
        if a <= x_center < b:
            return i
    # Outside every band -- a margin note, a page number, a hanging bullet.
    # Attach it to the nearest band rather than dropping it, so band
    # assignment can never lose a span.
    return min(range(len(bands)),
               key=lambda i: abs(x_center - (bands[i][0] + bands[i][1]) / 2))


def group_by_line(spans: list[Span], tol_frac: float = 0.5,
                  bands: list[tuple[float, float]] | None = None,
                  gutters: list[tuple[float, float, float]] | None = None) -> list[list[Span]]:
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
    if gutters:
        # Split by gutter, scoped to the rows where each gutter actually holds,
        # so full-width content above or below a columned block stays whole.
        buckets: dict[int, list[Span]] = {}
        for sp in spans:
            cx = (sp.bbox[0] + sp.bbox[2]) / 2
            cy = (sp.bbox[1] + sp.bbox[3]) / 2
            buckets.setdefault(_split_index(cx, gutters, cy), []).append(sp)
        out: list[list[Span]] = []
        for i in sorted(buckets):
            out.extend(group_by_line(buckets[i], tol_frac))
        return out

    if bands and len(bands) > 1:
        # COLUMN AWARENESS. On a two-column page the left and right columns
        # share baselines -- that is what "two columns" means -- so grouping by
        # y alone welds the left column's line to the right column's line at
        # the same height, yielding text that reads "...does not fully bound
        # radio resources are allocated among DRBs...". Nothing downstream can
        # undo it: by the time regions, tables and reading order are computed,
        # the two columns are already one string. This was the single root
        # cause behind a family of symptoms previously patched one at a time --
        # see the historical notes in detect_borderless_tables,
        # _detect_borderless_in_lines and parse_page, each of which describes
        # working around "group_by_line has no column awareness".
        #
        # Lines come out band by band (each band internally top-to-bottom)
        # rather than globally by y, because every consumer -- flow clustering,
        # borderless-table detection, paragraph assembly -- wants one column's
        # lines consecutively.
        buckets: dict[int, list[Span]] = {}
        for sp in spans:
            buckets.setdefault(
                _band_of((sp.bbox[0] + sp.bbox[2]) / 2, bands), []).append(sp)
        out: list[list[Span]] = []
        for i in sorted(buckets):
            out.extend(group_by_line(buckets[i], tol_frac))
        return out

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
                              tol: float, edge=lambda s: s.bbox[2],
                              indent_ratio: float = 1.8) -> list[list[list[Span]]]:
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
        return any(any(abs(edge(s) - cx) <= tol for cx in centers) for s in ln)

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


def _detect_row_wrapped_tables(lines: list[list[Span]], min_rows: int, tol: float, pad: float,
                                rule_fills: list[Fill]) -> list[Region]:
    """Recover a rule-framed key/value definition table whose trailing
    column (a description/VALUE) legitimately wraps across many lines per
    row -- a layout the alignment voter above can't see, because that voter
    looks for every column co-located on the SAME physical line, and here
    only a row's first line carries all its columns while the rest of the
    row is one long wrapped cell (confirmed real case: a PDF filter's
    parameter dictionary -- KEY | TYPE | VALUE -- where VALUE wraps 3-6
    lines; every row landed in its own singleton band under the small
    fixed-gap tolerance meant for short wrapped LABELS, and the whole table
    went undetected).

    The trick is WHICH span is allowed to vote: only each line's own
    LEFTMOST span, never any other. A row-start line's leading span (the
    KEY) recurs at a fixed x across rows -- clean signal. A continuation
    line's leading span (the wrapped VALUE's own text) recurs too, just at
    a different, wider x. Spans buried mid-sentence (an inline emphasized
    "true"/"false"/a cross-reference like "Rows") never win this vote no
    matter how short they are or how often they coincidentally align --
    that noise is exactly what made the generic per-span voter unusable
    here (confirmed real case: three unrelated short keywords on three
    different lines happened to right-align within tolerance and were
    read as a 2nd real column, corrupting row boundaries).

    Only tried when real drawn rules frame a region -- independent,
    author-provided evidence a table exists there -- since unlike the
    generic voter this has no numeric-fraction fallback guard to fall back
    on for a purely coincidental alignment.

    Rules are grouped by matching horizontal EXTENT (x0/x1) first, not by
    vertical proximity: taking the min/max Y of EVERY rule on the page as
    one global frame is fragile the instant a page carries more than one
    unrelated rule group -- a running page-header divider far above the
    real table, say -- which balloons the "framed" range to swallow
    unrelated prose above the table as if it were part of it (confirmed
    real case: an intro paragraph and worked examples above a real
    escape-sequence table got pulled into the same false table because the
    page's header rule sat 250pt above the table's own top rule). Grouping
    by y-proximity (as _cluster_rules does for fully gridded tables) is
    wrong here for the opposite reason: THIS table's own top/divider/bottom
    rules are deliberately far apart -- that's the whole point of a framed
    table with no internal row rules -- so a proximity cluster would split
    a single table's own frame into disconnected pieces. A table's frame
    rules, whatever their spacing, always share the same left/right extent
    (the same author-drawn box), which an unrelated rule elsewhere on the
    page essentially never coincidentally matches to within a few points.
    """
    # A tight tolerance here, deliberately NOT the wider `tol` used for text
    # alignment elsewhere in this function -- an unrelated rule can land
    # within a few points of a table's own extent by pure coincidence
    # (confirmed real case: a page-header rule's extent missed a real
    # table's own extent by only 5.8pt on both edges, well inside `tol`,
    # and got wrongly folded into the table's frame).
    x_tol = 3.0
    x_groups: list[list[Fill]] = []
    for f in rule_fills:
        x0, x1 = f.bbox[0], f.bbox[2]
        match = next((g for g in x_groups
                     if abs(g[0].bbox[0] - x0) <= x_tol and abs(g[0].bbox[2] - x1) <= x_tol), None)
        if match is not None:
            match.append(f)
        else:
            x_groups.append([f])

    page_bottom = max((s.bbox[3] for ln in lines for s in ln), default=0.0)

    out: list[Region] = []
    for cluster in x_groups:
        if len(cluster) < 2:
            continue
        result = _detect_row_wrapped_in_frame(lines, min_rows, tol, pad, cluster, page_bottom)
        if not result:
            # No visible bottom rule: the table's last page has no closing
            # border because it continues onto the NEXT page (confirmed
            # real case: a filter's parameter table opens with a top rule
            # and a header-divider rule but ends with the page itself, its
            # actual bottom rule sitting on a page that hasn't been parsed
            # yet). Retrying with the frame opened all the way to the
            # page's own bottom margin recovers it; the same min_rows +
            # KEY-diversity + non-wide-row guards above still gate this, so
            # ordinary prose below an unrelated short table still won't
            # get swept in as fake rows.
            result = _detect_row_wrapped_in_frame(lines, min_rows, tol, pad, cluster, page_bottom,
                                                  open_ended=True)
        out.extend(result)
    return out


def _detect_row_wrapped_in_frame(lines: list[list[Span]], min_rows: int, tol: float, pad: float,
                                  rule_cluster: list[Fill], page_bottom: float,
                                  open_ended: bool = False) -> list[Region]:
    rule_ys = [y for f in rule_cluster for y in (f.bbox[1], f.bbox[3])]
    framed_lo = min(rule_ys)
    framed_hi = page_bottom if open_ended else max(rule_ys)

    votes: list[tuple[float, int]] = []
    for li, ln in enumerate(lines):
        if not ln:
            continue
        cy = (min(s.bbox[1] for s in ln) + max(s.bbox[3] for s in ln)) / 2
        if not (framed_lo - tol <= cy <= framed_hi + tol):
            continue
        leftmost = min(ln, key=lambda s: s.bbox[0])
        if leftmost.text.strip():
            votes.append((leftmost.bbox[0], li))
    if len(votes) < min_rows:
        return []
    votes.sort()

    clusters: list[dict] = []
    cur: dict | None = None
    for x, li in votes:
        if cur is not None and x - cur["last"] <= tol:
            cur["xs"].append(x)
            cur["lines"].append(li)
            cur["last"] = x
        else:
            cur = {"xs": [x], "lines": [li], "last": x}
            clusters.append(cur)
    # A real drawn top+divider rule frame (guaranteed by the caller) is
    # independent, author-provided evidence a table exists here, strong
    # enough to accept as few as 2 recurring KEY entries (a header row plus
    # a single real data row) rather than requiring min_rows -- confirmed
    # real case: a filter's Table 3.9 opens with just ONE parameter ("K")
    # whose own description happens to run 9 lines, so its header + single
    # data row is genuinely only 2 KEY-column lines, and requiring 3 would
    # reject a real, unambiguous table outright. The diversity, non-wide-
    # row, and majority-multi-line-wrap checks below still gate this the
    # same as any other candidate.
    key_min_rows = min(2, min_rows)
    real = [c for c in clusters if len(c["lines"]) >= key_min_rows]
    if not real:
        return []

    # The KEY/label column is the LEFTMOST recurring margin -- any line
    # starting further right is a row's own wrapped continuation, not a
    # new row (a real table's label column always sits left of its data).
    key_cluster = min(real, key=lambda c: sum(c["xs"]) / len(c["xs"]))
    row_start_lines_all = sorted(key_cluster["lines"])
    if len(row_start_lines_all) < key_min_rows:
        return []
    key_x = sum(key_cluster["xs"]) / len(key_cluster["xs"])

    # This book's own convention reprints an identical literal header row
    # ("KEY TYPE VALUE") at the top of EVERY table, even when two entirely
    # separate tables happen to share the exact same page margins -- which
    # is the common case here, since nearly every parameter-dictionary
    # table in the whole book uses the same x0/x1. Grouping rules by
    # shared x-extent (the caller) then pulls BOTH tables' rules into one
    # cluster, and the leftmost-margin vote above pulls both tables' KEY
    # columns into one combined row_start_lines set -- silently welding two
    # unrelated tables into one (confirmed real case: Table 3.43 and Table
    # 3.44 on the same page, with genuinely different TYPE-column widths,
    # got merged this way and the whole detection aborted outright, since
    # neither table's own consistent column layout survived the merge).
    # An EXACT repeat of the very first row-start line's own text is what
    # marks a second table's own header beginning -- not just "this line's
    # cells all happen to look short," which a legitimately short DATA row
    # could also satisfy; requiring an exact match keeps this from ever
    # misfiring on a real row within a single table.
    def _row_text(li: int) -> str:
        return " ".join(s.text.strip() for s in sorted(lines[li], key=lambda s: s.bbox[0]))

    header_text = _row_text(row_start_lines_all[0])
    row_groups: list[list[int]] = [[row_start_lines_all[0]]]
    for li in row_start_lines_all[1:]:
        if _row_text(li) == header_text:
            row_groups.append([li])
        else:
            row_groups[-1].append(li)

    # A genuine KEY column names a DIFFERENT parameter/entry each row; a
    # bulleted list's leading marker ("•") recurs at a fixed left margin
    # too but is the SAME literal glyph every row, which is what actually
    # distinguishes a real label column from a bullet list wrapping across
    # many lines (confirmed real case: a bulleted list of stream-object
    # rules was read as a table whose single "KEY" was always "•").
    # A strict majority-unique test (not just "more than one distinct value
    # ever appears") is needed -- confirmed real case: a figure caption
    # line plus three bulleted sub-items had exactly 2 distinct key texts
    # ("FIGURE 9.15" and the bullet glyph) across 4 rows, which cleared a
    # bare ">1 distinct" bar while still being 75% the same repeated glyph.
    def _build_one(row_start_lines: list[int], group_last_line: int) -> Region | None:
        # A genuine KEY column names a DIFFERENT parameter/entry each row; a
        # bulleted list's leading marker ("•") recurs at a fixed left margin
        # too but is the SAME literal glyph every row, which is what actually
        # distinguishes a real label column from a bullet list wrapping across
        # many lines (confirmed real case: a bulleted list of stream-object
        # rules was read as a table whose single "KEY" was always "•").
        # A strict majority-unique test (not just "more than one distinct value
        # ever appears") is needed -- confirmed real case: a figure caption
        # line plus three bulleted sub-items had exactly 2 distinct key texts
        # ("FIGURE 9.15" and the bullet glyph) across 4 rows, which cleared a
        # bare ">1 distinct" bar while still being 75% the same repeated glyph.
        # key_diversity_ratio feeds the combined evidence score below
        # (alongside wrap presence) rather than vetoing here on its own --
        # a real KEY column names a DIFFERENT parameter/entry each row, so
        # low diversity is real negative evidence, but not on its own
        # decisive: see the score check further down for the case (a
        # filter's Table 3.9 opening with a single repeated-looking KEY)
        # this used to reject outright.
        key_texts_list = [min(lines[li], key=lambda s: s.bbox[0]).text.strip() for li in row_start_lines]
        key_diversity_ratio = len(set(key_texts_list)) / len(key_texts_list)

        # A repeated key that is a short, non-alphanumeric glyph (a bullet,
        # dash, or similar marker) is never a real parameter/entry name no
        # matter how the overall diversity ratio comes out -- this is a
        # categorical fact about what the KEY column contains, not a
        # strength-of-evidence question, so it stays a hard reject rather
        # than folding into the score (confirmed real case: a bulleted list
        # of stream-object rules was read as a table whose single "KEY" was
        # always "•"; a figure caption's own bullet list sat at the exact
        # same x as two ordinary paragraph lines just above it, and those
        # two genuinely-different sentences were enough "distinct" values
        # to inflate the diversity ratio despite the bullet itself
        # repeating 3 times).
        most_common_text, most_common_count = Counter(key_texts_list).most_common(1)[0]
        if most_common_count > 1 and len(most_common_text) <= 2 and not most_common_text.isalnum():
            return None

        # Consistent short columns after KEY (TYPE, and sometimes a further one
        # like an "OPI COMMENT" name), discovered one position at a time --
        # voted ONLY from row-start lines' own span at that position, so noise
        # from continuation-line prose keywords never enters this vote at all
        # (it only ever looks at lines already confirmed as row starts).
        # Confirmed real case: a 4-column dictionary -- KEY | TYPE | OPI COMMENT
        # | VALUE -- needs two extra columns found this way, not just one.
        extra_col_xs: list[float] = []
        pos = 1
        while True:
            pos_spans = []
            for li in row_start_lines:
                ln = sorted(lines[li], key=lambda s: s.bbox[0])
                if len(ln) > pos and len(ln[pos].text.strip()) <= _CELL_MAX_CHARS:
                    pos_spans.append(ln[pos])
            if len(pos_spans) < max(key_min_rows, int(len(row_start_lines) * 0.6)):
                break
            # A short span's OWN leading phrase (an italicized "(Optional)", a
            # PDF-version qualifier like "(Optional; PDF 1.2)") is routinely
            # split from the rest of the sentence into its own span purely by
            # styling -- that qualifier is still the START of the VALUE cell,
            # not a genuine further column. Every real TYPE/OPI-COMMENT-style
            # column value in this book's parameter dictionaries is a bare
            # word or identifier; a PARENTHESIZED qualifier is never one --
            # checking the candidate span's own leading character, rather than
            # trying to characterize whatever text happens to follow it (which
            # is unreliable: a real column's own trailing VALUE prose follows
            # it too, and looks the same locally), is what actually tells them
            # apart (confirmed real case: "(Optional)" was wrongly kept as an
            # extra column using a "does long prose follow" test, since real
            # TYPE columns like "boolean" are ALSO immediately followed by the
            # row's genuine long VALUE prose -- that pattern alone can't
            # distinguish a real last column from a fake one).
            if sum(1 for s in pos_spans if s.text.strip().startswith("(")) > len(pos_spans) * 0.5:
                break
            xs = sorted(s.bbox[0] for s in pos_spans)
            if xs[-1] - xs[0] > tol * 2:
                break
            extra_col_xs.append(sum(xs) / len(xs))
            pos += 1

        # This detector's whole premise is a genuine KEY/TYPE(/.../VALUE
        # definition table -- a real TYPE-like column recurring right after
        # KEY on every row. Without at least one, "recurring left margin" is
        # too weak a signal on its own and starts matching page furniture that
        # has nothing to do with a table at all (confirmed real case: a
        # figure's caption line and the NEXT section's heading happened to sit
        # at a similar indent below an image frame's top/bottom border rules,
        # with no TYPE-like column anywhere -- everything between them,
        # including an unrelated bulleted list, got welded into one bogus
        # "value" cell).
        if not extra_col_xs:
            return None

        # Abstain if the discovered extra column(s) ALSO recur on the wrapped
        # CONTINUATION lines (not just the row-start line) -- that means the
        # "extra column" is actually its own independently-wrapping data
        # column (several data columns, each listing its own values down
        # multiple lines in lockstep), a genuinely wider multi-column grid
        # this detector's model can't represent, not a "label(s) + one wrapped
        # trailing value" table (confirmed real case: a 4-column algorithm-
        # support matrix -- SubFilter value | three digest-algorithm columns --
        # had each of its 3 extra columns list several stacked values down
        # subsequent lines, and welding those into one trailing cell garbled
        # the whole table). In a genuine definition table, only the FINAL
        # (VALUE) cell ever continues past its row-start line; TYPE/OPI-COMMENT
        # -like columns appear exactly once per row.
        continuation_lines = [li for lo, hi in
                              [(row_start_lines[i], (row_start_lines[i + 1] - 1 if i + 1 < len(row_start_lines) else lo))
                               for i, lo in enumerate(row_start_lines)]
                              for li in range(lo + 1, hi + 1)]
        touches = sum(1 for li in continuation_lines
                     if any(abs(s.bbox[0] - cx) <= tol for s in lines[li] for cx in extra_col_xs))
        if continuation_lines and touches / len(continuation_lines) > 0.15:
            return None

        # Bounded by group_last_line, not just the raw frame's own extent --
        # a LATER group's row_start_lines (a second, genuinely separate
        # table sharing this same rule frame) must never be swallowed into
        # this group's own last row's trailing wrapped cell (confirmed real
        # case: Table 3.43's last row absorbed Table 3.44's header AND all
        # its data rows into one giant VALUE cell, because the frame-wide
        # last line was used for every group instead of stopping at the
        # boundary between them).
        frame_line_idxs = [li for li, ln in enumerate(lines) if ln and li <= group_last_line and
                           framed_lo - tol <= (min(s.bbox[1] for s in ln) + max(s.bbox[3] for s in ln)) / 2
                           <= framed_hi + tol]
        last_frame_line = max(frame_line_idxs) if frame_line_idxs else row_start_lines[-1]

        bands: list[tuple[int, int]] = []
        for i, lo in enumerate(row_start_lines):
            hi = row_start_lines[i + 1] - 1 if i + 1 < len(row_start_lines) else last_frame_line
            bands.append((lo, hi))

        # This detector exists specifically for rows whose trailing cell CAN
        # wrap across multiple lines -- if NONE of them ever do, the
        # leftmost-span voting has actually just rediscovered an ORDINARY
        # paragraph's own left margin (every line of a left-justified
        # paragraph starts at the same x, which trivially "recurs" and passes
        # the diversity check too, since each line begins with a different
        # word) rather than a real table (confirmed real case: a plain prose
        # paragraph below an unrelated small table, opened up by the no-
        # closing-rule retry above, got read as a table with one bogus 1-line
        # "row" per paragraph line). Requiring at least ONE real wrap, not a
        # MAJORITY, is what the check is actually testing for: an ordinary
        # misread paragraph can never show even a single wrapping band (by
        # construction, every one of its lines independently qualifies as
        # its own row-start), so any real wrap at all already rules that out
        # -- confirmed real case: a legitimate table mixing short one-line
        # entries (a boolean flag, a short date) with a couple of genuinely
        # long-wrapping ones had only 2 of 5 data rows wrap (40%), well under
        # a bare-majority bar, and was wrongly rejected outright. The table's
        # own HEADER band ("KEY TYPE VALUE") is excluded from this check --
        # it never wraps, even in a genuine table.
        # Zero wrapping stays a HARD, categorical reject rather than a
        # scored signal: an ordinary misread paragraph can never show even
        # ONE wrapping band (by construction, every one of its lines
        # independently qualifies as its own row-start), so zero wraps is
        # perfect evidence this is paragraph text, no matter how diverse
        # its "keys" look -- this is exactly the shape a purely-additive
        # score can't safely capture (a plain prose paragraph's leading
        # words are trivially all distinct, so a high key_diversity_ratio
        # would otherwise outweigh zero wrap evidence and wrongly pass;
        # confirmed real case: a plain prose paragraph below an unrelated
        # small table, opened up by the no-closing-rule retry above, got
        # read as a table with one bogus 1-line "row" per paragraph line).
        data_bands = bands[1:]
        if data_bands and not any(hi > lo for lo, hi in data_bands):
            return None

        # Beyond that hard floor, wrap_frac (how MUCH of the table wraps,
        # not just whether any of it does) becomes a scored signal
        # alongside key_diversity_ratio: a KEY column that's borderline on
        # diversity but wraps heavily is real positive evidence the
        # diversity-alone guard used to ignore entirely once it passed 0.5
        # -- confirmed real case: a legitimate table mixing short one-line
        # entries (a boolean flag, a short date) with a couple of
        # genuinely long-wrapping ones had only 2 of 5 data rows wrap
        # (40%), well under a bare-majority bar, and was previously
        # accepted only because diversity alone already cleared 0.5; this
        # score reaches the same accept with both signals contributing.
        wrap_frac = (sum(1 for lo, hi in data_bands if hi > lo) / len(data_bands)) if data_bands else 1.0
        score = _evidence_score(
            {"key_diversity": key_diversity_ratio, "wrap_frac": wrap_frac},
            {"key_diversity": 0.7, "wrap_frac": 0.3},
        )
        if score < 0.5:
            return None

        band_spans = [[s for li in range(lo, hi + 1) for s in lines[li]] for lo, hi in bands]
        all_band_spans = [s for spans in band_spans for s in spans]
        if not all_band_spans:
            return None

        key_members = [s for spans in band_spans for s in spans if abs(s.bbox[0] - key_x) <= tol]
        key_extent = (min(s.bbox[0] for s in key_members), max(s.bbox[2] for s in key_members))
        table_x0 = min(s.bbox[0] for s in all_band_spans)
        table_x1 = max(s.bbox[2] for s in all_band_spans)

        splits = [table_x0]
        prev_extent = key_extent
        for cx in extra_col_xs:
            members = [s for spans in band_spans for s in spans if abs(s.bbox[0] - cx) <= tol]
            if not members:
                continue
            extent = (min(s.bbox[0] for s in members), max(s.bbox[2] for s in members))
            splits.append((prev_extent[1] + extent[0]) / 2)
            prev_extent = extent
        splits.append(max(prev_extent[1] + pad, splits[-1] + pad))
        splits.append(max(splits[-1], table_x1))

        rbounds = [min(s.bbox[1] for s in all_band_spans) - pad]
        for i in range(len(bands) - 1):
            this_bot = max(s.bbox[3] for s in band_spans[i])
            next_top = min(s.bbox[1] for s in band_spans[i + 1])
            rbounds.append((this_bot + next_top) / 2)
        rbounds.append(max(s.bbox[3] for s in all_band_spans) + pad)

        table = Region(bbox=(splits[0], rbounds[0], splits[-1], rbounds[-1]), kind="table")
        for ri in range(len(rbounds) - 1):
            for ci in range(len(splits) - 1):
                table.cells.append(Region(
                    bbox=(splits[ci], rbounds[ri], splits[ci + 1], rbounds[ri + 1]),
                    kind="flow", table_row=ri, table_col=ci))
        return table

    global_frame_line_idxs = [li for li, ln in enumerate(lines) if ln and
                              framed_lo - tol <= (min(s.bbox[1] for s in ln) + max(s.bbox[3] for s in ln)) / 2
                              <= framed_hi + tol]
    global_last_line = max(global_frame_line_idxs) if global_frame_line_idxs else row_start_lines_all[-1]

    out: list[Region] = []
    for gi, row_start_lines in enumerate(row_groups):
        if len(row_start_lines) < key_min_rows:
            continue
        # This group's own last row must stop before the NEXT group's first
        # row-start line (a separate table's own header), not run all the
        # way to the shared frame's global extent.
        if gi + 1 < len(row_groups):
            next_start = row_groups[gi + 1][0]
            group_last_line = next_start - 1
            # Prefer this table's OWN closing rule, if one exists in the gap,
            # over blindly running all the way to just before the next
            # table's header -- an unrelated paragraph routinely sits
            # between the two (confirmed real case: "For Mac OS files, the
            # Mac entry..." plus the next table's own caption line, both
            # sitting between Table 3.43's real last row and Table 3.44's
            # header, got swallowed whole into that last row's VALUE cell
            # otherwise).
            lo_y = min(s.bbox[1] for s in lines[row_start_lines[-1]])
            hi_y = min(s.bbox[1] for s in lines[next_start]) if lines[next_start] else lo_y
            closing_ys = [f.bbox[1] for f in rule_cluster if lo_y - tol <= f.bbox[1] <= hi_y + tol]
            if closing_ys:
                cut_y = min(closing_ys)
                cut_line = next((li for li in range(next_start - 1, row_start_lines[-1] - 1, -1)
                                 if lines[li] and max(s.bbox[3] for s in lines[li]) <= cut_y + tol), None)
                if cut_line is not None:
                    group_last_line = cut_line
        else:
            group_last_line = global_last_line
        table = _build_one(row_start_lines, group_last_line)
        if table is not None:
            out.append(table)
    return out


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
    # Bands are EMPTY on the first pass and prose-derived on the second (see
    # parse_page). That ordering matters: a table's row is by definition a line
    # spanning the table's own columns, so splitting lines at a table's
    # internal gaps would make every row invisible. Splitting them at a PAGE
    # gutter is the opposite -- required, or a two-column page welds the left
    # column's prose into the right column's table and calls the result a row.
    # Deriving the bands from prose only is what separates the two cases
    # without a threshold: a table's own gaps cannot create a band, because a
    # table's spans are not in the sample the bands are computed from.
    lines = group_by_line(prim.spans)
    # group_by_line groups purely by y-proximity, with NO column awareness
    # at all -- on a multi-column page, a "line" at a given height can mix
    # spans from two DIFFERENT columns that merely share a y by coincidence
    # (confirmed real case: a 2-column arXiv paper's left-column running
    # prose and a table confined to the right column produced one merged
    # "line" pairing an unrelated sentence with that table's own row).
    # Splitting lines at the page's known column gutters was tried and
    # reverted: _column_boundaries requires a gutter to stay empty across
    # nearly the full content height, which a wide table spanning BOTH
    # columns elsewhere on the same page defeats (its own rows fill in
    # exactly the x-range a gutter would otherwise occupy), so it found no
    # gutter at all on the page that motivated this -- and gutter-splitting
    # still broke 4 previously-correct single-column tables in the
    # regression corpus, treating a real row's own wide cell-to-cell gap as
    # a column boundary. Column/2-column-mixing remains open; the fixes
    # below (band segmentation and per-band column re-voting) are
    # independent of it and validated separately.
    if len(lines) < min_rows:
        return []

    # 1. column votes: cell-like spans' edges, tagged by row. Tried BOTH
    #    right-edge (numeric/right-aligned columns, the common case) and
    #    left-edge (a table of short TEXT values, not numbers, is routinely
    #    left-aligned instead -- confirmed real case: a 4-column reference
    #    table where every cell starts at a fixed x but ends wherever its
    #    own text happens to). Whichever edge is wrong for a given table
    #    doesn't reliably fail closed: it can still find >= min_cols
    #    coincidentally-aligned edges and produce a plausible-looking but
    #    wrong grid (confirmed real case: right-edge "succeeded" with 6
    #    bogus columns on a left-aligned table, so a naive first-match-wins
    #    fallback never even tried left-edge). Scoring both by their own
    #    fill fraction and keeping the better-filled one is what actually
    #    tells a correct alignment from a coincidental one, since a wrong
    #    alignment scatters values into columns that don't line up as
    #    densely.
    rules = [f.bbox for f in prim.fills if f.is_rule]
    panels = [f.bbox for f in prim.fills if f.is_panel]

    # Definition-list-style tables (a wide trailing cell wraps across many
    # lines) are tried FIRST and take priority over the generic alignment
    # voter below when both fire on the same region. The voter's per-span
    # relaxation for long narrow spans (needed elsewhere for single-token
    # cells) can itself get fooled here: a standard boilerplate phrase
    # repeated at the start of nearly every VALUE cell (e.g. "(Optional;
    # PDF 1.2)") recurs at a consistent x purely because it's the same
    # phrase, not because it's a real column -- confirmed real case: the
    # voter "succeeded" with a plausible-looking but wrong 4-column split,
    # carving that boilerplate phrase into its own bogus column, while this
    # dedicated detector (which never lets ANY continuation-line span vote
    # at all, boilerplate or not) got the correct 3 columns for the same
    # rows.
    rule_fills = [f for f in prim.fills if f.is_rule]
    row_wrapped = _detect_row_wrapped_tables(lines, min_rows, tol, pad, rule_fills)

    # Right-edge and left-edge cover numeric/right-aligned and short-text/
    # left-aligned tables respectively -- but a simple 2-column summary
    # table (a colored header row, single key/value pairs) is routinely
    # CENTER-aligned instead, especially in generated/marketing-style
    # documents, and neither of the other two edges ever lines up for it
    # (confirmed real case: "Metric"/"Market Size"/"User Satisfaction"/
    # "Growth Rate" have left edges spanning 117-152 and right edges
    # spanning 205-240, yet all 4 share the exact same center x to within
    # a point). Center is tried as a third, equally-scored candidate.
    candidates: list[tuple[Region, float]] = []
    for edge in (lambda s: s.bbox[2], lambda s: s.bbox[0], lambda s: (s.bbox[0] + s.bbox[2]) / 2):
        candidates.extend(_detect_borderless_in_lines(lines, edge, min_rows, min_cols, tol, pad, rules,
                                                       prim.height, panels))

    # Different tables on the SAME page can each need a DIFFERENT edge
    # strategy (see above) -- picking one page-wide "best" edge and
    # discarding whatever the other two found was itself an un-scoped
    # decision, exactly the mistake this function's edge-selection is
    # supposed to avoid for a single table. An edge that correctly detects
    # THREE separate tables (each a solid but unspectacular fill_frac) used
    # to lose, in an averaged-score comparison across its own tables, to a
    # different edge that found only ONE strong-but-wrong table -- silently
    # discarding the other two genuine detections entirely (confirmed real
    # case: three stacked tables on one arXiv page). Keeping every
    # candidate from every edge and letting overlapping candidates compete
    # directly on their own fill_frac extends the SAME "best fill-fraction
    # wins" pattern already used for a single table's edge choice from "one
    # edge wins the whole page" to "the best candidate wins its own
    # region."
    voter_tables: list[Region] = []
    for table, _score in sorted(candidates, key=lambda ts: ts[1], reverse=True):
        if not any(containment(table.bbox, kept.bbox) > 0.3 or containment(kept.bbox, table.bbox) > 0.3
                   for kept in voter_tables):
            voter_tables.append(table)

    best_tables = list(row_wrapped)
    for t in voter_tables:
        if not any(containment(t.bbox, wt.bbox) > 0.3 or containment(wt.bbox, t.bbox) > 0.3
                   for wt in row_wrapped):
            best_tables.append(t)
    return _drop_column_straddlers(_drop_nested_page_straddlers(best_tables, prim), prim)


def _drop_nested_page_straddlers(tables: list[Region], prim: PagePrimitives) -> list[Region]:
    """Reject a table spanning both a nested page and its surroundings.

    A page printed inside another page is a separate physical page, and no
    table spans two pages. Checked separately from the column-gutter rule
    because that one returns early when the page has no gutters, and
    because such a table is exactly the one that looks legitimately
    full-width: confirmed real case (p99), a grid running x=43..679 across
    the inner page AND the teacher's margin swallowed the page's prose and
    dropped letter coverage to 0.457.
    """
    nested = nested_page_rect(prim)
    if nested is None:
        return tables
    nx0, _a, nx1, _b = nested
    out = []
    for t in tables:
        lo, hi = t.bbox[0] + 2.0, t.bbox[2] - 2.0
        if lo < nx0 < hi or lo < nx1 < hi:
            continue
        out.append(t)
    return out


def _drop_column_straddlers(tables: list[Region], prim: PagePrimitives) -> list[Region]:
    """Reject a borderless table that spans a page column gutter.

    group_by_line has no column awareness, so on a multi-column page it
    pools spans from BOTH columns into one "line". Alignments that are
    real within each column then look like one wide table, and the cells
    interleave unrelated content -- confirmed real case (Arabic teacher's
    guide p54): a lesson list in the left panel and a standards list in
    the right column fused into a 4-column grid whose cells read as
    scrambled fragments of both.

    A gutter is empty by construction, so nothing legitimately narrow
    straddles one. A genuinely full-width table does cross it, and stays:
    the test is crossing a gutter WITHOUT spanning essentially the whole
    content width.
    """
    if not tables:
        return tables
    boxes = [sp.bbox for sp in prim.spans]
    if not boxes:
        return tables
    bounds = _column_boundaries(boxes, prim.width, prim.height)
    if len(bounds) < 3:                       # single column -> no gutter
        return tables
    inner = [float(b) for b in bounds[1:-1]]
    x0 = min(b[0] for b in boxes)
    x1 = max(b[2] for b in boxes)
    content_w = x1 - x0
    if content_w <= 0:
        return tables
    kept = []
    for t in tables:
        w = t.bbox[2] - t.bbox[0]
        straddles = any(t.bbox[0] + 2.0 < g < t.bbox[2] - 2.0 for g in inner)
        if straddles and w < content_w * 0.9:
            continue
        # The gutters above are computed from EVERY span, the candidate's
        # own cells included -- so a table fusing two page columns supplies
        # the very evidence that makes it look full-width and exempt.
        # Recompute from the content OUTSIDE it: a gutter that survives
        # that is proven by unrelated prose, and nothing legitimately
        # spans one. Confirmed real case (p102): a "table" running the
        # full content width crossed a gutter at x=482 that the
        # surrounding prose keeps open, and it fused the teacher's margin
        # column with the lesson text.
        outside = [b for b in boxes
                   if not (t.bbox[0] - 2.0 <= (b[0] + b[2]) / 2 <= t.bbox[2] + 2.0
                           and t.bbox[1] - 2.0 <= (b[1] + b[3]) / 2 <= t.bbox[3] + 2.0)]
        if len(outside) > 3:
            ob = _column_boundaries(outside, prim.width, prim.height)
            if len(ob) >= 3 and any(t.bbox[0] + 2.0 < float(g) < t.bbox[2] - 2.0
                                    for g in ob[1:-1]):
                continue
        kept.append(t)
    return kept


def _detect_borderless_in_lines(lines: list[list[Span]], edge, min_rows: int, min_cols: int,
                                 tol: float, pad: float, rules: list[Rect] = (),
                                 page_bottom: float | None = None,
                                 panels: list[Rect] = ()) -> list[tuple[Region, float]]:
    # A genuine 2-column reference/glossary table (a symbolic code beside
    # its description, neither numeric) needs only 2 columns, but requiring
    # 3 unconditionally is what keeps pure coincidental alignment (two
    # prose columns that happen to line up) from being misread as a table.
    # Real drawn rules on the page are independent, author-provided
    # evidence that at least *some* genuine tabular structure exists here,
    # which is enough to safely relax that floor to 2 -- confirmed real
    # case: an escape-sequence table (SEQUENCE | MEANING) framed by real
    # top/bottom rules has exactly 2 columns and was rejected outright.
    min_cols = 2 if rules and min_cols > 2 else min_cols

    # The _CELL_MAX_CHARS cap on which spans may vote for a column exists to
    # keep wrapped PROSE lines from inventing spurious columns. But a table
    # cell can legitimately be a long *single token* (a hex-escaped literal
    # name like "/paired#28#29parentheses", 24 chars) -- capping by char
    # count alone drops those, so a middle row's cells stop voting, the row
    # stops counting as tabular, and the table's band breaks apart
    # (confirmed real case: a "LITERAL NAME | RESULT" example table went
    # undetected). Inside a rule-framed vertical span (strong, author-drawn
    # evidence of a real table there), also let a long span vote if it's
    # narrow relative to the page's content width -- a wrapped prose line
    # spans most of the width and still won't qualify, but a long-but-narrow
    # single-token cell does.
    all_spans = [s for ln in lines for s in ln if s.text.strip()]
    content_w = (max(s.bbox[2] for s in all_spans) - min(s.bbox[0] for s in all_spans)) if all_spans else 0.0
    rule_ys = [y for r in rules for y in (r[1], r[3])]
    framed_lo, framed_hi = (min(rule_ys), max(rule_ys)) if len(rule_ys) >= 2 else (0.0, -1.0)

    def votes_for_column(s: "Span", cell_w_frac: float = 0.5) -> bool:
        if len(s.text.strip()) <= _CELL_MAX_CHARS:
            return True
        cy = (s.bbox[1] + s.bbox[3]) / 2
        return (framed_lo <= cy <= framed_hi
                and (s.bbox[2] - s.bbox[0]) <= content_w * cell_w_frac)

    votes: list[tuple[float, int]] = []
    for li, ln in enumerate(lines):
        for s in ln:
            if votes_for_column(s):
                votes.append((edge(s), li))
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

    # A page can hold several DIFFERENT tables whose column x-positions
    # happen to partially coincide -- votes are pooled across the WHOLE
    # page (step 1 above), so a stray alignment between two unrelated
    # tables' columns can each independently clear min_cols and both land
    # in `tab`, even though they belong to different tables entirely
    # (confirmed real case: three stacked booktabs-style tables on one
    # arXiv page, each with 5-11 of its OWN numeric columns, shared just
    # enough coincidental column positions -- within a narrow paper's fixed
    # width -- that hits() alone couldn't tell them apart). A first attempt
    # fixed this by requiring adjacent tabular lines to share most of their
    # hit COLUMNS, not just clear the same count -- reverted after direct
    # testing found a real table it broke: a financial statement legitimately
    # alternates "$ amount" rows (7 populated columns) with "% growth rate"
    # rows (2 *different*, non-overlapping columns) for the SAME table, and
    # column-identity overlap can't tell that apart from two genuinely
    # different tables, since both look like ~0% column overlap between
    # adjacent rows. The actual distinguishing signal already exists one
    # level up: a new table always announces itself with its own numbered
    # caption line ("Table 2. ...", "Figure 3. ..."), which the sparse-row
    # case never has between it and its neighbor.
    def _has_section_break(prev: int, cur_li: int) -> bool:
        """A prose line between two tabular rows that clearly starts a NEW
        section -- a numbered "Table N."/"Figure N." caption, or a
        colon-terminated full sentence -- must hard-break a band even
        within the small gap otherwise tolerated for wrapped labels. A
        short colon-terminated LABEL ("Revenue:", "Costs and expenses:") is
        not this -- it's a legitimate divider row *within* the same table
        (confirmed against gold: it appears as its own row, blank-valued,
        inside one continuous table) and must not split anything. Word
        count is what tells them apart: a divider label is a few words; a
        real section intro is a full sentence (confirmed real case:
        "Share-based compensation expense included in costs and expenses:"
        at 9 words, introducing a wholly separate table two lines later,
        vs "Costs and expenses:" at 3 words, a divider inside the same
        table). The numbered-caption check is a second, independent signal
        (confirmed real case: three stacked "Table N."-captioned tables on
        one arXiv page, sharing coincidentally-similar column positions,
        need this to be told apart -- see above)."""
        # Scans through cur_li INCLUSIVE, not just strictly between --
        # group_by_line has no column awareness (see detect_borderless_
        # tables), so on a 2-column page a numbered caption belonging to
        # the right column can land merged onto the SAME "line" as an
        # unrelated left-column section heading, which is also cur_li
        # itself whenever that merged line happens to clear the min_cols
        # vote too (confirmed real case: "4.2. Training Details Table 3.
        # TapVid3D benchmark results..." -- the caption that should have
        # split table 2 from table 3 was sitting ON the boundary tabular
        # line, never checked by a range that stopped one line short of it).
        for li in range(prev + 1, cur_li + 1):
            ln = lines[li]
            if not ln:
                continue
            text = " ".join(s.text for s in sorted(ln, key=lambda s: s.bbox[0])).strip()
            if text.endswith(":") and len(text.split()) >= 6:
                return True
            # A genuine caption's number is followed immediately by a
            # period then non-digit ("Table 2.  Performance..."); an
            # ordinary cross-reference inside prose ("...see Table 8.109)")
            # continues into a section-numbered decimal instead -- the
            # trailing "\.(?!\d)" is what tells a real caption apart from a
            # citation that merely happens to contain the same two words
            # (confirmed real case: "(see Table 8.109)" inside a VALUE
            # cell's own description matched a search-anywhere check with
            # no such distinction, wrongly hard-breaking a real table in
            # the middle of one of its own rows).
            if re.search(r"(?:^|\s)(Table|Figure)\s+\d+\.(?!\d)", text):
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

    out: list[tuple[Region, float]] = []
    for band in bands:
        if len(band) < min_rows:
            continue
        lo, hi = band[0], band[-1]
        row_lines = lines[lo:hi + 1]                # include wrapped-label rows
        band_spans = [s for ln in row_lines for s in ln]

        # An unrelated intro paragraph just above the table (explaining an
        # abbreviation used in one of its columns, say) can share a short
        # coincidental alignment with a real column and get pulled into the
        # SAME band as a spurious leading "row" or two, pushing the band's
        # own top edge above the table's real top rule. Trim those leading
        # lines off BEFORE building the grid (not just relax the framing
        # check around them) so they don't render as garbled fake rows --
        # confirmed real case: a character-set table's first page had 2
        # intro lines ("U -- Undefined code point...") merged in this way;
        # every later continuation page of the SAME table, without that
        # intro, was already detected correctly.
        top_y = min(s.bbox[1] for s in band_spans)
        bot_y = max(s.bbox[3] for s in band_spans)
        # A colored PANEL enclosing this band (a header-row background, or
        # the table's own outer frame) is the same kind of independent,
        # author-drawn evidence as a rule -- some documents (generated
        # reports, marketing decks) draw table structure with filled
        # rectangles instead of stroked lines entirely, with no rule
        # anywhere near the table at all (confirmed real case: a simple
        # 2-column key/value table -- colored header background, bordered
        # outer panel -- had zero rule fills anywhere near it and was
        # rejected outright despite being unambiguously a real table).
        panel_frames = [(p[1], p[3]) for p in panels if p[1] <= top_y + tol * 2 and p[3] >= bot_y - tol * 2]
        top_framed = (any(abs(r[1] - top_y) <= tol * 5 or abs(r[3] - top_y) <= tol * 5 for r in rules)
                     or any(abs(py0 - top_y) <= tol * 5 for py0, py1 in panel_frames))
        bot_framed = (any(abs(r[1] - bot_y) <= tol * 5 or abs(r[3] - bot_y) <= tol * 5 for r in rules)
                     or any(abs(py1 - bot_y) <= tol * 5 for py0, py1 in panel_frames))
        if not bot_framed and page_bottom is not None:
            # Generous on purpose: a page's own bottom margin/footer area
            # (page number, running header) routinely eats 80-90pt, far
            # more than the tol*5 slack used for an actual drawn rule --
            # the min_rows/numeric-or-framed/fill-fraction guards further
            # below are what keeps this from accepting unrelated content,
            # not a tight distance here.
            bot_framed = (page_bottom - bot_y) <= tol * 15
        if not top_framed and bot_framed:
            interior_rule_ys = [r[1] for r in rules if top_y - tol <= r[1] <= bot_y + tol]
            if interior_rule_ys:
                # The TOPMOST interior rule is the table's own top border --
                # cutting there keeps the header row (right below it) as
                # part of the table, trimming only the unrelated content
                # further up that isn't bounded by any rule at all.
                cut_y = min(interior_rule_ys)
                new_lo = next((li for li in range(lo, hi + 1)
                              if lines[li] and min(s.bbox[1] for s in lines[li]) >= cut_y - tol), None)
                if new_lo is not None and lo < new_lo <= hi:
                    lo = new_lo
                    row_lines = lines[lo:hi + 1]
                    band_spans = [s for ln in row_lines for s in ln]
                    top_framed = True
        framed = top_framed and bot_framed

        # A table's actual column x-positions are best rediscovered from
        # its own rows alone, once its region is known -- the same
        # principle _cluster_rules already applies to ruled-table geometry
        # (confirmed real case: three stacked tables on one arXiv page
        # banded apart correctly, but each band's fill_frac still measured
        # a falsely low 0.27-0.53 using globally-derived centers, because a
        # different table's unrelated column pitch pulled them off-center).
        # But a local fit is *always* tighter than a global one, for a real
        # table or a coincidental one alike -- the same bias-variance
        # tradeoff any local-vs-global model fit has (confirmed real case:
        # the book's own back-of-book index, 2-column term/page-number
        # entries, and a stray math formula's subscript layout both scored
        # safely low under the global vote, using EITHER edge, but crept
        # past every fill_frac floor tried once centers were recomputed
        # locally). What actually and reliably distinguished every genuine
        # local-fit table found so far from every false positive found so
        # far is `framed`: real drawn rules independently confirm a table
        # exists there, and no false positive had enough real rules to
        # frame both edges of its band. So the local recompute is used ONLY
        # when framed, where it's validated safe; an unframed band falls
        # back to the ORIGINAL whole-page-scoped columns -- proven safe
        # across this whole corpus's history -- rather than trying to find
        # a fill_frac threshold that separates "framed" and "unframed" risk
        # profiles using the SAME locally-inflated metric (tried and
        # abandoned: no single floor worked, since local recompute inflates
        # fill_frac for coincidental unframed content too, and by varying
        # amounts across different kinds of coincidence).
        if framed:
            local_votes: list[tuple[float, int]] = []
            for li in range(lo, hi + 1):
                for s in lines[li]:
                    if votes_for_column(s):
                        local_votes.append((edge(s), li))
            local_votes.sort()
            local_clusters: list[dict] = []
            lcur: dict | None = None
            for x, li in local_votes:
                if lcur is not None and x - lcur["last"] <= tol:
                    lcur["xs"].append(x); lcur["lines"].add(li); lcur["last"] = x
                else:
                    lcur = {"xs": [x], "lines": {li}, "last": x}
                    local_clusters.append(lcur)
            # A small table (few voting rows) can lose a genuine column's
            # vote on just ONE row to source-PDF span merging: tight
            # kerning between adjacent numbers in a summary/total row
            # merges 2-3 logically separate values into one text run (see
            # _split_multi_cell_span), so that row casts fewer column
            # votes than it should. Requiring the fixed min_rows floor
            # from EVERY voting row, when the whole band only has a
            # handful of voting rows to begin with, makes that one row's
            # information loss fatal to any column it happened to merge
            # (confirmed real case: a 3-row reserves table -- 2 rows with
            # clean per-column spans, 1 with several merged -- lost every
            # column the merged row didn't separately vote for, cramming
            # several real columns' values into one cell). Scaling the
            # requirement to the band's OWN voting-row count (never below
            # 2, never above the global min_rows) tolerates exactly that
            # loss without loosening anything for a normal-sized table,
            # where the global floor is already <= this local one.
            # A line counts as a genuine DATA row here only if it casts
            # enough votes to look like one -- a wrapped row-label's own
            # continuation line ("Total" / "Affiliated" / "Companies" each
            # on their own line) contributes exactly ONE short-text vote,
            # which would otherwise inflate this count and push
            # local_min_rows right back up to the unrelaxed global floor,
            # silently defeating the whole point (confirmed while testing:
            # counting every voting line, label lines included, computed 6
            # "rows" for a table with only 3 real data rows, so local_min_
            # rows came out unchanged at 3 -- min_cols is the same bar
            # already used elsewhere to call a line tabular at all).
            line_vote_counts: dict[int, int] = {}
            for _, li in local_votes:
                line_vote_counts[li] = line_vote_counts.get(li, 0) + 1
            voting_lines = {li for li, cnt in line_vote_counts.items() if cnt >= min_cols}
            local_min_rows = min(min_rows, max(2, len(voting_lines) - 1))
            band_cols = [c for c in local_clusters if len(c["lines"]) >= local_min_rows]
            band_support = {id(c): len(c["lines"]) for c in band_cols}
        else:
            band_range = set(range(lo, hi + 1))
            band_cols = [c for c in real if c["lines"] & band_range]
            band_support = {id(c): len(c["lines"] & band_range) for c in band_cols}
            local_min_rows = min_rows
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
        max_support = max(band_support.values(), default=0)
        band_cols = [c for c in band_cols if band_support[id(c)] >= max(local_min_rows, max_support * 0.4)]
        if len(band_cols) < min_cols:
            continue
        centers = sorted(sum(c["xs"]) / len(c["xs"]) for c in band_cols)

        # Guard: the aligned cells must be predominantly numeric. This is what
        # separates a real data table from Arabic MCQ options / an answer key /
        # a two-column list that merely happens to line up. Without it, aligned
        # *words* get shredded into a garbage grid.
        aligned = [s for ln in row_lines for s in ln
                   if len(s.text.strip()) <= _CELL_MAX_CHARS
                   and any(abs(edge(s) - cx) <= tol for cx in centers)]
        if not aligned:
            continue
        numeric_frac = sum(_is_numeric_cell(s.text) for s in aligned) / len(aligned)
        # Guard: real drawn framing (or a strong majority-numeric grid) is
        # what separates a real data table from Arabic MCQ options / an
        # answer key / a two-column prose list that merely happens to line
        # up. Two attempts to fold this into a blended score with
        # fill_frac/column_support were tried and reverted after direct
        # testing against the corpus: (1) blending numeric_frac+framed with
        # fill/support let a PDF-syntax code example ("<< key1 value1 key2
        # value2 ... keyn valuen >>") pass, since a code illustration is
        # deliberately regular and scored fill=1.0/support=1.0 despite
        # numeric_frac=0.2 and framed=False; (2) blending fill_frac with
        # column_support alone (keeping this guard hard) still let the
        # book's own back-of-book INDEX pages pass as tables -- an index's
        # right-aligned page-number column has very high column_support
        # (0.92-0.94, it recurs on nearly every line) but a mostly-empty
        # grid otherwise (fill_frac 0.06-0.22), and support alone
        # outweighed that. Both were reverted; this stays a hard,
        # unscored gate, same as the original.
        if numeric_frac < 0.6 and not framed:
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
        # lines are excluded from the count). That case is already fixed by
        # the denominator restriction above, with no remaining confirmed
        # case of a real table sitting just under this floor -- so unlike
        # the definition-table detector's wrap/diversity signals below,
        # there's no evidence basis yet to soften this one into a score
        # (see the guard above for what happened when it was tried anyway).
        data_lines = [ln for li, ln in zip(range(lo, hi + 1), row_lines) if hits(li) >= min_cols]
        filled = sum(
            1
            for ln in data_lines
            for cx in centers
            if any(abs(edge(s) - cx) <= tol and len(s.text.strip()) <= _CELL_MAX_CHARS for s in ln)
        )
        fill_frac = filled / (len(data_lines) * len(centers)) if data_lines else 0.0
        if fill_frac < 0.5:
            continue

        # each column's left/right extent, from the spans that align to it
        extents = []
        for cx in centers:
            members = [sp for ln in row_lines for sp in ln if abs(edge(sp) - cx) <= tol]
            if not members:
                continue
            extents.append((min(sp.bbox[0] for sp in members), max(sp.bbox[2] for sp in members)))
        if len(extents) < min_cols:
            continue

        table_x0 = min(s.bbox[0] for s in band_spans)
        # vertical column splits: label | col0 | col1 | ... -- but only add a
        # leading "label" column if one actually exists (real, tol-sized gap
        # before the first detected column). Some tables have no such
        # column at all (their first detected column starts right at the
        # table's own left edge); always inserting one there produced a
        # degenerate, near-zero-width phantom first column and shifted
        # every real column's data over by one (confirmed real case: a
        # 4-column reference table came out as 5 columns, values in the
        # wrong cells).
        splits = ([min(table_x0, extents[0][0] - pad)] if extents[0][0] - pad - table_x0 <= tol
                 else [table_x0, extents[0][0] - pad])
        for j in range(len(extents) - 1):
            splits.append((extents[j][1] + extents[j + 1][0]) / 2)
        # The last column's right boundary, like the first column's left
        # boundary above, should reach the whole band's own extent, not
        # just its aligned members' -- a cell's content can spill into a
        # separate trailing span with a different style (an inline italic
        # run, say) that doesn't share the column's own alignment edge and
        # so never enters `extents`, and would fall outside a boundary
        # computed only from matched members (confirmed real case: "
        # (octal)" after an italic "ddd" landed outside its own cell and
        # was silently dropped).
        table_x1 = max(s.bbox[2] for s in band_spans)
        splits.append(max(extents[-1][1] + pad, table_x1))

        row_groups = _merge_wrapped_label_rows(row_lines, centers, tol, edge)
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
        out.append((table, fill_frac))
    return out


def _proximity_groups(boxes: list[Rect], xmult: float = 1.6, ymult: float = 2.2) -> list[list[Rect]]:
    """Union-find grouping of boxes that sit close enough to belong to the
    same figure. Distances are relative to the boxes' own median size, so
    this scales with the page's drawing scale rather than a fixed constant."""
    if not boxes:
        return []
    mw = float(np.median([b[2] - b[0] for b in boxes]))
    mh = float(np.median([b[3] - b[1] for b in boxes]))
    xtol, ytol = mw * xmult, mh * ymult
    parent = list(range(len(boxes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
            if dx <= xtol and dy <= ytol:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    out: dict[int, list[Rect]] = {}
    for i, b in enumerate(boxes):
        out.setdefault(find(i), []).append(b)
    return list(out.values())


def _chip_grid_tables(chip_boxes: list[Rect], min_rows: int = 2, min_cols: int = 3,
                      min_occupancy: float = 0.55) -> tuple[list[Region], set]:
    """Recover a REGULAR GRID of chips as a real table.

    A chip is a small filled box (see Fill.is_chip) -- an activity-number
    badge, but also, very commonly, one cell of a hand-drawn figure grid.
    Papers and Word-converted documents routinely draw a table as a field
    of uniformly-sized filled boxes with no rules at all, which the rule
    and alignment table detectors both miss (no rules to find; the text
    inside the boxes is short and irregular enough that column voting
    doesn't fire). Each box then surfaces as its own isolated region and
    the whole structure renders as a flat run of disconnected tokens --
    confirmed real cases: a BERT paper's input-representation figure
    (4 rows x 11 tokens: Input / Token / Segment / Position embeddings)
    and its fine-tuning figure, both of which came out as ~45 loose
    fragments in reading order with no row or column relationship left.

    The grid itself is the evidence, and it is strong: real rows share a
    y-centre, real columns share an x-centre, and a genuine grid fills
    most of its own row x column product. Scattered chips -- an ordinary
    document's activity badges, one per exercise -- satisfy none of that,
    which is what keeps this from firing on them.

    Returns (table regions, set of chip bboxes consumed).
    """
    used: set = set()
    tables: list[Region] = []
    for group in _proximity_groups(chip_boxes):
        if len(group) < min_rows * min_cols:
            continue
        mh = float(np.median([b[3] - b[1] for b in group]))
        mw = float(np.median([b[2] - b[0] for b in group]))
        rows = _cluster_1d_boxes(group, key=lambda b: (b[1] + b[3]) / 2, tol=mh * 0.6)
        cols = _cluster_1d_boxes(group, key=lambda b: (b[0] + b[2]) / 2, tol=mw * 0.6)
        if len(rows) < min_rows or len(cols) < min_cols:
            continue
        if len(group) < len(rows) * len(cols) * min_occupancy:
            continue
        row_centers = sorted(sum((b[1] + b[3]) / 2 for b in r) / len(r) for r in rows)
        col_centers = sorted(sum((b[0] + b[2]) / 2 for b in c) / len(c) for c in cols)
        rb = _midpoint_bounds(row_centers, min(b[1] for b in group), max(b[3] for b in group))
        cb = _midpoint_bounds(col_centers, min(b[0] for b in group), max(b[2] for b in group))
        table = Region(bbox=(cb[0], rb[0], cb[-1], rb[-1]), kind="table")
        for ri in range(len(rb) - 1):
            for ci in range(len(cb) - 1):
                table.cells.append(Region(bbox=(cb[ci], rb[ri], cb[ci + 1], rb[ri + 1]),
                                          kind="flow", table_row=ri, table_col=ci))
        tables.append(table)
        used.update(group)
    return tables, used


def _cluster_1d_boxes(boxes, key, tol):
    """Greedy proximity clustering of boxes along one axis."""
    groups: list[list] = []
    for b in sorted(boxes, key=key):
        if groups and key(b) - key(groups[-1][-1]) <= tol:
            groups[-1].append(b)
        else:
            groups.append([b])
    return groups


def _midpoint_bounds(centers: list[float], lo: float, hi: float) -> list[float]:
    """Cell boundaries midway between consecutive cluster centres."""
    return [lo] + [(a + b) / 2 for a, b in zip(centers, centers[1:])] + [hi]


def propose_regions(prim: PagePrimitives, min_panel_area: float = 2000.0) -> list[Region]:
    regions: list[Region] = []

    tables = detect_tables(prim)
    # borderless detection only where a drawn-rule table doesn't already cover
    # the area, so the two never fight over the same grid
    for bt in detect_borderless_tables(prim):
        # A real table's OWN subtotal/total row is routinely underlined with
        # an actual drawn rule -- detect_tables then finds a tiny "table"
        # covering just that one ruled row, entirely inside the bigger table
        # the borderless voter (correctly) found around it. Discarding the
        # borderless candidate whenever it overlaps ANY ruled table treated
        # that as a duplicate and kept only the tiny fragment (confirmed
        # real case: a 9-column/11-row employee-demographics table -- header
        # plus 6 data rows plus a 3-row ruled "Total" section -- collapsed to
        # just its own 3-row totals section, losing the header and all 6
        # data rows, because the totals section's own underline made
        # detect_tables see a small table there first). When the ruled
        # table(s) a candidate overlaps are near-fully CONTAINED inside it
        # (not just overlapping it) and it's strictly more complete, it's
        # the ruled table that's redundant, not the borderless one -- drop
        # the fragment(s) in favor of the region that actually has them
        # covered.
        #
        # The area check matters: high mutual containment alone doesn't
        # distinguish "small fragment inside a big table" from "same table,
        # both detectors found roughly the same extent" -- two boxes that
        # differ only by a few points of tolerance/padding slop are >=0.85
        # contained in EACH OTHER too. Only when the ruled table covers a
        # clearly smaller fraction of the candidate's area is it a genuine
        # sub-fragment; otherwise the ruled table's own drawn rules are
        # still the stronger evidence and should win as before (confirmed
        # real case: a definition-list table where the borderless voter
        # covered virtually the same box as the ruled table -- ratio 1.04,
        # not a fragment -- but split a wrapped description line into a
        # spurious extra column; the correctly-ruled 18-cell grid must win
        # over the borderless voter's worse 24-cell one there).
        def _area(b):
            return (b[2] - b[0]) * (b[3] - b[1])

        subsumed = [t for t in tables
                    if containment(t.bbox, bt.bbox) >= 0.85 and _area(t.bbox) <= 0.7 * _area(bt.bbox)]
        if subsumed and len(bt.cells) > sum(len(t.cells) for t in subsumed):
            for t in subsumed:
                tables.remove(t)
            tables.append(bt)
        elif not any(containment(bt.bbox, t.bbox) > 0.3 or containment(t.bbox, bt.bbox) > 0.3
                     for t in tables):
            tables.append(bt)
    regions.extend(tables)

    panels = _merge_nested([f for f in prim.fills if f.is_panel and f.area >= min_panel_area])
    for f in panels:
        regions.append(Region(bbox=f.bbox, kind="panel", fill_color=f.color))

    chip_boxes = [f.bbox for f in prim.fills if f.is_chip]
    grid_tables, gridded = _chip_grid_tables(chip_boxes)
    for gt in grid_tables:
        if not any(containment(gt.bbox, t.bbox) > 0.3 or containment(t.bbox, gt.bbox) > 0.3
                   for t in tables):
            regions.append(gt)
    for f in prim.fills:
        if f.is_chip and f.bbox not in gridded:
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


def _split_multi_cell_span(s: Span, tables: list[Region]) -> list[Span]:
    """Split a span holding multiple DIFFERENT table cells' values drawn as
    one PDF text run, e.g. "904 19,146" for two adjacent numeric columns
    whose gap was too tight for the PDF generator to break into separate
    text-showing runs.

    Unlike _split_repeated_span (identical value repeated N times, split
    evenly), here the values differ and the table's own already-detected
    cell boundaries are the only reliable place to split -- confirmed real
    case: a reserves table's per-region DETAIL rows (each column a separate,
    well-spaced span) column-align cleanly, but its "Total Consolidated"
    summary rows use tighter kerning between adjacent numbers, so PyMuPDF's
    own span extraction merges 2-3 of them into one run each; naive
    containment then dumps the merged run into whichever single cell it
    overlaps most, cramming several columns' worth of numbers into one and
    leaving the neighboring cells blank. Two guards keep this narrow:
    word count must EXACTLY match the number of cells the span
    geometrically straddles in its own row, AND every word must look
    numeric. The word-count match alone isn't enough -- a citation-bracketed
    model name ("DepthAnythingV3 [ 23 ]") or a reference table's descriptive
    cell ("0006 U+0006 (ACKNOWLEDGE)") can coincidentally have the same word
    count as the columns they straddle and are a single semantic value, not
    several -- confirmed real regressions where splitting either corrupted
    the cell instead of fixing it. A genuinely merged multi-column run is,
    like the columns it belongs to, numeric.
    """
    for t in tables:
        if containment(s.bbox, t.bbox) <= 0.6:
            continue
        cy = (s.bbox[1] + s.bbox[3]) / 2
        row_cells = sorted((c for c in t.cells if c.bbox[1] - 1 <= cy <= c.bbox[3] + 1),
                           key=lambda c: c.bbox[0])
        straddled = [c for c in row_cells if s.bbox[0] < c.bbox[2] and s.bbox[2] > c.bbox[0]]
        if len(straddled) < 2:
            continue
        words = s.text.split()
        if len(words) != len(straddled) or not all(_is_numeric_cell(w) for w in words):
            continue
        return [Span(text=w, bbox=(c.bbox[0], s.bbox[1], c.bbox[2], s.bbox[3]),
                     font=s.font, size=s.size, color=s.color, flags=s.flags, dir=s.dir)
               for w, c in zip(words, straddled)]
    return [s]


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
    def _row_match_frac(smaller: list[list[Span]], larger: list[list[Span]]) -> float:
        """Fraction of lines in `smaller` that have a same-row (y-matching)
        line in `larger`. High on a genuinely row-paired layout even when
        NEITHER side is short -- e.g. a code listing's variable-indent
        lines each paired with their own "% comment" on the same row,
        where the comment is routinely the longer of the two. This is what
        lets that case merge too, not just the short-marker one."""
        if not smaller:
            return 0.0
        hits = 0
        for ln in smaller:
            yc = sum((s.bbox[1] + s.bbox[3]) / 2 for s in ln) / len(ln)
            size = sum(s.size for s in ln) / len(ln)
            if any(abs(sum((t.bbox[1] + t.bbox[3]) / 2 for t in ln2) / len(ln2) - yc) <= size * 0.6
                   for ln2 in larger):
                hits += 1
        return hits / len(smaller)

    cols = sorted(by_col)
    for i in range(len(cols) - 1):
        a, b = cols[i], cols[i + 1]
        a_lines, b_lines = group_by_line(by_col[a]), group_by_line(by_col[b])
        if len(a_lines) < min_rows or len(b_lines) < min_rows:
            continue
        merged = False
        for src, dst, src_lines, dst_lines in ((a, b, a_lines, b_lines), (b, a, b_lines, a_lines)):
            short = sum(1 for ln in src_lines
                       if len(" ".join(s.text for s in ln).strip()) <= _CELL_MAX_CHARS)
            if short / len(src_lines) < short_frac:
                continue
            by_col[dst] = by_col[dst] + by_col[src]
            by_col[src] = []
            merged = True
            break
        if merged:
            continue
        # Neither side qualifies as a short marker column -- check for
        # strict row-correspondence instead (same signal, without the
        # length requirement). But row-correspondence ALONE isn't enough:
        # two ordinary, roughly-equal-width columns of running prose (an
        # academic paper's standard 2-column layout) naturally have most
        # of their rows land within a line-height of each other too, since
        # both columns share the same font size and leading throughout the
        # page -- that's an artifact of consistent typesetting, not
        # evidence the two columns are actually one row-paired unit
        # (confirmed real case: a 2-column arXiv paper's genuinely
        # independent columns scored >=0.7 row-match purely from shared
        # line-height, and got wrongly merged into one page-wide flow
        # region, collapsing the whole page to 1 detected column). A real
        # marker/content pairing (or code/comment pairing) is asymmetric
        # in WIDTH even when neither side is short enough to trip the
        # length check above -- requiring the narrower column to be
        # meaningfully narrower, not just "happens to have fewer or
        # shorter lines," is what a normal 2-column paper's near-equal
        # widths never satisfy.
        # Width is measured as the MEDIAN individual line width within
        # each column, not the column's overall bounding-box span (max x1
        # - min x0) -- the latter is fragile to a single outlier line
        # (confirmed real case: a table's own wide caption, already
        # excluded from gutter detection by a similar percentile filter in
        # _column_boundaries, still sat in this column's raw span list
        # here and alone inflated its bounding-box width enough to make
        # two ordinary, equal-width prose columns look asymmetric -- 0.68
        # ratio, under the 0.7 bar -- and get wrongly merged anyway). The
        # median is what most of a column's OWN lines actually look like,
        # immune to one outlier caption/heading line the same way it's
        # immune in _column_boundaries.
        smaller, larger, src, dst = ((a_lines, b_lines, a, b) if len(a_lines) <= len(b_lines)
                                     else (b_lines, a_lines, b, a))
        smaller_w = float(np.median([max(s.bbox[2] for s in ln) - min(s.bbox[0] for s in ln)
                                     for ln in smaller])) if smaller else 0.0
        larger_w = float(np.median([max(s.bbox[2] for s in ln) - min(s.bbox[0] for s in ln)
                                    for ln in larger])) if larger else 0.0
        if (larger_w > 0 and smaller_w / larger_w < 0.7
                and _row_match_frac(smaller, larger) >= 0.7):
            by_col[dst] = by_col[dst] + by_col[src]
            by_col[src] = []


def _find_disjoint_column_split(cells: list[Region], max_collision_frac: float = 0.2) -> int | None:
    """First column index k such that almost no row has content on both
    sides of it -- see split_disjoint_tables for what this signals.

    Not a strict zero: a wrapped 2-line entry's own continuation (e.g. a
    page number wrapping onto its own line) can coincidentally land on the
    same row-band as the NEXT column's unrelated entry, purely from
    row-clustering tolerance (confirmed real case: a table-of-contents
    entry's wrapped page number sat on the same row as an unrelated
    right-column "Note 18" entry -- 1 collision out of 9 rows). Requiring
    the collision rate to stay low, rather than absent, tolerates that kind
    of noise without accepting a genuinely single, correspondent table
    (which collides on nearly every row by definition).
    """
    if not cells:
        return None
    ncols = max(c.table_col for c in cells) + 1
    nrows = max(c.table_row for c in cells) + 1
    if ncols < 2:
        return None
    has = [[False] * ncols for _ in range(nrows)]
    for c in cells:
        if c.spans:
            has[c.table_row][c.table_col] = True
    for k in range(1, ncols):
        left_rows = [any(has[ri][ci] for ci in range(k)) for ri in range(nrows)]
        right_rows = [any(has[ri][ci] for ci in range(k, ncols)) for ri in range(nrows)]
        populated = sum(l or r for l, r in zip(left_rows, right_rows))
        collisions = sum(l and r for l, r in zip(left_rows, right_rows))
        if populated == 0 or collisions / populated > max_collision_frac:
            continue
        if sum(left_rows) >= 2 and sum(right_rows) >= 2:
            return k
    return None


# A recovered header row must sit within this multiple of the table's own row
# height above it, and cover this share of its columns. Both are expressed
# against the table, not in points, so the rule holds at any type size.
HEADER_GAP_ROWS = 1.6
HEADER_COL_COVER = 0.5


def _cell_shape(text: str) -> tuple:
    """Coarse character class of a cell, for comparing one row against another."""
    t = text.strip()
    return (bool(re.search(r"\d", t)),
            t.isupper() and len(t) > 1,
            0 if not t else (1 if len(t) <= 12 else (2 if len(t) <= 40 else 3)))


def _rows_alike(row_a: list[Region], row_b: list[Region], region: Region,
                frac: float = 0.6) -> bool:
    """Do two table rows share a character shape, column for column?"""
    def cells(row):
        return {c.table_col: "".join(sp.text for sp in (c.spans or [])).strip()
                for c in row}
    a, b = cells(row_a), cells(row_b)
    common = [k for k in a if k in b and (a[k] or b[k])]
    if not common:
        return False
    same = sum(1 for k in common if _cell_shape(a[k]) == _cell_shape(b[k]))
    return same >= max(1, len(common) * frac)


def recover_header_rows(tables: list[Region], prim: PagePrimitives) -> None:
    """Pull back a header row the rule clustering left outside the table.

    A table's header is very often styled differently from its body -- shaded,
    boxed, ruled with a heavier or a differently-drawn line -- and when that
    styling changes how its rules cluster, the header row falls outside the
    detected region. The first DATA row then becomes the header, and every
    value in the table keys to the wrong column.

    That failure is invisible in the text (every cell is extracted perfectly)
    and total in meaning: measured on a hand-verified gold set, it is the
    single largest table defect, worth 18.5 points of TableRecordMatch across
    43 tables. Confirmed case: a 5-row questionnaire whose region began 13pt
    below its own header, so "RQ1" became the column name for "RQ".

    The recovery is geometric and needs no styling rule: look at the strip
    immediately above the table, and take it only if its words line up with
    the columns the table already has. Alignment with an existing grid is
    strong evidence -- ordinary prose above a table does not land in its
    column bands -- and it is what keeps a caption or a paragraph out.
    """
    if not prim.spans:
        return
    for region in tables:
        if not region.cells:
            continue
        # Anchor on the first row that actually carries text. Rule clustering
        # routinely leaves one or two empty slivers at the top of a table, and
        # anchoring on those compares two blank rows and sizes the search band
        # from a 2pt height.
        by_row: dict[int, list[Region]] = {}
        for cell in region.cells:
            if cell.table_row is not None:
                by_row.setdefault(cell.table_row, []).append(cell)
        filled = [r for r in sorted(by_row)
                  if any("".join(sp.text for sp in (c.spans or [])).strip()
                         for c in by_row[r])]
        if len(filled) < 2:
            continue
        top_row = filled[0]
        first = by_row[top_row]
        if len(first) < 2:
            continue
        # Only recover when the table's own first row reads as DATA. If row 0
        # already looks unlike row 1 -- different character classes, different
        # lengths -- it is the header and there is nothing missing; reaching
        # above it then drags in the caption or the running head instead
        # ("TABLE 3.28 Entries in...", "SECTION 3.3"). A header is missing
        # precisely when row 0 is indistinguishable from the row beneath it,
        # as "RQ1" is from "RQ2".
        second = by_row[filled[1]]
        if not _rows_alike(first, second, region):
            continue
        # Row height from ALL the table's rows, not just its top one: rule
        # clustering can leave a 2pt sliver as row 0, and sizing the search
        # band from that looks 3pt above the table and finds nothing. Floored
        # by the page's own line height so a degenerate table still searches a
        # sensible strip.
        heights = [c.bbox[3] - c.bbox[1] for c in region.cells if c.bbox[3] > c.bbox[1]]
        line_h = statistics.median(
            [sp.bbox[3] - sp.bbox[1] for sp in prim.spans if sp.bbox[3] > sp.bbox[1]]
        ) if prim.spans else 10.0
        row_h = max(statistics.median(heights) if heights else 0.0, line_h)
        y1 = min(c.bbox[1] for c in first)
        y0 = y1 - row_h * HEADER_GAP_ROWS
        x0, x1 = region.bbox[0], region.bbox[2]

        band = [sp for sp in prim.spans
                if sp.bbox[3] <= y1 + 1 and sp.bbox[1] >= y0 - 1
                and sp.bbox[0] >= x0 - 4 and sp.bbox[2] <= x1 + 4
                and sp.text.strip()]
        if not band:
            continue
        # every column the strip actually lands in
        hit = set()
        for sp in band:
            cx = (sp.bbox[0] + sp.bbox[2]) / 2
            for c in first:
                if c.bbox[0] - 2 <= cx <= c.bbox[2] + 2:
                    hit.add(c.table_col)
                    break
        if len(hit) < max(2, len(first) * HEADER_COL_COVER):
            continue
        # A strip that also extends well beyond the table's own width is a
        # caption or running text that happens to pass overhead, not a header.
        if min(sp.bbox[0] for sp in band) < x0 - 6 or max(sp.bbox[2] for sp in band) > x1 + 6:
            continue

        new_top = min(sp.bbox[1] for sp in band)
        header_cells = []
        for c in first:
            cell = Region(bbox=(c.bbox[0], new_top, c.bbox[2], y1),
                          kind="table_cell", table_row=top_row - 1,
                          table_col=c.table_col)
            cell.spans = [sp for sp in band
                          if c.bbox[0] - 2 <= (sp.bbox[0] + sp.bbox[2]) / 2 <= c.bbox[2] + 2]
            header_cells.append(cell)
        if not any(c.spans for c in header_cells):
            continue
        region.cells.extend(header_cells)
        # Renumber so rows start at 0 again -- the grid builder sizes itself
        # from max(table_row) and indexes directly, so a negative row would
        # write the header into the last row instead of the first.
        shift = -min(c.table_row for c in region.cells if c.table_row is not None)
        if shift:
            for c in region.cells:
                if c.table_row is not None:
                    c.table_row += shift
        region.bbox = (region.bbox[0], new_top, region.bbox[2], region.bbox[3])
        region.spans = list(region.spans) + list(band)


def split_disjoint_tables(regions: list[Region]) -> list[Region]:
    """A page can lay two INDEPENDENT lists side by side purely to save
    space -- a table-of-contents' "Title .... page#" list beside an
    unrelated "Note N .... page#" list, say -- and column detection has no
    way to know they aren't one table, merging them into a single grid
    where every row is half-blank (confirmed real case: a 10-K's table of
    contents, two lists with different line-wrapping rhythms and no
    row-for-row relationship, merged into one 4-column table with every
    row populated on only one side). The tell is structural, not textual:
    a real table's rows populate multiple columns TOGETHER, by definition.
    If no row ever has content on both sides of some column boundary, this
    was never one table -- splitting there recovers two coherent tables
    instead of one riddled with holes. Must run after assign_spans (needs
    to know which cells actually have content) and before order_regions
    (so each half gets its own reading-order position).
    """
    out = []
    for r in regions:
        if r.kind != "table" or not r.cells:
            out.append(r)
            continue
        k = _find_disjoint_column_split(r.cells)
        if k is None:
            out.append(r)
            continue
        for cells in ([c for c in r.cells if c.table_col < k],
                      [c for c in r.cells if c.table_col >= k]):
            x0 = min(c.bbox[0] for c in cells); y0 = min(c.bbox[1] for c in cells)
            x1 = max(c.bbox[2] for c in cells); y1 = max(c.bbox[3] for c in cells)
            out.append(Region(bbox=(x0, y0, x1, y1), kind="table", cells=cells))
    return out


def assign_spans(prim: PagePrimitives, regions: list[Region], thresh: float = 0.6) -> list[Region]:
    """Drop every span into its tightest containing region; leftovers become
    free-flow regions clustered by line proximity, column by column."""
    tables = [r for r in regions if r.kind == "table"]
    chips = [r for r in regions if r.kind == "chip"]
    containers = [r for r in regions if r.kind in ("panel", "figure")]
    orphans: list[Span] = []

    spans = [sub2 for s in prim.spans
             for sub in _split_repeated_span(s, tables)
             for sub2 in _split_multi_cell_span(sub, tables)]
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
                    if c > best_score + 1e-6:
                        best_cell, best_score = cell, c
                    elif best_cell is not None and abs(c - best_score) <= 1e-6:
                        # A near-exact tie in containment -- a span
                        # straddling two adjacent cells almost evenly (e.g.
                        # a 5-digit value split ~50/50 across a column
                        # boundary) -- breaks in cell-iteration order with
                        # no signal, which can land it beside the wrong
                        # neighbor (confirmed real case: "1,062" tied at
                        # containment 0.5 between its own cell and the one
                        # to its left, landing left and cramming into that
                        # cell's value). A real table's numeric cells are
                        # right-aligned almost universally -- the same
                        # principle column detection itself already relies
                        # on -- so on a tie, prefer whichever cell's right
                        # edge the span's own right edge actually lines up
                        # with.
                        if abs(s.bbox[2] - cell.bbox[2]) < abs(s.bbox[2] - best_cell.bbox[2]):
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
    #
    # Boundaries are computed from `orphans` (the flowing prose text still
    # needing a column) rather than every span on the page -- a table
    # commonly breaks out to the FULL content width regardless of the
    # surrounding page's column layout, and its many individual (narrow)
    # cell spans collectively fill in the gutter region row after row, the
    # same way a wide single span would (confirmed real case: a WHO report
    # page with genuine 2-column running text above a full-width table
    # found NO gutter at all when table-cell spans were included, and the
    # whole page's prose got wrongly read as one column; excluding them
    # recovers the gutter the surrounding paragraphs actually have). A
    # table's own internal cell layout has nothing to say about whether
    # the PROSE around it is 1- or 2-column.
    # Text rotated 90 degrees is never part of the body reading flow -- on a
    # paper it is the publisher's margin stamp ("arXiv:2608.03477v1 [cs.DB]
    # 4 Aug 2026"), on a report a watermark or a side tab. It has to be kept
    # out of the horizontal flow clustering for two separate reasons. Its
    # bbox is a tall narrow strip spanning most of the page height, so it
    # both distorts gutter detection and, being vertically adjacent to
    # everything, merges into whichever body block it happens to touch --
    # confirmed real cases: a stamp swallowed into the document's own
    # "Contents" heading, and into a title, producing headings like
    # "arXiv:2608.03477v1 [cs.DB] 4 Aug 2026 Contents". Clustered on their
    # own they stay a separate region, which _fallback_role then types as
    # page furniture.
    rotated_orphans = [s for s in orphans if abs(s.dir[1]) > abs(s.dir[0])]
    if rotated_orphans:
        rot = set(map(id, rotated_orphans))
        orphans = [s for s in orphans if id(s) not in rot]
        for r in _cluster_flow(rotated_orphans, gutters=prim.column_gutters):
            r.kind = "rotated"
            regions.append(r)

    boundaries = _column_boundaries([s.bbox for s in orphans], prim.width, prim.height)
    by_col: dict[int, list[Span]] = {}
    for s in orphans:
        col = _column_of((s.bbox[0] + s.bbox[2]) / 2, boundaries)
        by_col.setdefault(col, []).append(s)
    _merge_marker_columns(by_col)
    flows: list[Region] = []
    for col, spans in by_col.items():
        for r in _cluster_flow(spans, gutters=prim.column_gutters):
            r.column = col
            flows.append(r)

    # A flow region whose bbox sits wholly inside another flow region's is
    # not a separate paragraph -- it is part of the same one, split off by
    # the column bucketing above. A full-width caption on a 2-column page
    # gets its spans bucketed by x like everything else, so the words that
    # happen to start past the gutter cluster into their own region nested
    # inside the caption's. Both then render: the outer one from the
    # geometric lines it owns (the whole caption) and the inner one from
    # its own raw spans (a subset of that same text), so those words come
    # out twice (confirmed real case: a BERT figure caption whose "[CLS] is
    # a special" / "[SEP] is a special separator token" fragments, set in a
    # different font mid-sentence, were emitted again under the full
    # caption). Merging the nested region into its container leaves one
    # region owning the geometric lines, which already carry that text.
    absorbed = set()
    for i, inner in enumerate(flows):
        for j, outer in enumerate(flows):
            if i == j or j in absorbed or i in absorbed:
                continue
            if containment(inner.bbox, outer.bbox) >= 0.99:
                outer.spans.extend(inner.spans)
                x0 = min(outer.bbox[0], inner.bbox[0]); y0 = min(outer.bbox[1], inner.bbox[1])
                x1 = max(outer.bbox[2], inner.bbox[2]); y1 = max(outer.bbox[3], inner.bbox[3])
                outer.bbox = (x0, y0, x1, y1)
                absorbed.add(i)
                break
    regions.extend(r for k, r in enumerate(flows) if k not in absorbed)

    return [r for r in regions if r.spans or r.kind in ("figure", "table")]


_LIST_MARKER_RE = re.compile(r"^[•‣◦●○▪▫∙·oO*\-–—]$|^\(?[0-9]{1,3}([.)]|[-–—][0-9]{1,3})?$|^\(?[a-zA-Z][.)]$")


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


LINE_STYLE_PURITY = 0.85


def _line_style(line: list[Span]) -> str | None:
    """The style that sets nearly all of one line, or None if it is mixed.

    Returning None for a mixed line is deliberate: a sentence with a bold term
    inside it is not a style change, and treating it as one would split every
    paragraph that emphasises a word.
    """
    tally: dict[str, int] = {}
    for sp in line:
        n = len(sp.text.strip())
        if n:
            tally[sp.style_key] = tally.get(sp.style_key, 0) + n
    total = sum(tally.values())
    if not total:
        return None
    key = max(tally, key=tally.get)
    return key if tally[key] >= total * LINE_STYLE_PURITY else None


def _cluster_flow(spans: list[Span], gap_mult: float = 1.6, size_ratio: float = 1.3,
                  bands: list[tuple[float, float]] | None = None,
                  gutters: list[tuple[float, float, float]] | None = None) -> list[Region]:
    """Greedy line-then-paragraph clustering for text outside any drawn box."""
    if not spans:
        return []
    ordered = group_by_line(spans, bands=bands, gutters=gutters)
    ordered.sort(key=lambda ls: min(s.bbox[1] for s in ls))
    heights = [np.median([s.bbox[3] - s.bbox[1] for s in ls]) for ls in ordered]
    lead = float(np.median(heights)) if heights else 10.0
    sizes = [float(np.median([s.size for s in ls])) for ls in ordered]

    groups: list[list[Span]] = []
    prev_bottom = None
    prev_xrange = None
    prev_size = None
    prev_style = None
    # the style that sets most of these spans' characters
    _tally: dict[str, int] = {}
    for _sp in spans:
        _n = len(_sp.text.strip())
        if _n:
            _tally[_sp.style_key] = _tally.get(_sp.style_key, 0) + _n
    body_style = max(_tally, key=_tally.get) if _tally else None
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
        # A STYLE change is a semantic break the size ratio cannot see. A
        # sub-heading set in the same family one step heavier and a sixth
        # larger (9.3pt bold over 8pt regular, a ratio of 1.16) sails under
        # any size threshold loose enough to be safe, and is then swallowed by
        # the paragraph beneath it -- confirmed case, four sub-headings on one
        # page absorbed into the following body text, invisible to every
        # downstream role rule because they were never their own region.
        #
        # Only a WHOLE line in a different style counts. An inline bold run
        # inside a sentence shares its line with body text and must not split
        # a paragraph, so the line's dominant style has to own nearly all of
        # it before the change is treated as structural.
        # Only a transition ACROSS the body style counts. A run of lines that
        # merely differ from the body -- a table's cells, a boxed sidebar, a
        # caption block -- are all in one non-body style, and splitting each
        # onto its own region turns every cell of an undetected table into a
        # short standalone block that then reads as a heading. Confirmed: an
        # Arabic Wikipedia page emitted 34 "headings", most of them table
        # cells. A heading is a break BETWEEN body text and something else,
        # so requiring one side of the transition to be the body style is
        # what makes the signal mean what it is meant to mean.
        style = _line_style(ls)
        if style is None or prev_style is None or style == prev_style:
            same_style = True
        else:
            same_style = body_style not in (style, prev_style)
        # A hanging-indent list marker (bullet, "1.", "a)") starting a line is
        # a new-item signal in its own right, independent of vertical gap: a
        # list's inter-item spacing is often barely larger than its intra-
        # paragraph line spacing (confirmed: 17.5pt actual gap vs a 17.6pt
        # threshold from line-height alone), so the gap check by itself can
        # merge two distinct items by a hair. A leading marker always starts
        # a new item regardless of how tight that gap happens to be.
        new_item = _starts_with_list_marker(ls)
        if groups and prev_bottom is not None and (top - prev_bottom) < lead * gap_mult and overlaps and same_size and same_style and not new_item:
            groups[-1].extend(ls)
        else:
            groups.append(list(ls))
        prev_bottom = max(s.bbox[3] for s in ls)
        prev_xrange = (x0, x1)
        prev_size = size
        prev_style = style

    out = []
    for g in groups:
        bbox = (min(s.bbox[0] for s in g), min(s.bbox[1] for s in g),
                max(s.bbox[2] for s in g), max(s.bbox[3] for s in g))
        out.append(Region(bbox=bbox, kind="flow", spans=g))
    return out


# ---------------------------------------------------------------------------
# reading order
# ---------------------------------------------------------------------------

# A gutter is whitespace measured in points, but the thing it has to be told
# apart from -- the gaps between words -- scales with the page's type size. A
# constant threshold therefore cannot be right for both a 7pt newsletter and a
# 14pt large-print report, and the constant that used to be here (10pt)
# rejected the commonest two-column layout in existence: LaTeX's default
# columnsep is 10pt, which after span-bbox padding and x-binning measures 8pt
# of clear space -- just under the bar, on every two-column paper ever
# submitted to arXiv.
#
# Measured over 74 two-column pages: real gutters ran 5-26pt (median 16),
# within-line word gaps had a p95 of 6pt, and line height a median of 10pt.
# Against the page's own line height the gutter is stable -- median 1.6x --
# which is why the floor below is a fraction of it. The absolute floor only
# guards degenerate input.
# A rule stopping short at one end still counts as part of the grid if it
# is contained in the majority extent and covers this much of it.
CONSISTENT_CONTAIN_FRAC = 0.6
GUTTER_MIN_FRAC = 0.55      # of the page's median line height
GUTTER_MIN_ABS = 3.0        # points, when no usable line metric exists

# A page column must be wide enough to SET RUNNING TEXT IN. This is what
# separates a real column gutter from the whitespace between a table's
# columns, which a projection cannot otherwise tell apart -- both are
# vertical bands empty over the full height of the content around them.
#
# Measured on the PDF 1.7 reference's operator tables: the candidate bands
# were 3-4 em wide (29-39pt at 9pt type), which holds about five characters.
# A two-column arXiv page's bands were 26 em. Typographic convention puts a
# readable measure at 20-35 em and even a narrow newspaper column near 12;
# below roughly 8 em nothing is running text, so a band that narrow is a
# table column, a label gutter or a margin, and its boundary is dropped.
# A five-column layout on A4 still clears this comfortably (~10 em), so the
# bar costs no real layout anything.
MIN_COLUMN_EM = 8.0

# Text lines each side of a gutter before it counts as a column boundary.
MIN_COLUMN_LINES = 8

# How much of the page's content height a gutter must stay clear for. A real
# column gutter runs the height of the columns it separates; whitespace that
# happens to line up for a few lines does not. Set low enough to admit the
# common mixed layout -- a full-width table or figure over a two-column body,
# where the gutter owns only the lower part of the page.
GUTTER_MIN_HEIGHT_FRAC = 0.25

# Horizontal whitespace, as a multiple of the median line height, that ends one
# block and starts the next in the XY-cut. Comfortably above paragraph leading
# (~0.2-0.6x) so prose is not shredded into bands, and well below the space a
# real layout leaves around a full-width table or a banner.
BLOCK_SPLIT_FRAC = 1.4


def _gutter_floor(bboxes: list[Rect]) -> float:
    """The narrowest vertical band this page may call a gutter."""
    heights = [b[3] - b[1] for b in bboxes if b[3] > b[1]]
    if not heights:
        return GUTTER_MIN_ABS
    return max(GUTTER_MIN_ABS, float(np.median(heights)) * GUTTER_MIN_FRAC)


def _column_boundaries(bboxes: list[Rect], page_width: float, page_height: float,
                       min_gap: float | None = None, max_width_frac: float = 0.92,
                       empty_thresh: float = 0.04,
                       _want_extents: bool = False) -> list:
    """2D-aware whitespace-gutter finder.

    `min_gap=None` derives the width floor from the page's own typography (see
    GUTTER_MIN_FRAC); passing a number overrides it, which the tests do to pin
    specific behaviour. Width is the weaker of the two guards anyway;
    `empty_thresh` (persistence across the page's full content height) is what
    actually tells a real gutter apart from an indent or list marker, since
    those are never empty for anywhere near the full column height the way a
    genuine gutter is -- a single line's word gap has ink at every OTHER line,
    so it scores a fill_frac near 1.0, not near 0.

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
        return [] if _want_extents else [0.0, page_width]
    if min_gap is None:
        min_gap = _gutter_floor(bboxes)
    content_left = min(b[0] for b in bboxes)
    content_right = max(b[2] for b in bboxes)
    content_width = max(content_right - content_left, 1.0)
    narrow = [b for b in bboxes if (b[2] - b[0]) <= content_width * max_width_frac]
    use = narrow if narrow else bboxes

    # A caption/subtitle that breaks out to span WIDER than either column
    # but narrower than max_width_frac's absolute cutoff still fills in a
    # real gutter at its own y-range, the same way a full-width element
    # does (confirmed real case: a table's own two-line caption, "TABLE 1
    # / Global targets set in 2023...", spanning 71% of a 2-column page's
    # content width -- under 92%, so the absolute cutoff above didn't
    # exclude it, and its single wide bbox was enough to drop the gutter's
    # fill_frac just over `empty_thresh`, hiding a real 2-column split in
    # the genuine body text around it). A fixed width-fraction can't catch
    # this without also excluding genuine full-width lines on an actual
    # 1-column page (confirmed case above: ~90%-wide body lines on a
    # business filing) -- the two scenarios need a page-relative measure,
    # not a shared constant. What distinguishes them: on the 2-column
    # page, the caption is a rare outlier against a tight, dominant
    # cluster of column-width lines (the page's own 90th-percentile width);
    # on the 1-column filing, wide lines ARE that dominant cluster, so
    # nothing exceeds it by a meaningful margin. Comparing each bbox
    # against the page's own 90th-percentile width (not a global content-
    # width fraction) is what makes this hold across both.
    if len(use) >= 5:
        p90 = float(np.percentile([b[2] - b[0] for b in use], 90))
        typical = [b for b in use if (b[2] - b[0]) <= p90 * 1.25]
        if typical:
            use = typical

    # A page title/heading sitting above the 2-column body can survive the
    # width filter above (its own second wrapped line can happen to be a
    # perfectly ordinary column-line width, just centered or indented
    # differently) while still crossing the gutter at its own y-range --
    # confirmed real case: "strategy and targets", the wrapped second line
    # of a 26pt heading, measured 232pt wide (well under the column-width
    # outlier bar above, since it's genuinely similar in WIDTH to a real
    # body line) but sat centered across the gutter anyway. A heading's
    # bbox HEIGHT (driven by its much larger font size -- 26pt vs 10pt
    # body text here) is the more reliable outlier signal in exactly this
    # case, using the same page-relative percentile idea as the width
    # filter: a real body line's height clusters tightly around the page's
    # own dominant line-height, and a heading stands out from that
    # cluster the same way its width-outlier caption sibling stood out
    # from the width cluster above.
    if len(use) >= 5:
        p90h = float(np.percentile([b[3] - b[1] for b in use], 90))
        typical_h = [b for b in use if (b[3] - b[1]) <= p90h * 1.25]
        if typical_h:
            use = typical_h
    if not use:
        return [] if _want_extents else [0.0, page_width]

    # 1pt x-bins, not 2pt: at 2pt resolution a 10pt gutter measures 8pt, and
    # that rounding was itself half the reason the old floor rejected real
    # columns. y stays coarse -- it only needs line resolution.
    xres, yres = 1.0, 4.0
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
        return [] if _want_extents else [0.0, page_width]

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

    # A column is gutter-like where it carries a long CONTIGUOUS empty run --
    # not where it is empty on average. The distinction is what admits mixed
    # layouts: a page with a full-width table above two columns of prose has a
    # gutter that is only clear over the lower part, so its average fill is
    # high and an average-based test rejects it outright. The longest-run test
    # subsumes the average one (a full-height gutter runs the whole way) and
    # additionally reports WHERE the gutter holds, which is what stops a
    # gutter belonging to the prose from slicing the table above it in half.
    nrows = content.shape[0]
    min_rows = max(2, int(nrows * GUTTER_MIN_HEIGHT_FRAC))

    def longest_empty(i: int) -> tuple[int, int, int]:
        col = ~content[:, i]
        best = (0, 0, 0)
        run = None
        for r, v in enumerate(col):
            if v:
                run = r if run is None else run
            elif run is not None:
                if r - run > best[0]:
                    best = (r - run, run, r)
                run = None
        if run is not None and nrows - run > best[0]:
            best = (nrows - run, run, nrows)
        return best

    # The longest-run criterion is used ONLY when extents are wanted, i.e. by
    # the XY-cut, which has already restricted the input to one band and so
    # cannot mistake a table's gap for a page column. The page-wide callers
    # keep the stricter average-emptiness test: relaxing it there let a
    # table's own KEY/TYPE gap be numbered as a page column on an ordinary
    # single-column page.
    spans_by_col: dict[int, tuple[int, int, int]] = {}
    gutterish = []
    for i in range(inked_cols[0], inked_cols[-1] + 1):
        best = longest_empty(i)
        spans_by_col[i] = best
        gutterish.append(best[0] >= min_rows if _want_extents
                         else fill_frac[i] <= empty_thresh)

    gaps, run = [], None
    for k, is_gut in enumerate(gutterish + [False]):
        i = inked_cols[0] + k
        if is_gut:
            run = i if run is None else run
            continue
        if run is None:
            continue
        if (i - run) * xres >= min_gap:
            # the gutter's own rows: the overlap of its columns' empty runs
            lo = max(spans_by_col[c][1] for c in range(run, i))
            hi = min(spans_by_col[c][2] for c in range(run, i))
            if hi - lo >= min_rows and \
                    max_run(inked_cols[0], run) >= min_run and \
                    max_run(i, inked_cols[-1] + 1) >= min_run:
                gaps.append((run * xres, i * xres, lo, hi))
        run = None

    bounds = [left] + [((a + b) / 2) for a, b, _lo, _hi in gaps] + [right + xres]
    bounds = _drop_narrow_bands(bounds, _body_size(bboxes) * MIN_COLUMN_EM)
    if _want_extents:
        # y-extent of each surviving gutter: the longest run of rows over which
        # it is actually clear. See page_gutters.
        top = inked_rows[0]
        kept = set(bounds[1:-1])
        return [(( a + b) / 2, (top + lo) * yres, (top + hi) * yres)
                for a, b, lo, hi in gaps if (a + b) / 2 in kept]
    return bounds


def _body_size(bboxes: list[Rect]) -> float:
    heights = [b[3] - b[1] for b in bboxes if b[3] > b[1]]
    return float(np.median(heights)) if heights else 10.0


def _drop_narrow_bands(bounds: list[float], min_width: float) -> list[float]:
    """Dissolve any band too narrow to be a column of running text.

    The narrowest band is merged into whichever neighbour is itself narrower,
    which keeps the merge local -- a stray label gutter next to a wide body
    column collapses into that column instead of restructuring the page.
    Repeats until every surviving band clears the bar or only one is left.
    """
    bounds = list(bounds)
    while len(bounds) > 2:
        widths = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
        j = min(range(len(widths)), key=lambda i: widths[i])
        if widths[j] >= min_width:
            break
        if j == 0:
            bounds.pop(1)                       # merge right
        elif j == len(widths) - 1:
            bounds.pop(len(bounds) - 2)         # merge left
        else:
            bounds.pop(j if widths[j - 1] <= widths[j + 1] else j + 1)
    return bounds


def _column_of(x_center: float, boundaries: list[float], rtl: bool = True) -> int:
    ncols = len(boundaries) - 1
    for i in range(ncols):
        if boundaries[i] <= x_center < boundaries[i + 1]:
            # column 0 is always "read first": rightmost for RTL, leftmost for LTR
            return ncols - 1 - i if rtl else i
    return 0


def detect_columns(regions: list[Region], page_width: float, page_height: float,
                   min_gap: float | None = None, rtl: bool = True,
                   spans: list[Span] | None = None,
                   gutters: list[tuple[float, float, float]] | None = None) -> int:
    """Whitespace-projection column finder. Returns number of columns and
    tags each region with its column index (0 = read first).

    Boundaries are computed from `spans` (the page's raw text spans) when
    given, not from `regions`' own bboxes -- by the time this runs, a
    genuinely 2-column page's text has often already been correctly
    clustered into just 2-3 wide flow regions (one per column, one for a
    stray footer), and re-deriving the gutter from those few COARSE boxes
    is a much lower-resolution signal than the hundreds of individual
    spans assign_spans already used to get the split right in the first
    place (confirmed real case: a 2-column arXiv paper's spans correctly
    split 85/130 across two columns, producing two properly-separated
    flow regions -- but re-running gutter detection on just those 2-3
    region bboxes here failed to find the same gutter, collapsing both
    back into one column and discarding the correct split). Falling back
    to region bboxes when spans aren't available keeps this usable in
    contexts (tests, etc.) that only have regions.
    """
    if not regions:
        return 0
    if gutters is not None:
        # The SAME gutters the line assemblers used. Deriving columns here from
        # a second projection is how a table's internal gap ended up numbered
        # as a page column on a single-column page: the two derivations
        # disagreed and nothing noticed. A gutter also only applies over its
        # own rows, so a region above a columned block is column 0 whatever
        # the block below it looks like.
        ncols = 1
        for r in regions:
            yc = (r.bbox[1] + r.bbox[3]) / 2
            active = [g for g in gutters if g[1] <= yc <= g[2]]
            idx = sum(1 for gx, _, _ in active if gx <= r.x_center)
            here = len(active) + 1
            r.column = (here - 1 - idx) if rtl else idx
            ncols = max(ncols, here)
        return ncols
    boundaries = _column_boundaries([s.bbox for s in spans] if spans else [r.bbox for r in regions],
                                    page_width, page_height, min_gap)
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


def nested_page_rect(prim: PagePrimitives) -> Rect | None:
    """The rectangle of a page reproduced INSIDE this page, if there is one.

    Some documents print one page inside another: a teacher's guide around
    a reduced student page, an annotated reprint, a facsimile edition, a
    slide with its notes. The two carry unrelated content -- an exercise
    and its answer key, a slide and its script -- and interleaving them by
    pure geometry produces chunks that mix both, which is worse for
    retrieval than either alone.

    The file states the structure: the inner page is drawn as its own
    background rectangle, in a different colour from the outer page's own
    background. Requiring text on BOTH sides is what keeps this from
    firing on an ordinary tinted callout box -- a callout has no content
    outside itself to be separated from.

    Nothing here is document-specific: it needs two background rectangles
    of different colours, one inside the other, with text in both.
    """
    if not prim.backgrounds or not prim.spans:
        return None
    page_area = prim.width * prim.height
    if page_area <= 0:
        return None

    def _area(b) -> float:
        return (b[2] - b[0]) * (b[3] - b[1])

    outers = [f for f in prim.backgrounds if _area(f.bbox) >= 0.9 * page_area]
    if not outers:
        return None
    inners = [f for f in prim.backgrounds
              if 0.15 * page_area <= _area(f.bbox) <= 0.85 * page_area]
    if not inners:
        return None

    def _differs(a, b) -> bool:
        return max(abs(x - y) for x, y in zip(a, b)) > 0.02

    for f in sorted(inners, key=lambda f: -_area(f.bbox)):
        if not any(_differs(f.color, o.color) for o in outers):
            continue
        x0, y0, x1, y1 = f.bbox
        inside = outside = 0
        for sp in prim.spans:
            if not sp.text.strip():
                continue
            cx = (sp.bbox[0] + sp.bbox[2]) / 2
            cy = (sp.bbox[1] + sp.bbox[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                inside += 1
            else:
                outside += 1
        if inside >= 3 and outside >= 3:
            return (x0, y0, x1, y1)
    return None


def _reading_key(r: Region, gutters: list[tuple[float, float, float]],
                 rtl: bool) -> tuple[float, int, float, float]:
    """Sort key that reads a page the way a person does: block by block down
    the page, and column by column inside a block.

    Splitting the lines correctly is only half of reading order -- with the
    columns separated but the regions still sorted by y, the output is a
    perfectly clean left paragraph, then a clean right paragraph, then the next
    left paragraph. Every block is right and the document is still unreadable.

    A region that lies outside every gutter's rows -- a full-width heading,
    a figure spanning the page, a footer -- sorts on its own y, so it lands
    between the blocks it sits between instead of being forced into a column.
    That is what makes mixed layouts read correctly without a special case.
    """
    yc = (r.bbox[1] + r.bbox[3]) / 2
    active = [g for g in gutters if g[1] <= yc <= g[2]]
    if not active:
        return (r.bbox[1], 0, r.bbox[1], r.bbox[0])
    band_top = min(g[1] for g in active)
    idx = sum(1 for gx, _, _ in active if gx <= r.x_center)
    col = (len(active) - idx) if rtl else idx
    return (band_top, col, r.bbox[1], r.bbox[0])


# A region must own this share of the declared ranks before the file's own
# order is trusted over geometry. Below it the tagging is partial -- a few
# decorative spans, or an artifact-heavy page -- and a partial order is worse
# than none, because it reorders some regions and leaves others where they lay.
DECLARED_ORDER_COVERAGE = 0.6


def _apply_declared_order(ordered: list[Region],
                          declared: list[tuple[tuple[float, float], int]]
                          ) -> list[Region] | None:
    """Reorder regions by the reading order the file itself declares.

    Returns None when the tags do not cover enough of the page, leaving the
    geometric order untouched. Ranks inside a region are reduced by MEDIAN, not
    minimum: one stray early-ranked span (a footnote marker, a stray artifact)
    should not drag a whole paragraph to the top.
    """
    if not declared or not ordered:
        return None
    per: list[list[int]] = []
    for r in ordered:
        x0, y0, x1, y1 = r.bbox
        per.append([rank for (px, py), rank in declared
                    if x0 <= px <= x1 and y0 <= py <= y1])
    covered = sum(1 for v in per if v)
    if covered < max(2, len(ordered) * DECLARED_ORDER_COVERAGE):
        return None
    # Regions with no declared rank keep their geometric neighbourhood: they
    # inherit the rank of the nearest preceding region that has one, so an
    # untagged figure stays where the layout put it instead of piling up first.
    filled: list[float] = []
    last = -1.0
    for v in per:
        if v:
            last = float(statistics.median(v))
        filled.append(last)
    return [r for _, _, r in sorted(
        ((rank, i, r) for i, (rank, r) in enumerate(zip(filled, ordered))),
        key=lambda t: (t[0], t[1]))]


def order_regions(regions: list[Region], page_width: float, page_height: float, rtl: bool = True,
                  spans: list[Span] | None = None,
                  nested: Rect | None = None,
                  declared: list[tuple[tuple[float, float], int]] | None = None,
                  gutters: list[tuple[float, float, float]] | None = None) -> list[Region]:
    detect_columns(regions, page_width, page_height, rtl=rtl, spans=spans,
                   gutters=gutters)
    if gutters:
        ordered = sorted(regions, key=lambda r: _reading_key(r, gutters, rtl))
    else:
        ordered = rtl_xy_cut(regions, rtl=rtl)
    # An explicit declaration beats an inference. Where the file ships a
    # structure tree, its order is the answer -- for any column count, for
    # sidebars, for a layout that changes shape halfway down the page -- and
    # geometry only has to cover the untagged majority.
    if declared:
        by_tags = _apply_declared_order(ordered, declared)
        if by_tags is not None:
            ordered = by_tags
    if nested is not None:
        # Emit the nested page's own content as one contiguous run, and the
        # surrounding page as another, instead of interleaving the two by
        # geometry. Each group keeps the order the xy-cut gave it, and the
        # group whose first region came first stays first, so this only
        # ever un-interleaves -- it never reorders within a page.
        x0, y0, x1, y1 = nested
        def _in(r: Region) -> bool:
            cx = (r.bbox[0] + r.bbox[2]) / 2
            cy = (r.bbox[1] + r.bbox[3]) / 2
            return x0 <= cx <= x1 and y0 <= cy <= y1
        a = [r for r in ordered if _in(r)]
        b = [r for r in ordered if not _in(r)]
        if a and b:
            # Inner page first, then the surrounding page. The nested page
            # is the document being reproduced; the margin around it is
            # commentary on it, and reading the commentary first inverts
            # the sense (an answer key ahead of its own exercise).
            ordered = a + b
    for i, r in enumerate(ordered):
        r.order = i
    return ordered
