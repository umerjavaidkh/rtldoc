"""Ground truth derived from the source PDF, not from a human.

The problem
-----------
ParseBench scores against ~169,000 hand-written rules over ~2,000
human-verified pages. That is the right way to build a benchmark and it is not
available here: nobody is going to hand-verify 43,597 pages, and a 20-page gold
set cannot tell you anything about a corpus this size.

The way out is to notice that a PDF states a great deal of ground truth
outright, in its own content stream, and that a parser is a different program
from the one reading those operators. Where the file is unambiguous, an
assertion generated from it is as binding as one a person wrote:

  * TABLE RECORDS. When a table is drawn with rules, the cell rectangles are
    determined by the line segments in the content stream. Intersecting them
    gives the true grid, and assigning words to cells by coordinate gives the
    true records. This is real ground truth for exactly the property RAG cares
    about -- does a value stay attached to the header that names it.

  * PRECEDENCE. Two text fragments in the same column, separated vertically,
    have an unambiguous reading order. Emitted as ParseBench-style pairwise
    assertions ("before" must precede "after" in the linearised output), they
    catch the reading-order shredding that character-coverage metrics cannot
    see. Rules are only generated where order is beyond argument -- same
    column, clear vertical separation -- so a violation is always a real bug.

  * DIGIT BAGS. The multiset of digits on a page is preserved by any correct
    extraction. ParseBench uses the same idea (`bag_of_digit_percent`); it is
    cheap and it is the single most load-bearing check for financial and
    numeric RAG, where a dropped or invented digit is a wrong answer with full
    confidence behind it.

Scope is English and Arabic: digit normalisation covers ASCII and Arabic-Indic
(U+0660-0669, U+06F0-06F9), and precedence is generated per column so it holds
for RTL pages as written.

What this CANNOT do, stated plainly, so the numbers are not over-read:
borderless tables have no ground truth here (nothing in the file says where the
columns are), and neither does semantic formatting. Those dimensions need the
small hand-verified gold set -- see RAGBENCH.md.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from dataclasses import dataclass, field

_AR_DIGITS = {**{chr(0x0660 + i): str(i) for i in range(10)},
              **{chr(0x06F0 + i): str(i) for i in range(10)}}
_DIGIT = re.compile(r"\d")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_AR_LETTER = re.compile(r"[\u0621-\u064a]")


def normalize(text: str) -> str:
    """Canonical form for every comparison in this module: NFKC, Arabic-Indic
    digits folded to ASCII, markdown furniture stripped, whitespace collapsed.
    Matching ParseBench's normalisation step -- without it, every metric
    measures formatting noise instead of content."""
    text = unicodedata.normalize("NFKC", text)
    # Strip tatweel/kashida (U+0640). Arabic justification stretches words with
    # it -- the raw text layer returns "يـــهدف" where the word is "يهدف" --
    # and rtldoc correctly removes it. Comparing without stripping it scored a
    # page at 3% recall when every word was in fact present and correct.
    text = text.replace("\u0640", "")
    # Symbol-font Private Use codepoints. The reference reads the raw stream,
    # where a Word bullet is still U+F0B7; rtldoc decodes it to "\u2022" via the
    # Adobe Symbol encoding. Comparing those scores the parser 0 for being
    # right, so both sides drop the block here.
    text = "".join(" " if 0xF000 <= ord(c) <= 0xF0FF else c for c in text)
    text = "".join(_AR_DIGITS.get(c, c) for c in text)
    # HTML tags must go BEFORE the markdown character strip -- stripping ">"
    # first leaves "<br" behind, which then survives into every cell value and
    # makes real matches fail. Confirmed on an IRS table whose cells are
    # <br>-joined.
    text = re.sub(r"<[^>]{0,20}>", " ", text)
    text = re.sub(r"[*_`~#>]|\|", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def digit_bag(text: str) -> collections.Counter:
    return collections.Counter(_DIGIT.findall(normalize(text)))


def word_bag(text: str) -> collections.Counter:
    return collections.Counter(w.lower() for w in _WORD.findall(normalize(text))
                               if len(w) > 1)


# --------------------------------------------------------------------------
# rule types
# --------------------------------------------------------------------------

@dataclass
class PrecedenceRule:
    """`before` must appear earlier than `after` in the linearised output.

    ParseBench's formulation: the rule passes iff the FIRST occurrence of
    `before` precedes the LAST occurrence of `after`. That asymmetry is
    deliberate -- it tolerates a fragment legitimately repeating (a running
    header, a term reused later) without letting a genuine order inversion
    pass."""
    before: str
    after: str
    page: int
    why: str = ""

    def check(self, linearised: str) -> bool | None:
        hay = normalize(linearised)
        b, a = normalize(self.before), normalize(self.after)
        i = hay.find(b)
        j = hay.rfind(a)
        if i < 0 or j < 0:
            return None          # not applicable: text is missing, CFS-text's job
        return i < j


@dataclass
class ContiguityRule:
    """Two lines adjacent in a column must land adjacent in the output.

    Precedence assertions cannot see interleaving, and that blind spot is
    total. When a two-column page is shredded -- left line, right line, left
    line -- the left column's lines still appear in the right ORDER relative to
    each other, so every precedence rule still passes while the text is
    unreadable. The defect is not inversion, it is INTRUSION: something from
    the other column landed in between.

    So this rule measures the gap. Two vertically consecutive lines of one
    column should be separated in the linearised output by roughly their own
    length, not by a foreign line's worth of text.
    """
    first: str
    second: str
    page: int
    slack: int = 120        # characters of separation tolerated between them

    def check(self, linearised: str) -> bool | None:
        hay = normalize(linearised)
        a, b = normalize(self.first), normalize(self.second)
        i = hay.find(a)
        if i < 0:
            return None
        j = hay.find(b, i + len(a))
        if j < 0:
            return None
        return (j - (i + len(a))) <= self.slack


@dataclass
class TableRecord:
    """One row of a ruled table, keyed by its column headers."""
    values: dict[str, str]
    page: int

    def keys(self) -> set[str]:
        return {k for k in self.values if k}


@dataclass
class PageTruth:
    page: int
    text: str
    digits: collections.Counter
    words: collections.Counter
    precedence: list[PrecedenceRule] = field(default_factory=list)
    contiguity: list[ContiguityRule] = field(default_factory=list)
    records: list[TableRecord] = field(default_factory=list)
    # the same records, kept grouped BY TABLE. Scoring needs the grouping: a
    # ground-truth table must be compared against the table rtldoc emitted for
    # it, not against every record on the page.
    record_groups: list[list[TableRecord]] = field(default_factory=list)
    n_ruled_tables: int = 0


# --------------------------------------------------------------------------
# building truth from a page
# --------------------------------------------------------------------------

def _rules_from_drawings(drawings) -> tuple[list[tuple], list[tuple]]:
    """Horizontal and vertical rule segments, from strokes and thin fills."""
    h, v = [], []
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                (x0, y0), (x1, y1) = item[1], item[2]
                if abs(y1 - y0) <= 1.0 and abs(x1 - x0) > 8:
                    h.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2))
                elif abs(x1 - x0) <= 1.0 and abs(y1 - y0) > 8:
                    v.append((min(y0, y1), max(y0, y1), (x0 + x1) / 2))
            elif item[0] == "re":
                r = item[1]
                w, ht = r.x1 - r.x0, r.y1 - r.y0
                if ht <= 2.0 and w > 8:
                    h.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
                elif w <= 2.0 and ht > 8:
                    v.append((r.y0, r.y1, (r.x0 + r.x1) / 2))
    return h, v


def _dedupe_axis(vals: list[float], tol: float = 3.0) -> list[float]:
    """Collapse near-coincident rules (double-ruled borders, overdrawn cells)
    into one line each, or every table gains phantom zero-height rows."""
    out: list[float] = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
        else:
            out[-1] = (out[-1] + v) / 2
    return out


def ruled_table_records(page, min_rows: int = 2, min_cols: int = 2) -> list[list[list[str]]]:
    """The true cell grids of every ruled table on the page.

    Cells come from the intersections of the drawn rules, and their contents
    from the words whose centres fall inside them. Nothing here consults the
    parser -- this is what the file says, which is what makes it usable as
    ground truth."""
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    h, v = _rules_from_drawings(drawings)
    if len(h) < min_rows + 1 or len(v) < min_cols + 1:
        return []

    words = [w for w in page.get_text("words") if w[4].strip()]
    grids: list[list[list[str]]] = []

    # group horizontal rules into vertically-contiguous bands = table candidates
    h.sort(key=lambda t: t[2])
    bands, cur = [], [h[0]]
    for line in h[1:]:
        if line[2] - cur[-1][2] <= 120:
            cur.append(line)
        else:
            bands.append(cur)
            cur = [line]
    bands.append(cur)

    for band in bands:
        if len(band) < min_rows + 1:
            continue
        ys = _dedupe_axis([b[2] for b in band])
        y0, y1 = ys[0], ys[-1]
        x_lo = min(b[0] for b in band)
        x_hi = max(b[1] for b in band)
        crossing = [c for c in v
                    if c[0] < y1 - 2 and c[1] > y0 + 2 and x_lo - 4 <= c[2] <= x_hi + 4]
        xs = _dedupe_axis([c[2] for c in crossing])
        if len(ys) < min_rows + 1 or len(xs) < min_cols + 1:
            continue

        grid = _grid_from_axes(words, xs, ys)
        if _is_table_shaped(grid):
            grids.append(grid)
    return grids


def _grid_from_axes(words, xs: list[float], ys: list[float]) -> list[list[str]]:
    """Fill the cells bounded by the rule positions `xs` x `ys` with the words
    whose centres fall inside each. Shared by the ground-truth builder and the
    gold-set collector so a verdict is always about the same grid the metric
    scored."""
    grid: list[list[str]] = []
    for r in range(len(ys) - 1):
        row: list[str] = []
        for c in range(len(xs) - 1):
            cx0, cx1, cy0, cy1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
            inside = [w for w in words
                      if cx0 <= (w[0] + w[2]) / 2 <= cx1
                      and cy0 <= (w[1] + w[3]) / 2 <= cy1]
            # Word order within a cell follows the cell's own script. Sorting
            # by ascending x is correct for Latin and exactly backwards for
            # Arabic: it turned "دون المعيار" into "المعيار دون", so every
            # Arabic table scored 0% against output that was in fact correct.
            # The direction is decided per cell, so a mixed table works too.
            rtl = _AR_LETTER.search("".join(w[4] for w in inside)) is not None
            inside.sort(key=lambda w: (round(w[1] / 3), -w[0] if rtl else w[0]))
            row.append(normalize(" ".join(w[4] for w in inside)))
        grid.append(row)
    return grid


# A drawn lattice is not the same thing as a table. Diagrams built from boxes,
# and the axis frames of matplotlib charts, produce exactly the same rule
# intersections -- confirmed on arXiv papers, where flowchart boxes and a
# plot's y-axis ticks were both being read as table ground truth and then
# scored against rtldoc's correct output, which put the whole TABLES dimension
# at 4% when the real figure was far higher.
#
# Measured on 12 lattices (5 hand-checked real tables, 7 hand-checked
# diagrams/charts), the two conditions below keep 4/5 real and reject 6/7 fake.
# The separation is good, not perfect: expect roughly one chart axis in seven
# to survive as false ground truth. The residual is acceptable ONLY because a
# rejected grid yields no records at all, so the dimension reports n/a rather
# than a wrong score -- the failure mode is silence, not a lie.
HEADER_FILL_MIN = 0.60      # a table's header row names most of its columns
EMPTY_ROW_MAX = 0.25        # alternating blank rows mean boxes, not rows


def _is_table_shaped(grid: list[list[str]]) -> bool:
    if len(grid) < 2 or not grid[0]:
        return False
    header_fill = sum(1 for c in grid[0] if c.strip()) / len(grid[0])
    empty_rows = sum(1 for r in grid if not any(c.strip() for c in r)) / len(grid)
    return header_fill >= HEADER_FILL_MIN and empty_rows <= EMPTY_ROW_MAX


def _merge_wrapped_rows(grid: list[list[str]]) -> list[list[str]]:
    """Fold a wrapped cell's continuation lines back into their own row.

    A ruled grid gives one grid row per pair of horizontal rules, but a cell
    whose text wraps produces extra rows that carry only the overflow: the row
    below has an empty first column and continues the sentence above it.
    Counting those as separate records inflates the reference row count -- an
    8-row Arabic rubric came back as 12 -- and then penalises a parser that
    got the row count right.

    Two rules, both structural: a wholly empty row is dropped, and a row whose
    leading column is empty while a later column has text is a continuation and
    is appended to the row above.
    """
    out: list[list[str]] = []
    for row in grid:
        if not any(c.strip() for c in row):
            continue
        leading_empty = not row[0].strip()
        if leading_empty and out:
            for i, cell in enumerate(row):
                if cell.strip() and i < len(out[-1]):
                    out[-1][i] = (out[-1][i] + " " + cell).strip()
            continue
        out.append(list(row))
    return out


_NUMERIC_CELL = re.compile(r"^[\s\-+(]*[\d\u0660-\u0669][\d\u0660-\u0669,.\s%()\-]*$")


def _numeric_row(row: list[str]) -> bool:
    cells = [c.strip() for c in row if c.strip()]
    if not cells:
        return False
    return sum(1 for c in cells if _NUMERIC_CELL.match(c)) / len(cells) >= 0.6


def _leaf_header_row(grid: list[list[str]]) -> int:
    """Which row of this grid actually names the columns?

    Not row 0. A statistical yearbook draws its caption INSIDE the table's
    ruled frame, so the grid's first rows are the title and its number,
    shredded across the columns, and the real header sits below them:

        row0  ['Health of Beds the','Sector, / Five','10,000 Years','KSA',...]
        row1  ['','Table','2-2','','','','','','','جدول 2-2','','']
        row2  ['','','','Year','','','','','العام','','','']
        row3  ['','2022G','','2021G','','2020G','','2019G','','2018G','','']

    Keying records on row 0 keys every value to a fragment of the title, and
    the table scores near zero however well it was read -- and it does so on
    BOTH sides, because the parser makes the same assumption. Measuring a
    parser against a reference that shares its defect cannot show a fix.

    The leaf header is the last text row above the body: walk down while rows
    are not yet numeric, and take the one immediately before the numbers
    start. A table with no numeric body has no type break to find, so it
    keeps row 0 -- the previous behaviour, unchanged.
    """
    if len(grid) < 3:
        return 0
    first_data = None
    for i, row in enumerate(grid):
        if _numeric_row(row):
            first_data = i
            break
    if first_data is None or first_data == 0:
        return 0
    # Need at least one data row left to score.
    if first_data >= len(grid) - 1:
        return 0
    return first_data - 1


def grid_to_records(grid: list[list[str]], page: int) -> list[TableRecord]:
    """Header row + data rows -> records keyed by header, ParseBench's
    TableRecordMatch shape. A table whose header is dropped or transposed
    produces records with the wrong keys, and scores near zero -- which is
    correct, because that is exactly the failure that makes an agent read the
    wrong column."""
    grid = _merge_wrapped_rows(grid)
    if len(grid) < 2:
        return []
    # Drop wholly-empty columns, exactly as pipeline._table_grid does for the
    # prediction. Keeping them here gave the two sides different column counts
    # on 11% of tables -- a phantom column shifts every header, the key sets
    # stop overlapping and the whole table scores zero however well it was
    # read.
    ncols = max(len(r) for r in grid)
    keep = [c for c in range(ncols)
            if any((r[c] if c < len(r) else "").strip() for r in grid)]
    if keep and len(keep) < ncols:
        grid = [[(r[c] if c < len(r) else "") for c in keep] for r in grid]
    hrow = _leaf_header_row(grid)
    header = [h or f"col{i}" for i, h in enumerate(grid[hrow])]
    out = []
    for row in grid[hrow + 1:]:
        if not any(c.strip() for c in row):
            continue
        out.append(TableRecord(
            values={header[i]: row[i] for i in range(min(len(header), len(row)))
                    if row[i].strip()},
            page=page))
    return out


def precedence_rules(page, max_rules: int = 6, min_len: int = 25,
                     contig: list | None = None) -> list[PrecedenceRule]:
    """Pairwise order assertions that are beyond argument.

    Only generated within a single column, between fragments separated by real
    vertical distance. On a two-column page a rule is emitted per column, never
    across the gutter -- cross-column order is a layout convention, not a fact
    the file states, and asserting it would fabricate failures."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    if len(words) < 60:
        return []

    # Group by the SAME block-scoped column model the parser uses, not by a
    # page-global mid-page strip. The crude version paired a caption in the
    # left column with body text in the right one and then asserted an order
    # between them -- a rule that is simply false, and that a correctly
    # column-ordered parser is bound to violate. A benchmark whose ground truth
    # is derived with weaker geometry than the system under test measures the
    # benchmark.
    # Computed HERE, never imported from rtldoc. Ground truth derived with the
    # code under test is not ground truth: improve the parser's column
    # detection and the rules move with it, so the benchmark can never catch
    # that change getting it wrong. This is a deliberately independent
    # reimplementation of the same idea -- blocks by horizontal whitespace,
    # columns inside each block.
    gutters = _independent_gutters([tuple(w[:4]) for w in words])
    groups: dict[tuple, list] = collections.defaultdict(list)
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        active = [g for g in gutters if g[1] <= cy <= g[2]]
        band = min((g[1] for g in active), default=-1.0)
        col = sum(1 for gx, _, _ in active if gx <= cx)
        groups[(band, col)].append(w)

    rules: list[PrecedenceRule] = []
    contig = contig if contig is not None else []
    for col, ws in sorted(groups.items(), key=lambda kv: kv[0]):
        lines = collections.defaultdict(list)
        for w in ws:
            lines[round(w[1] / 3)].append(w)
        ordered = [ " ".join(x[4] for x in sorted(v, key=lambda x: x[0]))
                    for _, v in sorted(lines.items()) ]
        ordered = [normalize(t) for t in ordered]
        ordered = [t for t in ordered if len(t) >= min_len]
        if len(ordered) < 4:
            continue
        # pair lines far apart in the column: adjacent lines can legitimately
        # merge or reflow, distant ones cannot swap without a real bug
        step = max(2, len(ordered) // (max_rules // max(1, len(groups)) + 1))
        for i in range(0, len(ordered) - step, step):
            a, b = ordered[i], ordered[i + step]
            if a and b and a != b:
                rules.append(PrecedenceRule(a[:80], b[:80], page.number + 1,
                                            why=f"column {col}, {step} lines apart"))
        # adjacent pairs, for the contiguity check
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            if a and b and a != b:
                contig.append(ContiguityRule(a[:70], b[:70], page.number + 1))
    return rules[:max_rules]


def _independent_gutters(boxes: list[tuple]) -> list[tuple[float, float, float]]:
    """Column gutters as (x, y_top, y_bottom), derived independently of rtldoc.

    Blocks first, on horizontal whitespace larger than one line height; then,
    inside each block, vertical strips no word crosses. A strip counts only if
    both sides hold enough lines to be running text -- otherwise a table's
    column gap becomes a column and the rules generated from it are false.
    """
    if len(boxes) < 40:
        return []
    heights = [b[3] - b[1] for b in boxes if b[3] > b[1]]
    scale = sorted(heights)[len(heights) // 2] if heights else 10.0

    events = sorted((b[1], b[3]) for b in boxes)
    bands, lo, hi = [], events[0][0], events[0][1]
    for y0, y1 in events[1:]:
        if y0 - hi > scale:
            bands.append((lo, hi))
            lo, hi = y0, y1
        else:
            hi = max(hi, y1)
    bands.append((lo, hi))

    out: list[tuple[float, float, float]] = []
    for by0, by1 in bands:
        band = [b for b in boxes if by0 - 1 <= (b[1] + b[3]) / 2 <= by1 + 1]
        if len(band) < 40:
            continue
        x0 = min(b[0] for b in band)
        x1 = max(b[2] for b in band)
        width = x1 - x0
        if width < 100:
            continue
        n = int(width) + 1
        occupied = [False] * (n + 1)
        for b in band:
            for i in range(max(0, int(b[0] - x0)), min(n, int(b[2] - x0)) + 1):
                occupied[i] = True
        i, runs = 0, []
        while i < n:
            if occupied[i]:
                i += 1
                continue
            j = i
            while j < n and not occupied[j]:
                j += 1
            if (j - i) >= scale * 0.55 and 0.10 * n < (i + j) / 2 < 0.90 * n:
                runs.append(x0 + (i + j) / 2)
            i = j + 1
        if not runs:
            continue
        edges = [x0] + runs + [x1]
        lines_in = lambda a, b: len({round(x[1]) for x in band
                                     if a <= (x[0] + x[2]) / 2 < b})
        if any(lines_in(a, b) < 8 or (b - a) < scale * 8
               for a, b in zip(edges, edges[1:])):
            continue
        out.extend((x, by0, by1) for x in runs)
    return out


def _gutter_x(words, rect) -> float | None:
    """x of a vertical strip no word crosses, near the middle of the text area.
    None when the page is not two-column."""
    if len(words) < 40:
        return None
    x0 = min(w[0] for w in words)
    x1 = max(w[2] for w in words)
    width = x1 - x0
    if width < 100:
        return None
    nb = max(8, int(width / 2))
    occ = [False] * (nb + 1)
    for w in words:
        a = max(0, int((w[0] - x0) / width * nb))
        b = min(nb, int((w[2] - x0) / width * nb))
        for i in range(a, b + 1):
            occ[i] = True
    best, i, end = (0, None), int(nb * 0.30), int(nb * 0.70)
    while i < end:
        if occ[i]:
            i += 1
            continue
        j = i
        while j < end and not occ[j]:
            j += 1
        if j - i > best[0]:
            best = (j - i, (i + j) / 2)
        i = j + 1
    run, centre = best
    if centre is None or run * (width / nb) < 10.0:
        return None
    return x0 + centre / nb * width


# LaTeX sets tables with booktabs: horizontal rules only, never verticals.
# A lattice WITH vertical rules in a LaTeX document is therefore never a table
# -- it is a flowchart, a plot frame, or a figure border. Measured on 40 arXiv
# papers, 7 of 8 sampled lattices were figures, which put the TABLES dimension
# at 9% while rtldoc's actual table output was fine.
#
# No per-grid threshold separated the two populations (fill density, rule span,
# drawing count and curve count were all tried and all overlapped). So the
# metric ABSTAINS on LaTeX documents instead of reporting a number known to be
# wrong. LaTeX table fidelity needs a hand-verified gold set; until there is
# one, RAGBench says n/a rather than guessing.
_LATEX_PRODUCER = re.compile(r"arxiv genpdf|pdftex|xetex|luatex|tex2pdf|latex", re.I)


def has_reliable_ruled_tables(doc) -> bool:
    """False when this document's tables cannot be recovered from drawn rules."""
    meta = doc.metadata or {}
    hay = f"{meta.get('producer', '')} | {meta.get('creator', '')}"
    return not _LATEX_PRODUCER.search(hay)


def build_page_truth(page, ruled_tables: bool = True) -> PageTruth:
    text = page.get_text()
    contig: list[ContiguityRule] = []
    truth = PageTruth(
        page=page.number + 1,
        text=text,
        digits=digit_bag(text),
        words=word_bag(text),
        precedence=precedence_rules(page, contig=contig),
    )
    truth.contiguity = contig[:40]
    if ruled_tables:
        for grid in ruled_table_records(page):
            truth.n_ruled_tables += 1
            recs = grid_to_records(grid, truth.page)
            if recs:
                truth.record_groups.append(recs)
                truth.records.extend(recs)
    return truth
