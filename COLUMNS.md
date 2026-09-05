# Column detection and reading order

## What was broken

On a two-column page rtldoc reported `columns: 1` and emitted the left and
right columns' lines alternating, one for one:

```
competes on the default bearer, but it does not fully bound radio resources
are allocated among DRBs and endpoints. residence time under saturation
because endpoints carrying This implementation choice is the source of ...
```

Every character is present, so character-coverage metrics scored the page
~1.00. It affected 62 of 254 sampled arXiv papers (7.5% of their pages), and it
was also the largest single cause of table damage: 10 of the 14 total-loss
cases in the table gold set were a table whose header had absorbed prose from
the column beside it.

## Why the previous approach failed

Three independent causes, each of the same kind — **an absolute constant
standing in for a quantity that scales with the document.**

1. **`min_gap = 10.0` in `_column_boundaries`.** LaTeX's default `\columnsep`
   is 10pt, which after span-bbox padding and 2pt x-binning measures 8pt of
   clear space. Every two-column paper on arXiv fell just under the bar. A
   gutter is whitespace in points, but the thing it must be told apart from —
   the gaps between words — scales with type size, so no constant is right for
   both a 7pt newsletter and a 14pt report.

2. **`col_gap_mult = 1.3` in `geobidi.group_baselines`.** The glyph-level line
   builder guessed at a gutter from one line's spacing: split where a gap
   exceeds 1.3× the font size. To avoid firing on wide word spacing that
   multiple has to sit above ~1.2×, which puts its floor (13pt on 10pt type)
   *above* the commonest real gutter. The two populations overlap, so no value
   of the constant separates them.

3. **Two line assemblers, neither column-aware.** `layout.group_by_line` (over
   spans) and `geobidi.group_baselines` (over glyphs) both grouped by y alone.
   On a two-column page the columns share baselines — that is what two columns
   means — so grouping by y welds the left column's line to the right column's.
   Nothing downstream can undo it: by the time regions, tables and reading
   order are computed, the two columns are already one string. Five separate
   call sites carried comments describing local workarounds for this.

Measured over 74 two-column pages: real gutters ran 5–26pt (median 16), within-
line word gaps had a p95 of 6pt, and line height a median of 10pt. Against the
page's own line height the gutter is stable at a median of 1.6×.

## The new approach

**Spatial segmentation first, then everything block-local.**

```
              raw WORD extents (never span bboxes)
                            │
                  recursive XY-cut
        (horizontal whitespace -> stacked blocks)
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
  block: full-width table              block: 2-column prose
  (no gutter of its own)               (gutter found at x=306)
         │                                     │
   line assembly with no gutter        line assembly split at x=306
         └──────────────────┬──────────────────┘
                            ▼
                  one shared block structure
      (line grouping, table detection, flow clustering,
       reading order all read it -- no second opinion)
```

Four properties make it general:

- **Word geometry, never span geometry.** A span bbox is a merged, padded run.
  On a Word-produced two-column page the same gutter measures 5pt through spans
  and 8pt through words; those 3pt of phantom ink hid the column.

- **Typography-relative thresholds.** The gutter floor is `0.55 ×` the page's
  median line height, not a constant. The block-split gap is a robust outlier
  test (median + 3 MAD) on the page's own gap distribution, floored at one line
  height — so it adapts to the document's leading instead of assuming one.

- **Gutters carry vertical extent.** A gutter is `(x, y_top, y_bottom)`: where
  it separates columns, not merely that it does. A page is rarely one layout
  all the way down, and a gutter belonging to the prose has no authority over
  the full-width table above it. This is what makes mixed layouts work with no
  special case — and it was a real defect: a three-table arXiv page had its
  full-width Table 1 sliced down the middle by the gutter of prose two-thirds
  further down.

- **A column must be wide enough to set text in.** A band narrower than 8 em is
  a table column, a label gutter or a margin — not a page column. Typographic
  convention puts a readable measure at 20–35 em; even a narrow newspaper
  column is ~12. Measured: the arXiv bands were 26 em, the PDF-reference
  operator table's "columns" were 3–4 em. A five-column A4 layout still clears
  the bar at ~10 em, so it costs no real layout anything.

### Handling tables

A table's internal whitespace is empty over the full height of the content
around it — exactly like a gutter. Five candidate discriminators (band width,
vertical coverage, line count, line-spacing regularity, fill ratio) were each
measured on both populations and each overlapped, so none was shipped.

The circularity breaks by **ordering**, not by tuning: tables are detected
first, from unsplit lines, with no gutters active — so the first pass behaves
exactly as it always has. Columns are then projected from the words those
tables do not own, so a table's own gaps cannot manufacture a column. A block
that is mostly table abstains entirely, since its few non-table words are
usually just the caption.

### Structural metadata

Where a file ships a tagged structure tree, `tags.reading_order_ranks` walks it
in document order and `order_regions` uses that instead of geometry. A
declaration beats an inference: the tree's order *is* the logical reading order
by definition, and it covers any column count, sidebars, and layouts that
change shape mid-page with no special case. It applies only when tags cover
≥60% of the page's regions, because a partial order is worse than none.

Coverage measured on 120 corpus documents: 0/106 arXiv papers are tagged (LaTeX
emits none), against 10/14 of the Word/publisher/web documents. So tags are
worth having and cannot be relied on — geometry still carries the bulk.

## Results

- `eval/regression.py`: **23/23** cell-level table fixtures (unchanged)
- `tests/test_columns.py`: **14** new tests — 1/2/3-column, mixed layout, RTL,
  table-vs-column, hanging markers, type-size scaling, declared order
- `eval/invariants.py` over 128 PDFs / 8,548 pages: 0 crashes, 0 presentation
  forms, 0 non-rectangular tables, 0 non-deterministic pages, coverage 0.999

## Known limitation

Letting the borderless-table detector see gutter-split lines raises the table
gold-set score from 63.8% to 68.4% (24/43 perfect instead of 22, 13 total
losses instead of 14) but introduces one false table on a two-column index
page, where three consecutive index entries align well enough to look tabular
once the column is isolated. That is a regression, so it is not enabled. The
fix belongs in `detect_borderless_tables` — a 2-column borderless candidate
sitting inside a single column band needs stronger evidence than one spanning
the page — not in column detection.
