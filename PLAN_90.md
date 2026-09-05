# Getting every dimension above 90%

Every number below is measured, and every "expected gain" is arithmetic over
the actual failing cases — not an estimate. Where a bottleneck turned out to be
different from what we assumed, the measurement is shown.

## Where we actually are

| dimension | now | measured bottleneck | not the bottleneck |
|---|---:|---|---|
| Citability | **100%** | — | — |
| Content faithfulness | 97.0% | `digit_integrity` 0.919 | recall/precision both ≥0.98 |
| Reading order | 0.971 | `contiguity` 0.955 | `precedence` 0.986 |
| Chunk integrity | 84.9% | `ends_complete` **0.498** | everything else ≥0.88 |
| Structure | 70.2% | `recall` **0.22** | precision 0.93, hierarchy 1.00 |
| Table record match | **63.8%** | header row dropped | cell text (0.8 pts total) |

Two assumptions were tested and are false:

- **"Structure is bound by the table score."** Correlation between a document's
  table score and its structure score is **+0.13**, and on the 208 documents
  with no table ground truth at all structure is **0.694** — identical to the
  overall mean. Structure is an independent problem.
- **"Table Record Match is 68.4%."** It is **63.8%**. 68.4% was a variant that
  was measured but *not shipped*, because it introduced a false table on a
  two-column index page.

---

## 1. Tables: 63.8% → 91.6%

The 43 hand-verified gold tables, classified by how the shape differs from
truth. This is the whole gap, in four buckets:

| bucket | cases | gain if fixed | running total |
|---|---:|---:|---:|
| **fewer rows than truth** | 8 | **+18.5** | 82.3% |
| no table emitted at all | 4 | +9.3 | 73.1% |
| more rows than truth (split cells) | 4 | +7.6 | 71.4% |
| right shape, cell text differs | 5 | +0.8 | 64.6% |

**Buckets 1 + 2 alone reach 91.6%.**

### 1a. Header-row recovery — the single highest-value fix (+18.5)

Most of the "fewer rows" cases are off by exactly one row, and it is always the
same row. Confirmed on `b6aa3e42` p6:

```
truth   : RQ  | Null hypothesis (H0)          | Alternative hypothesis (H1)
          RQ1 | There is no difference in ... | The spec-delta workflow ...
          RQ2 | ...

emitted : RQ1 | There is no difference in ... | The spec-delta workflow ...   <- header
          RQ2 | ...
```

The header row is excluded from the detected table region, so the first data
row becomes the header. Every cell is extracted correctly and the table still
scores **0.00**, because `TableRecordMatch` keys on the header — which is the
right behaviour for RAG, since an agent reading that table looks up the wrong
column.

Why it happens is the thing to find first: the header row is usually styled
differently (bold, shaded, ruled beneath), and the row grouper appears to treat
that difference as a region boundary rather than as evidence of a header. The
fix is to test the row immediately above a detected table for header-ness
(shared column x-origins with the body, different weight/fill, no numeric
cells) and extend the region upward when it matches — not to loosen the region
boundary generally, which would pull in captions.

### 1b. Table detection recall (+9.3)

Four gold tables produce no table block at all: `CVX_10-Q` p8 (6×3),
`arxiv_2608` p18 (8×2), `nist_ir_828` p18 (12×8), `97892415638` p17 (27×4). A
27×4 and a 12×8 going undetected is not a threshold miss; these need diagnosing
individually before any rule is written.

### 1c. Wrapped-cell row splitting (+7.6) — your proposal #1

Real, and correctly identified: `e3580b13` p6 emits **14 rows for a 5-row
table**, which is exactly wrapped cell text becoming separate rows. Worth
doing, but it is the third bucket, not the first. Note `layout.py` already has
`_merge_wrapped_label_rows`, so this is likely extending an existing pass
rather than a new one.

### On proposals 2 and 4

- **X-histogram column projection.** Only 2 of 21 imperfect cases have a column
  count mismatch (`19x2 → 18x1`). As a standalone change it is worth under 2
  points. It may well be *part of* the header fix — a spanning header is
  precisely what breaks a whitespace column finder — so fold it in there rather
  than running it as its own workstream.
- **Sidebar hoisting.** No evidence for it in the data: structure's deficit is
  heading recall, not block sequencing, and `hierarchy` is already 1.000.
  Defer until a measurement asks for it.
- **Borderless anchor rules (your #3).** Do this — it is what would unblock the
  +4.6 points currently withheld, by killing the index-page false positive that
  made letting the borderless detector see gutter-split lines a regression.

---

## 2. Structure: 70.2% → 90%+

`recall = 0.22` against `precision = 0.93` and `hierarchy = 1.000`. rtldoc is
not mislabelling headings; it is not finding them.

But the recall proxy is partly wrong, and that has to be fixed before the
parser is touched. Classifying all 209 spans the metric calls "missed
headings", over 117 pages:

| what it actually is | share | verdict |
|---|---:|---|
| title line | 30.6% | **real miss** — the document title should be h1 |
| other short large text | 23.4% | needs inspection |
| marginal stamp (rotated arXiv side-text) | 22.0% | **metric artifact** — exclude rotated text |
| person name (author block) | 21.1% | **metric artifact** — front matter is not a heading |
| section number alone (`"1"` split from `"Introduction"`) | 2.9% | **real miss** |

So ~43% of the deficit is my metric and ~34% is real. Order of work:

1. Exclude rotated/marginal text and front-matter author blocks from the
   heading-truth proxy. Measurement fix; expected to move recall a long way
   with no parser change.
2. Emit the document title as `heading1`. Largest real bucket.
3. Join a section number to its heading text (`"1"` + `"Introduction"`).
4. Re-classify the remaining 23% and decide.

---

## 3. Chunk integrity: 84.9% → ~94%

`ends_complete = 0.498` — **2,394 chunks end mid-sentence**, against every other
sub-metric ≥0.88. This is a bug in `eval/ragbench/chunking.py`, not in rtldoc:
the packer flushes as soon as a chunk reaches the target size and only breaks at
a sentence when it overflows the hard maximum. Making the flush seek the nearest
sentence end should take that sub-metric to ~0.95 and the dimension to ~94%.

Worth stating plainly: this is a benchmark fix, not a parser improvement. It
matters because until it is fixed, chunk integrity is measuring my chunker
rather than rtldoc's output.

---

## 4. Content: 97.0% → 98%+, Reading order: 0.971 → 0.99

- `digit_integrity` 0.919 is the weakest content sub-metric and the one that
  matters most for financial and numeric RAG. Diagnose which digits are being
  dropped or invented before proposing anything.
- `contiguity` 0.955 is the remaining reading-order gap — the column work moved
  it from 0.946, and the residue is where columns are still not detected.

---

## Method note: 43 cases is a small n

The table plan is built on 43 hand-verified tables from 13 documents, all
English, all ruled. Two consequences:

- Tuning until those 43 hit 90% risks fitting the sample rather than the
  problem. The gold set should grow — and it must gain an **Arabic slice**,
  since half the stated scope currently has no table number at all.
- Borderless tables have no ground truth here. They are unmeasured, not
  passing.

The order that respects this: fix the header row (a mechanism, not a
threshold), re-measure, *then* expand the gold set, and only then chase the
remaining buckets.

## Suggested order

1. Chunker sentence-boundary flush — small, isolated, +9 points, benchmark-only
2. Structure metric artifacts (rotated text, author blocks) — measurement truth
3. **Table header-row recovery** — the biggest single win, +18.5
4. Expand the gold set, add Arabic tables — before tuning further
5. Table detection recall on the 4 undetected cases
6. Document title as h1, section-number joining
7. Borderless anchor rules — unblocks the withheld +4.6
8. Wrapped-cell row merging
