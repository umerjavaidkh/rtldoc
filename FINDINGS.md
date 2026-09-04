# rtldoc — issues found and fixed

Written so you can **verify every claim yourself**. Each entry gives the
symptom, the root cause, what changed, and a command to check it.

- Baseline for all "before" numbers: `a935021` (v1.0.7)
- 13 commits. `v1.1.0` (7 commits) is **published to PyPI**; 6 commits after
  it are on `main` but **not pushed**.

---

## Headline results

| Measure | Scope | Before | After |
|---|---|---:|---:|
| Unusable headings | 1,462 docs / 31,354 pages | **17.1%** | **2.3%** |
| Documents ≥30% unusable headings | 1,361 docs with headings | **206** | **33** |
| Duplicated-text pages | 118 PDFs / 8,349 pages | ~240 | **193** |
| Presentation-form leaks (encoding) | 31,354 pages | — | **0** |
| Crashes | 1,462 docs | — | **2** |
| Table proposals, CVX 10-K | 264 pages | 96 | 146 |
| Golden regression fixtures | 23 cell-level | 23/23 | **23/23** |

---

## How to verify

```bash
cd /Users/umerjavaid/Documents/umerwork/AI/pdf_parser

python3 eval/regression.py                 # 23 cell-level table fixtures
python3 eval/invariants.py book/ corpus/   # property checks, 118 PDFs
```

To compare against old behaviour for any single claim:

```bash
git stash            # or: git checkout a935021 -- rtldoc/
# ...re-run the measurement...
git stash pop        # or: git checkout HEAD -- rtldoc/
```

---

# PART 1 — Table structure (released in v1.1.0)

Found by diffing real output against real documents: a 264-page SEC filing
(CVX 10-K), the BERT paper, the PDF 1.7 reference.

### 1.1 A small ruled row destroyed the large table around it — `69cfdbb`

**Symptom.** Tables lost their header and most data rows. On CVX p9 a
9-column × 11-row employee table collapsed to just its 3-row totals section.

**Root cause.** A table's own subtotal row is often underlined with a real
drawn rule. The ruled-table detector found a *tiny* table covering only that
row, sitting inside the correct borderless table. Conflict resolution treated
any overlap as a duplicate and kept the **tiny fragment**, discarding the
complete table.

**Fix.** The ruled table wins only when it covers roughly the same extent.
When it is a clearly smaller fragment (<70% of area) of a strictly more
complete candidate, the complete region wins.

**Affected pages found:** CVX 9, 118, 141, 155, 190, 202.

### 1.2 Kerning-merged numbers crammed into one cell — `f44a396`

**Symptom.** `'904 19,146'` in a single cell, neighbouring cells blank.

**Root cause.** Tight kerning in summary rows makes PyMuPDF emit 2–3
logically separate numbers as one text run. Cell assignment dumped the whole
run into whichever cell it overlapped most.

**Fix.** Split such a run at the table's own column boundaries — gated
tightly: word count must exactly match straddled cells **and** every word
must look numeric. (A looser first version corrupted a citation-bracketed
model name and a hex-reference row; caught by the regression corpus.)

Also: small tables no longer permanently reject a genuine column when one
row's merged span costs it a single vote, and a value straddling two cells
~50/50 now breaks the tie by right-edge alignment.

### 1.3 Two side-by-side lists merged into one holey grid — `69cfdbb`

**Symptom.** A contents page's titles and its "Note N" list merged into one
4-column table where every row was half empty.

**Fix.** If no row (allowing small tolerance) has content on both sides of a
column boundary, split into two coherent tables.

### 1.4 NEW: grids drawn as filled boxes are now real tables — `8eed163`

**Symptom.** BERT Figure 2 rendered as ~45 disconnected tokens; Figure 4 as
~80. All row/column relationships lost.

**Root cause.** Papers and Word exports often draw a table as a field of
uniformly sized filled boxes with **no rules at all**. The rule detector had
no rules to find; the alignment voter never fired because the text in each
box is short and irregular. Every box became its own isolated region.

**Fix.** The grid's own regularity is the evidence: boxes are grouped by
proximity (relative to their own median size), then must resolve into ≥2 row
clusters sharing a y-centre and ≥3 column clusters sharing an x-centre, and
fill ≥55% of their rows × columns product.

**Result.** Figure 2 now emits the actual 4×12 grid, row labels included:

| Input | [CLS] | my | dog | … |
|---|---|---|---|---|
| Token Embeddings | E[CLS] | Emy | Edog | … |
| Segment Embeddings | EA | EA | EA | … |
| Position Embeddings | E0 | E1 | E2 | … |

**Safety proof.** The Arabic teacher's guide is the corpus's heaviest user of
filled-box badges. It is **byte-identical across all 143 pages** (350
activity markers, 145 tables unchanged) — so the detector does not fire on
scattered badges.

---

# PART 2 — Duplicated text (released in v1.1.0)

Three distinct bugs, all the same class: two overlapping regions each
rendering the same text.

| # | Symptom | Root cause | Commit |
|---|---|---|---|
| 2.1 | BERT Fig. 2 tokens output twice, interleaved | A per-token chip and the enclosing panel both contain a line fully (containment 1.0 each) — winner decided by iteration order, panel won | `a0259fd` |
| 2.2 | BERT Fig. 1 diagram output twice (flat + as table) | A line poking slightly outside a table's bbox scored <1.0 against it but 1.0 against the enclosing panel | `c2c90ea` |
| 2.3 | Caption fragments repeated under the caption | Column bucketing split a full-width caption; the part past the gutter became a region **nested inside** the caption's own | `8ec937e` |

Fix 2.1 needed an "essentially contained" test (≥0.95), not exact-tie:
geobidi pads line bboxes, so a line genuinely inside a chip measured 0.994
while measuring exactly 1.000 against the roomier panel.

**Result:** duplicated-text pages 240 → 193 corpus-wide; BilArabi 4 → 0;
pdfreference1.7old 20 → 17; GOOGL 10-K 1 → 0.

### 2.4 One caption copied onto every nearby figure — (post-1.1.0)

**Symptom.** On a full-page architecture diagram built from ~15 icon images,
**all 15** received the same `Figure 1: …` caption — making it the worst
duplicated-text page in that document (letter excess 0.378).

**Fix.** A caption is claimed by at most one figure — the nearest. Figures
that lose carry no caption, which is the honest answer for an icon inside a
larger diagram. That document went to **0** flagged pages.

---

# PART 3 — Arabic

### 3.1 FIXED — ornate parentheses counted as encoding leaks — `2a427c7`

**Symptom.** `arwiki_computer.pdf` p1 tripped the HARD invariant
"presentation forms leaked into output".

**Root cause.** The check tested *block membership*: anything in
U+FB50–FDFF / U+FE70–FEFF counted as a leak. But **U+FD3E/U+FD3F are ornate
parentheses** `﴾ ﴿`, used to quote Qur'anic verses — ordinary characters with
**no decomposition**. The parser correctly preserved them; the checker was
wrong to call it a leak.

**Fix.** A genuine presentation form always has a compatibility decomposition
(`U+FEFB → "<isolated> 0644 0627"`). Parser and checker now share that test,
so they cannot disagree.

```bash
python3 -c "
from rtldoc import arabic
for c in '﴾﴿ﻻﺍ': print(repr(c), arabic.is_presentation_form(c))"
# ﴾ False   ﴿ False   ﻻ True   ﺍ True
```

### 3.2 NOT FIXED — reversed lam-alef in BilArabi

Your hint: 79% of lam-alef pairs reversed; الاصطناعي stored as االصطناعي.

**Where it actually is — your numbers point at BilArabi, not the new files:**

| File | correct | reversed | ligatures |
|---|---:|---:|---:|
| arwiki_ai.pdf | 118 | 0 | 0 |
| arwiki_education / engineering / syria | 1 each | 0 | 0 |
| **BilArabi_TG07.pdf** | **0** | **10** | **58** |

**What the parser does on BilArabi:** deshaping works (58 ligatures → 0) but
the **order** stays wrong (10 → 11 reversed).

**Root cause.** `arabic.normalize` already reverses *before* deshaping
precisely to keep lam-alef atomic. That guard never fires here because the
failing text has **zero presentation forms** — the ligature arrives already
split into two base letters in visual order. Sample raw line, page 64:

```
.ّبعد ذلك، حدّد لهم البحث عن معلوماتٍ حول الذكاء االصطناعي
```

Letters are logical (بعد = ب,ع,د), but the sentence period sits at the front
and the lam-alef is swapped.

**Why I did not fix it.** `ا ل` is also the definite article **ال**, and
occurs legitimately inside words (قال = ق,ا,ل). A blind swap corrupts correct
text. The one reliable signal I found is a word starting with **two alefs**
(اا), invalid in Arabic — that covers الاصطناعي, but I have not verified it
covers your other 600+ cases.

**What would settle it:** confirm whether to scope the fix to
double-alef-at-word-start, or give me expected output to validate a broader
rule against.

### 3.3 New Arabic files parse well

| file | pages | crashes | coverage | leaks |
|---|---:|---:|---:|---:|
| arwiki_ai | 15 | 0 | 0.999 | 0 |
| arwiki_arabic_language | 17 | 0 | 0.985 | 0 |
| arwiki_computer | 18 | 0 | 0.998 | 0 |
| arwiki_saudi | 32 | 0 | 0.993 | 0 |
| arwiki_chemistry | 13 | 0 | 0.998 | 0 |

HTML under `output/arwiki_*.html/`.

---

# PART 4 — Heading quality (the largest win)

Ranking source: your `agentic_graph_rag/eval/heading_quality_report.md`.

**Important process note.** I first measured on a 40-document sample and got
7%. The **full 1,462-document audit showed 17.1%** — the sample had hidden
the dominant failure mode completely (`mid-sentence fragment`: 4 in sample vs
**1,442** corpus-wide). All numbers below are full-corpus.

| stage | unusable | docs ≥30% bad |
|---|---:|---:|
| baseline | **17.1%** | 206 |
| after all four fixes | **2.3%** | **33** |

Failure modes, before → after (full corpus):

| mode | before | after |
|---|---:|---:|
| mid-sentence fragment | 1,442 | **0** |
| empty | 196 | 139 → **0** |
| line-break fragment | 178 | 58 |
| mostly digits | 12 | 10 |
| no words at all | 1 | 1 |
| body sentence (sample) | 46 | 2 |

### 4.1 Rotated text shredded into single characters — `5274fac`

**Symptom.** Headings like `7 1 0 2 n a J 0 3 ] G L .s c[ 9 v0 89 6. 2 1 41 :
v i X ra` — your report's #1 mode (749 headings) — plus corrupted body text.

**Root cause.** The geometric line rebuild assumes horizontal text: a line is
glyphs sharing a **y**, ordered by **x**, with word gaps measured along x.
Rotated text is the opposite — its glyphs share an **x** and advance down
**y**. So y-grouping made every glyph its own 1-character "line", and those
interleaved with real body lines in the same y band.

This is the vertical arXiv stamp **on essentially every arXiv PDF**:

```
before: '...such as ELMo (Peters / 2 / y / a / M / 4 / 2 / ] / downst...'
after : 'arXiv:1810.04805v2 [cs.CL] 24 May 2019'
```

**Fix.** Glyphs carry the writing direction PyMuPDF already reports. Rotated
glyphs group by shared x, order along y in writing direction, skip the bidi
x-sorts, and use y-based gap measurement.

### 4.2 Body paragraphs typed as headings — `d9a9d02`

**Symptom.** Whole paragraphs stored as headings; longest was **3,171
characters**.

**Root cause.** Heading = "dominant font size > 1.35 × page median", with no
structural check. That median is unstable: on pages with lots of small text
(figure labels, axis ticks, equation subscripts, dense tables) it is dragged
below real body size, so **ordinary paragraphs clear the bar**.

**Fix.** A heading must also be *shaped* like one — ≤200 chars, ≤3 lines,
<2 internal sentence breaks. Font-independent, so it holds exactly where the
size signal is poisoned. Measures **rendered** text, not raw spans (spans
alone still let an 1,818-char heading through).

**Longest heading: 3,171 → 189 chars.**

### 4.3 Rotated stamp merged into real headings — `4bdb6b3`

**Symptom.** Once 4.1 made stamps legible, they appeared *as* headings and
merged into real ones:

```
'arXiv:2608.03477v1 [cs.DB] 4 Aug 2026\nContents'
'arXiv:2608.05346v1 [cs.NI] 5 Aug 2026\ntency Communication'
```

**Root cause.** A stamp's bbox is a tall narrow strip spanning most of the
page height — it distorted gutter detection and, being vertically adjacent to
everything, merged into whichever body block it touched.

**Fix.** Rotated spans cluster into their own region, typed `page_furniture`.

### 4.4 Headings scattered around charts/figures — `a59a352`

**Symptom.** `'2 2'`, `'12'`, `'(a)\n(b)'`, `'∥q − x∥2 ≈ ∥q∥2 + …'` as headings.

**Root cause.** A large-font subscript or axis label cleared the size bar.

**Fix.** A heading must contain a run of 2+ letters. **Script-agnostic**
(`[^\W\d_]{2,}`, not `A-Za-z`) so Arabic and Greek headings still pass.

### 4.5 Continuation lines and empty headings — `9fa9a65`

**Symptom (a).** Body continuation lines as headings — 1,442 of 1,829
failures corpus-wide:

```
'research on state-of-the-art self-driving systems, cannot be'
'determines the shape of a DAG primarily by specifying either'
```

Short, single-line, has words — so they passed every other test.

**Fix (a).** A heading does not open mid-sentence: reject a lower-case first
character. Case-aware by construction — Arabic and other uncased scripts
report `islower()` False, so their headings still pass.

**Symptom (b).** 196 headings with **empty text**.

**Root cause (b).** `_dedupe_blocks` can empty a block when every line it held
was a duplicate owned better elsewhere. The textless block kept its role, and
an empty "heading" fabricates a structure boundary for retrieval to trust.

**Fix (b).** Drop empty blocks after dedupe; figures and tables exempt (their
payload is an image or cell grid, not text).

---

# PART 5 — Visual content: gap analysis (NOT yet implemented)

Measured to decide the architecture, not yet built.

**Vector vs raster is corpus-dependent — do not design around one ratio.**
In the arXiv sample: 156,722 vector draw ops vs 1,917 embedded images (82:1).
But scanned contracts, invoices and archives are ~100% raster. The parser
must route **per element**, never by a global assumption.

**What raster images actually contain** (1,917 images sampled):

| kind | count | share | information carrier |
|---|---:|---:|---|
| flat / line-art (rasterized charts, diagrams, screenshots) | 1,207 | **63%** | **text inside the image** |
| continuous-tone (photos, heatmaps, microscopy) | 710 | 37% | depicted content |

Median image ~355×355px — real figures, not icons, and large enough for OCR.

**Current output for an image is metadata, not information:**

```
caption: ''
ImageVisual: width 16, height 16, colors ['#f0f0f0','#d01010'], kind 'photo'
```

**Already working (no model, no GPU):** vector diagram extraction —
nodes, directed edges, labels, rendered as **Mermaid** in HTML:

```
nodes: Acrobat, Macintosh application, Windows application, Adobe PDF printer
edges: Acrobat → Adobe PDF printer; Adobe PDF printer → Macintosh application
```

**Recommended tiering** (cheapest and most universal first):

| tier | covers | tool | RAM | deterministic |
|---|---|---|---:|---|
| 1 | text already in the PDF | existing | 0 | yes |
| 2 | vector charts/diagrams | extend `visual.py` | 0 | yes |
| 3 | **text inside raster images** | RapidOCR (ONNX) | ~100MB | yes |
| 4 | photo semantics | small VLM, opt-in | ~350MB | **no** |

Tier 3 first: it is the biggest universal gap (images contribute *nothing*
today) and applies to every document type.

**Evidence against putting a layout model in front:** DocLayout-YOLO tested
this session — **0 of 169** pages where geometry found nothing did it
confidently disagree; ~0.9s/page (50–100× the vector pipeline); and it called
a real sparse table "plain text". Branch `experiment/learned-layout-classifier`,
not merged.

**Two requirements for multi-million scale:**
1. Store the canonical AST **per element with extractor name + version**, so
   improving the chart parser re-runs only affected elements, not the corpus.
2. Tag every element `vector` / `text` / `ocr` / `vlm` **plus confidence**. A
   VLM-guessed number and a vector-exact one must not look identical
   downstream.

---

# PART 6 — Known open, NOT fixed

Listed so nothing here is assumed done.

1. **Reversed lam-alef in BilArabi** (§3.2) — diagnosed; needs a scope
   decision before it is safe to fix.
2. **Remaining 2.3% bad headings** — `line-break fragment` (58),
   `mostly digits` (10), `no words at all` (1).
3. **2 crashes** in 1,462 documents — not yet diagnosed.
4. **IRS p969 index page** — a 3-column back-of-book index still interleaves.
   `_merge_marker_columns` folds a genuine narrow index column into its
   neighbour. Every signal measured (width ratio 0.47, row-match 0.97,
   shortness 0.83) is *identical* to a real marker column. No reliable
   discriminator found; changes reverted rather than shipped as a guess.
5. **Extreme-density tables** — a 12+ column, 3-header-level table can still
   crowd two values into one cell (CVX p232, TCO/Other rows). A fix existed
   but no threshold separated it from confirmed-bad cases (0.017 apart in
   score), so it was not shipped.
6. **Overprinted glyphs** — `[[CCLLSS]]`, `TTookk 11` on BERT p15. The source
   PDF draws those glyphs twice; primitive-level, untouched.
7. **Spurious figure captions** on BERT p3 — icon-sized figure regions inside
   a diagram still attract caption text.

---

# Commits

**Published — PyPI `pdf-rtldoc` 1.1.0:**

```
5273f98  Bump version to 1.1.0
8eed163  Recover a regular grid of chips as a real table          (new capability)
8ec937e  Merge a flow region nested wholly inside another
c2c90ea  Make a table authoritative over its own footprint
a0259fd  Fix duplicated/scrambled text where a chip sits inside a larger container
f44a396  Fix p232: small-table column support, right-aligned cell ties
69cfdbb  Fix three universal table-detection bugs
```

**On `main`, NOT pushed:**

```
9fa9a65  Reject continuation lines and textless blocks as headings
a59a352  A heading must contain at least one actual word
4bdb6b3  Keep 90-degree rotated margin text out of the body flow
d9a9d02  Require a heading to be shaped like one, not just set in a larger font
5274fac  Reconstruct 90-degree rotated text on its own axis
2a427c7  Don't count ornate parentheses as leaked presentation forms
```

Those 6 are a meaningful quality jump (headings 17.1% → 2.3%) and are
release-worthy whenever you want v1.2.0.

## Open: activity markers render on the wrong side (RTL)

Reported by the user, seen throughout the book, not yet fixed.

Numbered activity chips (1, 2, 3, ...) render flush LEFT in the HTML while
the page places them on the RIGHT, which is where an RTL reader expects a
list marker. The text blocks beside them are correct.

What is known:

- The markers are their own blocks (`role == "activity_marker"`), one per
  chip, and their geometry is right: on p112 every chip sits at
  x=446-464 on a 720pt page, i.e. to the RIGHT of the inner page's text
  column (x=69-434). So this is a rendering problem, not a layout one.
- A marker's text is a bare number. It contains no strong-RTL character,
  so its resolved direction is LTR and it aligns to the left edge of its
  container, unlike the `dir="rtl"` paragraphs around it.

Likely fix: emit `dir="rtl"` (or an explicit right alignment) on the
marker block the same way the paragraph blocks get it, rather than
letting the direction be inferred from a digit. Check the same for any
other block whose text is digits-only -- folios, table marker columns --
since they will have inherited the same defect.

Not attempted yet; noted for a later pass.
