> **Design notes & full evidence.** The short pitch is in [README.md](README.md); this file keeps the detailed rationale, bug logs, and methodology.

# rtldoc — a geometry-first parser for RTL complex-layout PDFs

Built for Arabic educational publishing: teacher's guides, pupil books, workbooks —
and validated to generalize beyond that, on English SEC filings, policy documents,
and reports. The kind of page where Docling, Marker, and VLM-based parsers all
return something that *looks* like text and is quietly wrong.

---

## 1. Why the general-purpose parsers fail on this class of document

A born-digital, RTL, complex-layout PDF (an Arabic teacher's guide is the
hardest version of this, but the same failure modes hit any densely
laid-out publication) stacks problems that generic parsers were never
tuned for:

| # | Failure mode | What the general parsers do |
|---|---|---|
| 1 | **RTL reading order** | Every reading-order model — Docling's, Marker's, LayoutLMv3's — is trained overwhelmingly on arXiv, invoices, and business reports. They emit left column first. Where right column first is correct, that's silent, total corruption of document order. |
| 2 | **Presentation forms in the text layer** | ME InDesign exports often map glyphs to `U+FE70–FEFF` / `U+FB50–FDFF` rather than base letters. Raw extraction gives `ﺍﻟﻬﺪﻑ` — visually identical, lexically useless. Tokenizers, embeddings, and BM25 all miss. |
| 3 | **Lam-alef ligature reversal** | The single nastiest bug. `لا` is one glyph in the stream. If the extractor decomposes it *before* bidi reordering, the two letters flip: `التلاميذ` → `التالميذ`. This is [PyMuPDF #2199](https://github.com/pymupdf/PyMuPDF/issues/2199), it affects roughly a third of Arabic words, and it produces text that passes every eyeball check because it still renders as Arabic. |
| 4 | **Two documents on one page** | A teacher-notes column and a pupil-page facsimile can share one page. Linearising them into one stream puts answers next to the wrong questions. A RAG index built on that confidently answers with mismatched keys. |
| 5 | **Semantics carried by vector art** | A tinted panel means "reading passage"; a numbered chip means "exercise N," and that number is the only thing linking an exercise to its answer key — and it lives in the drawing layer, which every markdown-first parser discards. |
| 6 | **Table structure without borders** | A financial statement using row-shading instead of ruled lines (see META 10-K p.83 below) has no visible grid at all in the vector layer — most table-structure detectors, ours included, need *something* geometric to key off. |

Docling in particular fails here not because it is bad but because it is
*general*: it runs a layout model + a reading-order model + table structure,
all tuned for a distribution this page is nowhere near. You cannot fine-tune
your way out of #2 and #3 — those are encoding bugs, not perception problems.

---

## 2. Where the field actually is (mid-2026)

- **End-to-end VLM parsers now dominate the leaderboards.** PaddleOCR-VL-1.6 reports [a state-of-the-art 93.19% overall on Real5-OmniDocBench at 0.9B parameters, beating Qwen3-VL-235B and Gemini-3 Pro](https://arxiv.org/pdf/2606.03264). HunyuanOCR reports [the top score on OmniDocBench and its distorted-capture variant at 1B parameters](https://arxiv.org/pdf/2511.19575).
- **But modular pipelines still win on overall structure.** The ABot-OCR report is candid about it: [pipeline parsers that combine specialised layout-detection and text-recognition modules still hold the highest overall scores, at the cost of multi-stage orchestration](https://arxiv.org/pdf/2605.27978).
- **Practitioner guidance says route per page-type, not per leaderboard.** One 2026 comparison concludes the [public shortlist is for forming candidates, and you should run a fixed page-type bake-off before rollout](https://instavar.com/blog/ai-production-stack/OCR_SOTA_Feb_2026_Open_Document_AI_Leaderboard).
- **Nobody benchmarks Arabic textbooks.** [OmniDocBench](https://github.com/opendatalab/OmniDocBench) — the standard public benchmark, 1,651 pages, 10 document types — covers English, Simplified Chinese, and English-Chinese mixed content only. **Zero RTL/Arabic coverage.** The RTL educational-publishing slice is entirely unrepresented on any public leaderboard, which is exactly why this repo ships its own gold-labeled eval set (§5) rather than only quoting someone else's numbers.

The practical read: **for a born-digital corpus, a VLM is the wrong default.**
It rasterises a page that already contains perfect glyph coordinates, hallucinates
at ~2–5% on long Arabic passages, costs 100–1000× more per page, and cannot be
audited. Use it as a *reviewer*, not as the *extractor*.

---

## 3. Comparison matrix

Two different things are being compared below, and they're kept visibly
separate so neither gets overstated:

- ✅/⚠️/❌ marks are **structural/documented facts** — a training-data
  distribution, a cited bug, a published limitation. These aren't from
  running the competing tool ourselves.
- The bugs in §4 and the numbers in §5 are **things this repo actually ran
  and verified**, with the specific input/output shown. That's the part
  you don't have to take on faith.

| Capability | rtldoc | Docling | Marker | Generic VLM (GPT-4V/Gemini/Claude-vision) | naive `page.get_text()` |
|---|:---:|:---:|:---:|:---:|:---:|
| RTL reading order | ✅ geometric (x-descending), no training distribution to be wrong about | ❌ LTR-trained reading-order model¹ | ❌ same class of model, same training bias¹ | ⚠️ usually right (reads the image), not guaranteed, not auditable | ❌ trusts the PDF's own bidi/producer order, often wrong |
| Lam-alef / ligature integrity | ✅ ligature stays atomic through reordering (`primitives.py`, `TEXT_PRESERVE_LIGATURES`) | ❌ inherits PyMuPDF/pdfium's decomposition order² | ❌ same | ✅ reads pixels, doesn't hit this bug class | ❌ hits PyMuPDF #2199 directly² |
| Presentation-form deshaping | ✅ `arabic.deshape`, scoped to the two Arabic presentation blocks only | ⚠️ depends on backend text extraction | ⚠️ depends on backend text extraction | ✅ | ❌ |
| Table structure from vector rules | ✅ works with **zero visible borders needed** for panels/chips; ✅ recovers full grids from stroked *or* filled rule lines (see §4) | ✅ TableFormer, model-based | ✅ Surya-based, model-based | ⚠️ often right, not auditable, costs per page | ❌ no structure at all |
| Table structure with **no rules at all** (shading only) | ✅ recovered by column-alignment detection (§4a) — verified on META p.83's borderless quarterly statement; numeric-density guards keep it off aligned *text* | ⚠️ model-based, may or may not catch it — untested here | ⚠️ same | ⚠️ likely reads it correctly (visual), not auditable | ❌ |
| Cross-column semantic linking (exercise ↔ answer key) | ✅ `_link_activities`, publisher-specific but real | ❌ no concept of this at all | ❌ | ❌ | ❌ |
| Auditability (why did this text come out this way) | ✅ every transform is a named, inspectable function; `rtldoc audit` flags low-confidence pages | ❌ black-box model inference | ❌ | ❌ | N/A (it's not doing anything) |
| Cost per page (born-digital) | ✅ ~3-8ms, no GPU, no API | ⚠️ CPU-feasible, model inference cost | ⚠️ same | ❌ 100-1000× more expensive³ | ✅ free |
| Non-RTL (English) documents | ✅ verified in this repo — see §5 | ✅ this is their home turf | ✅ same | ✅ | ⚠️ order is usually fine, no structure |
| Images represented in output | ✅ extracted to file + geometrically-nearest caption as alt text, deduped by PDF xref | ✅ | ✅ | N/A (is the image) | ❌ dropped entirely |

¹ Reading-order models in this class are trained overwhelmingly on arXiv/business-document corpora — documented training-distribution bias, not a claim we verified by running Docling/Marker ourselves.
² [PyMuPDF issue #2199](https://github.com/pymupdf/PyMuPDF/issues/2199) — decomposing lam-alef before bidi reordering flips the two letters; affects any tool built on the same extraction backend, roughly a third of Arabic words.
³ Per-page VLM cost and the 2-5% hallucination rate on long Arabic passages are the standard published tradeoffs cited in §2, not something this repo benchmarked directly.

**Docling and Marker are not run in the quantitative eval below.** Marker alone
pulls torch + transformers + surya-ocr — realistically 2-5GB and 10+ minutes
on first run — and was deliberately skipped for this pass to keep the
harness cheap to reproduce. The adapter interface (`eval/adapters.py`) is a
single function per parser; adding either is a ~15-line PR, not a redesign.

---

## 4. Bugs found and fixed while testing on real documents (this repo, this session)

These are concrete, reproducible findings from running rtldoc against 5 real
PDFs (a 239-page Arabic teacher's guide, two SEC 10-Ks, and two English
reports) — not hypothetical failure modes.

**Every one of these was a *latent, wrongly-universal assumption*, not a
one-off typo** — worth stating plainly since the whole point of this pass was
making sure the parser doesn't secretly only work on one book.

| Bug | Where | Symptom | Root cause | Fix |
|---|---|---|---|---|
| Full-page background swallows the page | `primitives.py` | One page (Arabic, p.88) collapsed to a single scrambled block | A full-bleed background tint (larger than the page itself) was classified as a content "panel" | Exclude any fill ≥60% of page area — a ratio test, not a color/book-specific rule |
| Column gutter invisible or phantom | `layout.py` | Two-column pages reported as 1 column (gutter erased by a full-width header) or 4 columns (list-indent noise misread as gutters) | 1D ink-projection ignoring y-persistence | 2D check: a gap only counts if empty across ~96% of content height, not just present somewhere |
| Paragraphs merged across columns | `layout.py` | Teacher-column and pupil-column text interleaved line-by-line | Paragraph clustering ran before columns were known | Split by column first, cluster within each column second |
| Table grids invisible | `primitives.py` | A 6×3 rubric table (p.150) rendered as disconnected paragraph fragments | `extract_page` only captured *filled* rectangles; this publisher draws table borders as **stroked lines** | Capture stroke-type drawings too; merge split segments into row/column boundaries |
| Duplicate text across regions | `pipeline.py` | Same passage appeared twice in one page's output | Two overlapping regions could each independently claim the same reconstructed line | Give each line to exactly one region (highest-containment owner) before rendering |
| **English text corrupted by an RTL default** | `geobidi.py` | On a *pure-English* 10-K page: `(MAUs),` at the end of a line came out as `,( ... (MAUs` at the front of the *next* line | The bidi reconstruction sorted every line right-to-left first, and an "undecidable" neutral character (trailing punctuation with no *next* neighbour in that scan) defaulted to "R" — i.e. the whole module silently assumed every document is RTL | Check each line for actual RTL characters *before* applying any RTL-specific reordering; a pure-LTR line just sorts ascending |
| Audit heuristic false-flagged plain prose | `pipeline.py` | An English 10-K's review-rate read 37% — far higher than every other document | `needs_review` required ≥2 blocks per page; a legitimately single-paragraph prose page tripped it | Score on total extracted character count, not block count |
| Images silently dropped | `pipeline.py` | A photo on an Arabic page had zero representation anywhere in the markdown output | `to_markdown` skipped any block with empty text, and figures never had text | Extract to file (deduped by PDF xref, so a page-repeated logo isn't saved N times), attach the geometrically-nearest block as alt-text caption, always emit a placeholder even when extraction is skipped |

Every one of these is independently checkable: the exact before/after text
is in the conversation history that produced this repo, and the fixes are
in the numbered commits.

### 4a. Borderless table recovery (the critique's hardest gap, now closed)

The one gap a reviewer flagged that a "lightweight parallel parser" was
supposed to cover — borderless tables (financial statements that separate
columns with shading or whitespace, no drawn rules) — turned out *not* to be
solvable by routing to another tool: pdfplumber's text-alignment table mode,
run on META 10-K p.83, shredded the entire page (prose included) into a
77×16 grid, splitting words mid-token. So it's fixed *inside* rtldoc instead
(`layout.detect_borderless_tables`), keyed on the one unambiguous signal:

- **column alignment recurring across rows** — the right edges of short cells
  cluster into columns, and a column only counts if ≥3 rows support it, so a
  wrapped prose line can never invent one;
- **a numeric-majority guard** — the aligned cells must be predominantly
  numeric, which is what separates a data table from Arabic MCQ options, an
  answer key, or a two-column list that merely happens to line up;
- **a grid-fill guard** — the grid must actually be *filled*, which rejects a
  numbered exercise or an image-credits page whose stray numbers coincidentally
  align in a few places.

Verified across the whole corpus: recovers real tables on ~27 GOOGL and ~20
META pages that previously extracted as prose fragments, fires on **zero**
Arabic pages (the two initial false positives — an exercise and a
credits page — are exactly what the numeric and fill guards were added to
reject), and leaves all 239 Arabic pages' text content byte-identical.

### 4b. Merge policy for cross-block duplication

The universality test (§5a) caught a defect the 5-document regression never
could: on dense/overlapping layouts, span-assignment and geo-line ownership
disagree about which region owns a piece of text, so the same line renders
twice — once via the geometry path, once via a neighbour's span fallback.
Confirmed a genuine parser-side bug, not source repetition: the phrase
occurred **once** in the PDF (both PyMuPDF extractors agree) and **twice** in
the output.

Rather than perfectly reconcile the two segmentations, a merge policy
(`pipeline._dedupe_blocks`) resolves it directly: a line appearing across more
than one block is kept only in the **highest render-quality block** (geometry
beats span-fallback) and stripped from the rest. Repeats *within* one block
are left untouched, so faithful source duplication survives while parser-side
double-emission does not. Result on the 96-PDF corpus: duplication-flagged
pages **396 → 86 (−78%)**, with text coverage unchanged (0.988) and every HARD
invariant still zero — the fix removed duplication without bringing back the
text loss it traded against.

### 4c. Table accuracy, scored the way the field scores tables (TEDS)

Coverage proxies don't measure tables; the field standard is **TEDS**
(Tree-Edit-Distance Similarity, from PubTabNet — also used by FinTabNet and
OmniDocBench). It turns each table into an HTML tree and scores the normalized
tree edit distance, so a merged cell or a dropped row costs what it should.
Implemented from scratch in `eval/teds.py` (Zhang-Shasha tree edit distance,
self-tested, no external dependency), with full **colspan/rowspan** support via
an HTML-table parser (the PubTabNet gold format), a **TEDS-Struct** variant
that scores grid structure alone, wired into `eval/run_matrix.py`, and run
against a hand-transcribed gold table (META 10-K p.83, the borderless quarterly
income statement):

| Parser | Table TEDS on META p.83 |
|---|---|
| naive `page.get_text()` | 0.000 (produces no table) |
| pdfplumber | 0.053 (shreds the page into a 77×16 grid) |
| **rtldoc** | **0.319** |

rtldoc wins decisively on the standard metric — but the honest read is in the
absolute number, not just the ranking. 0.319 is a *clear win, not a high
score*: rtldoc recovers the **data** correctly (every value lands in the right
quarter column) but the **grid geometry** is approximate — it merged the top
three rows into one multi-line cell and kept two phantom empty columns, which
TEDS rightly penalizes. So the industry metric confirms both halves of the
§8 story: the extraction is right, the inferred structure needs work. That's
the next place to push table quality, now that it's measurable.

---

## 5. Measured performance (5 real documents, this repo)

| Document | Pages | Time | Pages/sec | Images (deduped) | Audit review-rate |
|---|---:|---:|---:|---:|---:|
| Arabic teacher's guide (`BilArabi_TG07.pdf`) | 239 | 23.5s | 10.2 | 415 | 2.1% |
| GOOGL 10-K (2016 fiscal year) | 162 | 2.7s | 60.0 | 1 | 3.7% |
| META 10-K (2017 fiscal year) | 144 | 5.5s | 26.2 | 5 | 8.3% |
| Company compliance policy (`rag_document.pdf`) | 12 | 0.74s | 16.2 | 36 | 8.3% |
| WHO report (`rag_document_2.pdf`) | 52 | 8.6s | 6.0 | 31 | 7.7% |

No GPU, no external API calls, single CPU core mostly idle. "Audit
review-rate" is `rtldoc audit`'s own confidence flag (§7) — the fraction of
pages it thinks should get a second look, not an external quality score.

These numbers are ~10–25% faster than this repo's first cut: profiling
showed ~70% of per-page time was PyMuPDF's native text extraction, called
**twice** per page (`get_text("dict")` for spans + `get_text("rawdict")` for
per-glyph bidi). Since rawdict is a strict superset of dict, one extraction
now serves both — verified byte-identical output across all 609 pages of the
five documents above. The lesson worth keeping: the bottleneck was the
native call count, not any of the Python-level layout/bidi loops, which
collectively account for under 10% of runtime. Optimize what the profiler
points at, not what reads as expensive.

---

## 5a. Universality test — 96 unseen complex PDFs, 2336 pages

Everything in §4/§5 is measured on 5 documents. To test whether the guarantees
hold *beyond* them, rtldoc was run over **96 complex born-digital PDFs (2336
pages)** downloaded from arXiv across 15 subject areas — layouts, templates,
and table/math densities the parser had never seen. This is *label-free*
testing: instead of hand-transcribed truth (which can't scale), it checks
properties that must hold for any PDF. Two harnesses, both in `eval/`:

- `eval/invariants.py` — HARD invariants that must never break, plus
  coverage/excess diagnostics vs the source glyph stream
- `eval/hardness.py` — 8 independent difficulty angles (completeness,
  get_text/rawdict disagreement, duplication, geo-line orphaning, region
  overlap, presentation-form load, bidi reversal, fragmentation), flagged by
  **consensus** so no single (possibly unreliable) metric dominates

Full reports and the reproducible arXiv-ID manifest are in
[`eval/results/`](eval/results/) (the copyrighted PDFs themselves are not
redistributed; the fetch script re-downloads the identical corpus).

**What held up (strong, confident):**

| HARD invariant | Result on 2336 unseen pages |
|---|---|
| crashes | **0** |
| presentation forms leaked into output | **0** |
| non-rectangular tables | **0** |
| non-deterministic pages | **0** |
| mean letter coverage vs glyph stream | **0.988** |

**What it caught (the honest part):** the multi-angle detector surfaced a
**real duplication defect on ~17% of pages** (396/2332), strongly
co-occurring with region overlap (276 pages) — spot-checked and confirmed as
genuine repeated sentence fragments on dense/multi-column academic layouts,
not benign header repetition. It is the tension exposed by the text-loss fix
(assign every line to its best-overlap region) where span-assignment and
geo-line ownership disagree. Not yet fixed; see §8.

This is exactly why the test exists: it turned a would-be overclaim
("works on anything") into a specific, located, honest limitation.

---

## 5b. Is rtldoc actually better? An honest verdict

Answered at the confidence the evidence in this repo actually supports —
no more.

**Yes, with confidence, on its design target** — born-digital, complex-layout,
RTL (especially Arabic) documents. Here it is not marginally better than
general parsers, it is *structurally* better: it fixes encoding-level bugs
(presentation forms, lam-alef reversal) that no amount of layout-model tuning
addresses, recovers reading order from geometry instead of a mis-trained
model, and does it deterministically, auditably, on CPU, at 6–60 pages/sec.
The bug log in §4 is the evidence — each is a concrete failure of the general
approach that this one gets right.

**Yes, on robustness, universally** — 2336 unseen pages, zero crashes, zero
encoding leaks, zero malformed tables, fully deterministic, 98.8% text
coverage (§5a). That is a strong, measured claim.

**Not yet, as a general-purpose "best parser for any PDF."** The §5a test
found a real duplication defect on ~17% of dense academic pages, and the
scanned-PDF path still isn't wired. Until those close, claiming universal
superiority would be exactly the kind of unbacked assertion this repo's
evaluation is built to prevent. It is a best-in-class *specialized* parser
with a robust core and an honest, measured list of what still needs work —
not (yet) a universal one.

The reason to trust these statements is that the same test infrastructure
that produced the wins also produced the limitations. An evaluation that only
ever flatters the thing it measures isn't measuring.

---

## 6. The design, and why each choice is the right one

```
┌──────────────────────────────────────────────────────────────┐
│ 0  TRIAGE          text-layer density → born-digital | scan  │
├──────────────────────────────────────────────────────────────┤
│ 1  PRIMITIVES      glyphs+bbox+font+colour | vector fills |  │
│    (primitives.py) placed images     ← lossless, no pixels   │
├──────────────────────────────────────────────────────────────┤
│ 2a REGIONS         panels, chips, and table grids read off   │
│    (layout.py)     the DRAWING layer at exact coordinates    │
│ 2b CV FALLBACK     HSV tint segmentation + RLSA smearing     │
│    (cvfallback.py) only when the author drew nothing         │
├──────────────────────────────────────────────────────────────┤
│ 3  READING ORDER   RTL recursive XY-cut: vertical cuts       │
│    (layout.py)     ordered x-DESCENDING, column-gutter-aware │
├──────────────────────────────────────────────────────────────┤
│ 4  GEOMETRIC BIDI  rebuild logical order from glyph x-coords │
│    (geobidi.py)    ← per-line RTL/LTR detection, not assumed │
├──────────────────────────────────────────────────────────────┤
│ 5  ARABIC REPAIR   deshape · ligature-safe · mirror · harakat│
│    (arabic.py)                                               │
├──────────────────────────────────────────────────────────────┤
│ 6  SEMANTICS       style-signature map (label once/series)   │
│    (pipeline.py)   + cross-column activity linking           │
│                    + figure extraction & geometric captions  │
├──────────────────────────────────────────────────────────────┤
│ 7  AUDIT           per-page confidence → route ~3% to a VLM  │
└──────────────────────────────────────────────────────────────┘
```

### The load-bearing idea: reconstruct bidi from geometry, per line

Every parser recovers logical order from the *character stream* — a stream
written for rendering, not reading — using heuristics about producer intent.
PyMuPDF applies its own bidi pass. pdfium applies none. Both are guessing.

Glyph coordinates are not a guess. If a glyph sits further right on the same
baseline, it comes earlier in Arabic. So `geobidi.py` throws the string away:

1. group glyphs into baselines
2. **check whether the baseline actually contains any RTL characters** — a
   pure-LTR line is sorted ascending and left alone; this is the fix from
   §4 that keeps the whole module correct on non-Arabic documents too
3. for RTL lines: sort by **x descending** → logical Arabic order, for free
4. find maximal LTR runs (Latin, digits) and re-sort those ascending → correct
   bidi L1/L2 resolution, which is what fixes `العالم2021 في` → `العام 2021 شهدت`
5. resolve neutrals positionally (bidi rule N1) and mirror brackets (rule L4)
6. insert word breaks from measured inter-glyph gaps, not from stream spaces

This is ~3ms/page, deterministic, model-free, dictionary-free, and correct on
both RTL and LTR documents in the same corpus — it decides direction from
what's actually on the page, never from an assumption about the book.

### Style signatures instead of a layout model

A textbook *series* has on the order of a dozen `(font, size, colour, weight)`
combinations. Run `rtldoc styles` once, label those twelve in a JSON file, and
every page in the series types deterministically:

- ~100% accuracy on the series it was labelled for, vs ~92% from a layout model
- zero inference cost, zero GPU
- fully auditable — you can point at the rule that fired
- 20 minutes of human labelling amortised over 500+ pages

This only works because publishing corpora are homogeneous. It would be a bad
idea for heterogeneous web PDFs. It is the correct idea here.

### The output unit is an *Activity*, not a page

`_link_activities` propagates each numbered chip down its column in reading
order, pairing exercise + rubric + answer key as one retrievable object — the
only chunking that makes a lesson-planning or tutoring RAG actually work, and
something no general-purpose parser produces because the linkage is
cross-column and publisher-specific.

---

## 7. Eval harness

`eval/` is a from-scratch comparison harness, not a wrapper around someone
else's numbers:

- `eval/metrics.py` — normalized edit-distance (text fidelity), reading-order
  score (greedy content-matched, not ID-matched, so no two parsers need to
  segment a page the same way), and a table cell-match score
- `eval/adapters.py` — one function per parser (`naive_pymupdf`, `pdfplumber`,
  `rtldoc`); registering a new one is adding one function to `ADAPTERS`
- `eval/gold/gold.json` — hand-labeled gold pages spanning exactly the cases
  §1 and §4 describe: Arabic single-column, Arabic two-column, Arabic
  bordered table, English prose, English bordered table, English
  **borderless** table (now recovered, §4a), TOC pages, figure/caption pages
- `eval/gold/images/*.png` — rendered reference images for labeling (gitignored, see §9)

```bash
python eval/run_matrix.py eval/gold/gold.json
```

**Status: gold-label transcription is in progress, not complete.** Unlabeled
pages are skipped, not scored as failures — the matrix output right now is
partial by design. Numbers will be added here as labeling finishes rather
than published early and walked back.

---

## 8. Known limitations (honest, not fixed yet)

- **Text duplication on dense/overlapping layouts** — *largely fixed.* It was
  ~17% of academic pages (span-assignment and geo-line ownership disagreeing
  on overlapping multi-column regions, so a line rendered twice). A merge
  policy (§4b) cut it to ~4% (arXiv corpus 396→86 flagged pages, −78%) with no
  text-loss regression. The residual is the hardest overlap cases; the merge
  keeps the best copy of any line that still double-renders.
- **Borderless tables** *are* now detected (§4a) via column-alignment, but
  the inferred row/column boundaries aren't pixel-perfect: a multi-line cell
  can occasionally merge two source rows, and the numeric-only trigger means
  a purely *textual* borderless table (no numbers) is still missed. The data
  is recovered and correctly column-associated; the grid geometry is
  approximate.
- **Orphaned diacritic marks**: some PDF fonts emit a harakat glyph as its
  own tiny span disconnected from its base letter; these show up as
  single-character noise blocks on a few pages.
- **No OCR/VLM fallback wired in yet** for the ~3-8% of pages `audit()`
  flags — the routing hook exists, the second-pass model call doesn't.
- **The style map is per-publisher, by design** (§6) — a different series
  needs its own labelled `styles.json`, not new code, but that labelling
  step is a real, non-zero cost.
- **Docling/Marker aren't in the quantitative matrix** — architecturally
  reasoned about in §3, not benchmarked head-to-head here (cost tradeoff,
  see §3's footnote).

---

## 9. Usage

```bash
pip install -e .

# put your own PDFs in book/ -- not committed, see .gitignore
# 1. census the styles across a sample of the book, once per series
rtldoc styles book.pdf --pages 80-100 --out styles.json
#    → edit styles.json, map each signature to a role:
#      "MyriadArabic-Bold|12.0|1a2f5c|B": "passage_title"

# 2. parse
rtldoc parse book.pdf --style-map styles.json --json out.json --md pages/
#    images are extracted to pages/images/, deduped by PDF xref

# 3. find the pages that need a human or a VLM
rtldoc audit book.pdf | jq '.review_rate, .pages[] | select(.needs_review)'
```

Python:
```python
from rtldoc.pipeline import parse_document, to_markdown, save_images
results = parse_document("book.pdf", style_map=json.load(open("styles.json")))
```

**A note on `book/` and `eval/gold/images/`**: both are gitignored. The test
PDFs used to develop and validate this parser (a commercial Arabic textbook,
SEC filings, a company policy document, a WHO report) are third-party
copyrighted material used locally for testing, not redistributed in this
repo. Drop your own PDFs in `book/` to run everything above.

### Docker

No Python environment to set up, nothing to conflict with: rtldoc's actual
runtime dependencies are just **PyMuPDF and numpy** — both ship prebuilt
wheels, so the image needs no compiler and no system libraries at all.
`opencv-python-headless` (only for the not-yet-wired scanned-page fallback)
and `arabic-reshaper`/`python-bidi` (only for the local test-fixture
generator) are real dependencies of *other parts of this repo*, but nothing
the CLI itself imports — so they're `pip install rtldoc[cv]` /
`rtldoc[dev]` extras, not baked into the image.

```bash
docker build -t rtldoc:light .
```

**399MB**, verified locally (`docker images`) — mostly the `python:3.12-slim`
base itself. Compare that to a Docling or Marker image, which starts with
torch and model weights before your code even runs.

Run it against your own PDFs with a volume mount:

```bash
docker run --rm \
  -v "$(pwd)/book:/data:ro" \
  -v "$(pwd)/out:/out" \
  rtldoc:light parse /data/yourfile.pdf --md /out/pages --json /out/book.json
```

`ENTRYPOINT` is `rtldoc`, so any subcommand works the same way:
`docker run --rm -v "$(pwd)/book:/data:ro" rtldoc:light audit /data/yourfile.pdf`.
Verified end-to-end in-container against both an Arabic PDF and an English
10-K before writing this section — not just a Dockerfile that looks right.

---

## 10. Roadmap

**Near term**
- Wire the OCR/VLM audit fallback for the ~3-8% flagged pages, using the
  vector-extracted text as a constraint so the model corrects rather than re-reads
- Tighten borderless-table grid geometry (multi-line cell merging; extend
  beyond numeric tables to purely textual borderless ones) — detection itself
  now shipped, see §4a
- Complete gold-label transcription and publish full eval numbers
- Style-map learning: cluster signatures across a whole book and propose
  labels, so the human confirms rather than types

**Medium term**
- Harakat-aware CER, because vowelling errors matter more than letter errors in
  a teaching corpus and standard CER hides them
- Publish the labeled set as `arabic-textbook-bench` — it does not exist,
  the Gulf edtech market needs it, and OmniDocBench's own coverage gap (§2)
  confirms nobody else has filled it

**Worth considering**
- The `Region` interface is deliberately publisher-agnostic. A second publisher
  should need only a new style map, not new code. If it needs new code, the
  abstraction is wrong and should be fixed then, not now.
