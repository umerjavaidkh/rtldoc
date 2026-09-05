# rtldoc

**A geometry-first PDF parser for RTL & complex layouts.** It extracts text,
reading order, and tables from born-digital PDFs using the glyph coordinates
already in the file — **no OCR, no GPU, no API**. Built for the pages that
quietly break Docling, Marker, and VLM parsers: Arabic/RTL, multi-column,
and vector tables.

---

## Why it's better where it matters

- **RTL / Arabic done right.** Reading order and bidi are rebuilt from glyph
  *positions*, not from a reading-order model trained on English. It fixes the
  presentation-form and lam-alef bugs that silently corrupt ~⅓ of Arabic words
  in every general parser — encoding bugs no layout model can fine-tune away.
- **Tables without a model.** Reads table structure from the PDF's own vector
  rules at 100% (vs ~85–95% from a CNN), recovers *borderless* tables by
  column alignment, and recovers grids drawn as plain filled boxes — the way
  papers and Word exports often draw a table with no rules at all — from the
  regularity of the boxes themselves.
- **Deterministic & auditable.** Every output traces to a rule you can point
  at. CPU-only, **6–60 pages/sec**, $0/page — a VLM is 100–1000× the cost and
  can't be audited.
- **Page visual summary, no vision model.** Opt-in (`--visual`) geometry pass
  that turns a page's own vector drawings into a structured summary: image
  size/color stats, table dimensions, and — for vector flowcharts/diagrams —
  the actual boxes, their text labels, and which ones a connecting line joins.
  Built entirely from the PDF's own drawing commands (PyMuPDF `get_drawings()`),
  not a screenshot or a guess. HTML output renders any detected diagram as a
  real chart via Mermaid.js, fed the extracted nodes/edges directly.

## Proven at scale

Tested on **119 PDFs / 13,557 pages** it never saw during development, plus a
separate **18-document / 2,283-page Gulf corpus** (Saudi and UAE statistical
yearbooks, labour regulations, HR policy and service manuals) used as the
standing regression set — an
Arabic teacher's guide, two SEC 10-Ks, 96+ arXiv papers (15 fields), 5 OpenStax
physics/chemistry/calculus textbooks (figures, geometry, exercises), the
3,130-page PostgreSQL 18 manual (deeply-nested reference tables, code blocks),
and a growing set of real-world forms, reports, and scanned documents.
The checks are *property-based and label-free*, so they scale to any corpus:

| Property (must hold on every page) | Result |
|---|---:|
| crashes | **0** |
| encoding leaks (presentation forms in output) | **0** |
| malformed tables | **0** |
| non-deterministic pages | **0** |
| text coverage vs the PDF's own glyph stream | **~99%** |

**Table quality, scored with TEDS** (the PubTabNet/OmniDocBench standard) on a
borderless financial statement — where the whole point is a hard table:

| rtldoc | pdfplumber | naive `get_text` |
|---:|---:|---:|
| **0.942** | 0.061 | 0.000 |

Everything is reproducible in [`eval/`](eval/) (harnesses, arXiv manifest,
saved reports).

## Measured against other parsers

Not claims — scores on the same pages, with the harnesses in [`eval/`](eval/).

**Tables**, on 43 hand-verified tables (`eval/ragbench/gold/gold.json`), scored
with ParseBench's TableRecordMatch:

| | score | tables it returns nothing for |
|---|---:|---:|
| **rtldoc** | **0.625** | 9 |
| PyMuPDF `find_tables()` | 0.273 | 19 |
| Marker's projection alone | 0.356 | 4 |
| rtldoc + projection as fallback | **0.646** | 4 |

**Arabic**, rtldoc against [Marker](https://github.com/datalab-to/marker) on
the same 297-page Saudi statistical yearbook:

| | Marker | rtldoc |
|---|---:|---:|
| Arabic words extracted | 12,902 | **17,677** |
| Latin words extracted | 11,459 | **18,002** |
| tatweel (kashida) left in text | 2,846 | **0** |
| broken lam-alef ligatures | 1,078 | **11** |

The last two rows are the ones that matter for retrieval: Marker's output
*looks* right on screen, but `العـاج` will never match a query for `العلاج`,
and `االنتقال` never matches `الانتقال`. Thousands of words silently
unsearchable. On a different file (a Saudi labour regulation) Marker's Arabic
came out reversed entirely — 289 of rtldoc's 300 commonest words appeared
letter-reversed — so its RTL handling is font-dependent rather than absent.

**Headings** are the honest exception. Scored on 72 hand-adjudicated cases,
rtldoc and DocLayout-YOLO are a statistical dead heat — 57/72 each, McNemar
p = 1.0. Four attempts at improving heading detection have failed, the last
one by fitting weights on the headings PDFs declare in `/H1../H6`
(`eval/heading_fit.py`). The blocker is measured and it is labels, not
features: ten documents in the corpus carry a structure tree, yielding 149
positives of which 137 come from one file.

## What breaks in Arabic PDFs, and why geometry finds it

Three defects found by measurement, each invisible to a layout model because
none of them is a layout problem:

- **Chrome print-to-PDF stacks a hidden copy of every letter.** The Wikipedia
  print stylesheet, exported through Chrome, emits each Arabic letter a second
  time with no advance, all glyphs piled on one x. They draw nothing but sort
  into the line by position, welding glued letters onto the front of it:
  `علملوثةلعرقيحدُيطلق اسم علم الوراثة` for `ُيطلق اسم علم الوراثة`. Between
  42% and 55% of lines in the worst files, 12% of the stratum's text.
  Detected by *stacking*, not by zero width — many fonts render the lam-alef
  ligature as one glyph whose alef component has no advance, and filtering on
  width alone turns `الُجْغَرافّية` into `الَُْافّة`.
- **Symbol-font bullets arrive as Private Use codepoints.** U+F0B7 is a
  bullet in Adobe's Symbol encoding and tofu everywhere else; 134 of them in
  one UAE service manual.
- **Cell text cut at cell boundaries.** Clipping a rectangle to read a table
  cell makes MuPDF *cut* every line crossing the edge, and the offcuts land in
  the neighbouring cell. Lines are assigned to the cell containing their
  centre instead: a line belongs to one cell, it is never divided.

## Use it

```bash
pip install pdf-rtldoc
```

(Published on PyPI as `pdf-rtldoc` -- the plain `rtldoc` name was already
taken by an unrelated project. The installed command, and the module you
import in Python, are both still `rtldoc`.)

Or install straight from a specific release without going through PyPI:

```bash
pip install "git+https://github.com/umerjavaidkh/rtldoc.git@v1.2.1"
```

Then:

```bash
rtldoc parse book.pdf --md out/ --json out.json
rtldoc parse book.pdf --html out_html/   # real <table>/<figure>, RTL-aware dir=
rtldoc parse book.pdf --html out_html/ --visual  # + diagram/image/table visual summary
rtldoc audit book.pdf                    # flags low-confidence pages for review
```

Runtime deps are just PyMuPDF + numpy. (From a clone: `pip install -e .`.)

Zero-setup via Docker (399 MB, no compiler/GPU):

```bash
docker build -t rtldoc . && docker run --rm -v "$PWD:/d" rtldoc parse /d/book.pdf --md /d/out
```

Output: per-page Markdown (tables as GFM, images extracted + auto-captioned)
plus structured JSON, or a self-contained HTML page per PDF page.

## Honest limits

- Borderless-table *grid geometry* is approximate on the hardest wide,
  multi-level-header tables (occasional row/column structure mismatches
  — that's the 0.942 TEDS on our graded case, not 1.0).
- A page dominated by a figure/table spanning the full content width can
  still be under-counted as fewer columns than it visually has — the
  whitespace-gutter detector requires the gap to stay empty across most
  of the page's height, and a full-width element defeats that locally.
- A table cell whose own text wraps onto a later line, where that line's
  content coincidentally re-aligns with an earlier row's columns, can
  occasionally attach to the wrong row (`_merge_wrapped_label_rows` — a
  narrow, tracked edge case, not a general table-detection failure).
- Scanned / no-text-layer pages now OCR via Tesseract (needs the `tesseract`
  binary on PATH — `brew install tesseract` / `apt install tesseract-ocr`;
  no Python package required). Word-level positioning, not this repo's
  glyph-exact reading order; a page with no tesseract installed just gets
  no blocks, as before.
- **Heading detection is the weak axis: 58.6%** on the Gulf corpus, and four
  attempts to improve it have failed. See above -- the blocker is labelled
  data, not the algorithm, and saying so is more useful than a fifth attempt.
- Table *scores* are harder to trust than table output. The RAGBench TABLE
  axis derives its ground truth from another detector, and a 60-case hand
  audit found that reference's own grid wrong on 28% of its detections. Two
  measurement bugs in it were fixed (records keyed by a header that a
  continuation table does not have; truth keeping empty columns the prediction
  drops) and the axis moved 9.4% -> 20.5% on unchanged parser output. The
  number worth quoting is the 43 hand-verified tables, not that axis.
- Best semantic typing needs a one-time per-publisher style map (~20 min).
- Diagram detection reconstructs simple box-and-arrow flowcharts reliably;
  dense multi-level diagrams (deep tree/org-chart hierarchies with many
  branches) get correct node/box detection but not yet reliable connection
  tracing — a harder, separate problem noted for future work.

## Roadmap

- Reliable connection tracing for dense/branching diagrams (see above).
- A page-level chart/figure classification pass, so bar charts, legends,
  and gridlines are recognized and set aside before table/diagram
  detection runs, rather than relying on those detectors' own guards to
  reject them case by case.
- A cell-level golden regression corpus (`eval/golden/` + `eval/regression.py`)
  now exists and grows with each table-detection fix; still short of full
  coverage across document types.

## Publishing a release (maintainers)

Bump `version` in `pyproject.toml`, commit, tag (`git tag -a vX.Y.Z`), push
the tag, then publish a GitHub Release from it. `.github/workflows/publish.yml`
builds and uploads to PyPI automatically when the release is published, via
PyPI's Trusted Publisher (OIDC) mechanism — no API token stored anywhere.

One-time setup (already done for `pdf-rtldoc`): on pypi.org, under the
project's *Publishing* settings, add a trusted publisher with owner
`umerjavaidkh`, repository `rtldoc`, workflow filename `publish.yml`, and
environment name `pypi`.

---

*Design rationale, the full bug log, and methodology: **[DESIGN.md](DESIGN.md)**.*
