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
  rules at 100% (vs ~85–95% from a CNN), and recovers *borderless* tables by
  column alignment.
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

Tested on **119 PDFs / 13,557 pages** it never saw during development — an
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

## Use it

Install without cloning, pinned to the release:

```bash
pip install "git+https://github.com/umerjavaidkh/rtldoc.git@v1.0.5"
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

---

*Design rationale, the full bug log, and methodology: **[DESIGN.md](DESIGN.md)**.*
