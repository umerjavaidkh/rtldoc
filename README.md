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

## Proven at scale

Tested on **108 PDFs / 11,961 pages** it never saw during development — an
Arabic teacher's guide, two SEC 10-Ks, 96 arXiv papers (15 fields), 5 OpenStax
physics/chemistry/calculus textbooks (figures, geometry, exercises), and the
3,130-page PostgreSQL 18 manual (deeply-nested reference tables, code blocks).
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
| **0.503** | 0.053 | 0.000 |

Everything is reproducible in [`eval/`](eval/) (harnesses, arXiv manifest,
saved reports).

## Use it

Install without cloning, pinned to the release:

```bash
pip install "git+https://github.com/umerjavaidkh/rtldoc.git@v1.0.1"
```

Then:

```bash
rtldoc parse book.pdf --md out/ --json out.json
rtldoc parse book.pdf --html out_html/   # real <table>/<figure>, RTL-aware dir=
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

- Borderless-table *grid geometry* is approximate (data lands in the right
  columns, but wide multi-level-header tables can still lose structure —
  that's the 0.503 TEDS, not 1.0).
- Scanned / no-text-layer pages are flagged for OCR, not yet parsed inline.
- Best semantic typing needs a one-time per-publisher style map (~20 min).

---

*Design rationale, the full bug log, and methodology: **[DESIGN.md](DESIGN.md)**.*
