# rtldoc — what it does, and what it is measured at

Every number here is reproducible from this repository. Where a number is
weak or a claim is unproven, it says so.

---

## The problem it solves

Arabic documents break retrieval **silently**. A parser returns text that looks
correct on screen, and the search that should find it returns nothing, because
the characters are not the ones a user types.

Measured on one 297-page Saudi statistical yearbook, against
[Marker](https://github.com/datalab-to/marker), the leading open-source parser:

| | Marker | rtldoc |
|---|---:|---:|
| Arabic words extracted | 12,902 | **17,677** |
| Latin words extracted | 11,459 | **18,002** |
| tatweel (kashida) left in the text | 2,846 | **0** |
| broken lam-alef ligatures | 1,078 | **11** |

```
Marker:   العـاج      االنتقال      األعوام
correct:  العلاج       الانتقال      الأعوام
```

A user searching `العلاج` gets **zero results** from that output. Not ranked
lower — no match, because the strings differ by characters. Thousands of words,
silently unsearchable, in text that passes visual inspection.

On a different file (a Saudi labour regulation) Marker's Arabic comes out
**reversed entirely** — 289 of rtldoc's 300 commonest words appear
letter-reversed. Its RTL handling is font-dependent, not absent.

---

## Measured results

Scored with RAGBench (`eval/ragbench/`): page level, deterministic rules, no
LLM judge, ground truth derived from each PDF's own content stream.

### Gulf corpus — 18 documents / 2,283 pages

Saudi and UAE statistical yearbooks, labour regulations, HR policy, service
manuals. This is the standing regression set; the parser has been tuned against
it.

| axis | score | what it means |
|---|---:|---|
| CITATION | 98.9% | every chunk traceable to a page and a box |
| COLUMN | 97.0% | reading order correct across columns |
| PAGE | 95.9% | nothing dropped, nothing duplicated |
| TBL-FOUND | 95.7% | the table was found, once |
| TEXT | 83.7% | content faithfulness |
| HEADING | 54.1% | **weak — see limits** |
| TABLE | 29.5% | **understated — see limits** |
| **OVERALL** | **79.3%** | |

### HR corpus — 12 documents / 676 pages, never tuned against

HR policy and labour-law documents from UAE, Oman, Kuwait and Saudi Arabia.
The parser had never seen these files.

| axis | Gulf | **HR (unseen)** |
|---|---:|---:|
| CITATION | 98.9% | **100.0%** |
| COLUMN | 97.0% | **99.4%** |
| PAGE | 95.9% | **96.7%** |
| TEXT | 83.7% | **92.5%** |
| HEADING | 54.1% | 58.1% |

**Every axis holds or improves on documents never seen during development.**
That is the number that matters: it is generalisation, not fit.

### Against other tools

| task | rtldoc | alternative |
|---|---:|---|
| tables, 43 hand-verified | **0.625** | PyMuPDF `find_tables()` 0.273 |
| headings, 72 hand-adjudicated | **F1 0.889** | Marker 0.732 · DocLayout-YOLO tied (p=1.0) |

---

## How it works, and why that matters commercially

**Geometry-first, no model.** Reading order, bidi resolution and table structure
are rebuilt from the glyph coordinates already in the PDF. No OCR, no GPU, no
API call.

- **~4 pages/second on a laptop CPU.** A VLM parser is 2–10 seconds per page.
- **$0 per page**, and nothing leaves the machine — which matters for HR files,
  contracts and government documents.
- **Deterministic and auditable.** Every output traces to a rule you can point
  at. Re-running gives the same answer.
- **Apache-2.0 dependencies only.** No model weights with revenue caps.

---

## What was fixed, with evidence

Each of these was root-caused rather than patched, and each is a defect class
rather than one document.

| defect | evidence |
|---|---|
| White rules drawn on coloured bands were dropped as invisible | A 300-page yearbook reported **zero rules and zero tables** on pages full of them. Table coverage on its data half: 46% → **84%** of pages |
| A decorative page border became a table region | An HR regulation emitted 13- and 15-column "tables" from running prose. Pages reporting a table: 35 → **12** |
| Table lines were cut at the cell edge instead of assigned to a cell | An Arabic cell collected the neighbouring English title, sliced into fragments |
| Chrome print-to-PDF stacks a hidden zero-advance copy of every letter | 42–55% of lines affected in Arabic Wikipedia files; **12% of the stratum's text** was junk |
| Symbol-font bullets arrived as Private Use codepoints | 134 in one service manual, rendering as tofu; now decoded via the Adobe Symbol encoding |
| Figure captions swallowed headings and page numbers | A header banner bleeding off-page took the page's own heading |
| Stacked sub-tables with differing column counts collapsed to one column | A bilingual contract page read as `8x1` mush; now `1x2, 2x2, 5x3` |

Across the Gulf corpus, detector findings fell **3,324 → 2,735 (−17.7%)**:
`table_missed` 942 → 571, `repeated_run` 418 → 291, `table_single_column`
132 → 59.

---

## Limits — stated plainly

**Heading detection is weak: ~54%.** Five separate attempts to improve it have
all scored *below* the existing heuristic, including a fitted classifier and two
external models. The blocker is measured and it is **labelled data**: ten
documents in the corpus declare a structure tree, and one of them supplies 137
of 149 usable positives. This is the honest gap.

**Multi-level table headers are flattened.** A header spanning four columns is
sliced across them. Two attempts to fix it failed — absence of a rule does not
mean cells are merged, and plenty of tables simply do not rule their headers.

**Scanned pages are not the product.** ~10% of the Gulf corpus has no text
layer and routes to Tesseract, which is weak on Arabic. Geometry cannot help a
page with no glyphs.

**The TABLE score understates and should not be quoted.** Its reference reads
only *ruled* grids, and a 60-case hand audit found that reference's own grid
wrong on **28%** of its detections. The same parser scores **0.625** on 43
hand-verified tables against the benchmark's 29.5%. Three of the four
improvements to that axis this cycle were fixing the benchmark, not the parser.

**Two regressions are open:** `table_row_collapse` (97 → 148) and
`heading_flood` (307 → 346). Finding more tables surfaced more imperfect ones.

---

## What it is honest to claim

> Born-digital Arabic and bilingual documents, parsed on CPU at ~4 pages/second,
> with reading order, bidi and orthography correct enough to survive retrieval —
> **92.5% text faithfulness, 99.4% column order and 100% citability on
> documents it had never seen.**

What not to claim: table perfection, heading hierarchy, or scanned documents.

---

*Methodology and the full defect log: [DESIGN.md](DESIGN.md). Harnesses,
fixtures and saved reports: [`eval/`](eval/).*
