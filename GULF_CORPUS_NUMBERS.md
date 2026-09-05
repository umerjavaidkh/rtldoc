# Gulf Arabic corpus — all 18 documents, all 2,283 pages

No page cap, no sampling. **18 documents · 2,283 pages · 4,106 chunks · 0
crashes · 525s.** Saudi MHRSD (labour + HR executive regulations), Saudi MoH,
Saudi MoF, Oman NCSI, plus ministry policy and beneficiary reports.
All measured **per page**, on the parser's own blocks.

## The six axes

| axis | score | measured over |
|---|---:|---|
| **CITATION** | **99.2%** | 2,283 pages |
| **PAGE** | **96.9%** | 2,283 pages |
| **COLUMN** | **95.9%** | pages with a detected gutter, 6 docs |
| **TEXT** | **82.3%** | 2,283 pages |
| **HEADING** | **60.1%** | 2,283 pages |
| **TABLE** | **7.8%** | 2,854 records across 7 docs |
| **OVERALL** | **73.7%** | |

## Sub-metrics, worst first

| sub-metric | score |
|---|---:|
| TABLE · mean record similarity | 0.143 |
| **HEADING · recall** | **0.150** |
| HEADING · F₀.₅ | 0.201 |
| HEADING · precision | 0.480 |
| **TEXT · digit integrity** | **0.580** |
| TEXT · word precision | 0.785 |
| TEXT · word recall | 0.844 |
| PAGE · words intact | 0.902 |
| COLUMN · contiguity | 0.920 |
| CITATION · localization | 0.946 |
| PAGE · no empty blocks | 0.950 |
| PAGE · tables well-formed | 0.991 |
| COLUMN · precedence | 0.998 |
| PAGE · blocks produced | 1.000 |
| PAGE · roles assigned | 1.000 |
| CITATION · classification | 1.000 |
| HEADING · hierarchy valid | 1.000 |

## TABLE, per document (now real coverage, not one page)

| score | records | document |
|---:|---:|---|
| 0.005 | 494 | NCSI Statistical Yearbook 2023 |
| 0.022 | 604 | NCSI Statistical Yearbook 2021 |
| 0.026 | 644 | NCSI Statistical Yearbook 2022 |
| 0.067 | 165 | book-Statistics-2018 |
| 0.068 | 634 | user-manual-ar |
| 0.147 | 277 | اللائحة التنفيذية لنظام العمل |
| 0.208 | 36 | MoF Annual Report 2021 |

Average truth 8.2 records per table vs 6.65 predicted — rtldoc is consistently
producing fewer rows than the reference claims.

## How these numbers were verified

**Sampled 16 pages at random** (one per document, seeded), then adjudicated the
four outliers against the rendered page by eye:

| page | metric said | actually |
|---|---|---|
| `commu_p4` | TEXT 0.90 | **correct** — back cover, logo + URL, extracted exactly |
| `اللائ_p95` | recall **0.00** | **not a defect** — decorative back cover, nothing to extract |
| `Annua_p56` | digits **0.00** | **false positive** — the only digit is the page number "56", correctly dropped as furniture |
| `tqryr_p1` | TEXT 0.62 | **real defect** — title, subtitle and copyright each emitted twice; `الرب ع`, `الب رشية` (false spaces, wrong letters) |

**1 of 4 outliers is a genuine defect.** The other three are near-empty pages
where a two-word difference swings the percentage hard. The per-page mean gives
a 2-word cover page the same weight as a 500-word body page, which inflates the
apparent defect rate. That weighting is not yet fixed.

## What is trustworthy

- **CITATION, PAGE, COLUMN** — high, stable, and corroborated by the detector
  pass. `precedence` 0.998 and `contiguity` 0.920 mean reading order holds on
  Arabic.
- **TEXT 82.3%** — up from 77.0% after fixing a reference bug (the raw text
  layer returns kashida-stretched Arabic, `يـــهدف` for `يهدف`; rtldoc
  correctly removes it and was being penalised). Remaining gap is part real
  (duplication, false spaces inside words) and part near-empty-page noise.
- **HEADING 60.1%, recall 0.150** — the worst number and it is real. Verified
  directly: rtldoc emits **0 headings across 331 blocks over 15 pages** of the
  Labour Law Regulations. For HR retrieval this is the most damaging defect
  here — every chunk of a numbered legal regulation arrives with no section
  context.
- **TABLE 7.8%** now has genuine coverage (2,854 records, 7 documents) rather
  than the single table the 15-page cap allowed. But the reference itself has
  **not** been adjudicated at this scale — 60 Latin candidates were checked by
  eye and 26% were invalid. Until the same is done here, treat 7.8% as a
  floor, not a verdict.
