# What actually writes PDFs, and what that means for the parser

Measured over this repo's corpus (118 files: `book/` + `corpus/`), not from documentation.
Reproduce with `python -c "import fitz; from rtldoc.producers import identify; print(identify(fitz.open(PATH)))"`.

## The corpus

| producers | n |
|---|---|
| pdfTeX / LuaTeX (arXiv) | 90 |
| Adobe (PDF Library, Distiller, PDFMaker) | 8 |
| Microsoft Word / Print-to-PDF | 7 |
| WeasyPrint, ReportLab, pdf-lib, Apache FOP, Canva, LiveCycle, Quartz, Ghostscript | 14 |
| **Skia / Google Docs Renderer** | 1 |

Evidence available: **tags 11, marked-content 4, geometry-only 103.**

## The single most important finding

**`/Producer` lies.** 88 of 96 arXiv papers report `pikepdf 8.15.1` — arXiv rewrites the
field when it serves the file. The real generator survives only in `/Creator`
(`arXiv GenPDF (tex2pdf:…)`) and in the operator mix. Any producer-conditional logic keyed
on `/Producer` alone is keyed on a value the delivery pipeline is free to overwrite, so
`producers.py` falls back to `/Creator` and then to the content stream itself.

## How each family emits text

| producer | text ops | positioning | tags | ToUnicode | the trap |
|---|---|---|---|---|---|
| **Google Docs (Skia)** | `Tj` only, **1 glyph per op** | one `Td` per glyph, one `BT/ET` per **word** | yes | 100% | no word grouping exists in the file at all |
| Chrome print-to-PDF (Skia) | same | same | usually no | 100% | same, without the tag fallback |
| pdfTeX / LuaTeX | `TJ` only, never `Tj` | `Td`, rarely `Tm` | no | **~50%** | Type1 subsets with no `/ToUnicode` |
| Word / PDFMaker | `TJ`, one per line | `Tm` per line | **yes** | 22–50% | tags are excellent, fonts are not |
| WeasyPrint | `TJ` | `Tm`, y-flipped | no | 100% | clean text, zero structure |
| Apache FOP | `TJ` | `Tm`, y-flipped | no | **9%** | worst text fidelity in the corpus |
| ReportLab / pdf-lib | `Tj`, ~40 chars per op | `Tm` | no | **0%** | no `/ToUnicode` anywhere |
| Quartz (macOS) | mixed `Tj`/`TJ` | `Tm` | no | 100% | — |

**102 of 118 files have <90% of fonts carrying `/ToUnicode`.** In academic PDFs the dominant
risk is not layout at all — it is that glyph→character is lossy before layout even starts.

## Google Docs, specifically

Exporting a Word document through Google Docs does **not** convert the file — Docs re-lays it
out in its own engine and renders through Skia, so Word's fonts and Word's tags are discarded
and replaced. What comes out (verified on `corpus/2607.22513v1.pdf`):

```
BT /F4 21.333334 Tf  1 0 0 -1 0 25.1675 Tm      <- y-flipped text matrix
0 -19.3125 Td <0014> Tj                          <- one glyph
11.8616943 0 Td <0011> Tj                        <- next glyph, repositioned
ET
BT ... <0003> Tj ET                              <- the space is its own text object
```

Three consequences:

1. **One `Tj` per glyph, one `Td` per glyph, one `BT`/`ET` per word.** 9851 `Tj` ops across six
   sampled pages. Word and line grouping exist *only* as geometry — nothing in the file groups
   characters into words, so any logic that trusts emitted runs gets 1-character spans.
2. **The text matrix is `1 0 0 -1`** — Skia works top-left-origin and flips. Same for WeasyPrint,
   Apache FOP, Microsoft Print-to-PDF and Canva (9 files in the corpus). Baseline logic that
   assumes `d > 0` inverts on all of them.
3. **It is tagged** — `Table`/`TR`/`TD`/`TH`, `H1`–`H3`, `L`/`LI`. Which makes the per-glyph
   emission mostly moot: read the tags and the structure is exact.

## Why this changes the architecture

Blueprint principle 1 says never infer what the file already states. `book/sample-tables.pdf`
declares **28 tables** with full `TH`/`TD` roles and column spans. rtldoc's geometry path finds
**18**, and cannot know which cells are headers. The remaining 10 are not a detector bug — they
are a layer that does not exist yet.

On the Google Docs file, geometry already finds all 5 tagged tables (it draws real ruled
borders); there the tags add header roles and guaranteed cell boundaries rather than raw recall.

## What was added

- `rtldoc/producers.py` — Layer 0. `identify(doc) -> ProducerProfile`: family, best available
  evidence (`tags` / `marked` / `geometry`), per-glyph flag, y-flip flag, ToUnicode fraction.
- `rtldoc/tags.py` — Layer 1b. `tagged_tables(doc) -> [TaggedTable]` from `/StructTreeRoot`,
  with `TD`/`TH` roles and spans. Includes a content-stream tokenizer that recovers MCIDs
  (PyMuPDF exposes none) and joins them to PyMuPDF's decoded spans by pen origin.

Both are dependency-free beyond PyMuPDF, and **neither is wired into `pipeline.py` yet** — they
add no behaviour and cannot regress the invariants until they are deliberately consumed.
