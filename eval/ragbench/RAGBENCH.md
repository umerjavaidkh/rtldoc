# RAGBench — an ingestion-quality benchmark for rtldoc

> **Scope: English and Arabic only.** See `eval/EVAL.md`.

Modelled on [run-llama/ParseBench](https://github.com/run-llama/ParseBench):
five equally-weighted capability dimensions, deterministic rule-based scoring,
no LLM judge. Re-aimed at what rtldoc is actually for — feeding a retrieval
pipeline.

## Why the previous suite was measuring the wrong thing

`eval/detectors.py` asks *does this page look right to a person*. That was the
right question when the complaint was "I opened the PDF and it was broken", and
those detectors stay useful. But rtldoc's output is not read by a person. It is
chunked, embedded, retrieved, and pasted into a prompt. Three things follow:

1. **A defect only matters if it survives chunking.** A heading mis-typed on a
   page nobody retrieves costs nothing. A table split between two chunks costs
   you every question about that table.
2. **Some catastrophes are invisible at page level.** A chunk that holds
   `104,000,000` but not the word `Egypt` looks fine on the page it came from
   and is *worse than useless* in a retrieval index — it makes an agent answer
   confidently and wrongly.
3. **What matters is not fidelity but recoverability.** The question is not
   "did we reproduce the document", it is "can the fact be found and is it
   still attached to its meaning".

## The five dimensions

| ParseBench | RAGBench | why the change |
|---|---|---|
| Tables | **Table Record Fidelity** | kept — same metric shape (`TableRecordMatch`) |
| Content Faithfulness | **Content Faithfulness** | kept — omissions, hallucinations, reading order |
| Charts | **Chunk Integrity** | rtldoc parses no charts; the chunk boundary is what breaks RAG instead |
| Semantic Formatting | **Structure & Hierarchy** | the heading path is chunk *metadata*, not decoration |
| Visual Grounding | **Citability** | same idea: can an answer point back at a page and a rectangle |

Plus a sixth number reported separately — the **end-to-end retrieval probe**,
which is the question itself rather than a proxy for it.

### 1. Content Faithfulness (CFS)

```
CFS = (1.0 · S_text + 0.5 · S_order) / 1.5          [ParseBench weights]
S_text = mean(word_recall, digit_integrity, word_precision)
```

`digit_integrity` is ParseBench's `bag_of_digit_percent`: the multiset of
digits must survive extraction. It is the highest-value cheap check there is
for numeric RAG — a dropped or invented digit is a wrong answer delivered with
full confidence. Arabic-Indic digits (U+0660–0669, U+06F0–06F9) are folded to
ASCII first, so an Arabic page is scored on the same footing.

`S_order` is ParseBench's pairwise precedence assertion: a rule passes iff the
**first** occurrence of `before` precedes the **last** occurrence of `after`.
The asymmetry tolerates a fragment legitimately repeating without letting a
real inversion through.

### 2. Table Record Fidelity (TRF)

```
TableRecordMatch(G,P) = Σ RecordSim(g,p) / max(|G|,|P|)
RecordSim(g,p)        = |{k : g[k] = p[k]}| / |K(g) ∪ K(p)|
```

Tables as **bags of records keyed by column header**, matched optimally
(Hungarian). Insensitive to row and column order — a retriever does not care
which row came first — but scores near zero when the header is dropped or
shifted, because for RAG **the header is the meaning**.

Scored **per page**, never pooled across a document: pooling lets a record from
one table match a record from another, which both hides real damage and
invents it.

### 3. Chunk Integrity (CIS)

The dimension that only exists because this is an ingestion benchmark. Five
equally-weighted properties of each retrieval unit:

- `ends_complete` — does not stop mid-sentence
- `has_context` — carries a heading breadcrumb to disambiguate it
- `within_window` — fits a typical embedding window
- `table_self_contained` — a table chunk holds its header *and* data rows
- `carries_content` — is not pure page furniture

`chunking.py` deliberately implements the **ordinary** strategy (respect block
boundaries, never split a table, break prose at sentence ends). If rtldoc's
output needs an unusually clever chunker to survive, that is a finding about
rtldoc.

### 4. Structure & Hierarchy (SFS)

F-beta over heading detection with **β = 0.5**, following ParseBench's styling
metric: an invented heading poisons the breadcrumb of every chunk beneath it,
while a missed heading only costs context on its own section. The asymmetry is
real, so the metric carries it. Averaged with a hierarchy-validity term (no
level jumps).

### 5. Citability (EPR)

```
Pass(c) = L · C · A      EPR = (1/N) Σ Pass(c)
```

`L` non-degenerate bbox inside the page · `C` every block has a role ·
`A` text and page recorded. For a regulated deployment an answer that cannot be
traced to a page and a rectangle cannot be audited.

### The retrieval probe

Needles are generated automatically from ruled-table ground truth: for a record
`{"Country": "Egypt", "Population": "104,000,000"}` the probe asks
`"Egypt Population"`, retrieves BM25 over the chunks, and requires a top-5
chunk to contain **both** the row key and the value.

Three outcomes, and the middle one is the finding:

- **supported** — row key and value in the same retrieved chunk
- **orphaned** — the value came back without its row key. This is the dangerous
  case: the number is present with nothing binding it to the question.
- **missing** — not retrieved at all

BM25 is implemented inline (no dependency), and lexical retrieval is the right
choice for a *parser* benchmark: it responds to the tokens the parser emitted,
where a dense embedder would blur exactly the damage being measured.

## Where the ground truth comes from

ParseBench scores against ~169,000 hand-written rules over ~2,000
human-verified pages. That is not available here, and a 20-page gold set says
nothing about 43,597 pages. So RAGBench derives assertions from **the PDF's own
content stream** — a different program from the one being tested:

- **table records** ← intersecting the drawn rule segments gives the true cell
  grid; words are assigned to cells by coordinate
- **precedence** ← same-column, vertically-separated line pairs
- **digit and word bags** ← the page's own text layer

**What this cannot do, stated plainly:** borderless tables have no ground truth
here (nothing in the file says where the columns are), and neither does
semantic formatting (bold, super/subscript). Those two need a hand-verified
gold set, and until one exists RAGBench is silent about them rather than
guessing. Structure & Hierarchy uses font-size evidence as a proxy for heading
truth, which is good but not perfect — treat it as the softest of the five.

## Running it

```bash
python -m eval.ragbench.runner corpus_large --sample 200 --max-pages 8 --json run.json
python -m eval.ragbench.runner corpus_large --stratum web_print__arabic --verbose
python -m eval.ragbench.runner corpus_large --doc some.pdf --verbose
```

`--verbose` prints sample failures per dimension for the worst documents, which
is where you should always start.

## Verify it the same way as the other suite

Two of RAGBench's own metrics were wrong on their first run and were caught by
comparing against the source before trusting the number:

- `records_from_markdown` treated the `|---|---|` separator as a table
  boundary, discarding the header row so the first data row became the header.
  Every table scored **0%** — attributable to rtldoc if nobody had checked.
- `normalize` stripped `>` before stripping HTML tags, leaving `<br` inside
  every cell value and failing real matches.

The rule that caught both: **when a metric reports a catastrophe, reproduce it
by hand against the source before believing it.** A benchmark is a program, and
it has bugs at the same rate as the program it is measuring.

## First run — 221 documents, 1,715 pages, 4,890 chunks

```
content     96.8%      tables      22.5%      chunks      84.3%
structure   69.4%      citable    100.0%      OVERALL     74.6%
```

| stratum | docs | content | tables | chunks | structure | citable | overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| scanned__no_text | 1 | n/a | n/a | n/a | n/a | n/a | **0.0%** |
| web_print__arabic | 5 | 96.1% | 3.2% | 87.7% | 80.8% | 98.2% | **73.2%** |
| office__latin | 2 | 97.9% | 33.3% | 78.6% | 72.7% | 100% | 76.5% |
| publisher__latin | 12 | 97.8% | 16.7% | 87.6% | 82.9% | 100% | 77.0% |
| latex__latin | 2 | 96.2% | n/a | 78.6% | 56.9% | 100% | 82.9% |
| word__latin | 6 | 95.0% | 54.2% | 87.2% | 80.5% | 100% | 83.4% |
| publisher__arabic | 1 | 85.1% | n/a | 84.0% | 75.0% | 100% | 86.0% |
| arxiv__latin | 186 | 97.0% | n/a | 84.1% | 67.6% | 100% | 87.2% |
| web_print__latin | 3 | 95.6% | n/a | 81.7% | 80.9% | 100% | 89.5% |

Retrieval probe: 57 needles, **answer-support rate @5 = 56.1%**, 40.4% missing,
3.5% orphaned.

### How much of this to believe

- **citability 100%** and **content 96.8%** are solid. Both are computed against
  ground truth the file states directly, and both were spot-checked by hand.
- **structure 69.4%** is the softest dimension by construction: heading truth is
  inferred from font size, not stated. Treat it as directional.
- **57 needles** is too few to publish a retrieval number from. It is enough to
  show the probe works and that "missing" dominates "orphaned".
- **tables** — settled below.

---

## The table number, settled

60 candidate lattices were collected (`eval/ragbench/goldset.py`), rendered, and
adjudicated by eye against two questions kept deliberately separate:

| | | |
|---|---:|---|
| candidates judged | 60 | |
| actually tables | 58 (97%) | 2 were two-column prose caught by frame rules |
| of those, derived grid correct | 43 (74%) | 15 swallowed adjacent prose, truncated the table, or merged two tables into one |

Separating "is it a table" from "is the grid right" is the whole point: a single
verdict cannot tell a parser bug from a benchmark bug, which is what produced
the successive 4%, 9% and 22% figures, none of which meant anything.

### The number

On the 43 tables where the ground truth is verified valid, each scored against
the best-matching table rtldoc emitted on that page:

```
TableRecordMatch = 63.8%
perfect (>=99%)  = 22/43
total loss (<=1%)= 14/43
```

Bimodal, not spread: rtldoc either reproduces a table exactly or loses it
completely. Almost nothing lands in between.

### Why the corpus-wide figure reads 21.3% instead

Decomposed on the same gold sample, scoring exactly as the corpus does and then
removing invalid ground truth one class at a time:

| scored over | n | mean |
|---|---:|---:|
| all 60 candidates — what the corpus number does | 60 | 48.8% |
| minus the 2 that are not tables | 58 | 50.4% |
| minus the 15 with a wrong derived grid | 43 | **63.8%** |

So roughly 15 points of the corpus figure is ground truth that was never valid.
The remaining gap is sample composition — the gold set drew from table-rich
documents, the corpus sample includes `web_print__arabic` (3.2%) and
`publisher__latin` (16.7%).

**Quote 63.8%, on 43 hand-verified tables. Do not quote the corpus-wide 21.3%**
— roughly 26% of its ground truth is invalid and it is not comparable.

### What is actually breaking — the 14 total losses

| cause | n |
|---|---:|
| **table header merged with body prose — the two-column bug** | **10** |
| no table emitted at all | 3 |
| header wrong, no true columns present | 1 |

Ten of the fourteen are the same defect parked earlier: rtldoc reports
`columns: 1` on a two-column page, so left-column prose is absorbed into a
right-column table and becomes its header. A real example, `e3580b13` p6, where
the true header is `Mode | Sessions | Aggregate (Gbps) | ...`:

```
| concurrency is the number of flows on the fabric at any single | repetitions). | | | | |
```

**Fixing the column bug would move table fidelity from 63.8% to roughly 87%**
(ten cases moving from 0 to 1.0 across 43). That is the single highest-value
fix available, and it is now quantified rather than assumed.

### Re-running the gold set

```bash
python -m eval.ragbench.goldset collect corpus_large --sample 60 -o eval/ragbench/gold
python -m eval.ragbench.goldset review --gold eval/ragbench/gold/gold.json
python -m eval.ragbench.goldset score  --gold eval/ragbench/gold/gold.json
```

Verdicts live in `gold/gold.json`; re-adjudicate whenever the ground-truth
extractor changes, since the verdicts are about ITS output, not about rtldoc's.

---

## The Arabic slice

Built from `book/BilArabi_TG07.pdf` (a commercial Arabic teacher's guide), which
is the only Arabic source in the corpus with real ruled tables — the Arabic
Wikipedia pages are Chrome-printed and every candidate lattice there comes back
spanning the whole page, because those pages render infoboxes, nav boxes and
tables all with borders and a y-projection cannot separate them.

**9 candidates, 9 genuinely tables (100%), 7 with a correct grid.** Adjudicated
from `gold_ar/sheet.png`. Two captured the instruction paragraph above the table
as their header row.

### Score: 18.9% — and it is a floor, not a verdict

Every one of the 7 shows the same shape: the reference has 10–12 rows where
rtldoc emits 7–8, a consistent ~30% row surplus on the *reference* side. The
reference is over-splitting wrapped cells, not rtldoc losing rows.

### Three benchmark bugs the Arabic slice exposed

Finding them is what the slice was worth; all three also affected Latin.

1. **RTL word order inside a cell.** `_grid_from_axes` sorted a cell's words by
   ascending x, which is correct for Latin and exactly backwards for Arabic —
   `دون المعيار` came back as `المعيار دون`. Now decided per cell by script.
2. **Byte-equality on cell text.** The reference is read from the PDF's raw word
   stream, which still carries the defects rtldoc exists to fix: a lam-alef
   ligature arrives decomposed in visual order (`بطالقة` for `بطلاقة`) and
   bullets land on the wrong side of the first word. Comparing byte-for-byte
   scored the parser **0% for being more correct than the reference.** Cell
   comparison is now token-set near-equality.
3. **Exact header pairing.** Headers were intersected as a set, so a header
   whose words came back in visual order matched *nothing* and every value under
   that column was discarded. Headers are now paired by the same near-equality.

Fixing 2 and 3 moved the Latin set from 63.8% to **64.6%** (26 perfect, up from
22) with no parser change at all — those four tables had been correct all along.

### The limit of derived ground truth

`page.get_text("words")` cannot be the reference for Arabic. It has the ligature
ordering, bidi and word-order defects that rtldoc was written to correct, so a
correct parser is penalised against it. Near-equality papers over the worst of
it and hits a ceiling around row structure.

**Settling Arabic tables needs hand-transcribed cells** — a real gold set rather
than a derived one. Nine tables is a small enough number to transcribe once, and
until that exists, treat 18.9% as "not yet measured", not as rtldoc's Arabic
table quality.
