# Universality test evidence

Results of running rtldoc over a diverse, unseen corpus to test whether the
parser's guarantees hold beyond the 5 documents it was developed on.

## The corpus

96 complex born-digital PDFs, **2336 pages**, downloaded from arXiv across 15
subject areas (cs, math, stat, econ, q-fin, q-bio, physics, astro, cond-mat,
eess) — chosen because each area uses different LaTeX templates, column
counts, and table/figure/math density, so the corpus stresses layouts the
parser has never seen.

The PDFs themselves are **not committed** — they are third-party copyrighted
academic papers, and arXiv's default license does not grant redistribution.
Instead this directory carries everything needed to reproduce the test:

- `corpus_manifest.txt` — the exact 96 arXiv IDs
- `../fetch_corpus.py` — re-downloads the identical corpus
- `invariants_corpus.txt`, `hardness_corpus.txt` — the saved reports

Reproduce:
```bash
python eval/fetch_corpus.py corpus 100
python eval/invariants.py corpus/ | tee eval/results/invariants_corpus.txt
python eval/hardness.py corpus/ --top 30 | tee eval/results/hardness_corpus.txt
```

## What it showed

**HARD guarantees held universally** — across all 2336 unseen pages: 0 crashes,
0 presentation-form leaks, 0 non-rectangular tables, 0 non-determinism. Mean
letter coverage vs the true glyph stream: 0.988.

**The multi-angle detector found a real, systematic defect** the 5-document
regression testing could never have surfaced: text **duplication** on ~17% of
pages (396/2332 originally), co-occurring with region **overlap**. Spot-checked
and confirmed genuine (once in source, twice in output — a parser bug, not
source repetition). A merge policy (`pipeline._dedupe_blocks`) then cut it to
~4% (**396 → 86 flagged pages, −78%**) with text coverage unchanged (0.988) and
all HARD invariants still zero — the numbers in `invariants_corpus.txt` /
`hardness_corpus.txt` here are the post-fix state.

## Textbooks (physics / chemistry / calculus / organic / biology)

`invariants_textbooks.txt` and `hardness_textbooks.txt` are the same harnesses
run over 5 OpenStax textbooks (CC-BY), **4888 pages** of the hardest layout —
geometry diagrams, figures, tables, worked examples, exercises, chemical
structures. HARD invariants: all zero. Coverage 0.990.

## Table accuracy (TEDS)

`teds_meta_p83.txt` scores the borderless META 10-K income statement with
**TEDS** (the PubTabNet/OmniDocBench table metric, implemented in
`eval/teds.py`): rtldoc **0.503** vs pdfplumber 0.053 vs naive 0.000 — a clear
win, with the modest absolute number honestly reflecting approximate grid
geometry on wide, multi-level-header tables (right data, occasional row/
column structure mismatches).

That is the point of label-free universality testing plus one real
gold-standard metric: it produced both the wins and the limitations.
