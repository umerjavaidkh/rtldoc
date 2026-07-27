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
pages (396/2332), strongly co-occurring with region **overlap** (276 pages).
Spot-checked and confirmed genuine — real sentence fragments repeated on
dense/multi-column academic pages, not benign header repetition. This is the
honest limitation stated in the top-level README; it is the tension exposed by
the text-loss fix (assign every line to its best-overlap region) on layouts
where span-assignment and geo-line ownership disagree.

That is the point of label-free universality testing: it caught a defect that
would otherwise have shipped as an overclaim.
