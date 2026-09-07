# Table experiments — what was measured, and what it showed

Measurement scripts, not shipped code. Nothing here is imported by `rtldoc`.

Everything was scored against **FinTabNet.c** (723 hand-labelled tables,
`python -m eval.fintabnet_fetch`) except where noted, so the numbers are
comparable to each other and to `eval/fintabnet_score.py`.

## The result

Table quality decomposes into three stages, and this parser is strong at
exactly one of them:

| stage | rules | model | measured on |
|---|---:|---:|---|
| locate the table (IoU>0.5) | 52.2% | **92.5%** | 161 gold tables |
| build the grid (exact shape) | 7.7% | **52.7%** | 723 gold tables |
| build the grid (cells) | 24.4% | **65.8%** | 723 gold tables |
| read a cell, given geometry | **99.4%** | — | 161 gold cells |

Given a table's geometry this parser reads 160 of 161 cells correctly.
It does not know where the cells are. That is the whole gap.

## Dead ends, so nobody repeats them

| approach | cells recovered |
|---|---:|
| whitespace projection as the primary method | 7.4% |
| edge clustering (word starts) | 10.4% |
| edge clustering (both edges, overlap assignment) | 11.8% |
| TATR + naive rows x columns post-processing | 24.4% |
| rtldoc rules | 24.4% |

No rule-based method for inferring grid structure came close. Three
independent formulations landed between 7% and 12%.

## Arabic

The models transform Latin tables and barely move Arabic ones:

| | Latin | Arabic (Gulf, vs RAGBench reference) |
|---|---:|---:|
| rules | 24.4% | 28.9% |
| full model chain | 65.8% | 31.4% |

Meanwhile rtldoc's Arabic TEXT beats every tool tested: 7,507 Arabic
characters recovered against MinerU's 1,156 on the same document, and 0
kashida against raw PyMuPDF's 2,712. MinerU returns `عرب` for `عبر`.

There is no Arabic table ground truth in existence — a Hugging Face
search returns one layout model with zero downloads and no layout or
table-structure dataset at all. Building it is the only way to know
whether the chain helps Arabic.

## Licensing

These scripts import `mineru` (**AGPL-3.0**) as an optional *evaluation*
dependency, and load weights from `opendatalab/PDF-Extract-Kit-1.0`,
which **declares no licence**. Neither may be shipped.

For production the same models exist under Apache-2.0 from their authors:

* `PaddlePaddle/PP-DocLayout_plus-L` — layout, Apache-2.0
* `PaddlePaddle/SLANeXt_wired` — table structure, Apache-2.0
* the `slanet_plus` inference code carries the PaddlePaddle Apache-2.0 header

`pp_doclayoutv2.py` is "Copyright (c) Opendatalab" and must NOT be
vendored. The numbers above need re-confirming against the Apache-2.0
weights before anything is built on them.

## The scripts

| file | what it answers |
|---|---|
| `slanet.py` | SLANet+ geometry + this parser's text, scored on FinTabNet |
| `slanet_gulf.py` | the same on the Gulf corpus |
| `chain_gulf.py` | layout model + SLANet+ + our text, on Arabic |
| `gen_slanet.py`, `gen_chain.py` | generate HTML with the model in the table path |
| `tatr.py` | Table Transformer, for comparison |
| `route.py` | rules first, model when the grid collapses (42.8% vs 37.0%/27.6%) |
| `edgegrid.py` | the rule-based grid inference that failed |
| `diag.py`, `why.py`, `paired.py` | where the table score is lost, and why |
| `split.py` | corpus scores reported per script, never pooled |
