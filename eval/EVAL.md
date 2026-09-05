# The evaluation suite

> **Scope: English and Arabic only.** Bidi, deshaping and RTL reading order are
> the problems worth solving here. CJK and other scripts are explicitly out of
> scope — the corpus strata that matter are `*__latin` and `*__arabic`.

Built to answer one complaint: *the metrics said the output was good, and then
you opened it and the tables, headings and text were broken.*

That is not a tuning problem. It is a **measurement** problem, and it has two
causes, both of which this suite is designed around.

## Cause 1 — character metrics cannot see structure

The old check (`invariants.py`) measures how many of the source's characters
survive into the output. A shredded two-column page contains **every single
character**, in the wrong order. It scores 1.00. So does a table flattened into
prose, a heading demoted to body text, and a grid whose cells were shuffled.

Coverage answers *did we keep the text*. Nobody was asking that. The question
is *is the output usable*, and that is a question about structure.

So every detector in `detectors.py` is written to one rule:

> A detector may only fire on something a person would point at and call broken.

## Cause 2 — the corpus is a monoculture

85% of the corpus is arXiv papers from one generator (`arXiv GenPDF`). A mean
over that is an arXiv mean. It will sit at 0.98 no matter how badly Word
documents, IRS publications or Arabic textbooks are handled, because together
they are ~5% of the mass.

So nothing here is ever pooled. `corpus_build.py` stratifies by **producer
family × script** — what actually predicts failure — and `scorecard.py` reports
per stratum, worst first, never a single headline number.

---

## The four tools

| | what it does |
|---|---|
| `corpus_build.py` | dedupes by SHA-256, stratifies, copies, writes `manifest.json` |
| `detectors.py` | the label-free defect detectors |
| `scorecard.py` | runs them over the corpus, reports per stratum |
| `review.py` | renders page-beside-output HTML so a number can be checked by eye |
| `calibrate.py` | measures each detector's precision and recall against your own judgement |

Detectors are **label-free**: they compare the output against the source PDF's
own geometry, never against a hand-made answer key. That is what lets them run
over 43,000 pages instead of the 20 pages someone had time to transcribe.

---

## How to verify the results — the actual answer

A detector is a claim. An unchecked claim is exactly what got you here. So the
suite is built to be distrusted, in three steps.

### Step 1 — look at what it caught

```bash
python eval/scorecard.py corpus_large --sample 300 --max-pages 8 --json run.json
python eval/review.py corpus_large --json run.json --worst 30 -o worst.html
```

Open `worst.html`. Page image on the left, emitted blocks on the right,
labelled with the role rtldoc assigned. If a flagged page looks fine, the
**detector** is wrong and you tighten it. Two of the original detectors were
caught this way and fixed: `digit_in_word` was firing on "5G" and "Simu5G",
and `repeated_run` was counting markdown table pipes as repeated words.

### Step 2 — look at what it MISSED

This is the step people skip, and it is the one that matters.

```bash
python eval/review.py corpus_large --json run.json --random 40 -o audit.html
```

`--random` samples pages **without regard to whether anything fired**. A
worst-first list can only ever show you what the suite already caught; it can
never show you a defect class the suite is blind to. Only random sampling can.

### Step 3 — put error bars on it

Mark each page in `audit.html` broken / fine / unsure, copy the JSON from the
bottom of the page, then:

```bash
python eval/calibrate.py --run run.json --labels labels.json
```

You get, per detector, **precision** (when it fires, how often is the page
really broken) and, for the suite, **recall** (of the pages you called broken,
how many did anything catch).

- precision below ~0.6 → the detector is training you to ignore the report
- recall below ~0.7 → a defect class is missing a detector, and no threshold
  change will fix it

After that, the numbers have known error bars instead of assumed ones. That is
the difference between a metric and a claim.

---

## What it found on the first honest run

301 documents, 2,326 pages, stratified sample:

| stratum | docs | clean pages | worst defect |
|---|---:|---:|---|
| web_print__arabic | 7 | **0.0%** | `arabic_word_split` |
| publisher__arabic | 1 | 25.0% | `arabic_word_split` |
| publisher__latin | 16 | 57.0% | `table_missed` |
| arxiv__latin | 254 | 58.8% | `repeated_run` |
| latex__latin | 3 | 66.7% | `table_missed` |
| word__latin | 8 | 77.4% | `repeated_run` |
| web_print__latin | 5 | 82.5% | `heading_starved` |
| office__latin | 3 | 87.5% | `repeated_run` |

**The headline defect: two-column reading order.** On 62 of 254 arXiv papers
(7.5% of their pages) rtldoc reports `columns: 1` on a genuinely two-column
page and emits left-column and right-column lines alternating, one for one. The
text is shredded and unreadable. Character coverage scores those pages ~1.00,
which is precisely why it went unnoticed.

```bash
python eval/review.py corpus_large --json run.json --code column_interleave --worst 8 -o ci.html
```

These numbers are **uncalibrated** — step 3 has not been run yet, so the
detectors' precision is estimated from spot checks, not measured. Do not quote
them until it has.
