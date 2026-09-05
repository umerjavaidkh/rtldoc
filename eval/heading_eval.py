"""Score heading detection against the headings a PDF DECLARES.

Why this file exists
--------------------
The heading term inside RAGBench infers its truth from font size -- a heading
is "text set >=1.25x the body". Two documents measured this week break that
assumption in opposite directions: an Arabic Wikipedia print whose
sub-headings run 1.16x, and a Saudi labour regulation whose 281 section heads
are set LIGHTER and SMALLER than the body they interrupt. Against that
reference a parser is punished for finding real headings, which is the
opposite of what a benchmark is for.

/H1../H6 and /Title in the structure tree are the producer's own declaration.
They owe nothing to font size, weight, spacing or wording, so they can score a
detector that uses all four without the circularity of grading a system
against its own logic.

Only tagged documents can be scored this way, and that is stated rather than
worked around: the number below covers the subset of the corpus that carries a
structure tree, and says nothing about the rest.

Usage:
    python eval/heading_eval.py corpus_gulf corpus_large/web_print__arabic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline, tags  # noqa: E402
from eval.ragbench.rules import normalize  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:                                   # pragma: no cover
    fuzz = None

MATCH = 82          # token-set ratio; tolerates deshaping and ligature spelling


def _same(a: str, b: str) -> bool:
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y or x in y or y in x:
        return True
    return bool(fuzz) and fuzz.token_set_ratio(x, y) >= MATCH


def score_document(path: str, max_pages: int = 0) -> dict:
    fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(path)
    declared = tags.tagged_headings(doc)
    if not declared:
        doc.close()
        return {}
    by_page: dict[int, list[tuple[int, list[int]]]] = {}
    for page_no, level, mcids in declared:
        by_page.setdefault(page_no, []).append((level, mcids))

    n = doc.page_count if not max_pages else min(doc.page_count, max_pages)
    tp = fp = fn = 0
    missed, spurious = [], []
    for i in range(n):
        want = by_page.get(i, [])
        page = doc[i]
        try:
            result = pipeline.parse_page(page)
        except Exception:
            continue
        if not result.born_digital:
            continue
        got = [b.text for b in result.blocks if (b.role or "").startswith("heading")]
        truth = []
        if want:
            texts = tags.mcid_text(page)
            for _level, mcids in want:
                t = " ".join(texts.get(m, "") for m in mcids).strip()
                if t:
                    truth.append(t)
        used = set()
        for t in truth:
            hit = next((j for j, g in enumerate(got) if j not in used and _same(t, g)), None)
            if hit is None:
                fn += 1
                if len(missed) < 12:
                    missed.append(f"p{i+1} {t[:56]!r}")
            else:
                used.add(hit)
                tp += 1
        for j, g in enumerate(got):
            if j not in used:
                fp += 1
                if len(spurious) < 12:
                    spurious.append(f"p{i+1} {g[:56]!r}")
    doc.close()
    return {"pdf": path, "tp": tp, "fp": fp, "fn": fn,
            "missed": missed, "spurious": spurious}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    pdfs = []
    for r in a.roots:
        p = Path(r)
        pdfs.extend(sorted(str(x) for x in p.rglob("*.pdf")) if p.is_dir() else [str(p)])

    tot = {"tp": 0, "fp": 0, "fn": 0}
    scored = 0
    rows = []
    for f in pdfs:
        d = score_document(f, a.max_pages)
        if not d:
            continue
        scored += 1
        for k in tot:
            tot[k] += d[k]
        rows.append(d)
    if not scored:
        print("no tagged documents found")
        raise SystemExit(0)

    prec = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    rec = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"# heading detection vs the structure tree")
    print(f"  documents scored : {scored} of {len(pdfs)} (rest carry no tags)")
    print(f"  declared headings: {tot['tp'] + tot['fn']}")
    print(f"  precision        : {prec:.3f}   ({tot['tp']} of {tot['tp'] + tot['fp']} emitted are real)")
    print(f"  recall           : {rec:.3f}   ({tot['tp']} of {tot['tp'] + tot['fn']} declared were found)")
    print(f"  F1               : {f1:.3f}")
    if a.verbose:
        for d in rows[:4]:
            if d["missed"] or d["spurious"]:
                print(f"\n  {Path(d['pdf']).name[:44]}  tp={d['tp']} fp={d['fp']} fn={d['fn']}")
                for m in d["missed"][:4]:
                    print(f"     MISSED   {m}")
                for s in d["spurious"][:4]:
                    print(f"     SPURIOUS {s}")
