"""
Property-based invariant checks -- the label-free path to "works on any PDF".

The gold-set eval (eval/run_matrix.py) measures *correctness* against
hand-transcribed truth, which is accurate but tiny and expensive. This module
is the complement: it checks properties that must hold for EVERY born-digital
PDF, needs no gold labels at all, and therefore scales to as many documents as
you can point it at. Run it over a big, diverse corpus and it flags the pages
where rtldoc did something structurally wrong -- lost text, duplicated text,
emitted a broken table, crashed -- without anyone having to say what the right
answer was.

Two kinds of check:

  HARD invariants must never be violated on any input. A violation is a bug:
    - no crash on any page
    - no Arabic presentation-form codepoints survive into the output
      (the deshaping guarantee)
    - every rendered table is rectangular (all rows same column count)
    - parsing is deterministic (same bytes in -> same text out)

  SOFT diagnostics won't be perfect (rtldoc intentionally drops page
  furniture and reorders text), so they're reported as distributions and
  flagged past a threshold rather than failed outright:
    - text COVERAGE: fraction of the source's letters that survive into
      output -- low coverage means a region was dropped
    - text EXCESS: how much MORE of a letter appears in output than in source
      -- high excess means text was duplicated or invented (the exact bug
      class that made a passage appear twice on one page)

Usage:
    python eval/invariants.py path/to/file.pdf [more.pdf ...]
    python eval/invariants.py path/to/corpus_dir/       # recurses for *.pdf
"""

from __future__ import annotations

import collections
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402

_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")

# thresholds for soft diagnostics -- below/above these a page is flagged
COVERAGE_FLAG = 0.85     # <85% of source letters survived -> possible loss
EXCESS_FLAG = 0.15       # >15% extra letters vs source -> possible duplication


def _letters(s: str) -> collections.Counter:
    """Multiset of alphabetic characters, normalized so that presentation
    forms and composed/decomposed variants compare equal across source and
    output."""
    norm = unicodedata.normalize("NFKC", s)
    return collections.Counter(c for c in norm.lower() if c.isalpha())


def _coverage_excess(source: str, out: str) -> tuple[float, float]:
    src, dst = _letters(source), _letters(out)
    total = sum(src.values())
    if total == 0:
        return 1.0, 0.0
    kept = sum(min(src[c], dst[c]) for c in src)
    excess = sum(max(0, dst[c] - src[c]) for c in dst)
    return kept / total, excess / total


def _table_rows(text: str) -> list[int]:
    """Column counts of each rendered markdown-table row (skipping the
    |---|---| separator), for the rectangularity check."""
    counts = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        counts.append(line.count("|") - 1)
    return counts


def check_pdf(path: str, determinism_sample: int = 5) -> dict:
    doc = fitz.open(path)
    n = doc.page_count
    report = {
        "pdf": path, "pages": n,
        "crashes": [], "presentation_forms": [], "nonrect_tables": [],
        "nondeterministic": [], "low_coverage": [], "high_excess": [],
        "coverage": [], "excess": [], "needs_ocr": 0,
    }

    for i in range(n):
        page = doc[i]
        try:
            r = pipeline.parse_page(page)
        except Exception as e:  # HARD: robustness
            report["crashes"].append((i + 1, repr(e)))
            continue

        if not r.born_digital:
            report["needs_ocr"] += 1
            continue

        out = "\n".join(b.text for b in r.blocks)

        if _PRESENTATION.search(out):                       # HARD: deshaping
            report["presentation_forms"].append(i + 1)

        for b in r.blocks:                                  # HARD: rectangular tables
            if b.role == "table":
                rc = _table_rows(b.text)
                if rc and len(set(rc)) > 1:
                    report["nonrect_tables"].append((i + 1, rc))

        cov, exc = _coverage_excess(page.get_text(), out)   # SOFT diagnostics
        report["coverage"].append(cov)
        report["excess"].append(exc)
        if cov < COVERAGE_FLAG:
            report["low_coverage"].append((i + 1, round(cov, 3)))
        if exc > EXCESS_FLAG:
            report["high_excess"].append((i + 1, round(exc, 3)))

    # HARD: determinism, on a sample of pages
    for i in range(0, n, max(1, n // determinism_sample))[:determinism_sample]:
        try:
            a = "\n".join(b.text for b in pipeline.parse_page(doc[i]).blocks)
            b_ = "\n".join(b.text for b in pipeline.parse_page(doc[i]).blocks)
            if a != b_:
                report["nondeterministic"].append(i + 1)
        except Exception:
            pass

    doc.close()
    return report


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 1.0


def summarize(reports: list[dict]) -> str:
    lines = []
    total_pages = sum(r["pages"] for r in reports)
    hard = {"crashes": 0, "presentation_forms": 0, "nonrect_tables": 0, "nondeterministic": 0}
    for r in reports:
        for k in hard:
            hard[k] += len(r[k])

    lines.append(f"# invariant report: {len(reports)} PDF(s), {total_pages} pages\n")
    lines.append("## HARD invariants (any nonzero is a bug)")
    lines.append(f"  crashes                : {hard['crashes']}")
    lines.append(f"  presentation forms out : {hard['presentation_forms']}")
    lines.append(f"  non-rectangular tables : {hard['nonrect_tables']}")
    lines.append(f"  non-deterministic pages: {hard['nondeterministic']}")

    all_cov = [c for r in reports for c in r["coverage"]]
    all_exc = [e for r in reports for e in r["excess"]]
    low = sum(len(r["low_coverage"]) for r in reports)
    high = sum(len(r["high_excess"]) for r in reports)
    ocr = sum(r["needs_ocr"] for r in reports)
    lines.append("\n## SOFT diagnostics (distributions, flags are for review)")
    lines.append(f"  mean letter coverage   : {_mean(all_cov):.3f}  (1.0 = no text lost)")
    lines.append(f"  mean letter excess     : {_mean(all_exc):.3f}  (0.0 = nothing duplicated)")
    lines.append(f"  pages flagged low cov  : {low}  (<{COVERAGE_FLAG})")
    lines.append(f"  pages flagged high exc : {high}  (>{EXCESS_FLAG})")
    lines.append(f"  pages routed to OCR     : {ocr}  (no text layer; expected for scans)")

    # surface the worst offenders so they're actionable
    worst_cov = sorted(((c, r["pdf"], p) for r in reports for p, c in r["low_coverage"]))[:8]
    if worst_cov:
        lines.append("\n  worst coverage pages:")
        for c, pdf, p in worst_cov:
            lines.append(f"    {c:.3f}  {Path(pdf).name} p{p}")
    for r in reports:
        for p, e in r["crashes"]:
            lines.append(f"\n  CRASH {Path(r['pdf']).name} p{p}: {e}")
    return "\n".join(lines)


def _iter_pdfs(args: list[str]):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(str(x) for x in p.rglob("*.pdf"))
        elif p.suffix.lower() == ".pdf":
            yield str(p)


if __name__ == "__main__":
    pdfs = list(_iter_pdfs(sys.argv[1:]))
    if not pdfs:
        print("usage: python eval/invariants.py <pdf-or-dir> [...]", file=sys.stderr)
        sys.exit(2)
    reports = [check_pdf(p) for p in pdfs]
    print(summarize(reports))
    hard_total = sum(len(r[k]) for r in reports
                     for k in ("crashes", "presentation_forms", "nonrect_tables", "nondeterministic"))
    sys.exit(1 if hard_total else 0)
