"""
Multi-angle detector for hard / problematic extraction areas.

No single metric can be trusted to find where a parser struggles -- we already
saw PyMuPDF's own get_text() silently return a third of a page's text on some
documents, which would fool a coverage-vs-get_text check completely. So this
computes several *independent* signals from different angles and flags a page
by CONSENSUS: the more angles that agree a page is hard, the more likely it
genuinely is. Each signal is also reported on its own, so the output says not
just *which* pages are hard but *why*, and -- aggregated over a corpus -- which
failure modes are systematic rather than one-offs.

The angles (each normalized so higher = harder, each with a flag threshold):

  completeness  low letter coverage vs the rawdict glyph stream (the reliable
                baseline, not get_text) -- text is being dropped
  disagreement  get_text and rawdict disagree on how much text exists -- the
                page's encoding is itself ambiguous / hard
  duplication   the same line appears more than once -- text invented/repeated
  orphaning     geo lines that fall cleanly inside no region -- segmentation
                doesn't tile the page
  overlap       regions overlap each other a lot -- ambiguous segmentation
  presentation  Arabic presentation forms in the raw text -- the hard shaping
                / bidi path is exercised heavily
  reversal      many lines needed bidi reversal -- heavy RTL/mixed direction
  fragmentation many length-1 output tokens -- glyphs split from their word
                (orphaned diacritics, broken clustering)

Usage:
    python eval/hardness.py <pdf-or-dir> [--top N]
"""

from __future__ import annotations

import collections
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import arabic, geobidi, pipeline  # noqa: E402
from rtldoc.primitives import containment, extract_page, rawdict  # noqa: E402
from rtldoc.layout import (assign_spans, order_regions, propose_regions)  # noqa: E402

_PRESENTATION = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
_ARLETTER = re.compile(r"[؀-ۿ]")

# per-angle flag thresholds (signal in [0,1]; >= threshold => that angle fires)
THRESHOLDS = {
    "completeness": 0.15,   # >15% of glyph-stream letters missing from output
    "disagreement": 0.25,
    "duplication": 0.08,
    "orphaning": 0.30,
    "overlap": 0.35,
    "presentation": 0.10,
    "reversal": 0.30,
    "fragmentation": 0.30,
}


def _letters(s: str) -> collections.Counter:
    return collections.Counter(c for c in unicodedata.normalize("NFKC", s).lower() if c.isalpha())


def _rawdict_text(raw: dict) -> str:
    out = []
    for b in raw["blocks"]:
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            for sp in ln["spans"]:
                out.append("".join(ch["c"] for ch in sp.get("chars", [])))
    return " ".join(out)


def page_signals(page: "fitz.Page") -> dict | None:
    raw = rawdict(page)
    prim = extract_page(page, raw=raw)
    if not prim.is_born_digital:
        return None

    result = pipeline.parse_page(page)
    out = "\n".join(b.text for b in result.blocks)

    # --- angle: completeness (vs rawdict, the reliable baseline) ---
    truth = _letters(_rawdict_text(raw))
    got = _letters(out)
    tot = sum(truth.values()) or 1
    coverage = sum(min(truth[c], got[c]) for c in truth) / tot
    completeness = 1.0 - coverage

    # --- angle: get_text vs rawdict self-disagreement ---
    gt = sum(_letters(page.get_text()).values())
    disagreement = abs(gt - tot) / max(tot, 1)

    # --- angle: duplication ---
    lines = [l.strip() for l in out.splitlines() if len(l.strip()) > 12]
    dup = sum(c - 1 for c in collections.Counter(lines).values() if c > 1)
    duplication = dup / max(len(lines), 1)

    # --- angle: orphaning (segmentation doesn't tile the page) ---
    regions = order_regions(assign_spans(prim, propose_regions(prim)), prim.width, prim.height)
    leaf = list(regions)
    for r in regions:
        if r.kind == "table":
            leaf.extend(r.cells)
    geo = geobidi.page_lines(page, raw=raw)
    orphan = sum(1 for bb, _ in geo if max((containment(bb, r.bbox) for r in leaf), default=0) < 0.55)
    orphaning = orphan / max(len(geo), 1)

    # --- angle: region overlap (ambiguous segmentation) ---
    boxes = [r.bbox for r in regions if r.kind != "table"]
    ov = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ov += containment(boxes[i], boxes[j])
    overlap = min(ov / max(len(boxes), 1), 1.0)

    # --- angle: presentation forms in raw (hard shaping path) ---
    rawtext = _rawdict_text(raw)
    arletters = sum(1 for c in rawtext if _ARLETTER.match(c) or _PRESENTATION.match(c)) or 1
    presentation = sum(1 for c in rawtext if _PRESENTATION.match(c)) / arletters

    # --- angle: bidi reversal rate ---
    rev = sum(b.diagnostics.get("reversed_lines", 0) for b in result.blocks)
    reversal = min(rev / max(len(lines), 1), 1.0)

    # --- angle: fragmentation (glyphs split from words) ---
    toks = out.split()
    frag = sum(1 for t in toks if len(t) == 1 and t.isalpha())
    fragmentation = frag / max(len(toks), 1)

    sig = {
        "completeness": completeness, "disagreement": disagreement,
        "duplication": duplication, "orphaning": orphaning, "overlap": overlap,
        "presentation": presentation, "reversal": reversal, "fragmentation": fragmentation,
    }
    fired = [k for k, v in sig.items() if v >= THRESHOLDS[k]]
    return {"signals": sig, "fired": fired, "consensus": len(fired)}


def analyze(path: str) -> list[dict]:
    doc = fitz.open(path)
    rows = []
    for i in range(doc.page_count):
        try:
            s = page_signals(doc[i])
        except Exception as e:
            rows.append({"page": i + 1, "pdf": path, "error": repr(e), "consensus": 99, "fired": ["CRASH"]})
            continue
        if s is None:
            continue
        s["page"] = i + 1
        s["pdf"] = path
        rows.append(s)
    doc.close()
    return rows


def _iter_pdfs(args):
    for a in args:
        p = Path(a)
        if p.is_dir():
            yield from sorted(str(x) for x in p.rglob("*.pdf"))
        elif p.suffix.lower() == ".pdf":
            yield str(p)


def report(rows: list[dict], top: int = 25) -> str:
    graded = [r for r in rows if "error" not in r]
    lines = [f"# hardness report: {len(rows)} pages across {len({r['pdf'] for r in rows})} PDF(s)\n"]

    # corpus-level: which angles are systematically the parser's weak spots
    counts = collections.Counter(k for r in graded for k in r["fired"])
    crashes = [r for r in rows if "error" in r]
    lines.append("## which angles fire most (systematic weak spots)")
    for k in THRESHOLDS:
        n = counts.get(k, 0)
        bar = "#" * int(40 * n / max(len(graded), 1))
        lines.append(f"  {k:14s} {n:4d}  {bar}")
    lines.append(f"\n  crashes: {len(crashes)}")

    # the hardest pages, by how many angles agree
    hard = sorted(rows, key=lambda r: -r["consensus"])[:top]
    lines.append(f"\n## hardest pages (>=2 angles agree), top {top}")
    lines.append("consensus | page | fired angles")
    for r in hard:
        if r["consensus"] < 2:
            break
        lines.append(f"    {r['consensus']:2d}    | {Path(r['pdf']).name[:22]:22s} p{r['page']:<4d} | {', '.join(r['fired'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top = 25
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    pdfs = list(_iter_pdfs(args))
    if not pdfs:
        print("usage: python eval/hardness.py <pdf-or-dir> [--top N]", file=sys.stderr)
        sys.exit(2)
    rows = []
    for p in pdfs:
        rows.extend(analyze(p))
    print(report(rows, top))
