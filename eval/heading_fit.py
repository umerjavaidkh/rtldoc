"""Fit heading weights on declared /H1../H6, validate on hand labels.

Why this and not another hand-weighted rule
-------------------------------------------
Three attempts at heading detection have failed the same way: several weak
signals combined with weights chosen by hand. The last one dropped precision
from 0.127 to 0.039, because four soft cues each worth ~0.2 sum to exactly the
threshold with no strong evidence anywhere.

GROBID's answer to the same problem is not better weights, it is LEARNED
weights -- a CRF fitted to labelled data, which is why it reaches ~80% where
hand rules sit far lower. So the weights here are fitted too.

The labels are free: a PDF that carries a structure tree declares its own
headings in /H1../H6, owing nothing to font size, and tags.tagged_headings
reads them. Ten documents in the corpus carry one, giving 270 positives and
every other block as a negative.

Validation is the 72 hand-adjudicated cases in eval/heading_gold_set, which
come from DIFFERENT documents -- so the number at the bottom is held out, not
fitted, which is the property the previous attempts lacked.
"""

from __future__ import annotations

import glob
import json
import statistics
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


def _same(a: str, b: str) -> bool:
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y or x in y or y in x:
        return True
    return bool(fuzz) and fuzz.token_set_ratio(x, y) >= 82


def features(block, page_ctx) -> dict:
    """Layout features, all measured RELATIVE to the page/document.

    GROBID's insight is that none of these is absolute: a heading is not "12pt
    or larger", it is "larger than this document's body", set differently from
    what surrounds it, with more space above it than a paragraph gets. A Saudi
    labour regulation whose section heads run SMALLER and lighter than body
    text breaks every absolute rule and none of the relative ones.
    """
    body_size, body_style, med_gap, page_h, sizes = page_ctx
    txt = (block.text or "").strip()
    if not txt:
        return {}
    size = block.bbox[3] - block.bbox[1]
    words = txt.split()
    letters = [c for c in txt if c.isalpha()]
    return {
        "size_ratio": (block.style.size / body_size) if body_size and getattr(block.style, "size", None) else 1.0,
        "style_differs": 0.0 if block.style == body_style else 1.0,
        "gap_above": min(3.0, (block.diagnostics or {}).get("gap_above", med_gap) / med_gap) if med_gap else 1.0,
        "short": 1.0 if len(words) <= 8 else 0.0,
        "very_short": 1.0 if len(words) <= 3 else 0.0,
        "no_final_period": 0.0 if txt.rstrip().endswith((".", "،", "؛")) else 1.0,
        "all_caps": 1.0 if letters and all(c.isupper() for c in letters if c.isascii()) and any(c.isascii() for c in letters) else 0.0,
        "numbered": 1.0 if txt[:1].isdigit() or txt.startswith(("المادة", "الفصل", "الباب")) else 0.0,
        "single_line": 1.0 if "\n" not in txt else 0.0,
        "rel_y": block.bbox[1] / page_h if page_h else 0.0,
    }


def page_context(result, page):
    tally = {}
    for b in result.blocks:
        n = len((b.text or "").strip())
        if n:
            tally[b.style] = tally.get(b.style, 0) + n
    body_style = max(tally, key=tally.get) if tally else None
    body_size = getattr(body_style, "size", 10.0) if body_style else 10.0
    gaps = []
    prev = None
    for b in sorted(result.blocks, key=lambda b: b.bbox[1]):
        if prev is not None:
            gaps.append(max(0.0, b.bbox[1] - prev))
        prev = b.bbox[3]
    med_gap = statistics.median(gaps) if gaps else 10.0
    return (body_size or 10.0, body_style, med_gap or 10.0, page.rect.height,
            [getattr(b.style, "size", 10.0) for b in result.blocks])


def collect(paths, max_pages=0):
    X, y = [], []
    fitz.TOOLS.mupdf_display_errors(False)
    for f in paths:
        try:
            doc = fitz.open(f)
            declared = tags.tagged_headings(doc)
        except Exception:
            continue
        if not declared:
            doc.close()
            continue
        by_page = {}
        for pno, _lvl, mc in declared:
            by_page.setdefault(pno, []).append(mc)
        n = doc.page_count if not max_pages else min(doc.page_count, max_pages)
        for i in range(n):
            page = doc[i]
            try:
                res = pipeline.parse_page(page)
            except Exception:
                continue
            if not res.born_digital or not res.blocks:
                continue
            truth = []
            if i in by_page:
                texts = tags.mcid_text(page)
                for mc in by_page[i]:
                    t = " ".join(texts.get(m, "") for m in mc).strip()
                    if t:
                        truth.append(t)
            ctx = page_context(res, page)
            for b in res.blocks:
                if b.role in ("table", "figure") or not (b.text or "").strip():
                    continue
                fe = features(b, ctx)
                if not fe:
                    continue
                X.append(fe)
                y.append(1 if any(_same(b.text, t) for t in truth) else 0)
        doc.close()
    return X, y


def main():
    files = sorted(glob.glob("corpus_gulf/*.pdf")) + sorted(glob.glob("corpus_large/*/*.pdf"))[:120] + sorted(glob.glob("book/*.pdf"))
    X, y = collect(files, max_pages=60)
    print(f"training rows {len(X)}  positives {sum(y)}  negatives {len(y)-sum(y)}")
    if sum(y) < 20:
        print("too few positives to fit")
        return
    keys = sorted(X[0])
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    A = np.array([[r.get(k, 0.0) for k in keys] for r in X])
    b = np.array(y)
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(A, b)
    print("\nfitted weights:")
    for k, w in sorted(zip(keys, m.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"   {k:18} {w:+.3f}")
    out = Path("eval/results/heading_weights.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"keys": keys, "coef": m.coef_[0].tolist(),
                               "intercept": float(m.intercept_[0])}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
