"""Build and score a hand-verified gold set for tables.

Why
---
RAGBench's table dimension derives its ground truth from drawn rules. That
works, but it admits figures: a flowchart or a chart axis produces the same
rule intersections a table does. Five per-grid discriminators were tried and
none separated the populations cleanly, so the corpus-level table number
carries an unknown amount of false ground truth and cannot be quoted.

A gold set settles it. Each candidate lattice is rendered and judged once, by a
person, against three questions that are genuinely different:

  is_table   -- is this region a table at all, or a figure the rule-finder
                mistook for one? Answers "how much false ground truth is in
                the corpus number".
  gt_correct -- did the derived cell grid capture the table correctly? Answers
                "is the ground truth itself trustworthy where it fires".
  Only where BOTH are true does rtldoc's score count. That restricted score is
  the settled number.

Separating the three is the whole point. A single "is this right" verdict
cannot tell a parser bug apart from a benchmark bug, which is exactly the
confusion that produced the unusable 4%, 9% and 22% figures in turn.

Usage:
    python -m eval.ragbench.goldset collect corpus_large --sample 60 -o gold/
    # ... adjudicate: fill in verdicts in gold/gold.json ...
    python -m eval.ragbench.goldset score corpus_large --gold gold/gold.json
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402

from eval.ragbench import metrics, rules  # noqa: E402

CROP_ZOOM = 2.4
PAD = 6.0


def lattice_boxes(page) -> list[tuple]:
    """Bounding boxes of the lattices `rules.ruled_table_records` accepts.

    Recomputed here rather than returned from there so the gold set records
    WHERE each grid came from -- a verdict is worthless if you cannot point at
    the region it was about."""
    try:
        draws = page.get_drawings()
    except Exception:
        return []
    h, v = rules._rules_from_drawings(draws)
    if len(h) < 3 or len(v) < 3:
        return []
    words = [w for w in page.get_text("words") if w[4].strip()]
    h.sort(key=lambda t: t[2])
    bands, cur = [], [h[0]]
    for line in h[1:]:
        if line[2] - cur[-1][2] <= 120:
            cur.append(line)
        else:
            bands.append(cur)
            cur = [line]
    bands.append(cur)

    out = []
    for band in bands:
        if len(band) < 3:
            continue
        ys = rules._dedupe_axis([b[2] for b in band])
        y0, y1 = ys[0], ys[-1]
        x0 = min(b[0] for b in band)
        x1 = max(b[1] for b in band)
        cross = [c for c in v if c[0] < y1 - 2 and c[1] > y0 + 2 and x0 - 4 <= c[2] <= x1 + 4]
        xs = rules._dedupe_axis([c[2] for c in cross])
        if len(ys) < 3 or len(xs) < 3:
            continue
        grid = rules._grid_from_axes(words, xs, ys)
        if grid and rules._is_table_shaped(grid):
            out.append((x0, y0, x1, y1, grid))
    return out


def collect(root: Path, sample: int, out_dir: Path, seed: int = 0,
            stratum: str | None = None) -> dict:
    fitz.TOOLS.mupdf_display_errors(False)
    meta = {}
    mf = root / "manifest.json"
    if mf.exists():
        data = json.loads(mf.read_text())
        meta = {str(root / d["path"]): d for d in data["documents"] if "path" in d}

    pdfs = sorted(str(p) for p in root.rglob("*.pdf"))
    if stratum:
        # substring match, so "arabic" selects every Arabic stratum at once
        pdfs = [p for p in pdfs if stratum in meta.get(p, {}).get("stratum", "")]
    rng = random.Random(seed)
    rng.shuffle(pdfs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    cases = []

    for path in pdfs:
        if len(cases) >= sample:
            break
        try:
            doc = fitz.open(path)
        except Exception:
            continue
        if not rules.has_reliable_ruled_tables(doc):
            doc.close()
            continue
        for i in range(min(doc.page_count, 20)):
            if len(cases) >= sample:
                break
            page = doc[i]
            boxes = lattice_boxes(page)
            if not boxes:
                continue
            try:
                result = pipeline.parse_page(page)
                md = pipeline.to_markdown(result)
            except Exception:
                continue
            for x0, y0, x1, y1, grid in boxes:
                if len(cases) >= sample:
                    break
                cid = f"{Path(path).stem[:20]}_p{i+1}_{int(x0)}_{int(y0)}"
                clip = fitz.Rect(x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD) & page.rect
                pix = page.get_pixmap(matrix=fitz.Matrix(CROP_ZOOM, CROP_ZOOM), clip=clip)
                crop = out_dir / "crops" / f"{cid}.png"
                pix.save(crop)
                cases.append({
                    "id": cid,
                    "pdf": path,
                    "stratum": meta.get(path, {}).get("stratum", "unknown"),
                    "page": i + 1,
                    "bbox": [round(v, 1) for v in (x0, y0, x1, y1)],
                    "crop": str(crop.relative_to(out_dir)),
                    "gt_grid": grid,
                    "pred_records": metrics.records_from_markdown(md, i + 1),
                    "verdict": {"is_table": None, "gt_correct": None, "note": ""},
                })
        doc.close()

    gold = {"cases": cases, "sample": sample, "seed": seed}
    (out_dir / "gold.json").write_text(json.dumps(gold, indent=1))
    return gold


def score(gold: dict) -> str:
    """The settled number, plus the two rates that explain it."""
    cases = gold["cases"]
    judged = [c for c in cases if c["verdict"]["is_table"] is not None]
    if not judged:
        return "no adjudicated cases yet -- fill in gold.json verdicts first"

    real = [c for c in judged if c["verdict"]["is_table"]]
    usable = [c for c in real if c["verdict"]["gt_correct"]]

    L = [f"# table gold set: {len(judged)} adjudicated candidate(s)", ""]
    L.append("## how much of the ground truth was real")
    L.append(f"  candidates judged        : {len(judged)}")
    L.append(f"  actually tables          : {len(real)}  ({len(real)/len(judged):.0%})")
    L.append(f"  FALSE ground truth       : {len(judged)-len(real)}  "
             f"({1-len(real)/len(judged):.0%})  <- figures read as tables")
    if real:
        L.append(f"  of the real tables, grid correct : {len(usable)}/{len(real)}"
                 f"  ({len(usable)/len(real):.0%})")
    L.append("")

    if not usable:
        L.append("no case has both a real table AND a correct derived grid, so")
        L.append("there is nothing to score rtldoc against. Widen the sample.")
        return "\n".join(L)

    L.append("## rtldoc's table fidelity, on verified tables only")
    L.append("   Each ground-truth grid is matched against the BEST-scoring table")
    L.append("   rtldoc emitted on that page. Scoring it against every record on")
    L.append("   the page instead would punish rtldoc for other tables it got")
    L.append("   right, which measures the page, not the table.")
    fitz.TOOLS.mupdf_display_errors(False)
    per = []
    for c in usable:
        truth = rules.grid_to_records(c["gt_grid"], c["page"])
        try:
            doc = fitz.open(c["pdf"])
            md = pipeline.to_markdown(pipeline.parse_page(doc[c["page"] - 1]))
            doc.close()
            candidates = metrics.tables_from_markdown(md) or [[]]
        except Exception:
            candidates = [c["pred_records"]]
        best = max(metrics.table_record_fidelity(truth, cand).value
                   for cand in candidates)
        per.append((best, c))
    mean = sum(v for v, _ in per) / len(per)
    L.append(f"  cases                    : {len(per)}")
    L.append(f"  TableRecordMatch (mean)  : {mean:.1%}   <-- the settled number")
    perfect = sum(1 for v, _ in per if v >= 0.99)
    zero = sum(1 for v, _ in per if v <= 0.01)
    L.append(f"  perfect (>=99%)          : {perfect}/{len(per)}")
    L.append(f"  total loss (<=1%)        : {zero}/{len(per)}")

    by = collections.defaultdict(list)
    for v, c in per:
        by[c["stratum"]].append(v)
    L.append("")
    L.append("  by stratum:")
    for st, vals in sorted(by.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        L.append(f"    {st:24} {sum(vals)/len(vals):6.1%}  (n={len(vals)})")

    L.append("")
    L.append("  worst cases:")
    for v, c in sorted(per, key=lambda t: t[0])[:8]:
        L.append(f"    {v:6.1%}  {c['id'][:46]}  {c['verdict'].get('note', '')[:40]}")
    return "\n".join(L)


def make_review(gold: dict, out_dir: Path, path: Path) -> None:
    """One page showing crop, derived grid and rtldoc's records side by side."""
    parts = ["""<style>
body{font:13px/1.5 -apple-system,sans-serif;margin:0;background:#f6f6f4}
.c{background:#fff;margin:16px;border:1px solid #ddd;border-radius:6px}
.h{padding:8px 12px;background:#fafafa;border-bottom:1px solid #eee;font-weight:600}
.g{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}
.g>div{padding:10px;overflow:auto;max-height:60vh;border-left:1px solid #eee}
.g>div:first-child{border-left:0}
img{max-width:100%;border:1px solid #ccc}
table{border-collapse:collapse;font-size:11px}
td{border:1px solid #ddd;padding:2px 4px;max-width:150px;overflow:hidden}
h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;color:#888}
pre{font:11px ui-monospace,monospace;white-space:pre-wrap}
</style><h2 style="margin:16px">Table gold set — is it a table? is the grid right?</h2>"""]
    for c in gold["cases"]:
        img = (out_dir / c["crop"]).read_bytes()
        b64 = base64.b64encode(img).decode()
        gt = "".join("<tr>" + "".join(f"<td>{html.escape(x[:40])}</td>" for x in row) + "</tr>"
                     for row in c["gt_grid"][:12])
        pred = "\n".join(json.dumps({k[:24]: v[:24] for k, v in r.items()},
                                    ensure_ascii=False)
                         for r in c["pred_records"][:8]) or "(no records)"
        parts.append(f"""<div class="c"><div class="h">{html.escape(c['id'])}
 &nbsp;<span style="font-weight:400;color:#777">{html.escape(c['stratum'])}
 &middot; p{c['page']}</span></div><div class="g">
 <div><h4>the region</h4><img src="data:image/png;base64,{b64}"></div>
 <div><h4>derived ground truth</h4><table>{gt}</table></div>
 <div><h4>rtldoc records</h4><pre>{html.escape(pred)}</pre></div>
</div></div>""")
    path.write_text("\n".join(parts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "score", "review"])
    ap.add_argument("root", nargs="?", default="corpus_large")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--stratum", default=None,
                    help="substring of the stratum name, e.g. 'arabic'")
    ap.add_argument("--gold", default="eval/ragbench/gold/gold.json")
    ap.add_argument("-o", "--out", default="eval/ragbench/gold")
    args = ap.parse_args()

    if args.cmd == "collect":
        g = collect(Path(args.root), args.sample, Path(args.out), stratum=args.stratum)
        print(f"collected {len(g['cases'])} candidates into {args.out}")
    elif args.cmd == "review":
        g = json.loads(Path(args.gold).read_text())
        out = Path(args.gold).parent
        make_review(g, out, out / "review.html")
        print(f"wrote {out/'review.html'}")
    else:
        print(score(json.loads(Path(args.gold).read_text())))
