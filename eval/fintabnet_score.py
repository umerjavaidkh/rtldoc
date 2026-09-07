"""Score rtldoc against FinTabNet.c ground truth.

An independent instrument. FinTabNet.c is labelled by neither this parser
nor RAGBench's reference, so where the three disagree it is the one with
no stake in the answer. Its annotations carry row_nums and column_nums
per cell, which means SPANNING cells and multi-level headers are ground
truth -- the structure this parser has no concept of.

    python -m eval.fintabnet_score [--limit N]

Licence: FinTabNet.c is CDLA-Permissive-2.0. PDFs from the ComTQA mirror.
"""
from __future__ import annotations
import argparse, json, sys, statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz                                    # noqa: E402
from rtldoc import pipeline, arabic            # noqa: E402

GOLD = Path("eval/fintabnet/gold")
PDFS = Path("eval/fintabnet/pdfs")


def _norm(t: str) -> str:
    return " ".join(arabic.normalize(t or "")[0].split()).strip().lower()


def gold_grid(cells: list) -> tuple[list[list[str]], int]:
    """Ground-truth grid, plus how many rows are column header."""
    if not cells:
        return [], 0
    nrows = max(max(c["row_nums"]) for c in cells) + 1
    ncols = max(max(c["column_nums"]) for c in cells) + 1
    g = [["" for _ in range(ncols)] for _ in range(nrows)]
    for c in cells:
        # A spanning cell is written once, at its top-left, exactly as a
        # flat grid would carry it.
        g[min(c["row_nums"])][min(c["column_nums"])] = c.get("pdf_text_content") or ""
    hdr = {r for c in cells if c.get("is_column_header") for r in c["row_nums"]}
    return g, (max(hdr) + 1 if hdr else 0)


def _at(pred, r, c) -> str:
    return _norm(pred[r][c]) if 0 <= r < len(pred) and 0 <= c < len(pred[r]) else ""


def cell_score(gold: list[list[str]], pred: list[list[str]],
               align: bool = True) -> tuple[float, int]:
    """Fraction of ground-truth cells recovered, and the row offset used.

    Aligned by default. A parser that drops a table's header rows starts
    its grid several rows into the reference's, and scoring by absolute
    position then reports ~0 for a table whose every value is correct --
    measured on ADS_2007_page_106, where all six columns of every data row
    match and the naive score is 0.0 because our row 0 is the gold's row 4.
    Alignment separates "did we read the cells" from "did we keep the
    header", which are different defects with different fixes.
    """
    gt = [(r, c, _norm(v)) for r, row in enumerate(gold)
          for c, v in enumerate(row) if _norm(v)]
    if not gt:
        return None, 0
    offsets = range(-2, 9) if align else [0]
    best, best_dr = 0.0, 0
    for dr in offsets:
        hit = sum(1 for r, c, v in gt if _at(pred, r - dr, c) == v)
        if hit > best:
            best, best_dr = hit, dr
    return best / len(gt), best_dr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    stems = sorted(p.stem for p in GOLD.glob("*.json"))
    if args.limit:
        stems = stems[:args.limit]

    cells_acc, raw_acc, shape_ok, found, total, shifted = [], [], 0, 0, 0, 0
    hdr_gold, hdr_multi, hdr_multi_ok = [], 0, 0
    for stem in stems:
        pdf = PDFS / f"{stem}.pdf"
        if not pdf.exists():
            continue
        tables = json.loads((GOLD / f"{stem}.json").read_text())
        try:
            doc = fitz.open(pdf)
            res = pipeline.parse_page(doc[0])
            preds = [b.table_grid for b in res.blocks if b.table_grid]
            doc.close()
        except Exception:
            preds = []
        for t in tables:
            g, hdepth = gold_grid(t.get("cells") or [])
            if not g:
                continue
            total += 1
            hdr_gold.append(hdepth)
            if hdepth > 1:
                hdr_multi += 1
            if not preds:
                cells_acc.append(0.0); raw_acc.append(0.0)
                continue
            found += 1
            scored = [cell_score(g, p) for p in preds]
            best = max((v or 0.0) for v, _ in scored)
            raw = max((cell_score(g, p, align=False)[0] or 0.0) for p in preds)
            cells_acc.append(best)
            raw_acc.append(raw)
            if best > 0 and max((d for v, d in scored if v == best), default=0) != 0:
                shifted += 1
            if any(len(p) == len(g) and len(p[0]) == len(g[0]) for p in preds):
                shape_ok += 1

    n = max(len(cells_acc), 1)
    print(f"FinTabNet.c — {total} tables over {len(stems)} pages")
    print(f"  table found on the page   : {100*found/max(total,1):5.1f}%")
    print(f"  exact grid shape          : {100*shape_ok/max(total,1):5.1f}%")
    print(f"  cells recovered (aligned) : {100*sum(cells_acc)/n:5.1f}%")
    print(f"  cells at absolute position: {100*sum(raw_acc)/max(len(raw_acc),1):5.1f}%")
    print(f"  tables needing a row shift: {100*shifted/max(total,1):5.1f}%  (header rows dropped)")
    print(f"  median per-table          : {100*statistics.median(cells_acc):5.1f}%")
    if hdr_gold:
        print(f"  ground-truth header depth : mean {statistics.mean(hdr_gold):.2f}, "
              f"multi-level {100*hdr_multi/len(hdr_gold):.1f}% "
              f"({hdr_multi} of {len(hdr_gold)}) -- the parser always assumes 1")


if __name__ == "__main__":
    main()
