"""Score DocLayout-YOLO's `title` class against the hand-labelled heading gold set.

Why this comparison is shaped this way
--------------------------------------
The gold set stores a page and the heading's TEXT, not a box, because it was
adjudicated from crops by eye. DocLayout-YOLO emits boxes with no text. So the
two are joined through rtldoc: the parser already segments the page into blocks
with both text and geometry, and the block whose text matches the gold case
supplies the box that the YOLO prediction is tested against.

That join is the honest weak point and is reported: a gold case whose text no
longer matches any block is counted as unjoinable rather than silently scored,
because a miss there measures the join, not the detector.

A case is predicted `heading` when a `title` box covers most of the matched
block. Containment, not IoU -- YOLO draws one box around a multi-line heading
and rtldoc may split it, so an IoU threshold would punish agreement on where
the heading is over a disagreement about how to slice it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402
from eval.ragbench.rules import normalize  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:                                   # pragma: no cover
    fuzz = None

MATCH = 82          # same threshold heading_eval.py uses to join deshaped Arabic
COVER = 0.60        # fraction of the block's area a title box must cover
DPI = 144           # render scale; imgsz=1024 is applied by the model itself


def _score(block: str, gold: str) -> float:
    """How well a block's text answers to the gold text. 0 = no join."""
    x, y = normalize(block), normalize(gold)
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    # containment only counts when the gold text is most of the block, so a
    # short candidate cannot attach itself to a long paragraph containing it
    if (x in y or y in x) and min(len(x), len(y)) / max(len(x), len(y)) >= 0.6:
        return 0.9
    if fuzz and len(y) >= 4:
        r = fuzz.token_set_ratio(x, y) / 100.0
        if r >= MATCH / 100.0 and min(len(x), len(y)) / max(len(x), len(y)) >= 0.5:
            return r
    return 0.0


def _cover(inner, outer) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    w = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    h = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    area = (ix1 - ix0) * (iy1 - iy0)
    return (w * h / area) if area > 0 else 0.0


def main(gold_path: str, conf: float = 0.25) -> None:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(
        "juliozhao/DocLayout-YOLO-DocStructBench",
        "doclayout_yolo_docstructbench_imgsz1024.pt",
    )
    model = YOLOv10(weights)

    cases = json.load(open(gold_path))["cases"]
    root = Path(gold_path).resolve().parent.parent.parent
    by_page: dict[tuple[str, int], list[dict]] = {}
    for c in cases:
        by_page.setdefault((c["pdf"], c["page"]), []).append(c)

    fitz.TOOLS.mupdf_display_errors(False)
    rows = []
    for (pdf, page_no), group in sorted(by_page.items()):
        path = root / pdf
        if not path.exists():
            for c in group:
                rows.append({**c, "join": "missing-pdf"})
            continue
        doc = fitz.open(path)
        page = doc[page_no - 1]        # gold stores 1-based page numbers

        # --- YOLO side: render once, keep the title boxes in PDF points ---
        pm = page.get_pixmap(dpi=DPI)
        img = Path("/private/tmp/claude-501/_yolo_page.png")
        pm.save(img)
        det = model.predict(str(img), imgsz=1024, conf=conf, device="cpu", verbose=False)[0]
        s = 72.0 / DPI
        titles, tables = [], []
        for b, cls in zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist()):
            name = model.names[int(cls)]
            box = (b[0] * s, b[1] * s, b[2] * s, b[3] * s)
            if name == "title":
                titles.append(box)
            elif name == "table":
                tables.append(box)

        # --- rtldoc side: blocks give the gold text a geometry ---
        try:
            blocks = pipeline.parse_page(page).blocks
        except Exception:
            blocks = []

        for c in group:
            hit, best_join = None, 0.0
            for blk in blocks:
                if not blk.text:
                    continue
                sc = _score(blk.text, c["text"])
                if sc > best_join:
                    hit, best_join = blk, sc
            if hit is None:
                rows.append({**c, "join": "no-block"})
                continue
            best = max((_cover(hit.bbox, t) for t in titles), default=0.0)
            rows.append({
                **c,
                "join": "ok",
                "yolo_heading": best >= COVER,
                "yolo_cover": round(best, 2),
                "rtldoc_heading": bool((hit.role or "").startswith("heading")),
                "style": str(hit.style), "size": round(hit.bbox[3]-hit.bbox[1],1),
                "n_titles": len(titles),
                "n_tables": len(tables),
            })
        doc.close()

    ok = [r for r in rows if r["join"] == "ok"]
    print(f"joined {len(ok)}/{len(rows)} cases "
          f"({len(rows) - len(ok)} unjoinable, excluded)")

    def prf(key: str) -> tuple[float, float, float, int, int, int]:
        tp = sum(1 for r in ok if r[key] and r["verdict"] == "heading")
        fp = sum(1 for r in ok if r[key] and r["verdict"] != "heading")
        fn = sum(1 for r in ok if not r[key] and r["verdict"] == "heading")
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r_ / (p + r_) if p + r_ else 0.0
        return p, r_, f, tp, fp, fn

    print(f"\n{'':14} {'P':>6} {'R':>6} {'F1':>6}   tp/fp/fn")
    for label, key in (("rtldoc", "rtldoc_heading"), ("DocLayout-YOLO", "yolo_heading")):
        p, r_, f, tp, fp, fn = prf(key)
        print(f"{label:14} {p:6.3f} {r_:6.3f} {f:6.3f}   {tp}/{fp}/{fn}")

    agree = sum(1 for r in ok if r["yolo_heading"] == r["rtldoc_heading"])
    y_only = [r for r in ok if r["yolo_heading"] and not r["rtldoc_heading"]]
    r_only = [r for r in ok if r["rtldoc_heading"] and not r["yolo_heading"]]
    print(f"\nagree {agree}/{len(ok)}   yolo-only {len(y_only)}   rtldoc-only {len(r_only)}")
    for label, sub in (("YOLO only", y_only), ("rtldoc only", r_only)):
        for r in sub[:8]:
            mark = "RIGHT" if (r["verdict"] == "heading") == r["yolo_heading"] else "wrong"
            if label == "rtldoc only":
                mark = "RIGHT" if r["verdict"] == "heading" else "wrong"
            print(f"  {label:12} {mark:5} {r['verdict']:8} {r['text'][:52]!r}")

    out = Path(__file__).parent / "results" / "yolo_heading.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/heading_gold_set/gold.json")
