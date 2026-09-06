"""Label headings with DocLayout-YOLO, which is precise where it is not recallful.

Why this and not hand labelling
-------------------------------
Heading detection is blocked on labels: ten documents in the corpus carry a
structure tree, one of them supplies 137 of 149 usable positives, and a model
fitted on that learned that document rather than headings.

DocLayout-YOLO is useless as a DETECTOR here -- scored against 72 hand
adjudicated cases it ties rtldoc exactly, 57/72 each, McNemar p = 1.0. But the
same measurement showed its Arabic precision is 22/22, Wilson floor 0.851,
against poor recall. Precision without recall is a bad detector and a good
LABELLER: what it fires on is almost certainly a heading, and the ones it
misses simply do not become labels.

So positives come from YOLO's `title` boxes, negatives from blocks it declines
that rtldoc did not call headings either. The result needs spot-checking, not
adjudicating -- and the labels it cannot supply are exactly the hard cases a
human should spend their time on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402
from eval.heading_gold import TOUGH  # noqa: E402

DPI = 144
CONF = 0.25
COVER = 0.60          # fraction of a block a title box must cover to claim it


def _cover(inner, outer) -> float:
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (w * h / a) if a > 0 else 0.0


def main(out_path: str, max_pages: int = 25) -> None:
    from doclayout_yolo import YOLOv10
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download("juliozhao/DocLayout-YOLO-DocStructBench",
                              "doclayout_yolo_docstructbench_imgsz1024.pt")
    model = YOLOv10(weights)
    fitz.TOOLS.mupdf_display_errors(False)
    tmp = Path("/private/tmp/claude-501/_al.png")

    rows = []
    for rel, why in TOUGH:
        p = Path(rel)
        if not p.exists():
            continue
        doc = fitz.open(str(p))
        n = min(doc.page_count, max_pages)
        for i in range(n):
            page = doc[i]
            try:
                res = pipeline.parse_page(page)
            except Exception:
                continue
            if not res.born_digital or not res.blocks:
                continue
            page.get_pixmap(dpi=DPI).save(str(tmp))
            try:
                det = model.predict(str(tmp), imgsz=1024, conf=CONF,
                                    device="cpu", verbose=False)[0]
            except Exception:
                continue
            s = 72.0 / DPI
            titles = [tuple(v * s for v in b) for b, c in
                      zip(det.boxes.xyxy.tolist(), det.boxes.cls.tolist())
                      if model.names[int(c)] == "title"]
            for b in res.blocks:
                txt = (b.text or "").strip()
                if not txt or len(txt) > 160 or len(txt) < 3:
                    continue
                if b.role in ("table", "figure", "page_furniture"):
                    continue
                claimed = max((_cover(b.bbox, t) for t in titles), default=0.0)
                ours = (b.role or "").startswith("heading")
                if claimed >= COVER:
                    label = "heading"        # YOLO fired: precision 22/22
                elif not ours:
                    label = "not"            # neither called it one
                else:
                    continue                 # we say heading, YOLO does not: HARD
                rows.append({"pdf": rel, "page": i + 1, "text": txt,
                             "label": label, "yolo_cover": round(claimed, 2),
                             "rtldoc_heading": ours, "why_doc": why})
        doc.close()

    Path(out_path).write_text(json.dumps({"cases": rows}, ensure_ascii=False,
                                         indent=1))
    pos = sum(1 for r in rows if r["label"] == "heading")
    print(f"{len(rows)} auto-labels  positives {pos}  negatives {len(rows)-pos}")
    print(f"documents {len({r['pdf'] for r in rows})}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/results/heading_autolabels.json")
