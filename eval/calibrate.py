"""Measure how often each detector is right, instead of assuming it.

The argument
------------
Every detector in eval/detectors.py is a claim about a page. Claims need error
bars. Without them you are back where you started -- trusting a number because
it is a number.

Calibration is cheap and it only has to be done once per detector change:

  1. python eval/review.py ... --random 60 -o audit.html
     Sixty pages sampled WITHOUT regard to whether anything fired. Random
     sampling is what makes recall measurable; a worst-first list can only
     ever show you what the detectors already caught.
  2. Open audit.html, mark each page broken / fine / unsure, copy the JSON.
  3. python eval/calibrate.py --run run.json --labels labels.json

You get, per detector: precision (when it fires, how often is the page really
broken) and recall (of the pages you called broken, how many did anything
catch). A detector below ~0.6 precision is training you to ignore the report
and should be tightened or dropped; a suite below ~0.7 recall is missing a
defect class and needs a new detector, not a tuned threshold.

The labels file is just {"<sha8>_<page>": "broken"|"ok"|"unsure"}.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def load(run_path: Path, root: Path) -> dict[str, list[dict]]:
    run = json.loads(run_path.read_text())
    meta = {}
    mf = root / "manifest.json"
    if mf.exists():
        data = json.loads(mf.read_text())
        meta = {str(root / d["path"]): d for d in data["documents"] if "path" in d}
    by_page: dict[str, list[dict]] = collections.defaultdict(list)
    for doc in run["documents"]:
        sha = meta.get(doc["pdf"], {}).get("sha256", "x" * 8)[:8]
        for f in doc["findings"]:
            by_page[f"{sha}_{f['page']}"].append(f)
    return by_page


def report(by_page: dict[str, list[dict]], labels: dict[str, str]) -> str:
    judged = {k: v for k, v in labels.items() if v in ("broken", "ok")}
    if not judged:
        return "no usable labels (need at least one 'broken' or 'ok')"

    lines = [f"# calibration over {len(judged)} hand-judged pages",
             f"  broken: {sum(1 for v in judged.values() if v == 'broken')}   "
             f"ok: {sum(1 for v in judged.values() if v == 'ok')}   "
             f"unsure (excluded): {sum(1 for v in labels.values() if v == 'unsure')}",
             ""]

    # per-detector precision
    lines.append("## per detector")
    lines.append(f"{'code':22}{'fired':>7}{'right':>7}{'precision':>11}")
    stats = collections.defaultdict(lambda: [0, 0])
    for page, verdict in judged.items():
        for code in {f["code"] for f in by_page.get(page, [])}:
            stats[code][0] += 1
            if verdict == "broken":
                stats[code][1] += 1
    for code, (fired, right) in sorted(stats.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0])):
        p = right / fired if fired else 0.0
        flag = "   <-- noisy, tighten or drop" if fired >= 3 and p < 0.6 else ""
        lines.append(f"{code:22}{fired:>7}{right:>7}{p:>10.0%}{flag}")

    # suite-level precision and recall
    fired_pages = {p for p in judged if by_page.get(p)}
    broken_pages = {p for p, v in judged.items() if v == "broken"}
    tp = len(fired_pages & broken_pages)
    fp = len(fired_pages - broken_pages)
    fn = len(broken_pages - fired_pages)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    lines += ["", "## suite",
              f"  precision : {prec:.0%}   ({tp} of {tp+fp} flagged pages really are broken)",
              f"  recall    : {rec:.0%}   ({tp} of {tp+fn} broken pages were caught)"]
    if fn:
        lines.append("")
        lines.append(f"  {fn} broken page(s) NOTHING fired on -- these are the")
        lines.append("  defect classes the suite is blind to. Each one is a")
        lines.append("  missing detector, not a threshold to retune:")
        for p in sorted(broken_pages - fired_pages)[:15]:
            lines.append(f"    {p}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", default="../corpus_large")
    args = ap.parse_args()
    by_page = load(Path(args.run), Path(args.root))
    labels = json.loads(Path(args.labels).read_text())
    print(report(by_page, labels))
