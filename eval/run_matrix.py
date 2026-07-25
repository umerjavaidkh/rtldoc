"""Run every registered adapter against a gold-labeled page set and print a
parser x metric comparison matrix, faceted by doc_type.

Usage:
    python eval/run_matrix.py eval/gold/gold.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.adapters import ADAPTERS
from eval import metrics


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # pdf_parser/


def run(gold_path: str) -> dict:
    gold = json.loads(Path(gold_path).read_text())["pages"]
    base_dir = PROJECT_ROOT

    rows: dict[str, list[dict]] = {name: [] for name in ADAPTERS}
    skipped = 0
    for entry in gold:
        if not entry.get("blocks"):
            skipped += 1  # not labeled yet -- don't let an empty gold fake a score
            continue
        pdf_path = str(base_dir / entry["pdf"])
        for name, fn in ADAPTERS.items():
            try:
                pred = fn(pdf_path, entry["page"])
            except Exception as e:
                pred = {"blocks": [], "text": "", "_error": str(e)}
            text_score = metrics.text_edit_score("\n".join(entry["blocks"]), pred.get("text", ""))
            order_score = metrics.reading_order_score(entry["blocks"], pred.get("blocks", []))
            table_score = None
            if entry.get("table"):
                pred_tables = pred.get("tables") or []
                best = 0.0
                for t in pred_tables:
                    best = max(best, metrics.table_cell_score(entry["table"], t))
                table_score = best if pred_tables else 0.0
            rows[name].append({
                "id": entry["id"], "doc_type": entry.get("doc_type", "unknown"),
                "text": text_score, "order": order_score, "table": table_score,
                "error": pred.get("_error"),
            })
    return rows, skipped


def report(rows: dict[str, list[dict]]) -> str:
    lines = ["| Parser | Text edit-score | Reading-order score | Table cell score | Errors |",
            "|---|---|---|---|---|"]
    for name, results in rows.items():
        text_scores = [r["text"] for r in results]
        order_scores = [r["order"] for r in results]
        table_scores = [r["table"] for r in results if r["table"] is not None]
        errs = sum(1 for r in results if r["error"])
        t = f"{statistics.mean(text_scores):.3f}" if text_scores else "-"
        o = f"{statistics.mean(order_scores):.3f}" if order_scores else "-"
        tb = f"{statistics.mean(table_scores):.3f}" if table_scores else "n/a"
        lines.append(f"| {name} | {t} | {o} | {tb} | {errs} |")

    lines.append("")
    lines.append("### By document type")
    doc_types = sorted({r["doc_type"] for results in rows.values() for r in results})
    for dt in doc_types:
        lines.append(f"\n**{dt}**\n")
        lines.append("| Parser | Text edit-score | Reading-order score |")
        lines.append("|---|---|---|")
        for name, results in rows.items():
            subset = [r for r in results if r["doc_type"] == dt]
            if not subset:
                continue
            t = statistics.mean(r["text"] for r in subset)
            o = statistics.mean(r["order"] for r in subset)
            lines.append(f"| {name} | {t:.3f} | {o:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    gold_path = sys.argv[1] if len(sys.argv) > 1 else "eval/gold/gold.json"
    rows, skipped = run(gold_path)
    if skipped:
        print(f"(skipped {skipped} page(s) with no gold `blocks` yet)\n")
    print(report(rows))
