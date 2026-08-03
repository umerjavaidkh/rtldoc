"""Cell-level table-detection regression harness.

Separate from eval/gold/gold.json + run_matrix.py (a human-labeled,
TEDS-scored quality benchmark). This harness answers a different question:
"did a layout.py change just break a page that used to work?" -- fast,
no human labeling, no scoring threshold. Each fixture freezes the exact
table_grid(s) rtldoc produced for one page, at a point where that output
was manually verified correct (against the rendered page or the PDF's own
text) by direct inspection. A future run that produces a different grid is
a regression until proven otherwise -- diff it against the actual page
before assuming the fixture is stale.

Usage:
    python eval/regression.py                       # check all fixtures
    python eval/regression.py --freeze PDF PAGE      # (re)freeze one fixture
                                                      # after manually
                                                      # verifying PAGE is
                                                      # now correct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtldoc.pipeline import parse_page

GOLDEN_DIR = Path(__file__).parent / "golden"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _tables_for_page(pdf_path: str, page_no: int) -> list[list[list[str]]]:
    doc = fitz.open(str(PROJECT_ROOT / pdf_path))
    page = doc[page_no - 1]
    result = parse_page(page)
    return [b.table_grid for b in result.blocks if b.role == "table" and b.table_grid is not None]


def freeze(pdf_path: str, page_no: int, name: str | None = None, note: str = "") -> None:
    tables = _tables_for_page(pdf_path, page_no)
    fixture = {"pdf": pdf_path, "page": page_no, "note": note, "tables": tables}
    fname = name or f"{Path(pdf_path).stem}_p{page_no}.json"
    GOLDEN_DIR.mkdir(exist_ok=True)
    (GOLDEN_DIR / fname).write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"froze {fname}: {len(tables)} table(s)")


def check_all() -> bool:
    fixtures = sorted(GOLDEN_DIR.glob("*.json"))
    if not fixtures:
        print("no fixtures in eval/golden/ -- nothing to check")
        return True
    n_pass = n_fail = 0
    for f in fixtures:
        fixture = json.loads(f.read_text(encoding="utf-8"))
        expected = fixture["tables"]
        try:
            actual = _tables_for_page(fixture["pdf"], fixture["page"])
        except Exception as e:
            print(f"FAIL {f.name}: crashed -- {e}")
            n_fail += 1
            continue
        if actual == expected:
            n_pass += 1
            print(f"PASS {f.name}  ({len(expected)} table(s))")
        else:
            n_fail += 1
            print(f"FAIL {f.name}  expected {len(expected)} table(s), got {len(actual)}")
            for i in range(max(len(expected), len(actual))):
                e = expected[i] if i < len(expected) else None
                a = actual[i] if i < len(actual) else None
                if e != a:
                    print(f"  table[{i}] expected: {e!r}")
                    print(f"  table[{i}] actual:   {a!r}")
    print(f"\n{n_pass} passed, {n_fail} failed, {len(fixtures)} total")
    return n_fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", nargs=2, metavar=("PDF", "PAGE"))
    ap.add_argument("--name")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.freeze:
        pdf, page = args.freeze
        freeze(pdf, int(page), args.name, args.note)
    else:
        ok = check_all()
        sys.exit(0 if ok else 1)
