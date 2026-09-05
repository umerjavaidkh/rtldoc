"""Run the detectors over a corpus and report per stratum, worst case first.

Why not one number
------------------
This corpus is 85% arXiv LaTeX papers. A mean over it is an arXiv mean: it
will read ~0.98 no matter how badly Word documents, Arabic textbooks or
scanned reports are handled, because those are 3% of the mass. Every headline
"accuracy" number computed over a corpus like this is a monoculture average
wearing a general-purpose label.

So this reports:
  * per stratum (producer family x script), never pooled;
  * the DOCUMENT-level defect rate, not the page mean, because one broken
    document with 200 bad pages and 200 clean documents are not the same
    thing even when the page average matches;
  * the worst documents by name, so a number always leads somewhere you can
    open.

Usage:
    python eval/scorecard.py corpus_large/                 # whole corpus
    python eval/scorecard.py corpus_large/ --stratum word__latin
    python eval/scorecard.py corpus_large/ --sample 200    # quick pass
    python eval/scorecard.py corpus_large/ --json out.json --workers 8
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402

import detectors as D  # noqa: E402  (same directory)


def probe_page(page, result) -> D.PageProbe:
    raw = page.get_text("dict")
    spans = [(s["size"], s["font"], s["text"], s["bbox"])
             for b in raw.get("blocks", []) if b.get("type") == 0
             for l in b.get("lines", []) for s in l.get("spans", [])]
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    return D.PageProbe(
        number=page.number + 1,
        blocks=result.blocks,
        born_digital=result.born_digital,
        source_text=page.get_text(),
        words=page.get_text("words"),
        spans=spans,
        lattices=D.find_lattices(drawings, page.rect),
        page_rect=tuple(page.rect),
    )


def scan_document(path: str, max_pages: int = 0) -> dict:
    """All findings for one document, plus the counts needed to normalise."""
    out = {"pdf": path, "pages": 0, "ocr_pages": 0, "crashes": [], "findings": []}
    try:
        doc = fitz.open(path)
    except Exception as exc:
        out["crashes"].append((0, repr(exc)))
        return out
    n = doc.page_count if not max_pages else min(doc.page_count, max_pages)
    out["pages"] = n
    for i in range(n):
        page = doc[i]
        try:
            result = pipeline.parse_page(page)
        except Exception as exc:
            out["crashes"].append((i + 1, repr(exc)))
            continue
        if not result.born_digital:
            out["ocr_pages"] += 1
            continue
        try:
            probe = probe_page(page, result)
            for detector in D.ALL_DETECTORS:
                for finding in detector(probe):
                    out["findings"].append(finding.as_row())
        except Exception as exc:
            out["crashes"].append((i + 1, f"detector: {exc!r}"))
    doc.close()
    return out


def _worker(args):
    path, max_pages = args
    # PyMuPDF chatters on damaged xrefs; the census already recorded them.
    fitz.TOOLS.mupdf_display_errors(False)
    return scan_document(path, max_pages)


def load_manifest(root: Path) -> dict[str, dict]:
    mf = root / "manifest.json"
    if not mf.exists():
        return {}
    data = json.loads(mf.read_text())
    return {str(root / d["path"]): d for d in data["documents"] if "path" in d}


def run(root: Path, sample: int, stratum: str | None, workers: int,
        max_pages: int) -> dict:
    meta = load_manifest(root)
    pdfs = sorted(str(p) for p in root.rglob("*.pdf"))
    if stratum:
        pdfs = [p for p in pdfs if stratum in meta.get(p, {}).get("stratum", "")]
    if sample and sample < len(pdfs):
        # stratified sample: take proportionally from each stratum so a quick
        # pass still sees the small strata, which are where the bugs are
        by = collections.defaultdict(list)
        for p in pdfs:
            by[meta.get(p, {}).get("stratum", "unknown")].append(p)
        rng = random.Random(0)
        picked = []
        for s, group in sorted(by.items()):
            k = max(1, round(sample * len(group) / len(pdfs)))
            picked.extend(rng.sample(group, min(k, len(group))))
        pdfs = sorted(picked)

    print(f"scanning {len(pdfs)} documents with {workers} workers...", file=sys.stderr)
    started = time.time()
    docs = []
    with ProcessPoolExecutor(workers) as ex:
        futures = {ex.submit(_worker, (p, max_pages)): p for p in pdfs}
        for i, fut in enumerate(as_completed(futures), 1):
            docs.append(fut.result())
            if i % 25 == 0 or i == len(pdfs):
                rate = i / max(1e-6, time.time() - started)
                print(f"  {i}/{len(pdfs)}  {rate:.1f} doc/s", file=sys.stderr)

    for d in docs:
        d["stratum"] = meta.get(d["pdf"], {}).get("stratum", "unknown")
    return {"root": str(root), "documents": docs,
            "elapsed_s": round(time.time() - started, 1)}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(results: dict, top: int = 12) -> str:
    docs = results["documents"]
    lines: list[str] = []
    total_pages = sum(d["pages"] for d in docs)
    total_ocr = sum(d["ocr_pages"] for d in docs)
    crashes = sum(len(d["crashes"]) for d in docs)

    lines.append(f"# scorecard: {len(docs)} documents, {total_pages} pages "
                 f"({results['elapsed_s']}s)")
    lines.append(f"  pages routed to OCR : {total_ocr}")
    lines.append(f"  crashes             : {crashes}"
                 + ("   <-- HARD failure" if crashes else ""))
    lines.append("")

    # ---- per stratum, the number that matters: clean-page rate ----
    by_stratum = collections.defaultdict(list)
    for d in docs:
        by_stratum[d["stratum"]].append(d)

    lines.append("## clean-page rate by stratum")
    lines.append("   a page is clean when no detector fired on it. Read the")
    lines.append("   SMALL strata: the big one drowns them in any average.")
    lines.append("")
    lines.append(f"{'stratum':26}{'docs':>6}{'pages':>8}{'clean':>8}{'HARD':>7}{'worst code':>22}")
    rows = []
    for stratum, group in by_stratum.items():
        pages = sum(d["pages"] for d in group)
        bad_pages = set()
        hard = 0
        codes = collections.Counter()
        for d in group:
            for f in d["findings"]:
                bad_pages.add((d["pdf"], f["page"]))
                codes[f["code"]] += 1
                if f["severity"] == "HARD":
                    hard += 1
        clean = 1 - (len(bad_pages) / pages) if pages else 1.0
        worst = codes.most_common(1)[0][0] if codes else "-"
        rows.append((clean, stratum, len(group), pages, hard, worst))
    for clean, stratum, ndocs, pages, hard, worst in sorted(rows):
        flag = "  <--" if clean < 0.80 else ""
        lines.append(f"{stratum:26}{ndocs:>6}{pages:>8}{clean:>7.1%}{hard:>7}"
                     f"{worst:>22}{flag}")

    # ---- what is actually breaking ----
    codes = collections.Counter()
    code_docs = collections.defaultdict(set)
    for d in docs:
        for f in d["findings"]:
            codes[f["code"]] += 1
            code_docs[f["code"]].add(d["pdf"])
    lines.append("")
    lines.append("## defects by kind")
    lines.append(f"{'code':22}{'hits':>8}{'docs':>7}  what a reader sees")
    for code, count in codes.most_common():
        lines.append(f"{code:22}{count:>8}{len(code_docs[code]):>7}  "
                     f"{D.CODES.get(code, '')}")

    # ---- worst documents, so every number leads to a file you can open ----
    lines.append("")
    lines.append(f"## worst {top} documents by defective-page share")
    scored = []
    for d in docs:
        if d["pages"] < 3:
            continue
        bad = len({f["page"] for f in d["findings"]})
        scored.append((bad / d["pages"], bad, d["pages"], d["stratum"], d["pdf"]))
    for frac, bad, pages, stratum, pdf in sorted(scored, reverse=True)[:top]:
        lines.append(f"  {frac:6.1%}  {bad:>4}/{pages:<4} {stratum:22} {Path(pdf).name[:56]}")

    for d in docs:
        for page, exc in d["crashes"][:2]:
            lines.append(f"\n  CRASH {Path(d['pdf']).name} p{page}: {exc}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--stratum", default=None)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-pages", type=int, default=0, help="cap pages per document")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    res = run(Path(args.root), args.sample, args.stratum, args.workers, args.max_pages)
    if args.json:
        Path(args.json).write_text(json.dumps(res))
        print(f"wrote {args.json}", file=sys.stderr)
    print(report(res, args.top))
