"""Run RAGBench over a corpus and aggregate, stratified.

Usage:
    python -m eval.ragbench.runner corpus_large --sample 200 --json run.json
    python -m eval.ragbench.runner corpus_large --stratum web_print__arabic
    python -m eval.ragbench.runner corpus_large --doc path/to/one.pdf --verbose
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

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import fitz  # noqa: E402
from rtldoc import tags, pipeline  # noqa: E402

from eval.ragbench import chunking, metrics, retrieval, rules  # noqa: E402


def score_document(path: str, max_pages: int = 0) -> dict:
    fitz.TOOLS.mupdf_display_errors(False)
    doc_out = {"pdf": path, "pages": 0, "ocr_pages": 0, "crashes": [],
               "dims": {}, "retrieval": {}, "failures": []}
    try:
        doc = fitz.open(path)
    except Exception as exc:
        doc_out["crashes"].append((0, repr(exc)))
        return doc_out

    n = doc.page_count if not max_pages else min(doc.page_count, max_pages)
    doc_out["pages"] = n
    ruled_ok = rules.has_reliable_ruled_tables(doc)
    doc_out["ruled_table_truth"] = ruled_ok

    all_blocks, all_truth_records, all_pred_records = [], [], []
    page_scores = collections.defaultdict(list)
    # /H1../H6 per page, or None when the file carries no structure tree -- see
    # metrics.structure_score, which then reports headings as unmeasured.
    _declared_by_page = None
    try:
        _d = tags.tagged_headings(doc)
        if _d:
            _declared_by_page = collections.defaultdict(list)
            for _pno, _lvl, _mc in _d:
                _declared_by_page[_pno].append(_mc)
    except Exception:
        _declared_by_page = None
    page_rect = (0, 0, 612, 792)

    for i in range(n):
        page = doc[i]
        page_rect = tuple(page.rect)
        try:
            result = pipeline.parse_page(page)
        except Exception as exc:
            doc_out["crashes"].append((i + 1, repr(exc)))
            continue
        if not result.born_digital:
            doc_out["ocr_pages"] += 1
            continue

        truth = rules.build_page_truth(page, ruled_tables=ruled_ok)
        md = pipeline.to_markdown(result)

        # blocks carry their page so chunks can cite it
        for b in result.blocks:
            b.page = i + 1
        all_blocks.extend(result.blocks)

        # Table fidelity is scored PER PAGE, not pooled over the document.
        # Pooling lets the optimal matching pair a record from one table with a
        # record from another, which both hides real damage and invents it.
        page_pred_recs = metrics.records_from_markdown(md, i + 1)
        all_truth_records.extend(truth.records)
        all_pred_records.extend(page_pred_recs)
        # Each ground-truth table is scored against the BEST-matching table
        # rtldoc emitted on the page. Pooling every record on the page instead
        # measures the page, not the table, and penalises a correct table for
        # the existence of a different one beside it -- confirmed against the
        # hand-verified gold set, where pooling read 51.8% and per-table
        # matching read 63.8% on identical output.
        pred_tables = metrics.tables_from_markdown(md) or [[]]
        # Page-level: was the table FOUND at all, once. Robust to the
        # reference's own header and ordering errors -- see table_detection.
        if truth.record_groups or any(pred_tables):
            page_scores["table_detection"].append(
                metrics.table_detection(truth.record_groups, pred_tables))
        for group in truth.record_groups:
            best = max((metrics.table_record_fidelity(group, cand)
                        for cand in pred_tables), key=lambda sc: sc.value)
            page_scores["table_record_fidelity"].append(best)

        cfs = metrics.content_faithfulness(truth, md)
        page_scores["content_faithfulness"].append(cfs)
        page_scores["reading_order"].append(metrics.reading_order_score(truth, md))

        raw = page.get_text("dict")
        spans = [(s["size"], s["font"], s["text"], s["bbox"])
                 for blk in raw.get("blocks", []) if blk.get("type") == 0
                 for ln in blk.get("lines", []) for s in ln.get("spans", [])]
        _decl = None
        if _declared_by_page is not None:
            texts = tags.mcid_text(page)
            _decl = [" ".join(texts.get(m, "") for m in mc).strip()
                     for mc in _declared_by_page.get(i, [])]
        # A page with no text layer cannot be scored for headings. OCR returns
        # words and boxes and NO font, size or colour, so block.style is None
        # and every heading rule has nothing to test -- measured on the scanned
        # Saudi labour regulation, 46 OCR pages produced 0 headings while its 4
        # born-digital pages produced them normally. Scoring those pages counts
        # a structural impossibility as a parser failure. Scanned documents are
        # out of scope (see CAPABILITIES.md) and are marked unmeasured here.
        if result.born_digital:
            page_scores["structure"].append(
                metrics.structure_score(result.blocks, spans, declared=_decl))
        page_scores["page_integrity"].append(
            metrics.page_integrity(result, page.get_text()))
        page_scores["citability"].append(
            metrics.page_citability(result, tuple(page.rect)))
        doc_out["failures"].extend(f"p{i+1} {f}" for f in cfs.failures[:3])

    doc.close()

    chunks = chunking.chunk_blocks(all_blocks)
    doc_out["chunks"] = len(chunks)

    dims = {}
    for name in ("content_faithfulness", "structure", "table_detection",
                 "table_record_fidelity", "page_integrity", "citability",
                 "reading_order"):
        got = page_scores.get(name, [])
        live = [s for s in got if s.n > 0]
        dims[name] = metrics.Score(
            sum(s.value for s in live) / len(live) if live else 0.0,
            sum(s.n for s in got),
            _mean_parts(live),
            [f for s in got for f in s.failures][:20])


    doc_out["dims"] = {k: v.as_row() for k, v in dims.items()}
    doc_out["overall"] = metrics.overall(dims)

    needles = retrieval.needles_from_records(all_truth_records)
    doc_out["retrieval"] = retrieval.probe(needles, chunks)
    return doc_out


def _mean_parts(scores: list) -> dict:
    acc = collections.defaultdict(list)
    for s in scores:
        for k, v in s.parts.items():
            if isinstance(v, (int, float)):
                acc[k].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def _worker(args):
    return score_document(*args)


def load_manifest(root: Path) -> dict[str, dict]:
    mf = root / "manifest.json"
    if not mf.exists():
        return {}
    data = json.loads(mf.read_text())
    return {str(root / d["path"]): d for d in data["documents"] if "path" in d}


def run(root: Path, sample: int, stratum: str | None, workers: int,
        max_pages: int, only: str | None) -> dict:
    meta = load_manifest(root)
    pdfs = [only] if only else sorted(str(p) for p in root.rglob("*.pdf"))
    if stratum:
        # substring match, so "arabic" selects every Arabic stratum at once
        pdfs = [p for p in pdfs if stratum in meta.get(p, {}).get("stratum", "")]
    if sample and sample < len(pdfs):
        by = collections.defaultdict(list)
        for p in pdfs:
            by[meta.get(p, {}).get("stratum", "unknown")].append(p)
        rng, picked = random.Random(0), []
        for s, group in sorted(by.items()):
            k = max(1, round(sample * len(group) / len(pdfs)))
            picked.extend(rng.sample(group, min(k, len(group))))
        pdfs = sorted(picked)

    print(f"RAGBench: {len(pdfs)} documents, {workers} workers", file=sys.stderr)
    started = time.time()
    docs = []
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_worker, (p, max_pages)): p for p in pdfs}
        for i, fut in enumerate(as_completed(futs), 1):
            docs.append(fut.result())
            if i % 25 == 0 or i == len(pdfs):
                print(f"  {i}/{len(pdfs)}  "
                      f"{i/max(1e-6, time.time()-started):.1f} doc/s", file=sys.stderr)
    for d in docs:
        d["stratum"] = meta.get(d["pdf"], {}).get("stratum", "unknown")
    return {"root": str(root), "documents": docs,
            "elapsed_s": round(time.time() - started, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--stratum", default=None)
    ap.add_argument("--doc", default=None, help="score a single PDF")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None, help="per-document rows, for diffing runs")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    res = run(Path(args.root), args.sample, args.stratum, args.workers,
              args.max_pages, args.doc)
    if args.json:
        Path(args.json).write_text(json.dumps(res))
        print(f"wrote {args.json}", file=sys.stderr)
    from eval.ragbench.report import csv_report, text_report
    if args.csv:
        Path(args.csv).write_text(csv_report(res))
        print(f"wrote {args.csv}", file=sys.stderr)
    print(text_report(res, verbose=args.verbose))
