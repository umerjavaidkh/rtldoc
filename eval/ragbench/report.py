"""Reports: a scorecard you can read, and a CSV you can diff between runs."""

from __future__ import annotations

import collections
import csv
import io
from pathlib import Path

from eval.ragbench.metrics import DIMENSIONS

_SHORT = {"structure": "HEADING", "table_detection": "TBL-FOUND", "table_record_fidelity": "TABLE",
          "content_faithfulness": "TEXT", "page_integrity": "PAGE",
          "reading_order": "COLUMN",
          "citability": "CITATION"}


def _agg(docs: list[dict]) -> dict:
    """Mean per dimension over the documents where the dimension applied."""
    out = {}
    for dim in DIMENSIONS:
        vals = [d["dims"][dim]["score"] for d in docs
                if d.get("dims", {}).get(dim, {}).get("n", 0) > 0]
        out[dim] = sum(vals) / len(vals) if vals else None
    live = [v for v in out.values() if v is not None]
    out["overall"] = sum(live) / len(live) if live else 0.0
    return out


def text_report(results: dict, verbose: bool = False) -> str:
    docs = [d for d in results["documents"] if d.get("dims")]
    if not docs:
        return "no documents scored"
    L: list[str] = []
    pages = sum(d["pages"] for d in docs)
    chunks = sum(d.get("chunks", 0) for d in docs)
    crashes = sum(len(d["crashes"]) for d in docs)

    L.append(f"# RAGBench: {len(docs)} documents, {pages} pages, {chunks} chunks "
             f"({results['elapsed_s']}s)")
    L.append(f"  crashes: {crashes}" + ("   <-- HARD failure" if crashes else ""))
    L.append("")
    L.append("Five dimensions, unweighted mean, deterministic rules, no LLM judge.")
    L.append("Ground truth is derived from each PDF's own content stream, so every")
    L.append("number below is reproducible from the files themselves.")
    L.append("")

    overall = _agg(docs)
    L.append("## overall")
    for dim in DIMENSIONS:
        v = overall[dim]
        L.append(f"  {_SHORT[dim]:10} {'n/a' if v is None else f'{v:6.1%}'}")
    L.append(f"  {'OVERALL':10} {overall['overall']:6.1%}")

    # ---- per stratum ----
    by = collections.defaultdict(list)
    for d in docs:
        by[d["stratum"]].append(d)
    L.append("")
    L.append("## by stratum  (read the small strata -- the big one is 85% of the corpus)")
    L.append(f"{'stratum':24}{'docs':>5}" +
             "".join(f"{_SHORT[d]:>10}" for d in DIMENSIONS) + f"{'OVERALL':>9}")
    rows = []
    for stratum, group in by.items():
        a = _agg(group)
        rows.append((a["overall"], stratum, len(group), a))
    for ov, stratum, n, a in sorted(rows):
        cells = "".join(
            f"{'n/a':>10}" if a[d] is None else f"{a[d]:>9.1%} " for d in DIMENSIONS)
        L.append(f"{stratum:24}{n:>5}{cells}{ov:>8.1%}" +
                 ("  <--" if ov < 0.75 else ""))

    # ---- retrieval probe ----
    tot = collections.Counter()
    for d in docs:
        r = d.get("retrieval", {})
        for k in ("n", "supported", "orphaned", "missing"):
            tot[k] += r.get(k, 0)
    L.append("")
    L.append("## end-to-end retrieval probe (BM25 over the chunks)")
    if tot["n"]:
        L.append(f"  needles                 : {tot['n']}  "
                 f"(auto-generated from ruled-table ground truth)")
        L.append(f"  answer-support rate @5  : {tot['supported']/tot['n']:.1%}")
        L.append(f"  orphaned (value without its row key) : {tot['orphaned']}"
                 f"  ({tot['orphaned']/tot['n']:.1%})")
        L.append(f"  missing  (not retrieved at all)      : {tot['missing']}"
                 f"  ({tot['missing']/tot['n']:.1%})")
        L.append("  An orphan is the dangerous case: the number comes back with")
        L.append("  nothing tying it to the row that gives it meaning.")
    else:
        L.append("  no ruled tables in this sample -- no needles to probe")

    # ---- worst documents ----
    L.append("")
    L.append("## worst 12 documents by overall score")
    ranked = sorted(docs, key=lambda d: d.get("overall", 1.0))[:12]
    for d in ranked:
        L.append(f"  {d.get('overall', 0):6.1%}  {d['stratum']:22} "
                 f"{Path(d['pdf']).name[:52]}")

    if verbose:
        L.append("")
        L.append("## sample failures")
        for d in ranked[:4]:
            L.append(f"\n  {Path(d['pdf']).name}  ({d.get('overall', 0):.1%})")
            for dim in DIMENSIONS:
                for f in d["dims"].get(dim, {}).get("failures", [])[:3]:
                    L.append(f"    [{_SHORT[dim]}] {f}")
            for ex in d.get("retrieval", {}).get("examples", [])[:3]:
                L.append(f"    [retrieval] {ex}")
    return "\n".join(L)


def csv_report(results: dict) -> str:
    """Per-document rows, so two runs can be diffed to see what a change moved."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["pdf", "stratum", "pages", "n_chunks", *[_SHORT[d] for d in DIMENSIONS],
                "overall", "needles", "asr", "orphaned", "missing"])
    for d in results["documents"]:
        if not d.get("dims"):
            continue
        r = d.get("retrieval", {})
        # A dimension that ABSTAINED (no applicable rules) is written blank, not
        # 0.0. Writing zero would read as a total failure in every spreadsheet
        # and every diff -- the opposite of what abstention means.
        cells = ["" if d["dims"][x]["n"] == 0 else round(d["dims"][x]["score"], 4)
                 for x in DIMENSIONS]
        w.writerow([
            Path(d["pdf"]).name, d.get("stratum", ""), d["pages"], d.get("chunks", 0),
            *cells, round(d.get("overall", 0), 4), r.get("n", 0),
            "" if not r.get("n") else round(r.get("asr", 0), 4),
            r.get("orphaned", 0), r.get("missing", 0)])
    return buf.getvalue()
