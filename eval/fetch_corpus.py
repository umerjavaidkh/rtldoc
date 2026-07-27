"""Download a diverse set of complex born-digital PDFs for invariant testing.

Pulls from arXiv across many subject areas (each area tends to use different
LaTeX templates, column counts, and table/figure density), so the corpus
stresses the parser on layouts it has never seen -- the whole point of
universality testing. arXiv is used because it is a reliable bulk source of
genuinely complex, multi-column, table-and-math-heavy documents.

Usage: python eval/fetch_corpus.py <out_dir> <count>
"""
from __future__ import annotations

import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CATEGORIES = [
    "cs.CL", "cs.CV", "cs.LG", "cs.DC", "cs.CR",
    "math.PR", "math.AG", "stat.ME",
    "econ.EM", "q-fin.ST", "q-bio.NC",
    "physics.optics", "astro-ph.GA", "cond-mat.stat-mech", "eess.SP",
]
API = "http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results={n}&sortBy=submittedDate&sortOrder=descending"
ATOM = "{http://www.w3.org/2005/Atom}"


def ids_for(cat: str, n: int) -> list[str]:
    url = API.format(cat=cat, n=n)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        print(f"  ! {cat}: query failed ({e})", file=sys.stderr)
        return []
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        pid = entry.find(f"{ATOM}id").text.rsplit("/", 1)[-1]
        out.append(pid)
    return out


def main(out_dir: str, count: int):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_cat = max(1, count // len(CATEGORIES) + 1)

    ids: list[str] = []
    for cat in CATEGORIES:
        got = ids_for(cat, per_cat)
        ids.extend(got)
        print(f"  {cat}: {len(got)} ids", flush=True)
        time.sleep(3)          # arXiv API politeness
    ids = ids[:count]

    ok = 0
    for i, pid in enumerate(ids, 1):
        dest = out / f"{pid.replace('/', '_')}.pdf"
        if dest.exists() and dest.stat().st_size > 10000:
            ok += 1
            continue
        try:
            req = urllib.request.Request(f"https://arxiv.org/pdf/{pid}",
                                         headers={"User-Agent": "rtldoc-corpus/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 10000:
                print(f"  [{i}/{len(ids)}] {pid}: too small, skip", flush=True)
                continue
            dest.write_bytes(data)
            ok += 1
            print(f"  [{i}/{len(ids)}] {pid}: {len(data)//1024} KB", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(ids)}] {pid}: FAILED ({e})", flush=True)
        time.sleep(1.5)        # polite between PDF downloads

    print(f"\nDONE: {ok} PDFs in {out}/", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "corpus",
         int(sys.argv[2]) if len(sys.argv) > 2 else 100)
