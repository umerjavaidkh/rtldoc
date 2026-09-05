"""Put the rendered page next to the emitted text, so a number can be checked.

The whole point
---------------
A detector is a claim, and an unchecked claim is exactly what produced "the
metrics looked fine but the PDF was broken". This builds a single self-
contained HTML file: for each flagged page, the page as the reader sees it on
the left, what rtldoc emitted on the right, and the detector's reason between
them. Thirty pages takes a couple of minutes to review by eye.

Two ways to use it:

  --worst      the pages the detectors scream loudest about. Confirms real
               bugs and gives you something to fix.
  --random     pages sampled WITHOUT regard to whether anything fired. This
               is the one that catches a suite which is quietly missing
               defects -- the failure mode that a worst-first list can never
               show you, because it only ever shows you what it found.

Usage:
    python eval/review.py ../corpus_large --json run.json --worst 30 -o review.html
    python eval/review.py ../corpus_large --random 40 -o audit.html
"""

from __future__ import annotations

import argparse
import base64
import collections
import html
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402

import detectors as D  # noqa: E402
from scorecard import probe_page  # noqa: E402

ZOOM = 1.4          # page raster scale; readable without bloating the file
MAX_PNG_PX = 1400


def render(page) -> str:
    zoom = min(ZOOM, MAX_PNG_PX / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return base64.b64encode(pix.tobytes("png")).decode()


def block_html(blocks) -> str:
    """Emitted blocks, each labelled with the role rtldoc assigned, because a
    role error is invisible in plain text and obvious once shown."""
    parts = []
    for b in sorted(blocks, key=lambda b: b.order):
        role = html.escape(b.role or "?")
        text = html.escape(b.text or "")
        cls = "tbl" if b.role == "table" else (
              "hd" if (b.role or "").startswith("heading") else "pg")
        parts.append(f'<div class="blk {cls}"><span class="role">{role}</span>'
                     f'<pre>{text}</pre></div>')
    return "".join(parts) or '<div class="blk"><em>no blocks emitted</em></div>'


CSS = """
body{font:13px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
 margin:0;background:#f6f6f4;color:#1a1a1a}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
 padding:14px 20px;z-index:9}
h1{font-size:17px;margin:0 0 4px}
.sub{color:#666;font-size:12px}
.case{background:#fff;margin:18px;border:1px solid #e0e0e0;border-radius:6px;
 overflow:hidden}
.hdr{padding:10px 14px;background:#fafafa;border-bottom:1px solid #eee;
 display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
.hdr .file{font-weight:600}
.hdr .stratum{color:#666;font-size:12px}
.codes{padding:8px 14px;background:#fff8e6;border-bottom:1px solid #f0e0b0;
 font-size:12px;max-height:6.5em;overflow:auto}
.code{display:inline-block;background:#fff;border:1px solid #e0c98a;
 border-radius:3px;padding:1px 6px;margin:2px 4px 2px 0;font-family:ui-monospace,monospace}
.code.HARD{background:#ffe8e8;border-color:#e0a0a0}
.why{color:#555}
.ev{font-family:ui-monospace,monospace;background:#f4f4f2;padding:1px 4px;
 border-radius:2px;white-space:pre-wrap}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:0}
.cols>div{padding:12px;overflow:auto;max-height:78vh}
.cols>div+div{border-left:1px solid #eee}
img{max-width:100%;border:1px solid #ddd}
.blk{margin:0 0 8px;border-left:3px solid #ddd;padding-left:8px}
.blk .role{font-size:10px;text-transform:uppercase;letter-spacing:.05em;
 color:#888;font-family:ui-monospace,monospace}
.blk pre{margin:2px 0 0;white-space:pre-wrap;word-break:break-word;font:12px/1.45
 ui-monospace,SFMono-Regular,monospace}
.blk.hd{border-left-color:#3a7bd5}.blk.hd pre{font-weight:600}
.blk.tbl{border-left-color:#c77}.blk.tbl pre{background:#fbf7f7}
.verdict{padding:10px 14px;border-top:1px solid #eee;background:#fcfcfc;font-size:12px}
label{margin-right:14px;cursor:pointer}
"""

JS = """
// Records your verdict per case in the page URL hash, so a review survives a
// reload and can be pasted back to calibrate.py. Nothing is uploaded.
function save(){
  const v={};document.querySelectorAll('.case').forEach(c=>{
    const s=c.querySelector('input:checked');if(s)v[c.dataset.id]=s.value;});
  document.getElementById('out').value=JSON.stringify(v);
}
document.addEventListener('change',save);
"""


def build(root: Path, cases: list[dict], title: str) -> str:
    fitz.TOOLS.mupdf_display_errors(False)
    out = [f"<style>{CSS}</style>", f"<header><h1>{html.escape(title)}</h1>",
           f'<div class="sub">{len(cases)} page(s). Left: the page. Right: what '
           f'rtldoc emitted, labelled by role. Mark each case, then copy the '
           f'JSON at the bottom into <code>eval/calibrate.py</code>.</div></header>']

    by_pdf = collections.defaultdict(list)
    for c in cases:
        by_pdf[c["pdf"]].append(c)

    for pdf, group in by_pdf.items():
        try:
            doc = fitz.open(pdf)
        except Exception as exc:
            out.append(f'<div class="case"><div class="hdr">{html.escape(pdf)}: '
                       f'{html.escape(repr(exc))}</div></div>')
            continue
        for case in sorted(group, key=lambda c: c["page"]):
            pno = case["page"] - 1
            if not (0 <= pno < doc.page_count):
                continue
            page = doc[pno]
            try:
                result = pipeline.parse_page(page)
            except Exception as exc:
                out.append(f'<div class="case"><div class="hdr">CRASH '
                           f'{html.escape(Path(pdf).name)} p{case["page"]}: '
                           f'{html.escape(repr(exc))}</div></div>')
                continue
            cid = f'{case["sha"]}_{case["page"]}'
            codes = "".join(
                f'<span class="code {f["severity"]}">{html.escape(f["code"])}</span>'
                f'<span class="why">{html.escape(f["detail"])}</span>'
                + (f' <span class="ev">{html.escape(f["evidence"][:120])}</span>'
                   if f.get("evidence") else "")
                + "<br>"
                for f in case["findings"]) or "<em>nothing fired on this page</em>"
            out.append(f"""
<div class="case" data-id="{html.escape(cid)}">
 <div class="hdr"><span class="file">{html.escape(Path(pdf).name)}</span>
  <span>page {case['page']}</span>
  <span class="stratum">{html.escape(case.get('stratum',''))}</span></div>
 <div class="codes">{codes}</div>
 <div class="cols">
  <div><img src="data:image/png;base64,{render(page)}"></div>
  <div>{block_html(result.blocks)}</div>
 </div>
 <div class="verdict">Is the output on the right broken?
  <label><input type="radio" name="{html.escape(cid)}" value="broken"> yes, broken</label>
  <label><input type="radio" name="{html.escape(cid)}" value="ok"> no, fine</label>
  <label><input type="radio" name="{html.escape(cid)}" value="unsure"> unsure</label>
 </div>
</div>""")
        doc.close()

    out.append('<div style="margin:18px"><b>Paste this into calibrate.py:</b><br>'
               '<textarea id="out" style="width:100%;height:80px;font-family:monospace">'
               '</textarea></div>')
    out.append(f"<script>{JS}</script>")
    return "\n".join(out)


def cases_from_run(run: dict, root: Path, worst: int, random_n: int,
                   code: str | None) -> list[dict]:
    meta = {}
    mf = root / "manifest.json"
    if mf.exists():
        data = json.loads(mf.read_text())
        meta = {str(root / d["path"]): d for d in data["documents"] if "path" in d}

    pages: dict[tuple, dict] = {}
    for doc in run["documents"]:
        sha = meta.get(doc["pdf"], {}).get("sha256", "x" * 8)[:8]
        for f in doc["findings"]:
            key = (doc["pdf"], f["page"])
            entry = pages.setdefault(key, {
                "pdf": doc["pdf"], "page": f["page"], "sha": sha,
                "stratum": doc.get("stratum", ""), "findings": []})
            entry["findings"].append(f)

    if code:
        pages = {k: v for k, v in pages.items()
                 if any(f["code"] == code for f in v["findings"])}

    if random_n:
        # sample over ALL scanned pages, flagged or not -- the only way to see
        # what the detectors are missing
        rng = random.Random(0)
        universe = []
        for doc in run["documents"]:
            sha = meta.get(doc["pdf"], {}).get("sha256", "x" * 8)[:8]
            for p in range(1, doc["pages"] + 1):
                universe.append({"pdf": doc["pdf"], "page": p, "sha": sha,
                                 "stratum": doc.get("stratum", ""),
                                 "findings": pages.get((doc["pdf"], p), {}).get("findings", [])})
        return rng.sample(universe, min(random_n, len(universe)))

    ranked = sorted(pages.values(),
                    key=lambda c: (-sum(2 if f["severity"] == "HARD" else 1
                                        for f in c["findings"]), c["pdf"], c["page"]))
    return ranked[:worst]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", required=True, help="scorecard --json output")
    ap.add_argument("--worst", type=int, default=30)
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--code", default=None, help="only pages where this code fired")
    ap.add_argument("-o", "--out", default="review.html")
    args = ap.parse_args()

    root = Path(args.root)
    run = json.loads(Path(args.json).read_text())
    cases = cases_from_run(run, root, args.worst, args.random, args.code)
    title = (f"random audit ({args.random} pages)" if args.random
             else f"worst pages" + (f" for {args.code}" if args.code else ""))
    Path(args.out).write_text(build(root, cases, f"rtldoc review - {title}"))
    print(f"wrote {args.out}  ({len(cases)} cases)")
