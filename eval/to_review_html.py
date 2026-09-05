"""Render a document's extraction as one self-contained HTML file for review.

Built for reading, not for machines. Every block is labelled with the role
rtldoc assigned, because a role error -- a heading emitted as a paragraph, a
table flattened into prose -- is invisible in plain text and obvious the moment
it is labelled. Pages are anchored and numbered so a reviewer can hold the PDF
beside it and compare page for page.

Usage:
    python eval/to_review_html.py in.pdf [more.pdf ...] --out eval/review_out/html
"""

from __future__ import annotations

import argparse
import html as _h
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline  # noqa: E402

CSS = """
:root{--fg:#1a1a1a;--mut:#777;--line:#e2e2e2;--bg:#fafaf8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.7 -apple-system,BlinkMacSystemFont,'Segoe UI','Noto Naskh Arabic',serif}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);
 padding:12px 22px;z-index:5}
h1{font-size:17px;margin:0 0 4px}
.meta{color:var(--mut);font-size:12px}
.page{background:#fff;margin:18px auto;max-width:980px;border:1px solid var(--line);
 border-radius:6px;overflow:hidden}
.pnum{background:#f4f4f2;border-bottom:1px solid var(--line);padding:6px 14px;
 font:11px ui-monospace,monospace;color:var(--mut);letter-spacing:.04em}
.body{padding:16px 22px}
.blk{position:relative;margin:0 0 14px;padding-left:74px}
[dir="rtl"].blk,.blk:has([dir="rtl"]){padding-left:0;padding-right:74px}
.role{position:absolute;left:0;top:2px;width:66px;text-align:right;
 font:10px ui-monospace,monospace;color:#999;text-transform:uppercase;letter-spacing:.05em}
.blk:has([dir="rtl"]) .role{left:auto;right:0;text-align:left}
.blk p{margin:0 0 6px}
h1.x,h2.x,h3.x,h4.x{margin:.2em 0;line-height:1.3}
h2.x{font-size:20px}h3.x{font-size:17px}h4.x{font-size:15px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:2px 0}
th,td{border:1px solid #d8d8d8;padding:5px 8px;vertical-align:top}
th{background:#f2f2f0;font-weight:600}
.table-wrap{overflow-x:auto}
.fig{color:var(--mut);font-style:italic;font-size:13px}
.empty{color:#c00;font-size:12px;font-family:ui-monospace,monospace}
.ocr{background:#fff6e5;border-left:3px solid #e0b050;padding:2px 8px}
"""


def role_of(b) -> str:
    r = b.role or "?"
    return {"paragraph": "text", "list_item": "list", "activity_marker": "marker"}.get(r, r)


def render_block(b) -> str:
    from rtldoc.pipeline import _dir_attr
    d = _dir_attr(b.text or "")
    role = _h.escape(role_of(b))
    if b.role == "table" and b.table_grid:
        rows = []
        for ri, row in enumerate(b.table_grid):
            tag = "th" if ri == 0 else "td"
            rows.append("<tr>" + "".join(
                f"<{tag}{_dir_attr(c)}>{_h.escape(c)}</{tag}>" for c in row) + "</tr>")
        inner = f'<div class="table-wrap"><table>{"".join(rows)}</table></div>'
    elif b.role == "figure":
        inner = f'<div class="fig"{d}>[image{": " + _h.escape(b.text) if b.text else ""}]</div>'
    elif (b.role or "").startswith("heading"):
        lvl = b.role[-1] if b.role[-1].isdigit() else "2"
        inner = f'<h{lvl} class="x"{d}>{_h.escape(b.text or "")}</h{lvl}>'
    elif not (b.text or "").strip():
        inner = '<div class="empty">[empty block emitted]</div>'
    else:
        inner = "".join(f"<p{d}>{_h.escape(p)}</p>"
                        for p in (b.text or "").split("\n") if p.strip())
    return f'<div class="blk"><span class="role">{role}</span>{inner}</div>'


def convert(path: Path, out_dir: Path) -> Path:
    fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(str(path))
    parts, ocr_pages, empty_blocks, tables, headings = [], 0, 0, 0, 0
    started = time.time()
    for i in range(doc.page_count):
        page = doc[i]
        try:
            r = pipeline.parse_page(page)
        except Exception as exc:
            parts.append(f'<div class="page"><div class="pnum">page {i+1}</div>'
                         f'<div class="body"><div class="empty">CRASH: '
                         f'{_h.escape(repr(exc))}</div></div></div>')
            continue
        cls = ' class="ocr"' if not r.born_digital else ""
        if not r.born_digital:
            ocr_pages += 1
        blocks = sorted(r.blocks, key=lambda b: b.order)
        empty_blocks += sum(1 for b in blocks if not (b.text or "").strip())
        tables += sum(1 for b in blocks if b.role == "table")
        headings += sum(1 for b in blocks if (b.role or "").startswith("heading"))
        body = "".join(render_block(b) for b in blocks) or \
            '<div class="empty">[no blocks emitted for this page]</div>'
        parts.append(f'<div class="page" id="p{i+1}"><div class="pnum"{cls}>page {i+1}'
                     f'{"  ·  OCR fallback" if not r.born_digital else ""}'
                     f'  ·  {len(blocks)} blocks</div>'
                     f'<div class="body">{body}</div></div>')
    n = doc.page_count
    doc.close()
    out = out_dir / (path.stem[:60] + ".html")
    out.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{_h.escape(path.stem[:60])}</title>"
        f"<style>{CSS}</style>"
        f"<header><h1>{_h.escape(path.name)}</h1>"
        f"<div class='meta'>{n} pages · {headings} headings · {tables} tables · "
        f"{empty_blocks} empty blocks · {ocr_pages} pages via OCR · "
        f"rendered in {time.time()-started:.0f}s</div></header>" + "".join(parts),
        encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--out", default="eval/review_out/html")
    a = ap.parse_args()
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    for p in a.pdfs:
        f = convert(Path(p), d)
        print(f"{f}  ({f.stat().st_size//1024} KB)")
