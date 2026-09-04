"""Arabic born-digital extraction benchmark.

Why this exists: every public Arabic document benchmark (KITAB-Bench,
ArabiDoc) ships page IMAGES and measures OCR. rtldoc does not do OCR on
born-digital PDFs -- it reads the glyph stream the file already carries --
so those benchmarks cannot measure it, and scoring near Tesseract on them
would say nothing about the thing being sold.

For a born-digital PDF the file's own text is the reference, so accuracy
can be measured with no labelling at all. That is what this does, plus a
detector for each defect class actually found in this corpus. Every check
here was a real bug: the gate read 0.999 coverage while bullets were being
deleted book-wide and a literal '1' sat between every word, because
coverage cannot see a substitution.

Usage:  python eval/arabic_bench.py book/ corpus/
"""
from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtldoc import arabic
from rtldoc.pipeline import parse_page

ARABIC = re.compile(r"[؀-ۿ]")
# a digit wedged between two Arabic letters: the space-glyph corruption
DIGIT_IN_WORD = re.compile(r"[؀-ۿ]\d[؀-ۿ]")
# a lone letter fenced by spaces mid-sentence: a word split by a false space
SPLIT_WORD = re.compile(r"[؀-ۿ]{2,} [؀-ۿ] [؀-ۿ]{2,}")
LEADER_RUN = re.compile(r"([.·․])\1{5,}")


def _arabic_pages(doc, sample: int = 5) -> bool:
    chars = ar = 0
    for i in range(min(len(doc), sample)):
        t = doc[i].get_text()
        chars += len(t)
        ar += len(ARABIC.findall(t))
    return chars > 200 and ar / max(1, chars) > 0.25


def check(path: str) -> dict | None:
    doc = fitz.open(path)
    try:
        if not _arabic_pages(doc):
            return None
        r = {"pdf": os.path.basename(path), "pages": len(doc),
             "replacement": 0, "presentation": 0, "digit_in_word": 0,
             "split_word": 0, "leader_run": 0, "kept": 0, "total": 0,
             "flagged": []}
        for i in range(len(doc)):
            page = doc[i]
            out = "\n".join(b.text or "" for b in parse_page(page).blocks)
            src = page.get_text()
            if not src.strip():
                continue
            s = collections.Counter(c for c in src if not c.isspace())
            o = collections.Counter(c for c in out if not c.isspace())
            r["total"] += sum(s.values())
            r["kept"] += sum(min(s[c], o[c]) for c in s)

            bad = {
                "replacement": out.count("�"),
                "presentation": sum(arabic.is_presentation_form(c) for c in out),
                "digit_in_word": len(DIGIT_IN_WORD.findall(out)),
                "split_word": len(SPLIT_WORD.findall(out)),
                "leader_run": len(LEADER_RUN.findall(out)),
            }
            for k, v in bad.items():
                r[k] += v
            if any(bad.values()):
                r["flagged"].append((i + 1, {k: v for k, v in bad.items() if v}))
        return r
    finally:
        doc.close()


def main(roots: list[str]) -> None:
    paths = []
    for root in roots:
        for dp, _, fs in os.walk(root):
            paths += [os.path.join(dp, f) for f in fs if f.lower().endswith(".pdf")]
    paths.sort()

    rows = []
    for p in paths:
        try:
            r = check(p)
        except Exception as e:                       # a crash is itself a result
            print(f"  CRASH {os.path.basename(p)}: {e!r}")
            continue
        if r:
            rows.append(r)

    if not rows:
        print("no Arabic born-digital PDFs found")
        return

    tot_pages = sum(r["pages"] for r in rows)
    kept = sum(r["kept"] for r in rows)
    total = sum(r["total"] for r in rows)
    print(f"# Arabic born-digital benchmark: {len(rows)} PDFs, {tot_pages} pages\n")
    print(f"  character recall vs the file's own text : {kept / max(1, total):.4f}")
    print("  (1.0 = every character the PDF states was emitted; this is the")
    print("   accuracy an OCR benchmark reports as CER, measured without labels)\n")
    print("  defect detectors (each was a real bug in this corpus):")
    for k, label in (("replacement", "U+FFFD in output"),
                     ("presentation", "presentation forms leaked"),
                     ("digit_in_word", "digit wedged inside an Arabic word"),
                     ("split_word", "word split by a false space"),
                     ("leader_run", "uncollapsed dot-leader run")):
        n = sum(r[k] for r in rows)
        docs = sum(1 for r in rows if r[k])
        print(f"    {label:36} {n:6}   in {docs} doc(s)")

    print("\n  worst documents by recall:")
    for r in sorted(rows, key=lambda r: r["kept"] / max(1, r["total"]))[:5]:
        print(f"    {r['kept'] / max(1, r['total']):.4f}  {r['pdf'][:40]:40} "
              f"({r['pages']} pages, {len(r['flagged'])} flagged)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["book/", "corpus/"])
