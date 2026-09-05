"""Build the evaluation corpus: deduplicate, stratify, copy, manifest.

Why this exists
---------------
A corpus is not a pile of PDFs. Two things ruin an extraction benchmark, and
both were present in the raw pile this was built from:

  1. DUPLICATES. The same file under two names inflates every count and
     silently weights whatever it duplicates. Deduplication is by SHA-256 of
     the file bytes -- names lie, bytes do not.

  2. MONOCULTURE. 85% of the raw pile is arXiv papers emitted by one
     generator (arXiv GenPDF / tex2pdf). A corpus mean over that is an
     arXiv-LaTeX mean: it can sit at 0.98 while every Word document and
     every scanned report in the set is visibly broken. The fix is not to
     drop them -- it is to STRATIFY, so the scorecard reports per producer
     family and per script, and the bulk cannot hide the tail.

The stratum axes are producer family and script, because those are what
actually predict failure: a Chrome print-to-PDF lays glyphs out differently
from pdfTeX, and Arabic exercises code paths Latin never reaches.

Usage:
    python eval/corpus_build.py --dry-run              # census only, copies nothing
    python eval/corpus_build.py --out corpus_large/    # dedupe + stratify + copy
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402

# Where PDFs are collected from. Directories are searched recursively.
DEFAULT_SOURCES = [
    "/Users/umerjavaid/Documents/umerwork/AI/agentic_graph_rag/sample_data_to_test/unstructured",
    "book",
    "corpus",
    "corpus_books",
]

_AR = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LAT = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")

# Ordered: first match wins, so the specific patterns precede the generic ones.
# Matched against "producer | creator" lowercased.
_PRODUCER_FAMILIES = [
    ("arxiv genpdf", "arxiv"),
    ("antenna house", "publisher"),
    ("indesign", "publisher"),
    ("acrobat distiller", "publisher"),
    ("adobe pdf library", "publisher"),
    ("ghostscript", "publisher"),
    ("dvips", "publisher"),
    ("skia", "web_print"),
    ("weasyprint", "web_print"),
    ("prince", "web_print"),
    ("wkhtml", "web_print"),
    ("chromium", "web_print"),
    ("print to pdf", "word"),
    ("word", "word"),
    ("powerpoint", "office"),
    ("excel", "office"),
    ("libreoffice", "office"),
    ("openoffice", "office"),
    ("quartz", "office"),
    ("pdftex", "latex"),
    ("xetex", "latex"),
    ("luatex", "latex"),
    ("latex", "latex"),
    ("reportlab", "generated"),
    ("pdf-lib", "generated"),
    ("fpdf", "generated"),
    ("tcpdf", "generated"),
    ("itext", "generated"),
    ("pikepdf", "generated"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def producer_family(producer: str, creator: str) -> str:
    """Which generator made this PDF. The single best predictor of how its
    glyphs are laid out, and therefore of which extraction paths it exercises.

    Note the ordering dependency: files rewritten by a tool (pikepdf, iText)
    carry the REWRITER as producer and the original generator as creator, so
    both fields are searched together and the specific names must be tested
    before the generic rewriter names."""
    hay = f"{producer} | {creator}".lower()
    for needle, family in _PRODUCER_FAMILIES:
        if needle in hay:
            return family
    return "unknown"


def script_of(ar: int, lat: int, cjk: int) -> str:
    total = ar + lat + cjk
    if total < 20:
        return "no_text"
    if ar / total > 0.5:
        return "arabic"
    if cjk / total > 0.3:
        return "cjk"
    if ar / total > 0.05:
        return "mixed_ar"
    return "latin"


def census_one(path_str: str) -> dict:
    """Metadata + a 5-page sample. Deliberately cheap: this runs over
    thousands of files and only has to be good enough to stratify."""
    path = Path(path_str)
    rec: dict = {"path": path_str, "size": path.stat().st_size}
    try:
        doc = fitz.open(path_str)
    except Exception as exc:
        rec["broken"] = repr(exc)
        return rec
    try:
        meta = doc.metadata or {}
        rec["producer"] = (meta.get("producer") or "").strip()[:160]
        rec["creator"] = (meta.get("creator") or "").strip()[:160]
        rec["title"] = (meta.get("title") or "").strip()[:160]
        n = doc.page_count
        rec["pages"] = n
        sample = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1} & set(range(n)))
        ar = lat = cjk = chars = imgs = draws = text_pages = 0
        for i in sample:
            page = doc[i]
            text = page.get_text()
            chars += len(text)
            if len(text.strip()) > 40:
                text_pages += 1
            ar += len(_AR.findall(text))
            lat += len(_LAT.findall(text))
            cjk += len(_CJK.findall(text))
            try:
                imgs += len(page.get_images(full=True))
                draws += len(page.get_drawings())
            except Exception:
                pass
        k = max(1, len(sample))
        rec["chars_per_page"] = round(chars / k, 1)
        rec["imgs_per_page"] = round(imgs / k, 2)
        rec["draws_per_page"] = round(draws / k, 1)
        rec["text_page_frac"] = round(text_pages / k, 2)
        rec["script"] = script_of(ar, lat, cjk)
        rec["family"] = producer_family(rec["producer"], rec["creator"])
        # A page with no usable text layer is a scan, whatever made it -- that
        # routes to OCR and must be scored as its own stratum, never averaged
        # in with born-digital extraction.
        if rec["text_page_frac"] < 0.34:
            rec["family"] = "scanned"
            rec["script"] = "no_text" if rec["script"] == "no_text" else rec["script"]
    except Exception as exc:
        rec["broken"] = repr(exc)
    finally:
        doc.close()
    return rec


def collect(sources: list[str]) -> list[Path]:
    out: list[Path] = []
    for src in sources:
        p = Path(src)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf":
            out.append(p)
    return out


def build(sources: list[str], out_dir: Path | None, dry_run: bool) -> dict:
    paths = collect(sources)
    print(f"found {len(paths)} PDF files in {len(sources)} source(s)", file=sys.stderr)

    with ThreadPoolExecutor(8) as ex:
        digests = list(ex.map(sha256, paths))

    # Deduplicate by content. Ties are broken by shortest path then name, so
    # the winner is stable across runs -- a manifest that reshuffles on every
    # rebuild makes result diffs unreadable.
    by_sha: dict[str, list[Path]] = collections.defaultdict(list)
    for path, digest in zip(paths, digests):
        by_sha[digest].append(path)
    keep: list[tuple[str, Path, list[str]]] = []
    for digest, group in by_sha.items():
        group = sorted(group, key=lambda p: (len(p.parts), str(p)))
        keep.append((digest, group[0], [str(p) for p in group[1:]]))
    keep.sort(key=lambda t: str(t[1]))
    dupes = sum(len(aliases) for _, _, aliases in keep)
    print(f"{len(keep)} unique by sha256 ({dupes} duplicate copies collapsed)", file=sys.stderr)

    with ProcessPoolExecutor(8) as ex:
        recs = list(ex.map(census_one, [str(p) for _, p, _ in keep], chunksize=8))

    entries = []
    for (digest, path, aliases), rec in zip(keep, recs):
        rec["sha256"] = digest
        rec["source"] = str(path)
        rec["aliases"] = aliases
        rec["stratum"] = f"{rec.get('family', 'unknown')}__{rec.get('script', 'no_text')}"
        entries.append(rec)

    broken = [e for e in entries if "broken" in e]
    ok = [e for e in entries if "broken" not in e]

    if out_dir and not dry_run:
        for entry in ok:
            dest_dir = out_dir / entry["stratum"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            # Name by sha prefix + original stem: unique without a hash-only
            # name you cannot recognise when a result points at it.
            stem = re.sub(r"[^\w.\-]+", "_", Path(entry["source"]).stem)[:80]
            dest = dest_dir / f"{entry['sha256'][:8]}_{stem}.pdf"
            if not dest.exists():
                shutil.copy2(entry["source"], dest)
            entry["path"] = str(dest.relative_to(out_dir))
        manifest = {
            "unique_documents": len(ok),
            "duplicate_copies_collapsed": dupes,
            "unreadable": len(broken),
            "total_pages": sum(e.get("pages", 0) for e in ok),
            "sources": sources,
            "documents": ok + broken,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"copied {len(ok)} documents into {out_dir}", file=sys.stderr)
    else:
        manifest = {
            "unique_documents": len(ok),
            "duplicate_copies_collapsed": dupes,
            "unreadable": len(broken),
            "total_pages": sum(e.get("pages", 0) for e in ok),
            "sources": sources,
            "documents": ok + broken,
        }
    return manifest


def report(manifest: dict) -> str:
    docs = [d for d in manifest["documents"] if "broken" not in d]
    lines = [
        f"# corpus: {manifest['unique_documents']} unique documents, "
        f"{manifest['total_pages']} pages",
        f"  duplicate copies collapsed : {manifest['duplicate_copies_collapsed']}",
        f"  unreadable files           : {manifest['unreadable']}",
        "",
        f"{'stratum':34}{'docs':>7}{'pages':>9}{'share':>8}",
    ]
    by = collections.Counter(d["stratum"] for d in docs)
    pages = collections.Counter()
    for d in docs:
        pages[d["stratum"]] += d.get("pages", 0)
    total = sum(by.values())
    for stratum, count in by.most_common():
        lines.append(f"{stratum:34}{count:>7}{pages[stratum]:>9}{100*count/total:>7.1f}%")
    lines.append("")
    lines.append("The largest stratum's share is the number to watch: a corpus")
    lines.append("mean is meaningless when one stratum owns most of the mass.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_large")
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    out = None if args.dry_run else (root / args.out if not Path(args.out).is_absolute() else Path(args.out))
    m = build(args.source or DEFAULT_SOURCES, out, args.dry_run)
    print(report(m))
