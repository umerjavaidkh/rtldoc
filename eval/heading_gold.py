"""Sample heading candidates for hand adjudication.

Why sampled this way
--------------------
A gold set built only from what the parser already emits can measure precision
and is blind to recall -- it can never contain a heading the parser missed. So
candidates come from three sources deliberately:

  emitted     blocks rtldoc currently calls headings   -> finds false positives
  declared    /H1../H6 in the PDF structure tree       -> finds misses the file admits
  overlooked  short blocks in a rare, non-body style   -> finds misses nobody admits
              that rtldoc did NOT call headings

The third source is the one that matters. Both automatic references available
are wrong in known directions -- font-size truth credits any large text, the
structure tree tags real chapters as body -- so a set drawn from either alone
would inherit its blind spot.

Documents are chosen for difficulty, not coverage: a legal regulation whose
headings are set SMALLER and LIGHTER than its body, an infographic report where
prominent text is mostly not headings, a dense statistical yearbook, and a
two-column paper. Easy documents would agree with any rule and teach nothing.

Usage:
    python eval/heading_gold.py sample -o eval/heading_gold_set
    python eval/heading_gold.py score  --gold eval/heading_gold_set/gold.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fitz  # noqa: E402
from rtldoc import pipeline, tags  # noqa: E402
from eval.ragbench.rules import normalize  # noqa: E402

# Chosen for the ways they break heading detection, not for variety.
TOUGH = [
    ("corpus_gulf/f9b88f18_اللائحة_التنفيذية_لنظام_العمل_وملحقاتها.pdf",
     "headings set SMALLER and LIGHTER than body"),
    ("corpus_gulf/b4f9c94b_اللائحة_التنفيذية_للموارد_البشرية.pdf",
     "legal regulation, uniform type"),
    ("corpus_gulf/214a88f2_beneficiary-voice-report-for-the-first-quarter-of-2024.pdf",
     "infographic: prominent text that is NOT headings"),
    ("corpus_gulf/ad90d515_Statistical-Yearbook-2022.pdf",
     "dense statistical tables"),
    ("corpus_gulf/600bb49d_user-manual-ar.pdf", "manual, many ruled tables"),
    ("corpus_large/web_print__arabic/13cc4777_arwiki_agriculture.pdf",
     "three heading levels, sub-heads at 1.16x body"),
    ("corpus_large/web_print__arabic/7aeb0366_arwiki_engineering.pdf",
     "three heading levels"),
    ("corpus_large/publisher__latin/c51b97c8_irs_p561.pdf",
     "IRS publication, two columns + tables"),
    # Second wave. The first eight gave 72 labels, too few to fit anything and
    # too narrow to hold out from: 137 of the 149 usable positives came from a
    # single file. These seven widen the producers and the heading styles.
    ("book/BilArabi_TG07.pdf", "Arabic teacher's guide, coloured band headings"),
    ("corpus_gulf/8a1fac72_book-Statistics-2018.pdf",
     "borderless statistical tables, bilingual heads"),
    ("corpus_gulf/325b88ed_Statistical-Yearbook-2023.pdf",
     "yearbook, headings inside coloured bands"),
    ("corpus_gulf/bfbfbbac_healthybook.pdf", "health handbook, mixed styles"),
    ("corpus_gulf/f6f9913d_AnnualReport2021.pdf",
     "annual report, display type that is not headings"),
    ("corpus_gulf/3739b6b1_2020_Khibrat_Guide_AR.pdf", "guide, numbered sections"),
    ("corpus_large/web_print__arabic/15a1fefa_arwiki_iraq.pdf",
     "wiki print, three heading levels"),
]
PAD = 14.0
ZOOM = 2.0


def _rare_style(style, tally, total) -> bool:
    return bool(style) and total > 0 and tally.get(style, 0) <= total * 0.25


def sample(out: Path, per_doc: int = 9) -> dict:
    fitz.TOOLS.mupdf_display_errors(False)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    cases = []
    for rel, why in TOUGH:
        path = Path(rel)
        if not path.exists():
            continue
        doc = fitz.open(str(path))
        declared = {}
        try:
            for pno, level, mcids in tags.tagged_headings(doc):
                declared.setdefault(pno, []).append((level, mcids))
        except Exception:
            pass
        got = 0
        for i in range(doc.page_count):
            if got >= per_doc:
                break
            page = doc[i]
            try:
                res = pipeline.parse_page(page)
            except Exception:
                continue
            if not res.born_digital or not res.blocks:
                continue
            tally, total = {}, 0
            for sp in pipeline.extract_page(page).spans:
                n = len(sp.text.strip())
                if n:
                    tally[sp.style_key] = tally.get(sp.style_key, 0) + n
                    total += n
            body_style = max(tally, key=tally.get) if tally else None

            declared_txt = []
            if i in declared:
                texts = tags.mcid_text(page)
                for _l, mc in declared[i]:
                    t = " ".join(texts.get(m, "") for m in mc).strip()
                    if t:
                        declared_txt.append(normalize(t))

            for b in res.blocks:
                if got >= per_doc or not b.bbox:
                    continue
                if b.role in ("table", "figure"):
                    continue
                text = (b.text or "").strip()
                # A one- or two-character block is a page number, a list
                # marker or a chip -- never a heading, and its crop renders as
                # an apparently empty box that costs a human a decision.
                if not text or len(text) > 160 or len(text.strip()) < 3:
                    continue
                is_head = (b.role or "").startswith("heading")
                short_rare = (len(text) <= 80 and b.style != body_style
                              and _rare_style(b.style, tally, total))
                nt = normalize(text)
                is_declared = any(nt in d or d in nt for d in declared_txt if d)
                src = ("emitted" if is_head else
                       "declared" if is_declared else
                       "overlooked" if short_rare else None)
                if src is None:
                    continue
                cid = f"{path.stem[:16]}_p{i+1}_{len(cases)}"
                clip = fitz.Rect(b.bbox[0] - PAD, b.bbox[1] - PAD * 2.5,
                                 b.bbox[2] + PAD, b.bbox[3] + PAD * 3.5) & page.rect
                page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), clip=clip).save(
                    str(out / "crops" / f"{cid}.png"))
                cases.append({
                    "id": cid, "pdf": str(path), "page": i + 1, "why_doc": why,
                    "source": src, "emitted_as_heading": is_head,
                    "text": text[:150], "crop": f"crops/{cid}.png",
                    "verdict": None,
                })
                got += 1
        doc.close()
    payload = {"cases": cases}
    (out / "gold.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return payload


def score(gold: dict) -> str:
    judged = [c for c in gold["cases"] if c["verdict"] in ("heading", "not")]
    if not judged:
        return "no verdicts recorded yet"
    tp = sum(1 for c in judged if c["verdict"] == "heading" and c["emitted_as_heading"])
    fp = sum(1 for c in judged if c["verdict"] == "not" and c["emitted_as_heading"])
    fn = sum(1 for c in judged if c["verdict"] == "heading" and not c["emitted_as_heading"])
    tn = sum(1 for c in judged if c["verdict"] == "not" and not c["emitted_as_heading"])
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    lines = [f"# heading gold set: {len(judged)} adjudicated of {len(gold['cases'])}",
             f"  真 headings: {tp + fn}   non-headings: {fp + tn}",
             f"  precision : {prec:.3f}  ({tp} of {tp + fp} emitted are real)",
             f"  recall    : {rec:.3f}  ({tp} of {tp + fn} real were found)",
             f"  F1        : {f1:.3f}"]
    miss = [c for c in judged if c["verdict"] == "heading" and not c["emitted_as_heading"]]
    spur = [c for c in judged if c["verdict"] == "not" and c["emitted_as_heading"]]
    if miss:
        lines.append("\n  missed:")
        for c in miss[:8]:
            lines.append(f"    {c['text'][:56]!r}")
    if spur:
        lines.append("\n  spurious:")
        for c in spur[:8]:
            lines.append(f"    {c['text'][:56]!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("-o", "--out", default="eval/heading_gold_set")
    ap.add_argument("--gold", default="eval/heading_gold_set/gold.json")
    ap.add_argument("--per-doc", type=int, default=9)
    a = ap.parse_args()
    if a.cmd == "sample":
        g = sample(Path(a.out), a.per_doc)
        import collections
        print(f"{len(g['cases'])} candidates")
        print(collections.Counter(c["source"] for c in g["cases"]).most_common())
    else:
        print(score(json.loads(Path(a.gold).read_text())))
