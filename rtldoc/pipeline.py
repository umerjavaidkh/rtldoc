"""
Layer 3 -- semantics and assembly.

Semantic typing here is *rule-learned, not model-learned*. You label the
dozen style signatures of a textbook series once (style_profile -> YAML) and
every page in the series types deterministically, at zero inference cost,
with an audit trail. A layout model gives you 92% and no explanation; a
labelled style map gives you ~100% on the series it was labelled for and
tells you exactly which rule fired.

The output unit is an Activity, not a page: a numbered exercise plus the
teacher-column material that answers it. That is the retrieval unit a tutor
or lesson-planning RAG actually needs, and no general-purpose parser can
produce it because the linkage is cross-column and publisher-specific.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

import fitz

from . import arabic
from .layout import Region, assign_spans, order_regions, propose_regions
from .primitives import PagePrimitives, Span, containment, extract_page, style_profile

DIGITS = re.compile(r"^[\s\u0660-\u0669\u06F0-\u06F90-9]{1,3}$")

DEFAULT_STYLE_MAP: dict[str, str] = {}   # style_key -> role, supplied per series


@dataclass
class Block:
    role: str
    text: str
    bbox: tuple
    order: int
    column: int | None
    activity: int | None
    style: str | None
    diagnostics: dict = field(default_factory=dict)
    # populated only for role == "figure": the xref needed to pull the actual
    # image bytes back out of the PDF (see save_images). Before save_images
    # runs, `text` holds the geometrically-nearest caption, if any was found.
    image_xref: int | None = None


@dataclass
class PageResult:
    page: int
    columns: int
    born_digital: bool
    blocks: list[Block] = field(default_factory=list)
    activities: dict[str, dict] = field(default_factory=dict)


def _chip_number(region: Region) -> int | None:
    txt = "".join(s.text for s in region.spans).strip()
    if not txt or not DIGITS.match(txt):
        return None
    txt = txt.translate(arabic.ARABIC_INDIC).translate(arabic.EASTERN_INDIC)
    try:
        return int(txt)
    except ValueError:
        return None


def _region_text_geo(region: Region, geo_lines, opts) -> tuple[str, dict]:
    """Preferred path: take pre-reconstructed geometric lines that fall inside
    this region. Text order comes from glyph coordinates, not from the
    extractor's bidi guess."""
    picked = [(bb, t) for bb, t in geo_lines if containment(bb, region.bbox) > 0.55]
    if not picked:
        return "", {}
    picked.sort(key=lambda p: p[0][1])
    out, diags = [], {"reversed_lines": 0, "presentation_forms": 0}
    for _, raw in picked:
        clean, d = arabic.normalize(raw, opts)
        diags["presentation_forms"] += int(d["had_presentation_forms"])
        diags["reversed_lines"] += int(d["was_reversed"])
        if clean:
            out.append(clean)
    return "\n".join(out), diags


def _region_text(region: Region, opts: arabic.NormalizeOptions) -> tuple[str, dict]:
    """Fallback path: join spans into logical lines, then repair each line."""
    if not region.spans:
        return "", {}
    lines: dict[int, list[Span]] = {}
    for s in region.spans:
        key = round(((s.bbox[1] + s.bbox[3]) / 2) / max(s.size * 0.55, 1))
        lines.setdefault(key, []).append(s)

    out, diags = [], {"reversed_lines": 0, "presentation_forms": 0}
    for key in sorted(lines):
        row = lines[key]
        rtl = any(arabic.is_arabic(s.text) for s in row)
        row.sort(key=lambda s: -s.bbox[0] if rtl else s.bbox[0])
        raw = " ".join(s.text for s in row)
        clean, d = arabic.normalize(raw, opts)
        diags["reversed_lines"] += int(d["was_reversed"])
        diags["presentation_forms"] += int(d["had_presentation_forms"])
        if clean:
            out.append(clean)
    return "\n".join(out), diags


def _table_text(region: Region, owned: dict[int, list], opts: arabic.NormalizeOptions) -> tuple[str, dict]:
    """Render a detected table grid (see layout.detect_tables) as a GFM
    markdown table, one cell per (row, col) position. Cell text still goes
    through the same geometry-first bidi/repair path as everything else --
    only the grid layout itself comes from the vector rules."""
    if not region.cells:
        return "", {}
    nrows = max(c.table_row for c in region.cells) + 1
    ncols = max(c.table_col for c in region.cells) + 1
    grid = [["" for _ in range(ncols)] for _ in range(nrows)]
    diags = {"reversed_lines": 0, "presentation_forms": 0}
    for cell in region.cells:
        cell_lines = owned.get(id(cell), [])
        if cell_lines:
            text, d = _region_text_geo(cell, cell_lines, opts)
            if not text and cell.spans:
                text, d = _region_text(cell, opts)
        else:
            text, d = _region_text(cell, opts)
        diags["reversed_lines"] += d.get("reversed_lines", 0)
        diags["presentation_forms"] += d.get("presentation_forms", 0)
        grid[cell.table_row][cell.table_col] = text.replace("\n", "<br>").replace("|", "/").strip()

    lines = []
    for ri, row in enumerate(grid):
        lines.append("| " + " | ".join(row) + " |")
        if ri == 0:
            lines.append("|" + "|".join(["---"] * ncols) + "|")
    return "\n".join(lines), diags


def _dominant_style(region: Region) -> str | None:
    if not region.spans:
        return None
    tally: dict[str, int] = {}
    for s in region.spans:
        tally[s.style_key] = tally.get(s.style_key, 0) + len(s.text)
    return max(tally, key=tally.get)


def _fallback_role(region: Region, page: PagePrimitives) -> str:
    if region.kind == "table":
        return "table"
    if region.kind == "figure":
        return "figure"
    if region.kind == "chip":
        return "activity_marker"
    sizes = [s.size for s in region.spans] or [10]
    biggest = max(sizes)
    body = sorted(s.size for s in page.spans)[len(page.spans) // 2] if page.spans else 10
    if region.kind == "panel":
        return "heading" if biggest > body * 1.25 else "passage"
    if biggest > body * 1.35:
        return "heading"
    if len(region.spans) <= 2 and (region.bbox[3] > page.height * 0.93):
        return "page_furniture"
    return "paragraph"


def _nearest_caption(fig: "Block", blocks: list["Block"], max_chars: int = 120) -> str:
    """Attach whichever other block sits geometrically closest to a figure,
    as its caption -- captions are conventionally adjacent to the image they
    describe, and this needs no model call, just the geometry already on
    hand. Same-column candidates are strongly preferred (a caption belongs
    to its own column, not the neighbour's), but not ruled out entirely, in
    case a figure spans the full page width."""
    fx = (fig.bbox[0] + fig.bbox[2]) / 2
    fy = (fig.bbox[1] + fig.bbox[3]) / 2
    best, best_d = None, None
    for b in blocks:
        if b is fig or b.role in ("figure", "table") or not b.text.strip():
            continue
        bx, by = (b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2
        d = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5
        if b.column != fig.column:
            d *= 3
        if best is None or d < best_d:
            best, best_d = b, d
    return best.text.strip().replace("\n", " ")[:max_chars] if best else ""


def _link_activities(regions: list[Region]) -> None:
    """Propagate each numbered chip's id to the material it introduces.

    A chip owns everything that follows it in reading order within the same
    column, until the next chip in that column. This is what pairs the pupil
    exercise on the left page with its answer key on the right.
    """
    by_col: dict[int, list[Region]] = {}
    for r in sorted(regions, key=lambda r: r.order or 0):
        by_col.setdefault(r.column or 0, []).append(r)
    for col in by_col.values():
        current = None
        for r in col:
            n = _chip_number(r) if r.kind == "chip" else None
            if n is not None:
                current = n
            r.activity = current


def parse_page(page: "fitz.Page", style_map: dict[str, str] | None = None,
               opts: arabic.NormalizeOptions | None = None,
               geometry_bidi: bool = True) -> PageResult:
    opts = opts or arabic.NormalizeOptions()
    style_map = style_map or DEFAULT_STYLE_MAP

    # One native text extraction, shared by extract_page (spans) and geobidi
    # (per-glyph). rawdict is a superset of dict, so this single pass serves
    # both consumers instead of tokenizing the page twice.
    from .primitives import rawdict as _rawdict
    raw = _rawdict(page)

    prim = extract_page(page, raw=raw)
    result = PageResult(page=prim.number, columns=0, born_digital=prim.is_born_digital)

    if not prim.is_born_digital:
        # hand off to the OCR branch; kept separate so this stays importable
        # without torch installed.
        result.blocks = []
        return result

    regions = propose_regions(prim)
    regions = assign_spans(prim, regions)
    regions = order_regions(regions, prim.width, prim.height)
    result.columns = len({r.column for r in regions})
    _link_activities(regions)

    geo_lines = []
    if geometry_bidi:
        try:
            from . import geobidi
            geo_lines = geobidi.page_lines(page, raw=raw)
        except Exception:
            geo_lines = []

    # Give each reconstructed line to exactly one region -- whichever
    # contains the most of it. Matching every region against every geo_line
    # independently (the old behaviour) let two overlapping regions -- a
    # panel and a stray flow block covering the same text, say -- each
    # separately claim it, which duplicated that text in the output.
    owned: dict[int, list] = {}
    if geo_lines:
        leaf_regions = list(regions)
        for r in regions:
            if r.kind == "table":
                leaf_regions.extend(r.cells)
        for bb, gtext in geo_lines:
            best, best_score = None, 0.55
            for r in leaf_regions:
                c = containment(bb, r.bbox)
                if c > best_score:
                    best, best_score = r, c
            if best is not None:
                owned.setdefault(id(best), []).append((bb, gtext))

    for r in regions:
        if r.kind == "table":
            text, diag = _table_text(r, owned, opts)
        elif geo_lines:
            text, diag = _region_text_geo(r, owned.get(id(r), []), opts)
            if not text and r.spans:
                text, diag = _region_text(r, opts)
        else:
            text, diag = _region_text(r, opts)
        style = _dominant_style(r)
        role = style_map.get(style or "", None) or _fallback_role(r, prim)
        if role == "activity_marker" and not text:
            continue
        result.blocks.append(
            Block(role=role, text=text, bbox=tuple(round(v, 1) for v in r.bbox),
                  order=r.order or 0, column=r.column, activity=r.activity,
                  style=style, diagnostics=diag, image_xref=r.image_xref)
        )

    # A figure has no text of its own -- geometrically attach the nearest
    # other block's text as a caption, so a photo or chart isn't rendered
    # with nothing to say what it is. Purely positional: no model call.
    for b in result.blocks:
        if b.role == "figure":
            b.text = _nearest_caption(b, result.blocks)

    for b in result.blocks:
        if b.activity is None:
            continue
        key = str(b.activity)
        slot = result.activities.setdefault(key, {"activity": b.activity, "parts": []})
        slot["parts"].append({"role": b.role, "column": b.column, "text": b.text})

    return result


def parse_document(path: str, style_map: dict[str, str] | None = None,
                   pages: list[int] | None = None,
                   opts: arabic.NormalizeOptions | None = None,
                   geometry_bidi: bool = True) -> list[PageResult]:
    doc = fitz.open(path)
    idx = range(doc.page_count) if pages is None else pages
    out = [parse_page(doc[i], style_map, opts, geometry_bidi) for i in idx]
    doc.close()
    return out


def save_images(doc: "fitz.Document", results: list[PageResult], out_dir: str) -> int:
    """Pull each figure's actual bytes out of the PDF and write it to disk,
    turning `block.text` (until now just the geometric caption, or empty)
    into a markdown image tag with that caption as alt text.

    This is the one place in the module that touches disk, deliberately
    separate from parse_page/to_markdown -- those stay pure. Call it before
    to_markdown/to_json if you want images represented at all; skip it and
    figures still render as a `*[image: caption]*` placeholder rather than
    silently disappearing. Returns the number of *distinct* images written.

    A shared logo or banner embedded once and reused across many pages is
    ordinary PDF practice -- it carries the same xref everywhere it's
    placed, so it's written to disk exactly once here and every occurrence
    links to that one file, rather than duplicating it per page.
    """
    import os

    written_for_xref: dict[int, str] = {}
    for r in results:
        for b in r.blocks:
            if b.role != "figure" or b.image_xref is None:
                continue
            fname = written_for_xref.get(b.image_xref)
            if fname is None:
                try:
                    info = doc.extract_image(b.image_xref)
                except Exception:
                    continue
                if not info:
                    continue
                os.makedirs(out_dir, exist_ok=True)
                fname = f"img_{b.image_xref}.{info['ext']}"
                with open(os.path.join(out_dir, fname), "wb") as f:
                    f.write(info["image"])
                written_for_xref[b.image_xref] = fname
            alt = b.text.replace("]", ")").replace("[", "(")
            b.text = f"![{alt}](images/{fname})"
    return len(written_for_xref)


def to_markdown(result: PageResult) -> str:
    lines = [f"<!-- page {result.page} | {result.columns} column(s) -->", ""]
    for b in result.blocks:
        tag = f"<!-- activity {b.activity} -->" if b.activity else ""
        if b.role == "figure":
            # A figure has no ordinary text, so it must not fall through to
            # the empty-text skip below -- that used to make images vanish
            # from the output entirely with no trace they were ever there.
            if b.text.startswith("!["):
                lines.append(f"{b.text} {tag}".strip())
            else:
                alt = f": {b.text}" if b.text else ""
                lines.append(f"*[image{alt}]* {tag}".strip())
            lines.append("")
            continue
        if not b.text:
            continue
        if b.role == "table":
            lines.append(b.text)
        elif b.role == "heading":
            lines.append(f"## {b.text} {tag}".strip())
        elif b.role == "activity_marker":
            lines.append(f"\n### تمرين {b.text}\n")
        elif b.role == "passage":
            lines.append(f"> {b.text.replace(chr(10), chr(10) + '> ')} {tag}".strip())
        else:
            lines.append(f"{b.text} {tag}".strip())
        lines.append("")
    return "\n".join(lines)


def to_json(results: list[PageResult]) -> str:
    return json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2)


def audit(results: list[PageResult]) -> dict:
    """Confidence signals for routing pages to a VLM second pass."""
    flags = []
    for r in results:
        rev = sum(b.diagnostics.get("reversed_lines", 0) for b in r.blocks)
        pf = sum(b.diagnostics.get("presentation_forms", 0) for b in r.blocks)
        empty = sum(1 for b in r.blocks if not b.text.strip())
        chars = sum(len(b.text) for b in r.blocks)
        # Total extracted text, not block count, is the real "did extraction
        # basically work" signal -- a page that is legitimately one long
        # prose paragraph is one block and entirely fine; a page where
        # something went wrong (regions swallowed each other, a container
        # was misdetected) shows up as too little text coming out, whatever
        # the block count happens to be.
        flags.append({
            "page": r.page,
            "born_digital": r.born_digital,
            "blocks": len(r.blocks),
            "reversed_lines_repaired": rev,
            "presentation_form_lines": pf,
            "empty_blocks": empty,
            "needs_review": (not r.born_digital) or chars < 40 or empty > 3,
        })
    return {"pages": flags, "review_rate": sum(f["needs_review"] for f in flags) / max(len(flags), 1)}
