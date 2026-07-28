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
import statistics
from dataclasses import dataclass, field, asdict

import fitz

from . import arabic
from .layout import Region, assign_spans, group_by_line, order_regions, propose_regions
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
    # populated only for role == "table": the raw (row, col) text grid, so
    # to_html can emit a real <table> without re-parsing markdown pipes.
    table_grid: list[list[str]] | None = None


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
    """Preferred path: render the geometric lines this region owns, in reading
    order. Text order comes from glyph coordinates, not the extractor's bidi
    guess. Ownership is decided once by the caller (parse_page), so we do NOT
    re-filter by containment here -- an earlier redundant containment gate on
    top of an already-inflated line bbox silently dropped a line the region
    genuinely owned, losing that text entirely."""
    picked = list(geo_lines)
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
    lines = group_by_line(region.spans)

    out, diags = [], {"reversed_lines": 0, "presentation_forms": 0}
    for row in lines:
        rtl = any(arabic.is_arabic(s.text) for s in row)
        row.sort(key=lambda s: -s.bbox[0] if rtl else s.bbox[0])
        raw = " ".join(s.text for s in row)
        clean, d = arabic.normalize(raw, opts)
        diags["reversed_lines"] += int(d["was_reversed"])
        diags["presentation_forms"] += int(d["had_presentation_forms"])
        if clean:
            out.append(clean)
    return "\n".join(out), diags


def _table_grid(region: Region, owned: dict[int, list],
                opts: arabic.NormalizeOptions) -> tuple[list[list[str]], dict]:
    """Build the raw (row, col) text grid for a detected table (see
    layout.detect_tables). Cell text goes through the same geometry-first
    bidi/repair path as everything else -- only the grid layout itself comes
    from the vector rules. Shared by both the markdown and HTML renderers so
    neither has to re-parse the other's output to get the cells back."""
    if not region.cells:
        return [], {}
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
        grid[cell.table_row][cell.table_col] = text.strip()

    # Drop wholly-empty rows and columns. Borderless-table column/row
    # boundaries are inferred from alignment, so a slightly-misplaced split can
    # leave a phantom empty column or a blank row; trimming them costs nothing
    # and also tidies vector tables that had an unused frame line.
    keep_cols = [ci for ci in range(ncols) if any(grid[ri][ci] for ri in range(nrows))]
    keep_rows = [ri for ri in range(nrows) if any(grid[ri][ci] for ci in range(ncols))]
    if not keep_cols or not keep_rows:
        return [], diags
    return [[grid[ri][ci] for ci in keep_cols] for ri in keep_rows], diags


def _table_text(region: Region, owned: dict[int, list], opts: arabic.NormalizeOptions) -> tuple[str, dict, list]:
    """Render a detected table as a GFM markdown table. Returns (markdown,
    diagnostics, grid) -- the grid is exposed so to_html can build a real
    <table> instead of re-parsing markdown pipes back into cells."""
    grid, diags = _table_grid(region, owned, opts)
    if not grid:
        return "", diags, grid
    ncols = len(grid[0])
    lines = []
    for ri, row in enumerate(grid):
        cells = [c.replace("\n", "<br>").replace("|", "/") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if ri == 0:
            lines.append("|" + "|".join(["---"] * ncols) + "|")
    return "\n".join(lines), diags, grid


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
    # Dominant size (char-weighted median), not the single biggest span: a
    # region that's mostly body paragraph text with one bigger banner/label
    # span inside it (a "Learning Objectives" title over its own bullet list,
    # say) is still a passage -- one big span shouldn't out-vote the bulk of
    # the region's actual content and mislabel the whole thing as a heading.
    weighted = [s.size for s in region.spans for _ in range(max(len(s.text.strip()), 1))] or [10]
    dominant = statistics.median(weighted)
    body = sorted(s.size for s in page.spans)[len(page.spans) // 2] if page.spans else 10
    if region.kind == "panel":
        return "heading" if dominant > body * 1.25 else "passage"
    if dominant > body * 1.35:
        return "heading"
    if len(region.spans) <= 2 and (region.bbox[3] > page.height * 0.93):
        return "page_furniture"
    return "paragraph"


def _dedupe_blocks(blocks: list["Block"], quality: dict[int, int], min_len: int = 12) -> None:
    """Merge policy: a text line that appears across more than one block is
    kept only in the highest render-quality block and stripped from the rest.

    Cross-block only: identical lines *within* one block are genuine source
    repetition and left untouched. Ties in quality keep the earliest block in
    reading order. Short lines (< min_len) are ignored -- a bare number or a
    one-word heading can legitimately recur, and removing it would lose real
    content for no benefit.
    """
    per_block_lines: list[list[str] | None] = []
    occ: dict[str, list[tuple[int, int]]] = {}
    for bi, b in enumerate(blocks):
        if b.role in ("figure", "table") or not b.text:
            per_block_lines.append(None)
            continue
        lines = b.text.split("\n")
        per_block_lines.append(lines)
        for li, ln in enumerate(lines):
            key = ln.strip()
            if len(key) >= min_len:
                occ.setdefault(key, []).append((bi, li))

    remove: set[tuple[int, int]] = set()
    for spots in occ.values():
        blocks_involved = {bi for bi, _ in spots}
        if len(blocks_involved) < 2:
            continue                      # all in one block -> genuine repeat
        best_bi = max(sorted(blocks_involved), key=lambda bi: quality.get(id(blocks[bi]), 1))
        for bi, li in spots:
            if bi != best_bi:
                remove.add((bi, li))

    if not remove:
        return
    for bi, lines in enumerate(per_block_lines):
        if lines is None:
            continue
        kept = [ln for li, ln in enumerate(lines) if (bi, li) not in remove]
        blocks[bi].text = "\n".join(kept).strip()


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
    #
    # Assign to the *best-overlap* region, not to any region clearing a fixed
    # containment floor: geobidi's line bboxes are padded by a full font size
    # above the baseline, so a line the region genuinely owns can score well
    # under 0.55 and, with a hard floor, get orphaned and dropped. Argmax with
    # only a tiny epsilon keeps the single-owner property (so no duplicates
    # come back) while guaranteeing every line that overlaps *some* region is
    # rendered.
    owned: dict[int, list] = {}
    if geo_lines:
        leaf_regions = list(regions)
        for r in regions:
            if r.kind == "table":
                leaf_regions.extend(r.cells)
        for bb, gtext in geo_lines:
            best, best_score = None, 1e-6
            for r in leaf_regions:
                c = containment(bb, r.bbox)
                if c > best_score:
                    best, best_score = r, c
            if best is not None:
                owned.setdefault(id(best), []).append((bb, gtext))

    # Render quality per block, for the merge policy below: text recovered
    # from glyph geometry (correct bidi) beats the span-based fallback, and a
    # table cell grid is authoritative.
    quality: dict[int, int] = {}
    for r in regions:
        grid = None
        if r.kind == "table":
            text, diag, grid = _table_text(r, owned, opts)
            q = 3
        elif geo_lines:
            text, diag = _region_text_geo(r, owned.get(id(r), []), opts)
            q = 2
            if not text and r.spans:
                text, diag = _region_text(r, opts)
                q = 1
        else:
            text, diag = _region_text(r, opts)
            q = 1
        style = _dominant_style(r)
        role = style_map.get(style or "", None) or _fallback_role(r, prim)
        if role == "activity_marker" and not text:
            continue
        block = Block(role=role, text=text, bbox=tuple(round(v, 1) for v in r.bbox),
                      order=r.order or 0, column=r.column, activity=r.activity,
                      style=style, diagnostics=diag, image_xref=r.image_xref, table_grid=grid)
        result.blocks.append(block)
        quality[id(block)] = q

    # MERGE POLICY. When two overlapping regions disagree on ownership, the
    # same text can be rendered by both -- once via the geo path, once via a
    # neighbour's span fallback -- so a line that occurs ONCE in the PDF comes
    # out twice. Rather than perfectly reconcile the two segmentations, drop
    # the duplicate and keep the best copy: a line appearing across more than
    # one block is kept only in the highest render-quality block and removed
    # from the others. Repeats WITHIN a single block are left alone (those are
    # genuine), so faithful source duplication survives while parser-side
    # double-emission does not.
    _dedupe_blocks(result.blocks, quality)

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


_MD_IMAGE = re.compile(r"^!\[(.*)\]\((.*)\)$", re.S)


def _dir_attr(text: str) -> str:
    """`dir="rtl"` for predominantly-Arabic text, nothing for LTR -- so mixed
    Arabic/English books render each block in its own correct direction
    instead of one direction being forced on the whole page."""
    return ' dir="rtl"' if arabic.is_arabic(text) else ""


def _heading_levels(result: PageResult) -> dict[int, int]:
    """Rank heading blocks by font size into h1-h4, largest first. Best-effort:
    without a style map this is the only per-page signal available, but it's
    enough to give a page real structure instead of every heading flattening
    to the same tag regardless of size."""
    sizes: dict[int, float] = {}
    for b in result.blocks:
        if b.role != "heading" or not b.style:
            continue
        m = re.match(r"^.*\|([0-9.]+)\|", b.style)
        if m:
            sizes[id(b)] = float(m.group(1))
    ranked = sorted(set(sizes.values()), reverse=True)
    return {bid: min(ranked.index(sz) + 1, 4) for bid, sz in sizes.items()}


def to_html(result: PageResult) -> str:
    """Render one page as an HTML fragment: real <table>, <img>/<figure> for
    images, leveled headings, direction-aware paragraphs. Meant to be dropped
    into the <body> of a page shell (see cli.py's --html output), not a full
    document on its own -- so it composes into a single combined file too.
    """
    import html as _html

    levels = _heading_levels(result)
    out = [f'<section class="page" id="page-{result.page}" data-page="{result.page}">']
    for b in result.blocks:
        tag_attr = f' data-activity="{b.activity}"' if b.activity is not None else ""

        if b.role == "figure":
            m = _MD_IMAGE.match(b.text) if b.text else None
            if m:
                alt, src = _html.escape(m.group(1)), _html.escape(m.group(2))
                out.append(f'<figure{tag_attr}><img src="{src}" alt="{alt}" loading="lazy">'
                          f'<figcaption>{alt}</figcaption></figure>')
            elif b.text:
                out.append(f'<div class="figure-placeholder"{tag_attr}>[image: {_html.escape(b.text)}]</div>')
            else:
                out.append(f'<div class="figure-placeholder"{tag_attr}>[image]</div>')
            continue

        if not b.text:
            continue

        if b.role == "table" and b.table_grid:
            rows = ["<table>"]
            for ri, row in enumerate(b.table_grid):
                cell_tag = "th" if ri == 0 else "td"
                cells = "".join(f"<{cell_tag}{_dir_attr(c)}>{_html.escape(c)}</{cell_tag}>" for c in row)
                rows.append(f"<tr>{cells}</tr>")
            rows.append("</table>")
            out.append(f'<div class="table-wrap"{tag_attr}>{"".join(rows)}</div>')
        elif b.role == "heading":
            level = levels.get(id(b), 2)
            out.append(f'<h{level}{_dir_attr(b.text)}{tag_attr}>{_html.escape(b.text)}</h{level}>')
        elif b.role == "activity_marker":
            out.append(f'<h3 class="activity-marker"{tag_attr}>{_html.escape(b.text)}</h3>')
        elif b.role == "passage":
            paras = "".join(f"<p>{_html.escape(p)}</p>" for p in b.text.split("\n") if p.strip())
            out.append(f'<blockquote{_dir_attr(b.text)}{tag_attr}>{paras}</blockquote>')
        else:
            paras = "".join(f"<p>{_html.escape(p)}</p>" for p in b.text.split("\n") if p.strip())
            out.append(f'<div class="{b.role}"{_dir_attr(b.text)}{tag_attr}>{paras}</div>')
    out.append("</section>")
    return "\n".join(out)


_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{css_href}">
</head>
<body>
{nav}
{content}
{nav}
</body>
</html>
"""

_DEFAULT_CSS = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 900px;
      margin: 0 auto; padding: 1.5rem; line-height: 1.55; color: #1b1b1b; background: #fff; }
.page { border-bottom: 1px solid #ddd; padding-bottom: 2rem; margin-bottom: 2rem; }
.page::before { content: "Page " attr(data-page); display: block; font-size: 0.8rem;
                color: #888; margin-bottom: 0.75rem; }
h1, h2, h3, h4 { color: #0d3b66; line-height: 1.25; }
h1 { font-size: 1.6rem; } h2 { font-size: 1.3rem; } h3 { font-size: 1.1rem; } h4 { font-size: 1rem; }
.activity-marker { color: #b5471b; }
blockquote { border-inline-start: 4px solid #0d3b66; margin: 1rem 0; padding: 0.25rem 1rem;
            background: #f4f8fb; }
p { margin: 0.6rem 0; }
figure { margin: 1.2rem 0; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #eee; }
figcaption { font-size: 0.85rem; color: #555; margin-top: 0.35rem; }
.figure-placeholder { padding: 0.75rem; background: #f3f3f3; color: #777; font-style: italic;
                      text-align: center; margin: 1rem 0; }
.table-wrap { overflow-x: auto; margin: 1.2rem 0; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: start; font-size: 0.92rem; }
th { background: #f0f4f8; }
[dir="rtl"] { text-align: right; }
nav.page-nav { display: flex; justify-content: space-between; font-size: 0.9rem; margin: 1rem 0; }
nav.page-nav a { color: #0d3b66; text-decoration: none; }
"""


def save_html(results: list[PageResult], out_dir: str, css_href: str = "styles.css",
             title_prefix: str = "Page") -> list[str]:
    """Write one HTML file per page plus a shared stylesheet and an index.
    Returns the list of written page filenames."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "styles.css"), "w") as f:
        f.write(_DEFAULT_CSS)

    names = [f"page_{r.page:04d}.html" for r in results]
    for i, r in enumerate(results):
        prev = f'<a href="{names[i-1]}">&larr; prev</a>' if i > 0 else "<span></span>"
        nxt = f'<a href="{names[i+1]}">next &rarr;</a>' if i < len(results) - 1 else "<span></span>"
        nav = f'<nav class="page-nav">{prev}<a href="index.html">index</a>{nxt}</nav>'
        html_doc = _HTML_SHELL.format(title=f"{title_prefix} {r.page}", css_href=css_href,
                                      nav=nav, content=to_html(r))
        with open(os.path.join(out_dir, names[i]), "w") as f:
            f.write(html_doc)

    index_items = "".join(f'<li><a href="{n}">{title_prefix} {r.page}</a></li>'
                          for n, r in zip(names, results))
    index_html = _HTML_SHELL.format(title=f"{title_prefix}s", css_href=css_href,
                                    nav="", content=f"<h1>{title_prefix}s</h1><ol>{index_items}</ol>")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(index_html)
    return names


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
