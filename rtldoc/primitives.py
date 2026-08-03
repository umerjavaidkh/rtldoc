"""
Layer 1 -- primitive extraction.

The single most important design decision in this parser: on a born-digital
PDF we never OCR. The glyphs, their exact bounding boxes, their font, size,
weight and colour are all sitting in the content stream at infinite
resolution. OCR throws that away and re-derives a lossy approximation.

We pull three primitive streams:
  * spans    -- text runs with geometry + full style signature
  * fills    -- vector rectangles (the coloured panels and numbered chips
                that an InDesign-authored textbook uses as its real,
                author-intended semantic containers)
  * images   -- placed raster art
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Iterable

import fitz  # PyMuPDF

from .arabic import deshape

Rect = tuple[float, float, float, float]


def _iou(a: Rect, b: Rect) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def containment(inner: Rect, outer: Rect) -> float:
    """Fraction of `inner`'s area that falls inside `outer`."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area else 0.0


@dataclass
class Span:
    text: str
    bbox: Rect
    font: str
    size: float
    color: int
    flags: int
    dir: tuple[float, float] = (1.0, 0.0)

    @property
    def bold(self) -> bool:
        return bool(self.flags & 2 ** 4) or "bold" in self.font.lower()

    @property
    def italic(self) -> bool:
        return bool(self.flags & 2 ** 1) or "italic" in self.font.lower()

    @property
    def style_key(self) -> str:
        """Stable style signature. This is the backbone of semantic typing.

        A single textbook series has on the order of a dozen distinct
        (font, size, colour, weight) combinations. Label them once, apply
        them to every page in the series -- deterministically, with no model
        and no per-page inference cost.
        """
        return f"{self.font}|{self.size:.1f}|{self.color:06x}|{'B' if self.bold else ''}{'I' if self.italic else ''}"


@dataclass
class Fill:
    """A filled vector shape -- the author's own structural annotation."""
    bbox: Rect
    color: tuple[float, float, float]
    area: float = 0.0
    is_chip: bool = False        # small square -> activity number badge
    is_panel: bool = False       # large block -> content container
    is_rule: bool = False        # thin -> separator line
    is_stroke: bool = False      # drawn as a stroked outline, not a fill
    # Endpoints of the drawing's own path, not just its bounding rect --
    # needed to tell a flowchart's connecting line apart from its boxes
    # (both reduce to the same kind of bbox otherwise) and to find which
    # two shapes a line actually connects. None for filled rects, where
    # the bbox alone is the shape.
    points: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self):
        w, h = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        self.area = w * h
        ar = w / h if h else 999
        self.is_rule = min(w, h) < 3 and max(w, h) > 20
        self.is_chip = (not self.is_rule) and 0.55 < ar < 1.8 and 40 < self.area < 900
        self.is_panel = (not self.is_rule) and self.area >= 900


@dataclass
class ImageRef:
    """A placed raster image: its position plus the xref needed to pull the
    actual bytes back out of the PDF later (see pipeline.save_images)."""
    bbox: Rect
    xref: int


@dataclass
class PagePrimitives:
    number: int
    width: float
    height: float
    spans: list[Span] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(s.text.strip()) for s in self.spans)

    @property
    def is_born_digital(self) -> bool:
        """Route: real text layer, or does this page need OCR?"""
        return self.char_count > 60


def _drawing_points(d: dict) -> tuple[tuple[float, float], ...] | None:
    """Flatten a get_drawings() entry's own path segments into plain (x, y)
    points -- the bounding rect alone can't tell a diagram's connecting
    line apart from its boxes (both reduce to the same kind of rect), and
    can't say which two shapes a line actually joins."""
    pts: list[tuple[float, float]] = []
    for item in d.get("items", []):
        op = item[0]
        if op == "l":
            pts.append((item[1].x, item[1].y))
            pts.append((item[2].x, item[2].y))
        elif op == "re":
            r = item[1]
            pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
        else:
            for p in item[1:]:
                if hasattr(p, "x"):
                    pts.append((p.x, p.y))
    return tuple(pts) if pts else None


def _near_white(rgb: tuple[float, float, float], tol: float = 0.04) -> bool:
    return all(c > 1 - tol for c in rgb)


# PRESERVE_LIGATURES is essential, not cosmetic. If MuPDF expands a lam-alef
# glyph into its two components *before* applying bidi, the components get
# reversed independently and every word containing لا comes out as ال.
# Keeping the ligature atomic through reordering and decomposing it ourselves
# afterwards is the only correct sequence.
#
# We ask for "rawdict" (per-character), not "dict" (per-span), because
# geobidi needs the per-character boxes anyway and rawdict is a strict
# superset -- extracting it once and deriving spans from it here saves a
# second full native text-extraction pass over the page (~30ms), which
# profiling showed to be ~35% of total per-page time.
_RAWDICT_FLAGS = fitz.TEXTFLAGS_RAWDICT | fitz.TEXT_PRESERVE_LIGATURES


_SPACE_GLYPH_CACHE_ATTR = "_rtldoc_space_glyphs"


def _document_space_glyphs(doc: "fitz.Document", min_samples: int = 5,
                            space_frac_thresh: float = 0.6) -> frozenset:
    """Which (font, glyph-id) pairs are, by consensus, a whitespace glyph --
    tallied across the WHOLE document, not one page, and cached on the
    document object itself (fitz.Document can't be weakly referenced, so a
    WeakKeyDictionary cache raises on every lookup; a plain dict keyed by
    id(doc) risks a stale hit if a garbage-collected Document's id is
    reused by a later one -- an ordinary attribute on the object side-steps
    both and is freed automatically with the document).

    A font's ToUnicode corruption is a property of the embedded font itself,
    consistent across every page that uses it -- so the vote must be too.
    Tallying per page instead is fragile: confirmed real case, the exact
    same (font, glyph) pair that reads as whitespace 81% of the time across
    a whole document dipped to 57% on one specific page purely from small-
    sample noise (that page happened to have a locally worse ratio), missing
    a 0.6 per-page threshold that the document-wide signal clears easily.
    Cached since this is the only part that needs every page's texttrace --
    expensive to redo per page on a multi-thousand-page document, but a
    one-time cost per document.
    """
    cached = getattr(doc, _SPACE_GLYPH_CACHE_ATTR, None)
    if cached is not None:
        return cached

    tallies: dict[tuple[str, int], list[int]] = {}
    for page in doc:
        try:
            trace = page.get_texttrace()
        except Exception:
            continue
        for span in trace:
            font = span.get("font", "")
            for code, glyph, _origin, _bbox in span.get("chars", []):
                key = (font, glyph)
                t = tallies.setdefault(key, [0, 0])
                t[1] += 1
                if chr(code).isspace():
                    t[0] += 1

    space_glyphs = frozenset(
        k for k, (sp, tot) in tallies.items()
        if tot >= min_samples and sp / tot >= space_frac_thresh
    )
    setattr(doc, _SPACE_GLYPH_CACHE_ATTR, space_glyphs)
    return space_glyphs


def _fix_broken_space_glyphs(page: "fitz.Page", raw: dict) -> dict:
    """Repair space glyphs whose ToUnicode CMap is internally inconsistent.

    A glyph ID is a fixed visual shape within one embedded font -- it cannot
    legitimately mean whitespace in one occurrence and a printable character
    in another. Confirmed real case: an embedded Arabic font's blank-space
    glyph was mapped by its own (corrupt) ToUnicode CMap to a plain space
    most of the time, but to the digit '1' or an en-space in the rest,
    purely depending on which CID subrange the subsetting tool happened to
    consult -- rendering shows nothing there in every case, but the minority
    reading silently injected a literal '1' into the extracted text between
    nearly every word on the affected pages. Also confirmed on a *different*
    document with a *different* font ("LiberationSans"), confirming this is
    a general PDF-authoring-tool defect, not one file's quirk.

    General fix: a document-wide consensus (see _document_space_glyphs)
    decides which (font, glyph-id) pairs are genuinely whitespace; any
    occurrence of one of those pairs that doesn't itself read as whitespace
    is the corrupted minority and gets corrected to a plain space. Requires
    get_texttrace() (glyph IDs) since rawdict alone only exposes the
    ToUnicode text, not the underlying glyph identity; occurrences are
    matched back into `raw` by (font, origin) since both calls resolve the
    same underlying glyph run. A real digit '1' glyph is untouched: it has
    its own distinct glyph ID with its own (non-whitespace) tally.
    """
    try:
        doc = page.parent
        if doc is None:
            raise ValueError
        space_glyphs = _document_space_glyphs(doc)
        if not space_glyphs:
            return raw
        trace = page.get_texttrace()
    except Exception:
        return raw

    bad_positions = {
        (span.get("font", ""), round(origin[0], 1), round(origin[1], 1))
        for span in trace
        for code, glyph, origin, _bbox in span.get("chars", [])
        if (span.get("font", ""), glyph) in space_glyphs and not chr(code).isspace()
    }
    if not bad_positions:
        return raw

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                for ch in span.get("chars", []):
                    o = ch.get("origin")
                    if o is None:
                        continue
                    if (font, round(o[0], 1), round(o[1], 1)) in bad_positions:
                        ch["c"] = " "
    return raw


# Symbol/dingbat fonts whose character codes are pictographic, NOT the
# Latin letters those codes nominally are. When such a font lacks a proper
# ToUnicode map, extraction falls back to the raw code and emits a spurious
# Latin letter -- e.g. ZapfDingbats code 0x49 (a filled bar/box glyph, used
# as an overline for x-bar, as an emoji stand-in, or as a currency mark)
# comes out as the letter "I", injecting garbage into the middle of words
# and math ("compute x each time" -> "compute xI each time", "IIII" for a
# run of emoji). Confirmed across three independent documents.
_SYMBOL_FONTS = ("dingbat", "wingding", "webding")


def _drop_symbol_font_letters(raw: dict) -> dict:
    """Remove characters that a symbol/dingbat font emitted as bare ASCII
    letters -- those are pictographs mis-decoded as text, never real
    content. Scoped tightly: only ASCII letters (A-Z, a-z) from a known
    symbol font are dropped, so a symbol font's legitimately-mapped Unicode
    (Greek/math glyphs like the Symbol font's U+221A square root) is
    untouched."""
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "").lower()
                if not any(sf in font for sf in _SYMBOL_FONTS):
                    continue
                chars = span.get("chars")
                if not chars:
                    continue
                span["chars"] = [ch for ch in chars if not ch.get("c", "").isascii()
                                 or not ch.get("c", "").isalpha()]
    return raw


def rawdict(page: "fitz.Page") -> dict:
    raw = page.get_text("rawdict", flags=_RAWDICT_FLAGS)
    return _drop_symbol_font_letters(_fix_broken_space_glyphs(page, raw))


def _block_text(block: dict) -> str:
    return "".join(ch["c"] for ln in block.get("lines", []) for sp in ln.get("spans", []) for ch in sp.get("chars", []))


def _bbox_iou(a: Rect, b: Rect) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def _char_bag_similarity(a: str, b: str) -> float:
    """Jaccard overlap of two strings' letter multisets, after deshaping.

    Order-independent on purpose: a downstream reconstruction step can
    legitimately scramble one copy's word order (see dedupe_duplicate_blocks)
    without changing what letters are actually present, so comparing bags
    of letters survives that scrambling where a literal string comparison
    would not. Presentation-form glyphs are folded to base letters first so
    two copies that differ only in font/shaping still compare equal.
    """
    na = Counter(c for c in deshape(a) if c.isalpha())
    nb = Counter(c for c in deshape(b) if c.isalpha())
    if not na or not nb:
        return 0.0
    inter = sum((na & nb).values())
    union = sum((na | nb).values())
    return inter / union if union else 0.0


def _center(bbox: Rect) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def dedupe_duplicate_blocks(raw: dict, iou_thresh: float = 0.5, sim_thresh: float = 0.92,
                             shift_tol: float = 10.0) -> dict:
    """Drop text blocks that duplicate another block's content.

    Some PDFs contain genuinely duplicated draw operations for the same
    visual element: a page carried two overlapping copies of a small
    running-header (a "unit" label and, right next to it, a "lesson"
    label), alongside byte-identical background-fill rectangles at the
    same positions -- only one copy was ever visually apparent; the file's
    own content stream is redundant, not our extraction.

    Two techniques are combined because neither alone covers real pages:

    1. Direct pairwise match: two blocks whose bboxes overlap (IoU) and
       whose letter-bags are near-identical are a confirmed duplicate pair.
       This catches the common case but misses it when a duplicated label
       is itself split into multiple blocks with looser overlap (label
       block widths/kerning differ slightly between the two copies, so a
       downstream block in one copy can drift far enough that its IoU with
       its counterpart falls under threshold even though the copy as a
       whole is clearly a duplicate).

    2. Shift-consistency extension, the same principle used in copy-move
       (duplicated-region) forgery detection: once a confirmed pair
       establishes the translation vector between "original" and "copy"
       (e.g. the copy sits ~23pt to the right), any OTHER pair of blocks on
       the page with matching letter-bags AND a center-to-center offset
       within `shift_tol` of that same vector is confirmed too, even if
       their raw IoU is low or zero. A pure proximity/clustering merge was
       tried and rejected: on the page that surfaced this bug, the two
       *different* labels within one copy sit closer to each other (~20pt)
       than is safe to treat as "same visual unit", so naive distance-based
       grouping merged unrelated labels together. Requiring a *specific,
       already-confirmed* shift (not just "nearby") avoids that failure
       while still catching the split-label case.

    Between two duplicates, the one LATER in content-stream order is kept:
    PDF rendering is a painter's algorithm, so a later draw sits on top and
    is what a reader actually sees (confirmed against the render for the
    case that surfaced this). Applied once here, on the raw dict shared by
    both primitives.extract_page (spans) and geobidi.page_lines (glyphs),
    so neither consumer can see the duplicate and neither needs its own
    copy of this logic.

    Both thresholds were originally far looser (iou>=0.2, sim>=0.6) and
    silently deleted ~half the body text on ordinary paragraph pages: two
    consecutive lines of unrelated prose routinely clear 0.2 IoU (ascender/
    descender bbox overlap between tight line-spacing is normal typesetting,
    not duplication), and _char_bag_similarity is a letter-frequency measure
    -- any two lines of the same-language text share 50-77% of their letter
    bag purely from common-letter statistics, independent of actual words
    (confirmed on book/sample-500kb.pdf p2 and 8 pages of book/BilArabi_TG07.pdf,
    where unrelated sentence pairs scored iou 0.15-0.26/sim 0.51-0.74). Genuine
    duplicate content -- verified on BilArabi_TG07.pdf p53's actual repeated
    glyphs -- scores iou>=0.999/sim==1.0, since it's the same content stream
    bytes painted twice. The large gap between those two clusters is why the
    thresholds now sit at 0.5/0.92 rather than somewhere in between: false
    positives top out well below 0.3 IoU and 0.8 similarity in every case
    found so far, so there's wide margin without risking the true-positive case.
    """
    all_blocks = raw.get("blocks", [])
    blocks = [b for b in all_blocks if b.get("type") == 0]
    others = [b for b in all_blocks if b.get("type") != 0]
    if len(blocks) < 2:
        return raw

    texts = [_block_text(b) for b in blocks]
    bboxes = [tuple(b["bbox"]) for b in blocks]
    centers = [_center(bb) for bb in bboxes]
    valid = [len(t.strip()) >= 2 for t in texts]
    n = len(blocks)

    def sim(i: int, j: int) -> float:
        return _char_bag_similarity(texts[i], texts[j])

    drop: set[int] = set()
    matched: set[int] = set()
    shifts: list[tuple[float, float]] = []

    # Pass 1: strong pairwise anchors -- geometric overlap + content match.
    for i in range(n):
        if i in matched or not valid[i]:
            continue
        for j in range(i + 1, n):
            if j in matched or not valid[j]:
                continue
            if _bbox_iou(bboxes[i], bboxes[j]) < iou_thresh:
                continue
            if sim(i, j) >= sim_thresh:
                drop.add(i)  # i < j: i is earlier in paint order, j sits on top
                matched.update((i, j))
                shifts.append((centers[j][0] - centers[i][0], centers[j][1] - centers[i][1]))
                break

    # Pass 2: extend via shift-consistency for blocks pass 1 didn't resolve
    # (e.g. a duplicated label split across blocks whose individual IoU is
    # too low, but whose offset matches an already-confirmed duplicate's).
    if shifts:
        remaining = [i for i in range(n) if valid[i] and i not in matched]
        for a in range(len(remaining)):
            i = remaining[a]
            if i in matched:
                continue
            for b in range(a + 1, len(remaining)):
                j = remaining[b]
                if j in matched:
                    continue
                if sim(i, j) < sim_thresh:
                    continue
                dx, dy = centers[j][0] - centers[i][0], centers[j][1] - centers[i][1]
                if any(abs(dx - sx) <= shift_tol and abs(dy - sy) <= shift_tol for sx, sy in shifts):
                    drop.add(i)
                    matched.update((i, j))
                    break

    kept = [b for i, b in enumerate(blocks) if i not in drop]
    out = dict(raw)
    out["blocks"] = kept + others
    return out


def extract_page(page: "fitz.Page", drop_white_fills: bool = True,
                 drop_full_page_frac: float = 0.6, raw: dict | None = None) -> PagePrimitives:
    prim = PagePrimitives(
        number=page.number + 1,
        width=page.rect.width,
        height=page.rect.height,
    )

    if raw is None:
        raw = rawdict(page)
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                # rawdict carries chars, not a joined "text" -- rebuild it.
                text = "".join(ch["c"] for ch in span.get("chars", []))
                if not text.strip():
                    continue
                # A zero-width span containing only Arabic diacritics
                # (tashkeel) is a redundant echo, not real content: some
                # justified-Arabic typesetting draws a diacritic twice --
                # once properly combined into its word's own span (which
                # already carries it correctly), once again as a separate,
                # zero-width mark positioned exactly on top of the letter
                # for fine placement. Confirmed real case: the word's own
                # span already read correctly fully vocalized ("بسّامُ فتًى
                # ...عامًا"), while these extra echoes had no base letter of
                # their own at all -- left in, they become their own
                # orphaned one-character "paragraph" blocks (and, worse,
                # can get picked up as a bogus figure caption) with zero
                # added information.
                bbox = span["bbox"]
                if bbox[0] == bbox[2] and all(0x064B <= ord(c) <= 0x065F for c in text.strip()):
                    continue
                prim.spans.append(
                    Span(
                        text=text,
                        bbox=tuple(span["bbox"]),
                        font=span["font"],
                        size=span["size"],
                        color=span["color"],
                        flags=span["flags"],
                        dir=tuple(line.get("dir", (1.0, 0.0))),
                    )
                )

    for d in page.get_drawings():
        dtype = d.get("type")
        if dtype not in ("f", "fs", "s"):
            continue
        r = d["rect"]
        if r.is_infinite:
            continue
        # A stroked line legitimately has zero width *or* height -- that's
        # not "empty", that's a line. Only reject it if it has neither
        # dimension (a degenerate point). Filled shapes still need the
        # stricter check: a truly empty fill rect isn't a real shape.
        # "fs" (fill+stroke) gets the SAME exemption as pure "s": a thin
        # connector is routinely drawn as a fill+stroke path with a
        # degenerate (zero-width or zero-height) rect rather than a
        # stroke-only one -- confirmed real case: a flowchart's own
        # connecting lines (drawn this way) were silently dropped as
        # "empty fills" below, even though a PDF renderer draws the stroke
        # portion regardless of the fill area being zero, leaving those
        # connectors invisible to this extractor while still visibly
        # rendered on screen.
        is_degenerate_line = dtype in ("s", "fs") and (r.width < 1e-6 or r.height < 1e-6)
        if dtype in ("s", "fs"):
            if r.width < 1e-6 and r.height < 1e-6:
                continue
        elif r.is_empty:
            continue
        # Table grids are very often drawn as stroked lines (type "s"), not
        # filled rectangles -- the stroke colour is the one that matters for
        # them, since a stroke-only path has no fill at all. A degenerate
        # "fs" line is the same story: its "fill" is meaningless (there is
        # no area to fill), so the STROKE colour is what's actually visible
        # -- using the fill colour there would routinely pick up incidental
        # white and get the whole line dropped by the white-fill filter
        # below (confirmed real case: the degenerate connector above had
        # fill=white/stroke=dark-gray; using fill would've silently
        # discarded a fully visible dark line as if it were blank).
        rgb = d.get("color") if (dtype == "s" or is_degenerate_line) else d.get("fill")
        if rgb is None:
            continue
        if drop_white_fills and _near_white(rgb):
            # A flowchart node is routinely drawn white-filled with a dark
            # (or otherwise non-white) BORDER -- "fs" with a real, visibly
            # non-white stroke is a genuine bordered box even though its
            # own fill happens to be white, and dropping it here treats a
            # perfectly visible shape as if it were blank (confirmed real
            # case: an org-chart diagram's ~15 node boxes were ALL drawn
            # this way -- white fill, dark stroke, real 2D area, not
            # degenerate -- and every one of them vanished from the
            # extracted geometry, leaving only the 2 shapes that happened
            # to use a non-white fill instead). A genuine blank background
            # tint (no meaningful border, or a border that's ALSO
            # near-white) still gets dropped exactly as before.
            stroke_rgb = d.get("color") if dtype == "fs" else None
            if stroke_rgb is not None and not _near_white(stroke_rgb):
                rgb = stroke_rgb
            else:
                continue
        # A full-page background tint (sometimes drawn with bleed, extending
        # past the crop box entirely) is not a semantic container -- a real
        # content panel (a reading-passage callout, an answer-key box) never
        # covers the majority of the page. Left in, it swallows nearly every
        # span on the page into one "panel" and destroys reading order.
        if (r.x1 - r.x0) * (r.y1 - r.y0) >= drop_full_page_frac * prim.width * prim.height:
            continue
        points = _drawing_points(d)
        prim.fills.append(Fill(bbox=(r.x0, r.y0, r.x1, r.y1), color=tuple(rgb),
                               is_stroke=(dtype == "s" or is_degenerate_line), points=points))

    for info in page.get_images(full=True):
        xref = info[0]
        try:
            for r in page.get_image_rects(xref):
                prim.images.append(ImageRef(bbox=(r.x0, r.y0, r.x1, r.y1), xref=xref))
        except Exception:
            continue

    return prim


def extract_document(path: str, pages: Iterable[int] | None = None) -> list[PagePrimitives]:
    doc = fitz.open(path)
    idx = range(doc.page_count) if pages is None else pages
    out = [extract_page(doc[i]) for i in idx]
    doc.close()
    return out


def style_profile(prims: list[PagePrimitives]) -> list[dict]:
    """Corpus-wide style census. Run once, label once, reuse forever."""
    census: dict[str, dict] = {}
    for p in prims:
        for s in p.spans:
            e = census.setdefault(
                s.style_key,
                {"style": s.style_key, "font": s.font, "size": round(s.size, 1),
                 "color": f"#{s.color:06x}", "bold": s.bold, "spans": 0,
                 "chars": 0, "sample": ""},
            )
            e["spans"] += 1
            e["chars"] += len(s.text)
            if len(e["sample"]) < 40:
                from .arabic import deshape
                e["sample"] = (e["sample"] + " " + deshape(s.text).strip())[:60]
    return sorted(census.values(), key=lambda e: -e["chars"])
