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

from dataclasses import dataclass, field, asdict
from typing import Iterable

import fitz  # PyMuPDF


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


def rawdict(page: "fitz.Page") -> dict:
    return page.get_text("rawdict", flags=_RAWDICT_FLAGS)


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
        # Table grids are very often drawn as stroked lines (type "s"), not
        # filled rectangles -- the stroke colour is the one that matters for
        # them, since a stroke-only path has no fill at all.
        rgb = d.get("color") if dtype == "s" else d.get("fill")
        if rgb is None:
            continue
        if drop_white_fills and _near_white(rgb):
            continue
        r = d["rect"]
        if r.is_infinite:
            continue
        # A stroked line legitimately has zero width *or* height -- that's
        # not "empty", that's a line. Only reject it if it has neither
        # dimension (a degenerate point). Filled shapes still need the
        # stricter check: a truly empty fill rect isn't a real shape.
        if dtype == "s":
            if r.width < 1e-6 and r.height < 1e-6:
                continue
        elif r.is_empty:
            continue
        # A full-page background tint (sometimes drawn with bleed, extending
        # past the crop box entirely) is not a semantic container -- a real
        # content panel (a reading-passage callout, an answer-key box) never
        # covers the majority of the page. Left in, it swallows nearly every
        # span on the page into one "panel" and destroys reading order.
        if (r.x1 - r.x0) * (r.y1 - r.y0) >= drop_full_page_frac * prim.width * prim.height:
            continue
        prim.fills.append(Fill(bbox=(r.x0, r.y0, r.x1, r.y1), color=tuple(rgb)))

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
