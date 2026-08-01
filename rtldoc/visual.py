"""
Layer 4 -- page visual summary (description without vision).

Every fact here comes from exact geometry already sitting in the PDF's own
content stream -- positions, sizes, drawn shapes, pixel colors -- turned
into structured data and a readable natural-language description. No
model, no API call, no new required dependency (PyMuPDF + numpy only,
both already runtime dependencies).

This deliberately cannot say what a photo *depicts* ("a boy holding a
phone") -- that needs a vision model, which is out of scope here by
design, not oversight. What it can do: exact position/size/color for every
image, and for vector diagrams (flowcharts, boxes-and-arrows) specifically,
a structural reconstruction -- shapes, their text labels, and which shapes
are connected -- since PDF drawings carry that geometry exactly, and
rtldoc already reads it for tables via the same union-find clustering
pattern (see layout._cluster_rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .primitives import Fill, PagePrimitives, Rect, containment


@dataclass
class ImageVisual:
    bbox: Rect
    width_px: int
    height_px: int
    dominant_colors: list[str]
    kind: str  # "photo" | "illustration" | "scan"


@dataclass
class DiagramShape:
    kind: str  # "box" | "line"
    bbox: Rect
    color: tuple[float, float, float]
    label: str = ""


@dataclass
class DiagramVisual:
    bbox: Rect
    shapes: list[DiagramShape] = field(default_factory=list)
    connections: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class TableVisual:
    bbox: Rect
    rows: int
    cols: int
    headers: list[str] = field(default_factory=list)


@dataclass
class PageAppearance:
    columns: int
    num_images: int
    num_diagrams: int
    num_tables: int
    num_headings: int
    dominant_panel_colors: list[str] = field(default_factory=list)


@dataclass
class PageVisual:
    images: list[ImageVisual] = field(default_factory=list)
    diagrams: list[DiagramVisual] = field(default_factory=list)
    tables: list[TableVisual] = field(default_factory=list)
    appearance: PageAppearance | None = None
    description: str = ""


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c * 255))) for c in rgb)


def _pixmap_colors(pix: "object", k: int = 3) -> tuple[list[str], float]:
    """Dominant colors and a 0-1 "color diversity" score from a rendered
    pixmap's own samples -- no PIL, no new dependency, just numpy on the
    raw bytes PyMuPDF already gives us. Diversity is the fraction of
    quantized color buckets that are actually distinct: a flat-color
    illustration or line-art diagram has very few; a photograph, with its
    gradients and noise, has many.
    """
    n = pix.n - (1 if pix.alpha else 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    arr = arr[:, :, :3] if n >= 3 else np.repeat(arr[:, :, :1], 3, axis=2)
    step_y, step_x = max(1, arr.shape[0] // 48), max(1, arr.shape[1] // 48)
    small = arr[::step_y, ::step_x].reshape(-1, 3)
    if small.size == 0:
        return [], 0.0
    quantized = (small // 32) * 32 + 16
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(-counts)[:k]
    top = [_hex(tuple(float(c) / 255 for c in colors[i])) for i in order]
    diversity = len(colors) / max(1, small.shape[0] // 8)
    return top, min(1.0, diversity)


def describe_image(page: "object", bbox: Rect, zoom: float = 1.0) -> ImageVisual:
    """Render just this figure's own region to get exact pixel dimensions
    and color stats -- cheaper and more precise than parsing the original
    embedded image's own encoding/colorspace by hand."""
    import fitz
    pix = page.get_pixmap(clip=fitz.Rect(*bbox), matrix=fitz.Matrix(zoom, zoom))
    colors, diversity = _pixmap_colors(pix)
    kind = "photo" if diversity > 0.35 else "illustration"
    return ImageVisual(bbox=bbox, width_px=pix.width, height_px=pix.height,
                       dominant_colors=colors, kind=kind)


def _is_box(f: Fill) -> bool:
    """Does this stroked shape's own path reduce to (approximately) its
    own bbox's four corners -- i.e. is it a rectangle, not a connecting
    line? A flowchart node is drawn as a rect; a connector (even an
    elbowed, multi-segment one) traces a path that doesn't match its own
    bbox corners this way."""
    if not f.points:
        return False
    x0, y0, x1, y1 = f.bbox
    corners = {(round(x0, 1), round(y0, 1)), (round(x1, 1), round(y0, 1)),
              (round(x1, 1), round(y1, 1)), (round(x0, 1), round(y1, 1))}
    pts = {(round(x, 1), round(y, 1)) for x, y in f.points}
    return pts <= corners


def _nearest_label(bbox: Rect, spans, max_dist: float = 60.0) -> str:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    best, best_d = "", max_dist
    for s in spans:
        if not s.text.strip():
            continue
        sx, sy = (s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2
        d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = s.text.strip(), d
    return best


def detect_diagrams(prim: PagePrimitives, claimed: list[Rect], min_shapes: int = 3,
                    pad: float = 10.0) -> list[DiagramVisual]:
    """Recover a flowchart/diagram's structure -- boxes, their labels, and
    which shapes a connecting line joins -- from the page's own stroked
    vector drawings. Table grids and ordinary panels are excluded via
    `claimed` (their regions), so a real table's border lines are never
    double-counted as a diagram.
    """
    stroke_fills = [f for f in prim.fills if f.is_stroke and f.points
                   and not any(containment(f.bbox, c) > 0.5 for c in claimed)]
    if len(stroke_fills) < min_shapes:
        return []

    # cluster nearby stroked shapes (same union-find pattern as
    # layout._cluster_rules / _cluster_images)
    n = len(stroke_fills)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def close(a: Rect, b: Rect) -> bool:
        return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])

    for i in range(n):
        for j in range(i + 1, n):
            if close(stroke_fills[i].bbox, stroke_fills[j].bbox):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[Fill]] = {}
    for i, f in enumerate(stroke_fills):
        groups.setdefault(find(i), []).append(f)

    out = []
    for members in groups.values():
        if len(members) < min_shapes:
            continue
        all_boxes = [f for f in members if _is_box(f)]
        # A box that contains two or more of the *other* boxes is the
        # diagram's own outer frame/border, not a real node -- confirmed
        # real case: a flowchart's containing rectangle got labeled with
        # whatever text happened to sit nearest its (page-spanning)
        # centroid, and being a giant bbox, matched every line's endpoint
        # too, swallowing all real connection detection.
        boxes = [f for f in all_boxes
                if sum(containment(g.bbox, f.bbox) > 0.9 for g in all_boxes if g is not f) < 2]
        lines = [f for f in members if f not in all_boxes]
        if len(boxes) < 2:
            continue
        shapes = []
        for f in boxes:
            label = _nearest_label(f.bbox, prim.spans)
            shapes.append(DiagramShape(kind="box", bbox=f.bbox, color=f.color, label=label))
        for f in lines:
            shapes.append(DiagramShape(kind="line", bbox=f.bbox, color=f.color))

        # infer connections: a line's own endpoints, each close to some
        # box's own boundary, mean that line joins those boxes
        connections: set[tuple[str, str]] = set()
        for f in lines:
            touched = []
            for x, y in f.points:
                for shp in shapes:
                    if shp.kind != "box":
                        continue
                    bx0, by0, bx1, by1 = shp.bbox
                    if bx0 - pad <= x <= bx1 + pad and by0 - pad <= y <= by1 + pad:
                        touched.append(shp.label or f"box@{round(bx0)},{round(by0)}")
                        break
            for a, b in zip(sorted(set(touched)), sorted(set(touched))[1:]):
                if a != b:
                    connections.add((a, b))

        x0 = min(f.bbox[0] for f in members); y0 = min(f.bbox[1] for f in members)
        x1 = max(f.bbox[2] for f in members); y1 = max(f.bbox[3] for f in members)
        out.append(DiagramVisual(bbox=(x0, y0, x1, y1), shapes=shapes,
                                 connections=sorted(connections)))
    return out


def describe_table(bbox: Rect, grid: list[list[str]]) -> TableVisual:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    headers = grid[0] if grid else []
    return TableVisual(bbox=bbox, rows=rows, cols=cols, headers=headers)


def _render_description(v: PageVisual) -> str:
    parts = []
    a = v.appearance
    if a is not None:
        layout = f"{a.columns}-column" if a.columns else "single-column"
        parts.append(f"This is a {layout} page.")
    if v.images:
        kinds = ", ".join(f"{i.width_px}×{i.height_px}px {i.kind}" for i in v.images)
        parts.append(f"It contains {len(v.images)} image(s): {kinds}.")
    if v.diagrams:
        for d in v.diagrams:
            boxes = [s.label or "unlabeled box" for s in d.shapes if s.kind == "box"]
            desc = f"A diagram with {len(boxes)} shapes ({', '.join(boxes)})"
            if d.connections:
                conns = "; ".join(f"{a} → {b}" for a, b in d.connections)
                desc += f", connected as: {conns}"
            parts.append(desc + ".")
    if v.tables:
        for t in v.tables:
            hdr = f", headers: {', '.join(t.headers)}" if t.headers else ""
            parts.append(f"A table with {t.rows} rows × {t.cols} columns{hdr}.")
    return " ".join(parts)


def describe_page(page: "object", prim: PagePrimitives, regions, table_grids: dict[int, list],
                  columns: int) -> PageVisual:
    """Tier-1 entry point: build a PageVisual purely from geometry already
    computed elsewhere in the pipeline (regions, table grids, column
    count) plus a fresh pass over drawings/images for diagram/image
    detection."""
    v = PageVisual()

    table_regions = [r for r in regions if r.kind == "table"]
    # Only tables are excluded here, not generic "panel" regions --
    # propose_regions already turns every is_panel Fill (including a
    # flowchart's own stroked boxes) into its own panel Region, so
    # excluding panels wholesale would exclude the diagram's own shapes
    # right along with them.
    claimed = [r.bbox for r in regions if r.kind == "table"]

    for r in table_regions:
        grid = table_grids.get(id(r))
        if grid:
            v.tables.append(describe_table(r.bbox, grid))

    for r in regions:
        if r.kind == "figure" and not r.composite and r.image_xref is None:
            continue
        if r.kind == "figure":
            try:
                v.images.append(describe_image(page, r.bbox))
            except Exception:
                continue

    v.diagrams = detect_diagrams(prim, claimed)

    panel_colors = sorted({_hex(f.color) for f in prim.fills if f.is_panel and not f.is_stroke})
    v.appearance = PageAppearance(
        columns=columns, num_images=len(v.images), num_diagrams=len(v.diagrams),
        num_tables=len(v.tables),
        num_headings=sum(1 for r in regions if r.kind == "panel"),
        dominant_panel_colors=panel_colors[:5],
    )
    v.description = _render_description(v)
    return v
