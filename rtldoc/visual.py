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
    # A real box has genuine 2D extent. A connector line has zero width or
    # height, which collapses its four bbox "corners" onto its two
    # endpoints -- so a line would otherwise match the corner test and be
    # miscounted as a box (confirmed real case: a flowchart's horizontal
    # and vertical connector lines all counted as boxes, leaving nothing
    # as a line to detect connections from).
    if min(x1 - x0, y1 - y0) < 3:
        return False
    corners = {(round(x0, 1), round(y0, 1)), (round(x1, 1), round(y0, 1)),
              (round(x1, 1), round(y1, 1)), (round(x0, 1), round(y1, 1))}
    pts = {(round(x, 1), round(y, 1)) for x, y in f.points}
    return pts <= corners


def _nearest_label(bbox: Rect, spans, max_dist: float = 60.0) -> str:
    # A box's own label routinely wraps across 2+ lines ("Content" /
    # "stream") -- picking only the SINGLE nearest span truncates it to
    # whichever line happens to sit closest to the box's centroid
    # (confirmed real case: an org-chart's boxes came out labeled "entry",
    # "stream", "destinations", "threads" -- each missing its own first
    # word). Every span whose own center falls inside the box belongs to
    # its label; join them in reading order.
    x0, y0, x1, y1 = bbox
    pad = 2.0
    inside = [s for s in spans if s.text.strip()
             and x0 - pad <= (s.bbox[0] + s.bbox[2]) / 2 <= x1 + pad
             and y0 - pad <= (s.bbox[1] + s.bbox[3]) / 2 <= y1 + pad]
    if inside:
        inside.sort(key=lambda s: (round(s.bbox[1]), s.bbox[0]))
        return " ".join(s.text.strip() for s in inside)
    # No text sits inside this shape at all (an unboxed diagram node, or a
    # connector with a nearby external label) -- fall back to the single
    # closest span outside it.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
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
    # Node/connector candidates: a flowchart node is either a stroked box
    # (outline only) OR a filled colored panel (confirmed real cases of
    # both -- one document draws its boxes as outlines, another fills them
    # with pale colors); a connector is a thin rule or stroked line. Skip
    # anything already claimed by a real table, and skip a near-full-page
    # fill (a background tint is not a diagram node).
    # is_rule fills are ALSO included -- a diagram's own boxes are routinely
    # drawn as thin rule-bordered rectangles (the exact same drawing style
    # tables use for their cell borders), not stroke-only outlines or
    # filled panels (confirmed real case: a public-key-encryption diagram's
    # boxes were all is_rule fills, and detect_diagrams found nothing at
    # all -- not even a wrong guess -- since none of its shapes passed the
    # is_stroke/is_panel filter). This is safe against misreading a real
    # TABLE as a diagram for two separate reasons: (1) `claimed` already
    # excludes any region the table detector -- itself extensively
    # validated -- already claimed; (2) _is_box below requires genuine 2D
    # extent (min(w,h) >= 3), which a table's rule segments essentially
    # never have on their own (a row/column divider is drawn as a single
    # degenerate thin line, not an enclosed rectangle) -- so a rule-bordered
    # TABLE's individual dividers still can't masquerade as diagram boxes.
    page_area = prim.width * prim.height
    stroke_fills = [f for f in prim.fills
                   if f.points and (f.is_stroke or f.is_panel or f.is_rule)
                   and f.area < page_area * 0.5
                   and not any(containment(f.bbox, c) > 0.5 for c in claimed)]
    if len(stroke_fills) < min_shapes:
        return []
    # The clustering below is O(n^2), and the connection-inference further
    # down is O(members^2 x points^2) within a cluster -- fine for a real
    # flowchart's handful of boxes/connectors, but a genuinely complex
    # vector graphic (a detailed map, chart, or technical illustration) can
    # carry 1000+ small stroked/filled paths on one page and take over a
    # minute to churn through, for a result that was never going to be a
    # real diagram anyway (confirmed real case: a WHO report page with
    # 1425 candidate shapes -- almost certainly a data map -- took 62s;
    # every genuine flowchart found so far, including a 17-box one, stayed
    # well under 200). Bailing out early here is the same "prefer under-
    # structuring" choice as the unlabeled-box guard below: a page this
    # dense was never going to render as a labeled node-and-edge diagram,
    # so there's nothing lost by skipping it, and a lot of time saved.
    if len(stroke_fills) > 200:
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

        # infer connections: a line's own points, each close to some box's
        # own boundary, mean that line joins those boxes.
        def _touched_boxes(f: Fill) -> list[tuple[str, Rect]]:
            touched: list[tuple[str, Rect]] = []
            seen: set[str] = set()
            for x, y in f.points:
                for shp in shapes:
                    if shp.kind != "box":
                        continue
                    bx0, by0, bx1, by1 = shp.bbox
                    if bx0 - pad <= x <= bx1 + pad and by0 - pad <= y <= by1 + pad:
                        key = shp.label or f"box@{round(bx0)},{round(by0)}"
                        if key not in seen:
                            seen.add(key)
                            touched.append((key, shp.bbox))
                        break
            return touched

        def _is_straight(f: Fill) -> bool:
            # A STRAIGHT path (its own drawn length barely exceeds the
            # straight-line distance between its first and last point) is
            # an unambiguous direct connector. A path that DETOURS well
            # beyond that (down, across, and back up again -- a bracket/
            # T-junction shape) only proves those boxes sit near a SHARED
            # trunk, not that they're directly linked to each other
            # (confirmed real case: a bracket-shaped connector down from
            # two side-by-side source boxes, across, and back up, was a
            # detour 70%+ longer than a direct line between them, and
            # asserting a direct edge from it invented a connection between
            # two boxes sharing no real edge at all -- while a DIFFERENT
            # diagram's simple straight same-row connectors, which must
            # still connect normally, need to survive this check).
            if len(f.points) < 2:
                return True
            path_len = sum(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                          for (x1, y1), (x2, y2) in zip(f.points, f.points[1:]))
            (sx, sy), (ex, ey) = f.points[0], f.points[-1]
            straight = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            return straight == 0 or path_len <= straight * 1.3

        def _point_seg_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
            dx, dy = x2 - x1, y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 == 0:
                return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len2))
            cx, cy = x1 + t * dx, y1 + t * dy
            return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

        def _lines_touch(f1: Fill, f2: Fill) -> bool:
            # A short connector's own ENDPOINT routinely lands partway ALONG
            # another line's segment (e.g. a bracket's flat horizontal run),
            # not exactly on one of that line's own corner points -- plain
            # point-to-point proximity misses this entirely (confirmed real
            # case: a vertical bridge segment's top endpoint sits at the
            # midpoint of a bracket's 190pt-long horizontal run, over 90pt
            # from either of the bracket's own corners, yet visibly meets it
            # dead center). Point-to-SEGMENT distance is what actually
            # captures "touches this line," not just "touches one of its
            # explicitly recorded vertices."
            for px, py in f1.points:
                for (x1, y1), (x2, y2) in zip(f2.points, f2.points[1:]):
                    if _point_seg_dist(px, py, x1, y1, x2, y2) <= pad:
                        return True
            return False

        touched_by_line = {id(f): _touched_boxes(f) for f in lines}
        connections: set[tuple[str, str]] = set()
        for f in lines:
            if not _is_straight(f):
                continue
            direct = touched_by_line[id(f)]
            # A short straight connector that only reaches ONE box directly
            # can still be the real bridge onward from a shared trunk/hub
            # it touches (a detour line excluded from direct pairing above,
            # or another straight segment) -- borrow whatever boxes THAT
            # other line touches, without pairing those borrowed boxes
            # with EACH OTHER (only this line's own direct box gets paired
            # with them): confirmed real case: a short vertical segment
            # from a bracket's own midpoint down into a single target box
            # is what actually carries "both siblings feed this box" --
            # the bracket alone (with no such bridging segment) correctly
            # stays silent about where its trunk goes.
            direct_labels = {label for label, _ in direct}
            borrowed_labels: set[str] = set()
            for g in lines:
                if g is f or not _lines_touch(f, g):
                    continue
                borrowed_labels.update(label for label, _ in touched_by_line[id(g)])
            borrowed_labels -= direct_labels
            for i in range(len(direct)):
                for j in range(i + 1, len(direct)):
                    a, _ = direct[i]
                    b, _ = direct[j]
                    if a != b:
                        connections.add(tuple(sorted((a, b))))
            for a in direct_labels:
                for b in borrowed_labels:
                    if a != b:
                        connections.add(tuple(sorted((a, b))))

        # Require at least one real LINE shape in this cluster (an actual
        # stroked connector, not just colored rectangles) -- this is the
        # signal that separates a genuine flowchart from a page that merely
        # has a couple of decorative colored panels near a separator rule.
        # Deliberately NOT requiring a successfully-inferred connection on
        # top of that: a real connector can exist in the PDF without this
        # detector being able to pin down exactly which boxes it joins
        # (confirmed real case above), and reporting "N shapes, connections
        # unknown" is still strictly more useful than dropping the whole
        # diagram back to unstructured text -- text content is never lost
        # either way, since nodes are always also captured as their own
        # ordinary passage blocks regardless of what this detector decides.
        if not lines:
            continue
        # A genuine flowchart node is drawn to be read, and virtually
        # always carries a real text label -- an "unlabeled box" is much
        # more likely a decorative UI element inside a stock illustration
        # (a browser-window mockup, a bar-chart icon) whose line-art
        # strokes happen to pass the box/line shape tests too, without
        # being a real diagram at all (confirmed real case: a marketing
        # slide's decorative infographic illustration produced 5
        # "unlabeled box" shapes, all pairwise "connected" to every other
        # one -- a pattern no genuine flowchart produces -- since its
        # decorative strokes touched multiple UI-mockup rectangles at
        # once). If NONE of this cluster's boxes have a real label,
        # there's nothing useful to report and a real risk of asserting a
        # fake structure, so skip it entirely.
        if not any(s.label for s in shapes if s.kind == "box"):
            continue
        x0 = min(f.bbox[0] for f in members); y0 = min(f.bbox[1] for f in members)
        x1 = max(f.bbox[2] for f in members); y1 = max(f.bbox[3] for f in members)
        out.append(DiagramVisual(bbox=(x0, y0, x1, y1), shapes=shapes,
                                 connections=sorted(connections)))
    return out


def render_diagram_mermaid(d: DiagramVisual) -> str:
    """Emit Mermaid flowchart syntax for a detected diagram -- an actual
    auto-laid-out chart from a real diagramming library (rendered
    client-side by Mermaid.js), not a pixel copy of the page and not a
    from-scratch coordinate reconstruction. Only the NODE LABELS and the
    detected CONNECTIONS feed this; Mermaid does its own layout entirely.
    """
    boxes = [s for s in d.shapes if s.kind == "box"]
    node_id = {}
    lines = ["flowchart TD"]
    for i, s in enumerate(boxes):
        node_id[s.label or f"box{i}"] = chr(ord("A") + i) if i < 26 else f"N{i}"
    for label, nid in node_id.items():
        safe = label.replace('"', "'") or "unlabeled"
        lines.append(f'    {nid}["{safe}"]')
    for a, b in d.connections:
        if a in node_id and b in node_id:
            lines.append(f"    {node_id[a]} --> {node_id[b]}")
    return "\n".join(lines)


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
            else:
                # A real connector line was found (that's what qualified
                # this as a diagram at all), but which boxes it joins
                # couldn't be pinned down -- saying so plainly beats
                # silently omitting the clause, which reads as "no
                # connections exist" rather than "not determined."
                desc += ", connections not determined"
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
