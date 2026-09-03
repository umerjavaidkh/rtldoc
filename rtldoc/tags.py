"""
Layer 1b -- the structure the file already states.

rtldoc's premise is that geometry beats vision. The corollary, easy to miss,
is that an *explicit declaration* beats geometry: when a PDF ships a tagged
structure tree, the table grid, the heading levels and the list nesting are
facts recorded by the authoring tool, not inferences drawn from ink. Inferring
them anyway is strictly worse -- it can only lose information.

Roughly a third of real-world PDFs carry usable tags (Word, Google Docs,
Adobe PDFMaker, InDesign accessibility exports, anything PDF/UA). On
`book/sample-tables.pdf` the file declares 28 tables with full TH/TD roles and
column spans; geometry alone recovers 18 of them and knows nothing about which
cells are headers.

Reading them takes two pieces, because the tree and the text live apart:

    /StructTreeRoot   the logical tree: Table > TR > TD, with /MCID leaves
    content stream    BDC /P <</MCID 4>> ... EMC around the glyphs

so this module walks the tree for structure, tokenizes the content stream for
the pen origin of every text op under each MCID, and joins the two on
position. PyMuPDF exposes no MCID of its own, which is why the tokenizer here
exists rather than deferring to it.

Everything is best-effort: a malformed tree degrades to "no tags found" and
the geometric path stands unchanged. This layer only ever *adds* certainty.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import fitz  # PyMuPDF

Matrix = tuple[float, float, float, float, float, float]
_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# a minimal PDF object reader, over the strings fitz hands back
# --------------------------------------------------------------------------

class Ref(int):
    """An indirect reference (`12 0 R`), kept distinct from a plain integer."""


_TOKEN = re.compile(rb"""
      (?P<ref>\d+\s+\d+\s+R\b)
    | (?P<real>[-+]?\d*\.\d+)
    | (?P<int>[-+]?\d+)
    | (?P<name>/[^\s/\[\]<>(){}]*)
    | (?P<dopen><<) | (?P<dclose>>>)
    | (?P<aopen>\[) | (?P<aclose>\])
    | (?P<hex><[0-9A-Fa-f\s]*>)
    | (?P<str>\((?:\\.|[^\\()])*\))
    | (?P<kw>[A-Za-z]+)
""", re.VERBOSE)


def _parse(data: bytes, pos: int = 0):
    """Parse one PDF object from `data` at `pos` -> (value, next_pos)."""
    m = _TOKEN.search(data, pos)
    if m is None:
        return None, len(data)
    k, raw = m.lastgroup, m.group()
    end = m.end()
    if k == "ref":
        return Ref(int(raw.split()[0])), end
    if k == "real":
        return float(raw), end
    if k == "int":
        return int(raw), end
    if k == "name":
        return raw.decode("latin-1"), end
    if k == "dopen":
        d, p = {}, end
        while True:
            m2 = _TOKEN.search(data, p)
            if m2 is None or m2.lastgroup == "dclose":
                return d, (m2.end() if m2 else len(data))
            if m2.lastgroup != "name":       # malformed -- skip the token
                p = m2.end()
                continue
            key = m2.group().decode("latin-1")
            val, p = _parse(data, m2.end())
            d[key] = val
    if k == "aopen":
        a, p = [], end
        while True:
            m2 = _TOKEN.search(data, p)
            if m2 is None or m2.lastgroup == "aclose":
                return a, (m2.end() if m2 else len(data))
            val, p = _parse(data, p)
            a.append(val)
    if k == "kw":
        return {b"true": True, b"false": False, b"null": None}.get(raw, raw.decode("latin-1")), end
    return raw, end


class Objects:
    """Cached xref -> parsed-object accessor."""

    def __init__(self, doc: "fitz.Document"):
        self.doc = doc
        self._cache: dict[int, object] = {}

    def get(self, xref: int):
        if xref in self._cache:
            return self._cache[xref]
        try:
            src = self.doc.xref_object(xref, compressed=True).encode("latin-1", "replace")
            val, _ = _parse(src)
        except Exception:
            val = None
        self._cache[xref] = val
        return val

    def resolve(self, v):
        """Follow indirect references (bounded, so a cyclic file cannot hang us)."""
        for _ in range(32):
            if not isinstance(v, Ref):
                return v
            v = self.get(int(v))
        return None


# --------------------------------------------------------------------------
# content stream -> where each MCID's glyphs were drawn
# --------------------------------------------------------------------------

_CS_TOKEN = re.compile(rb"""
      (?P<num>[-+]?[0-9]*\.?[0-9]+)
    | (?P<name>/[^\s/\[\]<>(){}]*)
    | (?P<str>\((?:\\.|[^\\()])*\))
    | (?P<hex><[0-9A-Fa-f\s]*>)
    | (?P<dopen><<) | (?P<dclose>>>)
    | (?P<aopen>\[) | (?P<aclose>\])
    | (?P<op>[A-Za-z'"*][A-Za-z0-9'"*]*)
""", re.VERBOSE)


def _mul(a: Matrix, b: Matrix) -> Matrix:
    return (a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5])


def _properties(page: "fitz.Page") -> dict[str, int]:
    """{/Pr1: mcid} for the BDC form that names a property list instead of inlining it."""
    out: dict[str, int] = {}
    try:
        kind, val = page.parent.xref_get_key(page.xref, "Resources/Properties")
        if kind != "dict":
            return out
        for m in re.finditer(r"(/[^\s/]+)\s+(\d+) 0 R", val):
            k = page.parent.xref_get_key(int(m.group(2)), "MCID")
            if k[0] == "int":
                out[m.group(1)] = int(k[1])
    except Exception:
        pass
    return out


def mcid_origins(page: "fitz.Page") -> list[tuple[int | None, str, tuple[float, float]]]:
    """(mcid, tag, page-space origin) for every text-showing operation.

    Only the *starting* pen position is tracked, never the advance, so no font
    metrics are needed: every producer we have seen re-positions with Td/Tm
    before each run, which is exactly the origin we want.
    """
    try:
        data = page.read_contents()
    except Exception:
        return []
    props = _properties(page)
    ctm: Matrix = _IDENTITY
    stack: list[Matrix] = []
    tm = tlm = _IDENTITY
    leading = 0.0
    marks: list[tuple[str, int | None]] = []
    out: list[tuple[int | None, str, tuple[float, float]]] = []
    operands: list = []

    for m in _CS_TOKEN.finditer(data):
        kind = m.lastgroup
        if kind != "op":
            operands.append(float(m.group()) if kind == "num" else m.group())
            continue
        op = m.group().decode("latin-1")
        a = operands
        operands = []
        try:
            if op == "q":
                stack.append(ctm)
            elif op == "Q":
                ctm = stack.pop() if stack else ctm
            elif op == "cm" and len(a) >= 6:
                ctm = _mul(tuple(a[-6:]), ctm)          # type: ignore[arg-type]
            elif op == "BT":
                tm = tlm = _IDENTITY
            elif op == "Tm" and len(a) >= 6:
                tm = tlm = tuple(a[-6:])                # type: ignore[assignment]
            elif op == "TL" and a:
                leading = a[-1]
            elif op in ("Td", "TD") and len(a) >= 2:
                if op == "TD":
                    leading = -a[-1]
                tlm = _mul((1, 0, 0, 1, a[-2], a[-1]), tlm)
                tm = tlm
            elif op == "T*":
                tlm = _mul((1, 0, 0, 1, 0, -leading), tlm)
                tm = tlm
            elif op == "BDC" and len(a) >= 2:
                tag = a[0].decode("latin-1") if isinstance(a[0], bytes) else str(a[0])
                mcid = None
                for i, t in enumerate(a):               # [/P, <<, /MCID, 5, >>]
                    if t == b"/MCID" and i + 1 < len(a) and isinstance(a[i + 1], float):
                        mcid = int(a[i + 1])
                        break
                if mcid is None and isinstance(a[-1], bytes) and a[-1].startswith(b"/"):
                    mcid = props.get(a[-1].decode("latin-1"))
                marks.append((tag.lstrip("/"), mcid))
            elif op == "BMC":
                marks.append((str(a[-1]) if a else "?", None))
            elif op == "EMC":
                if marks:
                    marks.pop()
            elif op in ("Tj", "TJ", "'", '"'):
                if op in ("'", '"'):
                    tlm = _mul((1, 0, 0, 1, 0, -leading), tlm)
                    tm = tlm
                cur = next(((t, i) for t, i in reversed(marks) if i is not None), ("", None))
                full = _mul(tm, ctm)
                out.append((cur[1], cur[0], (full[4], full[5])))
        except Exception:
            continue
    return out


def mcid_text(page: "fitz.Page", tol: float = 15.0) -> dict[int, str]:
    """{mcid: text} -- joins tokenizer origins to PyMuPDF spans by nearest origin.

    PyMuPDF decodes the glyphs correctly (encodings, ToUnicode, ligatures); the
    tokenizer knows only *where* each op started. Matching on origin lets each
    do the half it is good at.
    """
    pts = [(o[2][0], o[2][1], o[0]) for o in mcid_origins(page) if o[0] is not None]
    if not pts:
        return {}
    height = page.rect.height
    acc: dict[int, list[str]] = defaultdict(list)
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return {}
    for b in blocks:
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                ox, oy = span["origin"]
                best, best_d = None, 1e9
                for px, py, mid in pts:
                    # tokenizer works in PDF space (y up); get_text is y down
                    d = min(abs(px - ox) + abs(py - oy),
                            abs(px - ox) + abs((height - py) - oy))
                    if d < best_d:
                        best_d, best = d, mid
                if best is not None and best_d <= tol:
                    acc[best].append(span["text"])
    return {k: "".join(v) for k, v in acc.items()}


# --------------------------------------------------------------------------
# the structure tree
# --------------------------------------------------------------------------

@dataclass
class TaggedCell:
    kind: str                       # "TD" or "TH"
    text: str = ""
    colspan: int = 1
    rowspan: int = 1
    mcids: list[int] = field(default_factory=list)


@dataclass
class TaggedTable:
    page: int
    rows: list[list[TaggedCell]] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), max((len(r) for r in self.rows), default=0)


def _pageno(doc: "fitz.Document") -> dict[int, int]:
    """{page xref: page number} -- /Pg points at a page object, not an index."""
    return {doc[i].xref: i for i in range(doc.page_count)}


def tagged_tables(doc: "fitz.Document", resolve_text: bool = True) -> list[TaggedTable]:
    """Every /Table the file declares, with cell roles and spans.

    Returns [] for an untagged file -- callers keep their geometric path.
    """
    cat = doc.pdf_catalog()
    try:
        kind, val = doc.xref_get_key(cat, "StructTreeRoot")
    except Exception:
        return []
    if kind == "null":
        return []
    root_xref = int(val.split()[0]) if kind == "xref" else None
    if root_xref is None:
        return []

    objs = Objects(doc)
    pages = _pageno(doc)
    root = objs.get(root_xref)
    if not isinstance(root, dict):
        return []

    def kids(node) -> list:
        k = objs.resolve(node.get("/K")) if isinstance(node, dict) else None
        if k is None:
            return []
        return k if isinstance(k, list) else [k]

    def page_of(node) -> int | None:
        pg = node.get("/Pg") if isinstance(node, dict) else None
        return pages.get(int(pg)) if isinstance(pg, Ref) else None

    def first_page(node, depth: int = 0) -> int | None:
        if depth > 6 or not isinstance(node, dict):
            return None
        p = page_of(node)
        if p is not None:
            return p
        for c in kids(node):
            c = objs.resolve(c)
            if isinstance(c, dict):
                p = first_page(c, depth + 1)
                if p is not None:
                    return p
        return None

    def collect_mcids(node, into: list[int], depth: int = 0) -> None:
        if depth > 8 or not isinstance(node, dict):
            return
        for c in kids(node):
            c = objs.resolve(c)
            if isinstance(c, int) and not isinstance(c, Ref):
                into.append(int(c))
            elif isinstance(c, dict):
                if c.get("/Type") == "/MCR":
                    mc = c.get("/MCID")
                    if isinstance(mc, int):
                        into.append(int(mc))
                elif c.get("/Type") != "/OBJR":
                    collect_mcids(c, into, depth + 1)

    def span_of(node, key: str) -> int:
        a = objs.resolve(node.get("/A"))
        if isinstance(a, list):
            a = next((x for x in (objs.resolve(i) for i in a) if isinstance(x, dict)), None)
        if isinstance(a, dict):
            v = objs.resolve(a.get(key))
            if isinstance(v, int):
                return max(1, int(v))
        return 1

    def build_row(tr) -> list[TaggedCell]:
        cells = []
        for td in kids(tr):
            td = objs.resolve(td)
            if isinstance(td, dict) and td.get("/S") in ("/TD", "/TH"):
                mc: list[int] = []
                collect_mcids(td, mc)
                cells.append(TaggedCell(kind=td["/S"].lstrip("/"), mcids=mc,
                                        colspan=span_of(td, "/ColSpan"),
                                        rowspan=span_of(td, "/RowSpan")))
        return cells

    tables: list[TaggedTable] = []

    def gather_rows(node, out: list[list[TaggedCell]]) -> None:
        for c in kids(node):
            c = objs.resolve(c)
            if not isinstance(c, dict):
                continue
            s = c.get("/S")
            if s == "/TR":
                row = build_row(c)
                if row:
                    out.append(row)
            elif s in ("/THead", "/TBody", "/TFoot"):
                gather_rows(c, out)

    def walk(node, depth: int = 0) -> None:
        node = objs.resolve(node)
        if not isinstance(node, dict) or depth > 48:
            return
        if node.get("/S") == "/Table":
            rows: list[list[TaggedCell]] = []
            gather_rows(node, rows)
            if rows:
                tables.append(TaggedTable(page=first_page(node) or 0, rows=rows))
        for c in kids(node):
            c = objs.resolve(c)
            if isinstance(c, dict):
                walk(c, depth + 1)

    walk(root)

    if resolve_text and tables:
        by_page: dict[int, dict[int, str]] = {}
        for t in tables:
            if t.page not in by_page:
                by_page[t.page] = mcid_text(doc[t.page])
            lut = by_page[t.page]
            for row in t.rows:
                for cell in row:
                    cell.text = "".join(lut.get(m, "") for m in cell.mcids).strip()
    return tables
