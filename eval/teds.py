"""
TEDS (Tree-Edit-Distance-based Similarity) -- the field-standard table metric.

TEDS (Zhong et al., PubTabNet) scores a predicted table against gold by turning
each into an HTML-style tree (table -> tr -> td, each td carrying colspan,
rowspan and text) and computing the normalized tree edit distance between them:

    TEDS(pred, gold) = 1 - TreeEditDistance(pred, gold) / max(|pred|, |gold|)

1.0 is a perfect match; a missing row, a merged cell, or wrong cell text each
lower it in a way a flat cell-overlap score can't capture. It's the metric
PubTabNet, FinTabNet and OmniDocBench report, so a number here means the same
thing as a number on those leaderboards.

The tree edit distance is the classic Zhang-Shasha ordered-tree algorithm,
implemented here (no external dependency) and self-tested against known cases
at the bottom of this file (`python eval/teds.py`).

Inputs:
  * `teds(pred_grid, gold_grid)`  -- plain row/col grids (rtldoc's output form)
  * `teds_html(pred_html, gold_html)` -- full HTML tables WITH colspan/rowspan,
    the PubTabNet gold format; thead/tbody are transparent so an HTML gold and
    a grid prediction compare directly, and `th` normalizes to `td`
  * `struct_only=True` on either -> TEDS-Struct, scoring grid structure alone
    and ignoring cell text (the standard companion metric)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable

from rapidfuzz.distance import Levenshtein


@dataclass
class Node:
    tag: str
    content: str = ""
    colspan: int = 1
    rowspan: int = 1
    children: list["Node"] = field(default_factory=list)


def _norm_lev(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def _rename_cost(a: Node, b: Node) -> float:
    """TEDS node substitution cost: different tags cost 1; two cells with the
    same span cost the normalized edit distance of their text; a cell whose
    colspan/rowspan differs costs 1 (a structural mismatch, not a text one)."""
    if a.tag != b.tag:
        return 1.0
    if a.tag == "td":
        if a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        return _norm_lev(a.content, b.content)
    return 0.0


def _rename_cost_struct(a: Node, b: Node) -> float:
    """TEDS-Struct: score structure only (tags + spans), ignore cell text.
    Standard companion metric -- isolates 'did we get the grid right' from
    'did we read the cells right'."""
    if a.tag != b.tag:
        return 1.0
    if a.tag == "td" and (a.colspan != b.colspan or a.rowspan != b.rowspan):
        return 1.0
    return 0.0


def _postorder(root: Node) -> tuple[list[Node], list[int]]:
    """Return (nodes in postorder, lld) where lld[i] is the 1-based postorder
    index of node i+1's leftmost-leaf descendant."""
    nodes: list[Node] = []
    lld: list[int] = []

    def visit(n: Node) -> int:
        left_leaf = None
        for c in n.children:
            fl = visit(c)
            if left_leaf is None:
                left_leaf = fl
        nodes.append(n)
        idx = len(nodes)                 # 1-based postorder index
        lld.append(idx if left_leaf is None else left_leaf)
        return lld[-1]

    visit(root)
    return nodes, lld


def _keyroots(lld: list[int]) -> list[int]:
    seen: dict[int, int] = {}
    for i in range(1, len(lld) + 1):
        seen[lld[i - 1]] = i             # largest index for each lld value
    return sorted(seen.values())


def tree_edit_distance(rootA: Node, rootB: Node,
                       rename_cost: Callable[[Node, Node], float] = _rename_cost) -> float:
    A, Al = _postorder(rootA)
    B, Bl = _postorder(rootB)
    la, lb = len(A), len(B)
    treedist = [[0.0] * (lb + 1) for _ in range(la + 1)]

    for i in _keyroots(Al):
        for j in _keyroots(Bl):
            li, lj = Al[i - 1], Bl[j - 1]
            m, n = i - li + 2, j - lj + 2
            fd = [[0.0] * n for _ in range(m)]
            for x in range(1, m):
                fd[x][0] = fd[x - 1][0] + 1            # delete
            for y in range(1, n):
                fd[0][y] = fd[0][y - 1] + 1            # insert
            for x in range(1, m):
                for y in range(1, n):
                    ni, nj = li + x - 1, lj + y - 1
                    if Al[ni - 1] == li and Bl[nj - 1] == lj:
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[x - 1][y - 1] + rename_cost(A[ni - 1], B[nj - 1]),
                        )
                        treedist[ni][nj] = fd[x][y]
                    else:
                        p = Al[ni - 1] - li
                        q = Bl[nj - 1] - lj
                        fd[x][y] = min(
                            fd[x - 1][y] + 1,
                            fd[x][y - 1] + 1,
                            fd[p][q] + treedist[ni][nj],
                        )
    return treedist[la][lb]


def _count(root: Node) -> int:
    return 1 + sum(_count(c) for c in root.children)


def grid_to_tree(grid: list[list[str]]) -> Node:
    """A plain row/col grid (no spans) -> TEDS tree."""
    table = Node("table")
    for row in grid:
        tr = Node("tr")
        for cell in row:
            tr.children.append(Node("td", content=(cell or "").strip()))
        table.children.append(tr)
    return table


class _TableHTMLParser(HTMLParser):
    """Build a TEDS tree from an HTML table, honoring colspan/rowspan.

    thead/tbody are treated as transparent (their rows attach straight to the
    table) so an HTML gold table and a plain grid prediction are directly
    comparable -- otherwise the wrapper nodes would show up as pure structural
    difference on every table. `th` is normalized to `td`.
    """

    _KEEP = {"table", "tr", "td", "th"}
    _TRANSPARENT = {"thead", "tbody", "tfoot"}

    def __init__(self):
        super().__init__()
        self.root: Node | None = None
        self.stack: list[Node] = []
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._TRANSPARENT:
            return
        if tag not in self._KEEP:
            return
        a = dict(attrs)
        node = Node(tag="td" if tag == "th" else tag)
        if node.tag == "td":
            node.colspan = int(a.get("colspan", 1) or 1)
            node.rowspan = int(a.get("rowspan", 1) or 1)
            self._text = []
        if self.root is None:
            self.root = node
        elif self.stack:
            self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_data(self, data):
        if self.stack and self.stack[-1].tag == "td":
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag in self._TRANSPARENT or tag not in self._KEEP:
            return
        if not self.stack:
            return
        node = self.stack.pop()
        if node.tag == "td":
            node.content = "".join(self._text).strip()
            self._text = []


def html_to_tree(html: str) -> Node:
    p = _TableHTMLParser()
    p.feed(html)
    return p.root or Node("table")


def md_to_grid(md: str) -> list[list[str]]:
    """Parse a GFM pipe table (rtldoc's rendered table output) into a grid."""
    grid = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        grid.append([c.strip() for c in line.strip("|").split("|")])
    return grid


def teds_trees(pred: Node, gold: Node, struct_only: bool = False) -> float:
    cost = _rename_cost_struct if struct_only else _rename_cost
    np_, ng = _count(pred), _count(gold)
    if np_ == 1 and ng == 1:              # both empty tables
        return 1.0
    dist = tree_edit_distance(pred, gold, cost)
    return 1.0 - dist / max(np_, ng)


def teds(pred_grid: list[list[str]], gold_grid: list[list[str]],
         struct_only: bool = False) -> float:
    """TEDS between two plain row/col grids (no spans)."""
    return teds_trees(grid_to_tree(pred_grid), grid_to_tree(gold_grid), struct_only)


def teds_html(pred_html: str, gold_html: str, struct_only: bool = False) -> float:
    """TEDS between two HTML tables, with full colspan/rowspan support."""
    return teds_trees(html_to_tree(pred_html), html_to_tree(gold_html), struct_only)


# --------------------------------------------------------------------------
# self-test: run `python eval/teds.py`
# --------------------------------------------------------------------------
if __name__ == "__main__":
    def approx(a, b, tol=1e-9):
        return abs(a - b) < tol

    g = [["a", "b"], ["c", "d"]]
    # identical -> 1.0
    assert approx(teds(g, g), 1.0), teds(g, g)
    # one cell wrong text: single td substitution cost = normlev("d","x")=1;
    # tree has table(1)+2 tr + 4 td = 7 nodes -> TEDS = 1 - 1/7
    g2 = [["a", "b"], ["c", "x"]]
    assert approx(teds(g2, g), 1 - 1 / 7), teds(g2, g)
    # missing a whole row: delete tr + 2 td = 3 nodes; max nodes = 7
    g3 = [["a", "b"]]
    assert approx(teds(g3, g), 1 - 3 / 7), teds(g3, g)
    # totally different structure vs empty-ish
    assert 0.0 <= teds([["z"]], g) <= 1.0
    # symmetry
    assert approx(teds(g2, g), teds(g, g2))

    # --- colspan/rowspan via HTML ---
    # identical HTML with a colspan -> 1.0
    h = "<table><tr><td colspan='2'>hdr</td></tr><tr><td>a</td><td>b</td></tr></table>"
    assert approx(teds_html(h, h), 1.0), teds_html(h, h)
    # same text/layout but the span differs -> penalized (td sub cost = 1)
    h2 = "<table><tr><td colspan='3'>hdr</td></tr><tr><td>a</td><td>b</td></tr></table>"
    assert teds_html(h2, h) < 1.0, teds_html(h2, h)
    # thead/tbody are transparent: wrapping rows must not change the score
    h_wrapped = "<table><thead><tr><td colspan='2'>hdr</td></tr></thead><tbody><tr><td>a</td><td>b</td></tr></tbody></table>"
    assert approx(teds_html(h_wrapped, h), 1.0), teds_html(h_wrapped, h)
    # TEDS-Struct ignores cell text: two grids differing only in content -> 1.0
    assert approx(teds(g2, g, struct_only=True), 1.0), teds(g2, g, struct_only=True)

    print("TEDS self-tests passed:")
    print(f"  identical              = {teds(g, g):.3f}")
    print(f"  one cell wrong         = {teds(g2, g):.3f}  (expect {1-1/7:.3f})")
    print(f"  one row missing        = {teds(g3, g):.3f}  (expect {1-3/7:.3f})")
    print(f"  colspan match (HTML)   = {teds_html(h, h):.3f}")
    print(f"  colspan mismatch       = {teds_html(h2, h):.3f}")
    print(f"  thead/tbody transparent= {teds_html(h_wrapped, h):.3f}")
    print(f"  TEDS-Struct (text-blind)= {teds(g2, g, struct_only=True):.3f}")
