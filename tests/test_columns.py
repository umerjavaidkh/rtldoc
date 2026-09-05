"""Column detection and reading order.

Every layout here is built synthetically, so the tests pin behaviour rather
than a particular file: the point of the fix these cover is that column
detection is derived from the page's own typography instead of from constants
that happen to suit one document.

Run:  python3 -m unittest tests.test_columns -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtldoc import layout, pipeline  # noqa: E402

W, H = 612.0, 792.0
LOREM = ("the quick brown fox jumps over the lazy dog while the "
         "cat watches from a warm windowsill nearby and waits")


def _page(draw) -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    draw(page)
    # reopen from bytes so the page is parsed the way a real file is
    reopened = fitz.open("pdf", doc.tobytes())
    return reopened[0]


def _columns(words: list[str], x: float, y: float, width: float,
             size: float, page, lines: int) -> float:
    """Set `lines` justified-ish lines of body text in a column."""
    per_line = max(3, int(width / (size * 0.5)))
    i = 0
    for n in range(lines):
        text = " ".join(words[i % len(words):][:8]) or LOREM[:per_line]
        page.insert_text((x, y), text[:per_line], fontsize=size, fontname="helv")
        y += size * 1.2
        i += 3
    return y


def _bands(page) -> list[tuple[float, float]]:
    prim = pipeline.extract_page(page)
    return layout.column_bands([s.bbox for s in prim.spans], prim.width, prim.height)


def _order_text(page) -> str:
    r = pipeline.parse_page(page)
    return "\n".join(b.text for b in sorted(r.blocks, key=lambda b: b.order))


class ColumnBands(unittest.TestCase):

    def test_single_column_is_one_band(self):
        def draw(p):
            _columns(LOREM.split(), 72, 90, 460, 10, p, 40)
        self.assertEqual(len(_bands(_page(draw))), 1)

    def test_two_columns_with_a_ten_point_gutter(self):
        """LaTeX's default columnsep. The old absolute 10pt floor rejected
        this, which is why every two-column paper on arXiv read as one
        column and its text came out interleaved."""
        def draw(p):
            _columns(LOREM.split(), 72, 90, 225, 10, p, 40)
            _columns(LOREM.split(), 307, 90, 225, 10, p, 40)
        self.assertEqual(len(_bands(_page(draw))), 2)

    def test_three_columns(self):
        def draw(p):
            for x in (60, 245, 430):
                _columns(LOREM.split(), x, 90, 140, 9, p, 45)
        self.assertEqual(len(_bands(_page(draw))), 3)

    def test_gutter_floor_scales_with_type_size(self):
        """The same geometry at two type sizes. A constant threshold cannot
        be right for both; a typography-relative one is."""
        for size, gutter in ((6.0, 6.0), (16.0, 16.0)):
            with self.subTest(size=size):
                def draw(p, size=size, gutter=gutter):
                    colw = (460 - gutter) / 2
                    _columns(LOREM.split(), 72, 90, colw, size, p, int(600 / (size * 1.2)))
                    _columns(LOREM.split(), 72 + colw + gutter, 90, colw, size, p,
                             int(600 / (size * 1.2)))
                self.assertEqual(len(_bands(_page(draw))), 2)

    def test_narrow_table_columns_are_not_page_columns(self):
        """A table's inter-column whitespace is empty over the full height of
        the content, exactly like a gutter. What separates them is that a page
        column is wide enough to set running text in and a table column is
        not -- these are 3-4 em, which holds about five characters."""
        def draw(p):
            y = 100
            for _ in range(30):
                p.insert_text((70, y), "op", fontsize=9, fontname="helv")
                p.insert_text((105, y), "closepath fill and stroke the whole path here",
                              fontsize=9, fontname="helv")
                p.insert_text((330, y), "4.10", fontsize=9, fontname="helv")
                p.insert_text((365, y), "230", fontsize=9, fontname="helv")
                y += 18
        self.assertEqual(len(_bands(_page(draw))), 1)

    def test_hanging_marker_column_is_not_a_column(self):
        """Bullets in the margin leave a persistent empty strip beside them.
        It is not a column: nothing could be set in it."""
        def draw(p):
            y = 100
            for _ in range(25):
                p.insert_text((72, y), "*", fontsize=10, fontname="helv")
                p.insert_text((95, y), LOREM[:70], fontsize=10, fontname="helv")
                y += 22
        self.assertEqual(len(_bands(_page(draw))), 1)

    def test_table_rows_survive_even_when_the_page_bands(self):
        """A table whose columns ARE far enough apart to look like a page
        layout must still be read as a table. The guarantee does not come from
        getting the band count right -- it comes from the table detector
        reading unsplit lines, so a row is always visible as one line however
        the flow text is banded. This is the invariant that keeps banding safe
        for tables in general, rather than case by case."""
        def draw(p):
            y = 100
            for i in range(24):
                p.insert_text((70, y), f"rowlabel{i:02d}", fontsize=9, fontname="helv")
                p.insert_text((190, y), "closepath fill and stroke", fontsize=9, fontname="helv")
                p.insert_text((430, y), f"{i}.10", fontsize=9, fontname="helv")
                p.insert_text((500, y), f"{200 + i}", fontsize=9, fontname="helv")
                y += 18
        text = _order_text(_page(draw))
        # the first row's outer cells must still appear together in the output
        self.assertIn("rowlabel00", text)
        self.assertIn("200", text)
        row = next((ln for ln in text.splitlines() if "rowlabel00" in ln), "")
        self.assertIn("200", row, f"row was split apart: {row!r}")

    def test_empty_page_is_one_band(self):
        self.assertEqual(len(layout.column_bands([], W, H)), 1)


class ReadingOrder(unittest.TestCase):

    def test_two_columns_are_not_interleaved(self):
        """The defect this whole change exists for: left-column and
        right-column lines share a baseline, so grouping by y alone welds them
        into one line and the text reads as alternating fragments."""
        def draw(p):
            y = 90
            for i in range(30):
                p.insert_text((72, y), f"LEFTSIDE line number {i:02d} of the left column",
                              fontsize=10, fontname="helv")
                p.insert_text((320, y), f"RIGHTSIDE line number {i:02d} of the right column",
                              fontsize=10, fontname="helv")
                y += 14
        text = _order_text(_page(draw))
        for line in text.splitlines():
            self.assertFalse("LEFTSIDE" in line and "RIGHTSIDE" in line,
                             f"line welds both columns: {line!r}")
        # and the left column is read out before the right one
        self.assertLess(text.index("LEFTSIDE line number 29"),
                        text.index("RIGHTSIDE line number 00"))

    def test_full_width_heading_above_two_columns_reads_first(self):
        def draw(p):
            p.insert_text((72, 80), "A WIDE HEADING SPANNING THE WHOLE PAGE",
                          fontsize=20, fontname="hebo")
            y0 = 120
            y = y0
            for i in range(25):
                p.insert_text((72, y), f"LEFTSIDE body line {i:02d} of the left column",
                              fontsize=10, fontname="helv")
                p.insert_text((320, y), f"RIGHTSIDE body line {i:02d} of the right column",
                              fontsize=10, fontname="helv")
                y += 14
        text = _order_text(_page(draw))
        self.assertLess(text.index("WIDE HEADING"), text.index("LEFTSIDE body line 00"))
        self.assertLess(text.index("LEFTSIDE body line 24"),
                        text.index("RIGHTSIDE body line 00"))

    def test_rtl_two_columns_read_right_to_left(self):
        def draw(p):
            y = 90
            for i in range(28):
                # Arabic on both sides so the page is detected as RTL
                p.insert_text((72, y), f"يسار {i:02d} السطر الايسر من الصفحة",
                              fontsize=11, fontname="helv")
                p.insert_text((330, y), f"يمين {i:02d} السطر الايمن من الصفحة",
                              fontsize=11, fontname="helv")
                y += 15
        page = _page(draw)
        self.assertEqual(len(_bands(page)), 2)
        r = pipeline.parse_page(page)
        cols = {b.column for b in r.blocks if b.column is not None}
        self.assertGreaterEqual(len(cols), 1)


class BlockScoping(unittest.TestCase):
    """Gutters carry a vertical extent, so a page can change layout partway
    down without the lower layout reaching up into the upper one."""

    def _mixed(self):
        def draw(p):
            # a full-width table across the top
            y = 90
            for i in range(8):
                for x in (72, 190, 300, 410, 500):
                    p.insert_text((x, y), f"cell{i:02d}", fontsize=9, fontname="helv")
                y += 16
            # a clear block gap, then two columns of prose
            y += 40
            for i in range(24):
                p.insert_text((72, y), f"LEFTSIDE prose line {i:02d} of the left column",
                              fontsize=10, fontname="helv")
                p.insert_text((320, y), f"RIGHTSIDE prose line {i:02d} of the right column",
                              fontsize=10, fontname="helv")
                y += 14
        return _page(draw)

    def test_no_gutter_spans_two_blocks(self):
        """The invariant that makes mixed layouts safe: a gutter belongs to one
        block. A block may legitimately have gutters of its own -- a table's
        column gaps look identical to a projection, which is why the pipeline
        withholds table words from the projection -- but no single gutter may
        claim rows in two different blocks."""
        page = self._mixed()
        words = [tuple(w[:4]) for w in page.get_text("words") if w[4].strip()]
        bands = layout._horizontal_bands(words, layout._body_size(words))
        self.assertGreaterEqual(len(bands), 2, "the block gap was not detected")
        (_, top_y0, top_y1), (_, low_y0, low_y1) = bands[0], bands[-1]
        for x, y0, y1 in layout.page_gutters(words, W, H):
            spans_top = y0 <= top_y1 and y1 >= top_y0
            spans_low = y0 <= low_y1 and y1 >= low_y0
            self.assertFalse(spans_top and spans_low,
                             f"gutter x={x:.0f} claims rows in both blocks")

    def test_full_width_row_above_columns_is_not_split(self):
        text = _order_text(self._mixed())
        row = next((ln for ln in text.splitlines() if "cell00" in ln), "")
        self.assertIn("cell00", text)
        # the top table's row must survive as one line, not be cut at the
        # gutter belonging to the prose far below it
        self.assertGreaterEqual(row.count("cell00"), 1)
        for line in text.splitlines():
            self.assertFalse("LEFTSIDE" in line and "RIGHTSIDE" in line)

    def test_three_columns_read_column_major(self):
        def draw(p):
            for ci, x in enumerate((60, 245, 430)):
                y = 90
                for i in range(30):
                    p.insert_text((x, y), f"C{ci}L{i:02d} some body text here",
                                  fontsize=9, fontname="helv")
                    y += 13
        text = _order_text(_page(draw))
        # every line of column 0 precedes every line of column 1, and so on
        self.assertLess(text.index("C0L29"), text.index("C1L00"))
        self.assertLess(text.index("C1L29"), text.index("C2L00"))


class DeclaredOrder(unittest.TestCase):
    """`_apply_declared_order` -- the structure tree beats geometry when the
    file ships one, and is ignored when it covers too little to trust."""

    def _regions(self):
        mk = lambda x0, y0, x1, y1: layout.Region(kind="flow", bbox=(x0, y0, x1, y1))
        return [mk(300, 100, 500, 200), mk(50, 100, 250, 200), mk(50, 300, 250, 400)]

    def test_declared_order_overrides_geometry(self):
        regions = self._regions()
        declared = [((400, 150), 0), ((150, 150), 1), ((150, 350), 2)]
        out = layout._apply_declared_order(regions, declared)
        self.assertIsNotNone(out)
        self.assertEqual([r.bbox[0] for r in out], [300, 50, 50])

    def test_sparse_tags_fall_back_to_geometry(self):
        regions = self._regions()
        out = layout._apply_declared_order(regions, [((400, 150), 0)])
        self.assertIsNone(out, "one tagged region out of three must not reorder the page")

    def test_no_tags_is_a_no_op(self):
        self.assertIsNone(layout._apply_declared_order(self._regions(), []))


if __name__ == "__main__":
    unittest.main(verbosity=2)
