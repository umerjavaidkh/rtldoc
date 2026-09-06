"""Tesseract TSV parsing.

Run:  python3 -m unittest tests.test_ocr_tsv -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rtldoc import ocr  # noqa: E402

HEADER = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
          "left\ttop\twidth\theight\tconf\ttext")


def _row(word, line=1, num=1):
    return f"5\t1\t1\t1\t{line}\t{num}\t100\t200\t50\t20\t92.5\t{word}"


class TestParseTsv(unittest.TestCase):

    def test_a_bare_quote_does_not_swallow_the_rest_of_the_file(self):
        """Tesseract never escapes its text column, so a recognised `"`
        arrives bare. Read with csv's default quotechar it opens a quoted
        field and every following row is absorbed into it."""
        lines = [HEADER, _row("first"), _row('"', num=2)]
        lines += [_row(f"w{i}", line=2, num=i) for i in range(20)]
        rows = ocr.parse_tsv("\n".join(lines) + "\n")

        self.assertEqual(len(rows), len(lines) - 1)
        texts = [r["text"] for r in rows]
        self.assertIn("w19", texts)
        for r in rows:
            # no row may carry another row's fields smuggled into its text
            self.assertNotIn("\t", r["text"] or "")
            self.assertLess(len(r["text"] or ""), 40)

    def test_ordinary_rows_still_parse(self):
        rows = ocr.parse_tsv("\n".join([HEADER, _row("وزارة"), _row("العمل", num=2)]))
        self.assertEqual([r["text"] for r in rows], ["وزارة", "العمل"])
        self.assertEqual(rows[0]["left"], "100")
        self.assertEqual(rows[0]["conf"], "92.5")


if __name__ == "__main__":
    unittest.main()
