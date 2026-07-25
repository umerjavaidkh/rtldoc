"""Metrics shared by every eval track (custom gold set and OmniDocBench subset).

Definitions deliberately mirror what OmniDocBench itself scores (normalized
edit distance for text, edit distance over the predicted order sequence for
reading order, TEDS-style structural accuracy for tables) so a row computed
here means the same thing as a row pulled from a published leaderboard.
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz.distance import Levenshtein


def _normalize_text(s: str) -> str:
    """NFC + collapse whitespace + strip tatweel, so two transcriptions that
    differ only in incidental Unicode form or spacing don't get penalized."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("ـ", "")  # tatweel
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_edit_score(gold: str, pred: str) -> float:
    """1 - normalized Levenshtein distance. 1.0 = perfect, 0.0 = worthless."""
    g, p = _normalize_text(gold), _normalize_text(pred)
    if not g and not p:
        return 1.0
    dist = Levenshtein.distance(g, p)
    return max(0.0, 1.0 - dist / max(len(g), len(p), 1))


def reading_order_score(gold_blocks: list[str], pred_blocks: list[str], threshold: float = 0.5) -> float:
    """Reading-order accuracy from raw text, not shared block IDs -- no two
    parsers segment a page the same way, so IDs can't line up directly.

    Each predicted block is matched (greedily, by text-edit similarity) to
    its best unclaimed gold block; then we ask whether the matched gold
    indices come out in ascending order, via edit distance against their own
    sorted version. A block rtldoc merges, splits, or drops just shows up as
    a smaller matched set rather than corrupting the whole score.
    """
    if not gold_blocks:
        return 1.0
    matched: list[int] = []
    used: set[int] = set()
    for pb in pred_blocks:
        best_i, best_s = None, threshold
        for i, gb in enumerate(gold_blocks):
            if i in used:
                continue
            s = text_edit_score(gb, pb)
            if s > best_s:
                best_i, best_s = i, s
        if best_i is not None:
            matched.append(best_i)
            used.add(best_i)
    ideal = sorted(matched)
    dist = Levenshtein.distance(matched, ideal)
    missed = len(gold_blocks) - len(matched)
    return max(0.0, 1.0 - (dist + missed) / len(gold_blocks))


def table_cell_score(gold_grid: list[list[str]], pred_grid: list[list[str]]) -> float:
    """Fraction of gold cells whose text (edit-score > 0.7) was found
    *somewhere* in the predicted grid, regardless of exact row/col index --
    a cheap proxy for TEDS that doesn't need a tree-edit-distance library,
    tolerant to the predicted grid having a different row/col count than
    gold (e.g. a header spanning two columns gets split differently)."""
    gold_cells = [c.strip() for row in gold_grid for c in row if c.strip()]
    pred_cells = [c.strip() for row in pred_grid for c in row if c.strip()]
    if not gold_cells:
        return 1.0 if not pred_cells else 0.0
    matched = 0
    remaining = list(pred_cells)
    for gc in gold_cells:
        best_i, best_s = None, 0.7
        for i, pc in enumerate(remaining):
            s = text_edit_score(gc, pc)
            if s > best_s:
                best_i, best_s = i, s
        if best_i is not None:
            matched += 1
            remaining.pop(best_i)
    return matched / len(gold_cells)
