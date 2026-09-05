"""The chunker under test.

Why the benchmark owns a chunker
--------------------------------
rtldoc does not chunk -- it emits blocks, and something downstream splits them
into retrieval units. That split is where parse quality turns into RAG quality
or fails to, so a benchmark that stops at the parsed page is measuring the
wrong artifact. A table that parses perfectly and then gets cut in half between
its header and its rows retrieves as garbage, and no page-level metric will
ever say so.

So RAGBench chunks the output the way a normal ingestion pipeline would, and
scores the chunks. The chunker deliberately uses the ordinary, obvious strategy
-- respect block boundaries, pack to a token budget, keep a heading breadcrumb
-- because the point is to measure the parser, not to show off a clever
chunker. If rtldoc's output needs an unusually smart chunker to survive, that
is a finding about rtldoc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Token budget in characters. Real tokenizers vary; a character budget keeps
# this dependency-free and is within ~15% of a BPE count for both English and
# Arabic, which is well inside the resolution this benchmark needs.
TARGET_CHARS = 1800
MAX_CHARS = 2600
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    text: str
    pages: list[int]
    blocks: list[int] = field(default_factory=list)   # indices into the block list
    heading_path: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    bboxes: list[tuple] = field(default_factory=list)

    @property
    def has_table(self) -> bool:
        return "table" in self.roles

    @property
    def citable(self) -> bool:
        """A chunk a RAG answer can cite: it knows its page and where on it."""
        return bool(self.pages) and any(
            b and len(b) == 4 and b[2] > b[0] and b[3] > b[1] for b in self.bboxes)


def _heading_level(role: str, text: str) -> int | None:
    if not role or not role.startswith("heading"):
        return None
    tail = role[-1]
    return int(tail) if tail.isdigit() else 1


def chunk_blocks(blocks: list, target: int = TARGET_CHARS,
                 hard_max: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[Chunk]:
    """Pack blocks into retrieval units.

    Three rules, all of which a competent ingestion pipeline would have:

      1. A table is never split. It is emitted whole, even past the budget --
         splitting a table separates values from the header that names them,
         which is the single most damaging thing you can do to tabular RAG.
      2. A heading starts a new chunk and joins the breadcrumb, so every chunk
         carries the section context a retriever needs to disambiguate it.
      3. Prose overflowing the budget breaks at a sentence end, never
         mid-sentence, with a small overlap.
    """
    chunks: list[Chunk] = []
    path: list[str] = []
    cur: Chunk | None = None

    def flush():
        nonlocal cur
        if cur and cur.text.strip():
            chunks.append(cur)
        cur = None

    def start(page: int) -> Chunk:
        return Chunk(text="", pages=[page], heading_path=list(path))

    for i, b in enumerate(blocks):
        text = (b.text or "").strip()
        if not text:
            continue
        page = getattr(b, "page", 0)
        level = _heading_level(b.role, text)

        if level is not None:
            flush()
            path = path[: max(0, level - 1)] + [text]
            cur = start(page)
            cur.heading_path = list(path)
            cur.text = text
            cur.roles.append(b.role)
            cur.blocks.append(i)
            cur.bboxes.append(tuple(b.bbox) if b.bbox else ())
            continue

        if b.role == "table":
            # rule 1: a table is its own chunk, whole, with the breadcrumb
            flush()
            t = start(page)
            t.heading_path = list(path)
            t.text = text
            t.roles.append("table")
            t.blocks.append(i)
            t.bboxes.append(tuple(b.bbox) if b.bbox else ())
            chunks.append(t)
            continue

        if cur is None:
            cur = start(page)
        if len(cur.text) + len(text) + 1 > hard_max and cur.text:
            tail = _sentence_tail(cur.text, overlap)
            flush()
            cur = start(page)
            cur.heading_path = list(path)
            cur.text = tail
        cur.text = f"{cur.text}\n{text}".strip() if cur.text else text
        cur.roles.append(b.role or "paragraph")
        cur.blocks.append(i)
        cur.bboxes.append(tuple(b.bbox) if b.bbox else ())
        if page not in cur.pages:
            cur.pages.append(page)
        if len(cur.text) >= target:
            flush()
    flush()
    return chunks


_SENT = re.compile(r"[.!?؟۔]\s")


def _sentence_tail(text: str, n: int) -> str:
    """The last <=n characters, cut at a sentence start so the overlap that
    carries into the next chunk is readable rather than a fragment."""
    tail = text[-n:]
    m = _SENT.search(tail)
    return tail[m.end():] if m else tail
