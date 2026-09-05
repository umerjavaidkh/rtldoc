"""End-to-end probe: can a retriever actually find the fact?

Why this dimension exists
-------------------------
Every other score in RAGBench is a proxy. This one is the question itself: a
fact is in the document, someone asks about it, does the chunk that comes back
contain the answer with its meaning intact?

The needles are generated from the ruled-table ground truth in `rules.py`, so
there is no hand-labelling and no LLM in the loop. For a record
{"Country": "Egypt", "Population": "104,000,000"} the probe asks for
"Egypt Population", retrieves over the parsed-and-chunked output with BM25, and
requires that a top-k chunk contain BOTH the row key and the value. Requiring
both is the whole point -- a chunk holding "104,000,000" with no trace of
"Egypt" is exactly the retrieval that makes an agent answer confidently and
wrongly, and a value-only check would score it as a success.

BM25 is implemented here rather than pulled in, so the benchmark has no
retrieval dependency and its scores cannot drift when someone upgrades a
library. Lexical retrieval is also the right choice for a PARSER benchmark: it
responds directly to the tokens the parser emitted, where a dense embedder
would blur exactly the damage being measured.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass

from . import rules as R

K1, B = 1.5, 0.75
TOP_K = 5


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", R.normalize(text).lower(), re.UNICODE)


class BM25:
    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        self.n = len(docs)
        self.len = [len(d) for d in docs]
        self.avg = sum(self.len) / self.n if self.n else 0.0
        self.tf = [collections.Counter(d) for d in docs]
        df: collections.Counter = collections.Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def top(self, query: list[str], k: int = TOP_K) -> list[int]:
        if not self.n:
            return []
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in query:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.len[i] / max(1e-9, self.avg))
                s += self.idf.get(t, 0.0) * f * (K1 + 1) / denom
            scores.append((s, i))
        scores.sort(reverse=True)
        return [i for s, i in scores[:k] if s > 0]


@dataclass
class Needle:
    query: str
    row_key: str
    value: str
    page: int


def needles_from_records(records: list, max_per_page: int = 6) -> list[Needle]:
    """One probe per table row: the row's own label plus a column header, with
    that cell's value as the answer.

    Rows whose value is short or purely structural are skipped -- "1" or "N/A"
    would match half the document and measure nothing."""
    out: list[Needle] = []
    for rec in records:
        items = [(k, v) for k, v in rec.values.items() if v and len(v) >= 3]
        if len(items) < 2:
            continue
        row_key = items[0][1]              # first populated cell labels the row
        for header, value in items[1:]:
            if len(value) < 3 or value == row_key:
                continue
            if not re.search(r"[\w؀-ۿ]", value):
                continue
            out.append(Needle(query=f"{row_key} {header}", row_key=row_key,
                              value=value, page=rec.page))
            if len(out) >= max_per_page:
                return out
    return out


def probe(needles: list[Needle], chunks: list, k: int = TOP_K) -> dict:
    """Answer-Support Rate @k.

    Three outcomes per needle, and the distinction between the last two is the
    finding:
      supported   -- a retrieved chunk holds the row key and the value
      orphaned    -- the value was retrieved but its row key was not, so the
                     answer is present with nothing binding it to the question
      missing     -- neither, or the chunk never surfaced
    """
    if not needles or not chunks:
        return {"n": 0, "supported": 0, "orphaned": 0, "missing": 0,
                "asr": 1.0 if not needles else 0.0, "examples": []}

    texts = [R.normalize(c.text) for c in chunks]
    index = BM25([tokenize(t) for t in texts])

    supported = orphaned = missing = 0
    examples = []
    for nd in needles:
        hits = index.top(tokenize(nd.query), k)
        key_n, val_n = R.normalize(nd.row_key), R.normalize(nd.value)
        got_both = any(key_n in texts[i] and val_n in texts[i] for i in hits)
        got_val = any(val_n in texts[i] for i in hits)
        if got_both:
            supported += 1
        elif got_val:
            orphaned += 1
            if len(examples) < 15:
                examples.append(f"p{nd.page} ORPHAN {nd.query!r}: value "
                                f"{nd.value[:30]!r} retrieved without its row key")
        else:
            missing += 1
            if len(examples) < 15:
                examples.append(f"p{nd.page} MISS {nd.query!r}: "
                                f"{nd.value[:30]!r} not in top-{k}")
    n = len(needles)
    return {"n": n, "supported": supported, "orphaned": orphaned,
            "missing": missing, "asr": supported / n, "examples": examples}
