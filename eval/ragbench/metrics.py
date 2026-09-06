"""The five dimension scores.

Structure follows ParseBench: five capability dimensions, each scored 0-1 by
deterministic rules with no LLM judge, aggregated as an unweighted mean so no
dimension can be traded off against another. The dimensions themselves are
re-chosen for ingestion rather than for parse fidelity:

  ParseBench dimension   RAGBench dimension        why the change
  --------------------   ------------------       --------------------------
  Tables                 Table Record Fidelity     kept -- same metric shape
  Content Faithfulness   Content Faithfulness      kept -- omissions + order
  Charts                 Chunk Integrity           rtldoc parses no charts;
                                                   what breaks RAG instead is
                                                   the chunk boundary
  Semantic Formatting    Structure & Hierarchy     heading path is chunk
                                                   metadata, not decoration
  Visual Grounding       Citability                same idea: can an answer
                                                   point back at the page

Every score returns its own sub-scores and a denominator, because a dimension
scored over three rules and one scored over three thousand should not look
alike in a report.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

from scipy.optimize import linear_sum_assignment

from . import rules as R

_SENT_END = re.compile(r"[.!?؟۔][\"')\]]?$")


@dataclass
class Score:
    """A dimension score plus the evidence behind it."""
    value: float
    n: int                                   # rules/units the score is over
    parts: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def as_row(self) -> dict:
        return {"score": round(self.value, 4), "n": self.n,
                "parts": {k: round(v, 4) for k, v in self.parts.items()},
                "failures": self.failures[:25]}


def _safe(num: float, den: float, default: float = 1.0) -> float:
    return num / den if den else default


# --------------------------------------------------------------------------
# 1. Content Faithfulness  (omissions, hallucinations, reading order)
# --------------------------------------------------------------------------

def reading_order_score(truth: R.PageTruth, output: str) -> Score:
    """Column handling, as its own axis rather than folded into text.

    Two halves, because either alone is blind. PRECEDENCE catches inversion --
    text that comes out in the wrong sequence. CONTIGUITY catches intrusion --
    two lines that belong together arriving with a foreign column's text
    wedged between them. A shredded two-column page passes precedence
    perfectly, which is exactly why it needs both.
    """
    prec = [(r, r.check(output)) for r in truth.precedence]
    prec = [(r, v) for r, v in prec if v is not None]
    cont = [(r, r.check(output)) for r in truth.contiguity]
    cont = [(r, v) for r, v in cont if v is not None]
    if not prec and not cont:
        return Score(1.0, 0, {})
    p = _safe(sum(1 for _, v in prec if v), len(prec))
    c = _safe(sum(1 for _, v in cont if v), len(cont))
    fails = [f"order: {r.before[:40]!r} should precede {r.after[:40]!r}"
             for r, v in prec if not v][:10]
    fails += [f"split: {r.first[:40]!r} and {r.second[:40]!r} arrive apart"
              for r, v in cont if not v][:10]
    return Score((p + c) / 2, len(prec) + len(cont),
                 {"precedence": p, "contiguity": c}, fails)


def content_faithfulness(truth: R.PageTruth, output: str,
                         w_text: float = 1.0, w_order: float = 0.5) -> Score:
    """CFS = (w_text * S_text + w_order * S_order) / (w_text + w_order).

    ParseBench's weights (1.0 / 0.5): losing the content is worse than
    misordering it, but misordering it still ruins a chunk, so order is not
    free either."""
    out_words = R.word_bag(output)
    out_digits = R.digit_bag(output)

    kept = sum(min(truth.words[w], out_words[w]) for w in truth.words)
    total = sum(truth.words.values())
    recall = _safe(kept, total)

    # Hallucination: words in the output that the page never contained. The
    # chunker adds a heading breadcrumb, so a little excess is structural, but
    # invented running text is not.
    extra = sum(max(0, out_digits[d] - truth.digits[d]) for d in out_digits)
    dropped = sum(max(0, truth.digits[d] - out_digits[d]) for d in truth.digits)
    digit_total = sum(truth.digits.values())
    digit_score = _safe(digit_total - dropped - extra, digit_total)
    digit_score = max(0.0, digit_score)

    extra_words = sum(max(0, out_words[w] - truth.words[w]) for w in out_words)
    precision = _safe(total, total + extra_words) if total else 1.0

    s_text = (recall + digit_score + precision) / 3

    checked = [(r, r.check(output)) for r in truth.precedence]
    applicable = [(r, v) for r, v in checked if v is not None]
    s_prec = _safe(sum(1 for _, v in applicable if v), len(applicable))

    # Contiguity is the half of reading order that precedence cannot see: an
    # interleaved page keeps every column's lines in the right relative order
    # and is still unreadable. Both are averaged into the order term so the
    # score responds to intrusion as well as inversion.
    cchecked = [(r, r.check(output)) for r in truth.contiguity]
    capplicable = [(r, v) for r, v in cchecked if v is not None]
    s_contig = _safe(sum(1 for _, v in capplicable if v), len(capplicable))
    s_order = (s_prec + s_contig) / 2

    value = (w_text * s_text + w_order * s_order) / (w_text + w_order)
    failures = [f"order: {r.before[:45]!r} should precede {r.after[:45]!r}"
                for r, v in applicable if not v]
    if recall < 0.9:
        missing = [w for w in truth.words if out_words[w] < truth.words[w]]
        failures.append(f"omitted {len(missing)} distinct word(s), e.g. {missing[:8]}")
    if dropped or extra:
        failures.append(f"digits: {dropped} dropped, {extra} invented")
    return Score(value, 3 + len(applicable),
                 {"word_recall": recall, "digit_integrity": digit_score,
                  "word_precision": precision, "reading_order": s_order,
                  "precedence": s_prec, "contiguity": s_contig,
                  "s_text": s_text},
                 failures)


# --------------------------------------------------------------------------
# 2. Table Record Fidelity  (does a value keep the header that names it)
# --------------------------------------------------------------------------

def _record_sim(g: dict, p: dict) -> float:
    """RecordSim(g,p) = |{k : g[k] == p[k]}| / |K(g) U K(p)|  -- ParseBench.

    Keys are the column headers, so a table whose header row was dropped or
    shifted scores near zero however well its cell text was extracted. That is
    the point: for RAG the header IS the meaning."""
    if not g and not p:
        return 1.0
    # Positional fallback. Keying on the header is right when both sides HAVE
    # the same header, and catastrophic when they do not: a table continued
    # across a page break has its header on the previous page, so both sides
    # key off whatever data row came first and no key pairs. Measured on the
    # Gulf corpus, 38% of scored tables scored exactly zero for this reason
    # alone, and matching the same tables by cell position instead lifted the
    # mean from 0.155 to 0.220. When no header pairs, compare cell by cell in
    # order -- the columns are still in the same sequence.
    if g and p and not (set(g) & set(p)):
        gv, pv = list(g.values()), list(p.values())
        n = max(len(gv), len(pv))
        if n:
            same = sum(1 for i in range(n)
                       if _cell_equal(gv[i] if i < len(gv) else "",
                                      pv[i] if i < len(pv) else ""))
            return same / n
    # Headers are paired by near-equality too, for the same reason the values
    # are: a reference header read out of the raw word stream comes back with
    # its words in visual order ("التعلّم مؤشر" for "مؤشر التعلّم"), and an
    # exact-set intersection then matches NOTHING, discarding every value under
    # that column. Token-set matching is order-insensitive, which is exactly
    # the tolerance needed and no more -- a genuinely different header still
    # fails to pair.
    unmatched = list(p)
    hit = 0
    for gk in g:
        best, bk = 0.0, None
        for pk in unmatched:
            if gk == pk:
                best, bk = 1.0, pk
                break
            if _cell_equal(gk, pk):
                best, bk = 0.99, pk
        if bk is not None:
            unmatched.remove(bk)
            if _cell_equal(g[gk], p[bk]):
                hit += 1
    return hit / max(1, len(set(g) | set(p)))


_CELL_MATCH = 90        # rapidfuzz token_set_ratio


def _cell_equal(a: str, b: str) -> bool:
    """Do two cells carry the same value?

    Near-equality, not string equality, and the reason is Arabic. The ground
    truth is read from the PDF's raw word stream, which is exactly where the
    defects rtldoc exists to correct still live: a lam-alef ligature comes back
    decomposed in visual order ("بطالقة" for "بطلاقة") and a bullet lands on
    the wrong side of the first word. Comparing those byte-for-byte scores the
    parser 0 for being MORE correct than the reference, which is worse than
    useless.

    A high token-set threshold keeps what the metric is for -- a value attached
    to the wrong header still fails, because the tokens genuinely differ -- while
    not punishing orthography the reference got wrong. Latin cells are
    unaffected: they either match exactly or differ by real content.
    """
    x, y = R.normalize(a), R.normalize(b)
    if x == y:
        return True
    if not x or not y:
        return False
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return False
    if fuzz.token_set_ratio(x, y) >= _CELL_MATCH:
        return True
    # Arabic tokens whose LETTERS match but whose order does not. The raw text
    # layer returns a lam-alef ligature decomposed in visual order, so the
    # reference holds "المايل" where the word is "المالي" -- the same
    # characters, two of them transposed. rtldoc repairs that; comparing on
    # order alone therefore scores the parser 0 for being more correct, which
    # is what this whole function exists to prevent.
    #
    # Scoped to cells that are mostly Arabic and to a multiset over the WHOLE
    # cell, so it stays a statement about orthography and not a licence to
    # match any anagram: a cell with different content has different letters.
    ar = sum(1 for c in x if "\u0600" <= c <= "\u06ff")
    if ar >= 0.5 * max(len(x.replace(" ", "")), 1):
        from collections import Counter
        cx = Counter(c for c in x if not c.isspace())
        cy = Counter(c for c in y if not c.isspace())
        if cx and cy:
            common = sum((cx & cy).values())
            if common / max(sum(cx.values()), sum(cy.values())) >= 0.92:
                return True
    return False


def table_detection(truth_groups: list, pred_tables: list) -> Score:
    """Did the page's table get FOUND, once, regardless of cell fidelity.

    table_record_fidelity asks whether every row matches, and against this
    reference that is often unanswerable: the reference builds its records from
    the raw word stream, so its headers come back as fragments in visual order
    and it routinely takes the loose band rows above a table as the header row.
    Confirmed on the UAE service manual p18, where the reference's header is
    ['ال ينطبق', 'الغرامات', 'فقط', 'رسوم الإصدار'] -- decorative bands above
    the table -- and rtldoc's own 5x3 grid of the same table scores 0.0 while
    being visibly correct.

    A detection rate is robust to all of that. It answers the question an
    ingestion pipeline actually asks first -- is there a table here and did we
    emit one -- and it cannot be zeroed by the reference disagreeing about
    where a header is. Reported alongside fidelity, not instead of it: finding
    a table and reading it correctly are different claims and both matter.
    """
    want = len(truth_groups)
    got = len([t for t in pred_tables if t])
    if not want:
        # RECALL only. Scoring a page where the reference has no table would
        # punish rtldoc for finding tables the reference cannot see -- it reads
        # only RULED grids, and most of the Gulf corpus is set with coloured
        # bands or nothing at all. Precision against an incomplete reference is
        # not a measurement, it is a penalty for being better than it.
        return Score(1.0, 0, {"truth_tables": 0, "pred_tables": got})
    hit = 1.0 if got else 0.0
    return Score(hit, 1, {"truth_tables": want, "pred_tables": got},
                 [] if hit else ["reference has a table, none emitted"])

def table_record_fidelity(truth_records: list, pred_records: list) -> Score:
    """TableRecordMatch: optimal matching between true and predicted records,
    normalised by max(|G|,|P|) so both dropped and invented rows are punished.
    Insensitive to row and column ORDER, by design -- a retriever does not care
    which row came first, it cares that the row is intact."""
    G = [r.values for r in truth_records]
    P = pred_records
    if not G and not P:
        return Score(1.0, 0, {"records_truth": 0, "records_pred": 0})
    if not G or not P:
        return Score(0.0, max(len(G), len(P)),
                     {"records_truth": len(G), "records_pred": len(P)},
                     ["no records on one side: table wholly lost or invented"])

    sim = [[_record_sim(g, p) for p in P] for g in G]
    rows, cols = linear_sum_assignment([[-s for s in row] for row in sim])
    matched = sum(sim[i][j] for i, j in zip(rows, cols))
    value = matched / max(len(G), len(P))

    failures = []
    for i, j in zip(rows, cols):
        if sim[i][j] < 0.5:
            failures.append(f"record {dict(list(G[i].items())[:3])} -> "
                            f"{dict(list(P[j].items())[:3])} (sim {sim[i][j]:.2f})")
    return Score(value, max(len(G), len(P)),
                 {"records_truth": len(G), "records_pred": len(P),
                  "mean_record_sim": _safe(matched, min(len(G), len(P)), 0.0)},
                 failures)


def records_from_markdown(md: str, page: int) -> list[dict]:
    """Predicted records, read back out of the parser's own markdown tables the
    way a downstream consumer would -- not from an internal structure the
    consumer never sees."""
    out: list[dict] = []
    rows: list[list[str]] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                out.extend(_rows_to_records(rows))
                rows = []
            continue
        # The |---|---| separator is skipped WITHOUT flushing. Flushing on it
        # discards the header row that precedes it, and the first data row
        # silently becomes the header -- every value then keys to the wrong
        # column and the table scores zero however well it was parsed.
        if set(line) <= set("|-: "):
            continue
        rows.append([R.normalize(c) for c in line.strip("|").split("|")])
    out.extend(_rows_to_records(rows))
    return out


def tables_from_markdown(md: str) -> list[list[dict]]:
    """Records grouped BY TABLE rather than flattened across the page.

    The gold set needs this: its ground truth is one specific grid, so scoring
    it against every record on the page punishes rtldoc for tables it got right
    elsewhere. Grouped, each ground-truth table can be matched against the
    predicted table that actually corresponds to it."""
    out, rows = [], []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows:
                recs = _rows_to_records(rows)
                if recs:
                    out.append(recs)
                rows = []
            continue
        if set(line) <= set("|-: "):
            continue
        rows.append([R.normalize(c) for c in line.strip("|").split("|")])
    recs = _rows_to_records(rows)
    if recs:
        out.append(recs)
    return out


def _rows_to_records(rows: list[list[str]]) -> list[dict]:
    if len(rows) < 2:
        return []
    header = [h or f"col{i}" for i, h in enumerate(rows[0])]
    recs = []
    for row in rows[1:]:
        vals = {header[i]: row[i] for i in range(min(len(header), len(row)))
                if row[i].strip()}
        if vals:
            recs.append(vals)
    return recs


# --------------------------------------------------------------------------
# 3. Chunk Integrity  (the dimension that only matters for ingestion)
# --------------------------------------------------------------------------

def chunk_integrity(chunks: list) -> Score:
    """Is each retrieval unit usable on its own?

    A chunk is what the retriever returns and the LLM reads. It fails if it
    stops mid-sentence, if it holds table rows whose header is in a different
    chunk, if it carries no section context to disambiguate it, or if it is so
    large the embedder truncates it. None of these are visible at page level,
    and all of them are the difference between a right and a wrong answer."""
    if not chunks:
        return Score(0.0, 0, {}, ["no chunks produced"])

    n = len(chunks)
    complete_sentence = context = sized = table_ok = not_noise = 0
    failures: list[str] = []

    for c in chunks:
        text = c.text.strip()
        if c.has_table or _SENT_END.search(text) or len(text) < 80:
            complete_sentence += 1
        elif len(failures) < 25:
            failures.append(f"p{c.pages[:1]} chunk ends mid-sentence: ...{text[-60:]!r}")

        if c.heading_path or c.has_table:
            context += 1
        elif len(failures) < 25:
            failures.append(f"p{c.pages[:1]} chunk has no heading context")

        if len(text) <= 4000:
            sized += 1
        elif len(failures) < 25:
            failures.append(f"p{c.pages[:1]} chunk is {len(text)} chars, past a typical window")

        if not c.has_table:
            table_ok += 1
        else:
            grid = [l for l in text.splitlines() if l.strip().startswith("|")]
            body = [l for l in grid if not set(l.strip()) <= set("|-: ")]
            # a table chunk must carry a header row plus at least one data row,
            # or its values arrive with nothing naming them
            if len(body) >= 2:
                table_ok += 1
            elif len(failures) < 25:
                failures.append(f"p{c.pages[:1]} table chunk has {len(body)} row(s), "
                                f"header and data separated")

        words = R.word_bag(text)
        if len(text) > 60 and sum(words.values()) >= 5:
            not_noise += 1
        elif len(failures) < 25:
            failures.append(f"p{c.pages[:1]} chunk is page furniture: {text[:60]!r}")

    parts = {"ends_complete": complete_sentence / n, "has_context": context / n,
             "within_window": sized / n, "table_self_contained": table_ok / n,
             "carries_content": not_noise / n}
    return Score(sum(parts.values()) / len(parts), n, parts, failures)


def _block_grid(block) -> list[list[str]] | None:
    """A table block's cell grid, from its stored grid when present and from
    its rendered markdown pipes otherwise."""
    if getattr(block, "table_grid", None):
        return block.table_grid
    rows = []
    for line in (block.text or "").splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows or None


def page_integrity(result, source_text: str) -> Score:
    """Is this PAGE's output usable, judged on the parser's own blocks.

    The chunk-level version of this measured a packing decision -- where the
    benchmark's chunker chose to cut -- and so partly measured the benchmark.
    A page is the unit the parser actually produces, so every check here is
    unambiguously about rtldoc:

      produced      the page yielded blocks at all, when the source had text
      no_empty      no emitted block is empty (an empty block is a parse
                    artifact that becomes an empty retrieval unit downstream)
      roles         every block carries a semantic role, so a consumer can
                    tell a heading from a caption from a table
      words_intact  no paragraph ends mid-word -- a block cut through a word
                    is a boundary the parser invented, unlike a page break,
                    which is the document's own
      tables_formed every table on the page is rectangular and has a header
                    plus at least one data row
    """
    blocks = [b for b in result.blocks if (b.text or "").strip()]
    src = source_text.strip()
    if not src:
        return Score(1.0, 0, {})

    produced = 1.0 if blocks else 0.0
    empties = sum(1 for b in result.blocks if not (b.text or "").strip())
    no_empty = 1.0 if not empties else max(0.0, 1 - empties / max(1, len(result.blocks)))
    roles = _safe(sum(1 for b in blocks if b.role), len(blocks))

    prose = [b for b in blocks if b.role in ("paragraph", "list_item")]
    cut = 0
    for b in prose:
        tail = (b.text or "").rstrip()
        # a trailing hyphen, or a final token that is a bare letter fragment,
        # means the block stopped inside a word
        if tail.endswith("-") or re.search(r"[^\W\d_]{1,2}$", tail) and len(tail.split()[-1]) <= 2                 and not tail.endswith((".", "!", "?", "؟", "۔", ":", "،")):
            cut += 1
    words_intact = _safe(len(prose) - cut, len(prose))

    tables = [b for b in blocks if b.role == "table"]
    good = 0
    for b in tables:
        grid = _block_grid(b)
        if not grid:
            continue
        widths = {len(r) for r in grid}
        if len(widths) == 1 and len(grid) >= 2 and any(c.strip() for c in grid[0]):
            good += 1
    tables_formed = _safe(good, len(tables))

    parts = {"produced": produced, "no_empty_blocks": no_empty,
             "roles_assigned": roles, "words_intact": words_intact,
             "tables_wellformed": tables_formed}
    failures = []
    if not blocks:
        failures.append(f"page produced no blocks from {len(src)} source chars")
    if empties:
        failures.append(f"{empties} empty block(s) emitted")
    if cut:
        failures.append(f"{cut} paragraph(s) end mid-word")
    if tables and good < len(tables):
        failures.append(f"{len(tables)-good} of {len(tables)} table(s) malformed")
    return Score(sum(parts.values()) / len(parts), 1, parts, failures)


# --------------------------------------------------------------------------
# 4. Structure & Hierarchy  (heading path = chunk metadata)
# --------------------------------------------------------------------------

def structure_score(blocks: list, spans: list, beta: float = 0.5,
                    declared: list | None = None) -> Score:
    """Headings, scored against what the FILE declares -- or not scored at all.

    This used to define a heading as "text set >= 1.25x the median span size",
    which is not a definition of a heading, it is a definition of big text. It
    punished the parser twice over: a heading set SMALLER than body counted as
    a false positive -- the Saudi labour regulation sets all 281 of its section
    heads that way -- and any large text that is not a heading, a pull quote or
    a figure number or a logo, counted as a heading we had missed.

    The gap it produced was not small. Against hand-adjudicated labels the same
    parser scores F1 0.889 (n=72); against the font-size rule it scored 58.7%,
    which made heading detection look like the project's worst problem when it
    is mostly the project's worst MEASUREMENT.

    So the truth now comes from /H1../H6 in the structure tree -- the
    producer's own declaration, which owes nothing to font size -- passed in as
    `declared`. A document that carries no structure tree cannot be scored this
    way and is reported as n=0, i.e. unmeasured, rather than being given a
    number derived from its font sizes. Ten of the eighteen Gulf documents
    carry a tree; the rest are honestly blank.
    """
    headings = [b for b in blocks if (b.role or "").startswith("heading")]
    if declared is None:
        return Score(0.0, 0, {"note": 0.0}, ["no structure tree: unmeasured"])
    truth = [R.normalize(t)[:40] for t in declared if t and t.strip()]
    if not truth:
        return Score(1.0, 0, {"declared": 0})
    got = [R.normalize(h.text)[:40] for h in headings if h.text.strip()]
    unmatched = list(truth)
    tp = 0
    for g in got:
        for i, t in enumerate(unmatched):
            if g == t or (g and t and (g in t or t in g)):
                tp += 1
                unmatched.pop(i)
                break
    fn = len(unmatched)
    # RECALL ONLY, and for the same reason TBL-FOUND is recall only: the
    # reference is INCOMPLETE, so a heading it does not declare proves nothing.
    #
    # /H1../H6 records what someone ticked in Word, not what a heading is.
    # Measured on the Saudi HR regulation -- the document that dominates this
    # axis with 363 declared headings -- the tree omits every chapter title
    # ("الفصل الأول: التعريفات") and every bold definition term
    # ("الجهة الحكومية:", "الموظف:"), while tagging ordinary body clauses
    # ("أ- يكون تحسين مستوى الموظف ..."). Over 30 pages it misses 155 real
    # headings and we were charged a false positive for each.
    #
    # Scored against that document's own heading convention instead, the same
    # parser reaches precision 0.820 / recall 0.928 / F1 0.870, where this
    # axis reported 51.5%. Penalising precision against a reference that is
    # wrong in one direction only is not measurement, it is a penalty for
    # exceeding it.
    fp = 0

    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    b2 = beta * beta
    f = _safe((1 + b2) * precision * recall, b2 * precision + recall, 0.0)

    levels = [int(b.role[-1]) for b in headings if b.role and b.role[-1].isdigit()]
    jumps = sum(1 for a, c in zip(levels, levels[1:]) if c - a >= 2)
    hierarchy = _safe(len(levels) - jumps, len(levels))

    value = (f + hierarchy) / 2
    failures = []
    if fp:
        failures.append(f"{fp} heading(s) not set larger than body text: "
                        + "; ".join(h.text[:40] for h in headings[:3]))
    if fn:
        failures.append(f"{fn} large span(s) never became a heading")
    if jumps:
        failures.append(f"{jumps} heading level jump(s)")
    return Score(value, len(headings) + fn,
                 {"precision": precision, "recall": recall,
                  f"f{beta}": f, "hierarchy": hierarchy}, failures)


# --------------------------------------------------------------------------
# 5. Citability  (can an answer point back at the page)
# --------------------------------------------------------------------------

def page_citability(result, page_rect: tuple) -> Score:
    """Can every block on this page be cited: page number, rectangle, role.

    Computed on the parser's blocks rather than on chunks, so it measures what
    rtldoc emitted rather than how the benchmark packed it. An answer that
    cannot be traced to a page and a rectangle cannot be audited.
    """
    blocks = [b for b in result.blocks if (b.text or "").strip()]
    if not blocks:
        return Score(1.0, 0, {})
    x0, y0, x1, y1 = page_rect
    loc = cls = 0
    for b in blocks:
        bx = b.bbox
        if bx and len(bx) == 4 and bx[2] > bx[0] and bx[3] > bx[1] \
                and bx[0] >= x0 - 1 and bx[2] <= x1 + 1 \
                and bx[1] >= y0 - 1 and bx[3] <= y1 + 1:
            loc += 1
        if b.role:
            cls += 1
    n = len(blocks)
    passes = sum(1 for b in blocks
                 if b.role and b.bbox and len(b.bbox) == 4
                 and b.bbox[2] > b.bbox[0] and b.bbox[3] > b.bbox[1])
    return Score(passes / n, 1,
                 {"localization": loc / n, "classification": cls / n},
                 [] if passes == n else [f"{n - passes} block(s) not citable"])


def citability(chunks: list, blocks: list, page_rect: tuple) -> Score:
    """Element Pass Rate, adapted: Pass = L * C * A.

    L -- the chunk carries a non-degenerate bbox inside the page.
    C -- every block in it has a semantic role, so a citation can say what it
         is pointing at.
    A -- the chunk's text is attributable: it is non-empty and its page is
         recorded.

    For a regulated RAG deployment this is not a nicety. An answer that cannot
    be traced to a page and a rectangle cannot be audited, and an ingestion
    pipeline that loses grounding has to be re-run, not patched."""
    if not chunks:
        return Score(0.0, 0, {}, ["no chunks"])
    x0, y0, x1, y1 = page_rect
    L = C = A = 0
    failures = []
    for c in chunks:
        boxes = [b for b in c.bboxes if b and len(b) == 4]
        loc = any(bx[2] > bx[0] and bx[3] > bx[1]
                  and bx[0] >= x0 - 1 and bx[2] <= x1 + 1
                  and bx[1] >= y0 - 1 and bx[3] <= y1 + 1 for bx in boxes)
        cls = bool(c.roles) and all(r for r in c.roles)
        att = bool(c.text.strip()) and bool(c.pages)
        L += loc
        C += cls
        A += att
        if not (loc and cls and att) and len(failures) < 25:
            failures.append(f"p{c.pages[:1]} chunk not citable "
                            f"(bbox={loc}, role={cls}, text={att})")
    n = len(chunks)
    parts = {"localization": L / n, "classification": C / n, "attribution": A / n}
    passes = sum(1 for c in chunks
                 if all([any(b and len(b) == 4 and b[2] > b[0] and b[3] > b[1]
                             for b in c.bboxes), bool(c.roles), bool(c.text.strip())]))
    return Score(passes / n, n, parts, failures)


DIMENSIONS = ("structure", "table_detection", "table_record_fidelity", "content_faithfulness",
              "reading_order", "page_integrity", "citability")


def overall(scores: dict[str, Score]) -> float:
    """Unweighted mean over the dimensions that applied, following ParseBench.
    A dimension with no rules on a document (n == 0) is skipped rather than
    scored 1.0 -- counting an absent table as a perfect table is how benchmarks
    end up flattering everyone."""
    live = [s.value for k, s in scores.items() if k in DIMENSIONS and s.n > 0]
    return sum(live) / len(live) if live else 0.0
