---
name: pdf-forensics
description: Root-cause methodology for PDF extraction and layout bugs — garbled or duplicated text, wrong reading order, broken tables, mis-typed headings, RTL/Arabic corruption. Use when output looks wrong and the cause is not yet known, when a fix needs a discriminator between two look-alike cases, or when asked why a page parses badly.
---

# PDF forensics

The rule that produces every real fix here: **the signal is geometric, not
linguistic.** Text-level rules guess. Glyph coordinates, advance widths and
drawing operators are facts the file states outright.

Corollary: **measure before fixing.** Every failed attempt in this codebase
started by guessing a rule and testing it on one page.

## Where a bug can live — bisect the pipeline

```
PDF content stream
  └── rawdict / texttrace     ← extraction (phantom or missing glyphs)
      └── Span (primitives)   ← span text assembly
      └── Glyph (geobidi)     ← position-based line rebuild
          └── lines           ← baseline grouping, bidi runs, word gaps
              └── regions     ← columns, tables, panels (layout)
                  └── blocks  ← roles, dedupe, captions (pipeline)
```

Find the **lowest** layer where output is already wrong. Fixing above that
point papers over it and usually breaks something else.

## Extraction layer: rawdict vs texttrace

These two disagree, and the disagreement is diagnostic. `get_texttrace()`
reports the glyphs **actually drawn**, with glyph IDs; rawdict reports
characters after PyMuPDF's own processing.

```python
for sp in page.get_texttrace():
    for cp, gid, org, adv, *_ in sp["chars"]:
        print(chr(cp), gid, org)   # org = (x, y) baseline origin
```

Real case: a numbered list showed the number twice. texttrace reported one
`'1'` (gid 20) at x=672.3 and a **space** at x=659.5; rawdict reported a
second `'1'` there. The phantom digit entered before any layout code ran —
so no amount of layout fixing would have helped.

**If output has a character the page does not visibly show, compare the two
before touching anything downstream.**

## Glyph widths carry the truth

Advance width distinguishes cases that look identical as text:

- **zero-width glyph** = a mark, or half of a ligature that is drawn as one
  piece. Real case: the lam-alef ligature emits a zero-width alef plus a
  **double-width** lam (6.08 vs 3.05 normal) carrying both letters.
- **same char, overlapping boxes** = a duplicated layer (drop shadow / faux
  bold). Real case: headings drawn twice, offset dx 0.23 / dy 1.05pt, merged
  into one baseline and interleaved character by character.
- **same char, clearly separate boxes** = genuinely doubled letters. Arabic
  `اللغة` has two real lams. Do not "fix" these.

That last contrast is the whole method: find the property that separates the
broken population from the legitimate one, and *count both* before coding.

## Finding a discriminator — the required step

Before writing a fix, produce a table like this:

```
zero-width alef followed by lam   :  4218   <- the corruption
zero-width alef followed by other :     6   <- noise
normal-width alef followed by lam : 16897   <- legitimate, must not change
lam width after zero-width alef   :  6.08 (median)
lam width after normal alef       :  3.05 (median)
```

If the two populations do not separate cleanly, **you do not have a fix yet**.
Two cases from this codebase where they did not separate, and the change was
reverted rather than shipped on a guess:

- a table-candidate selection where the confirmed-good and confirmed-bad
  options sat **0.017 apart** in score
- a multi-column index whose real column matched a marker column on every
  measured signal (width ratio 0.47, row-match 0.97, shortness 0.83)

Reverting is the correct outcome there. Shipping a threshold that cannot
separate the cases just moves the bug.

## Render the region and look at it

When geometry is confusing, stop reasoning and render:

```python
page.get_pixmap(matrix=fitz.Matrix(6, 6), clip=fitz.Rect(x0, y0, x1, y1)).save("/tmp/crop.png")
```

Then Read the PNG. A crop settled the phantom-digit case in one step after
several rounds of speculation about the text stream.

## Duplicated text: it is always two regions claiming one line

Every duplication bug in this parser had the same shape — two overlapping
regions each rendering the same content. Check ownership, not the renderer:

- a small region and the container enclosing it **both** score containment
  1.0, so the winner is decided by iteration order
- geobidi pads line bboxes, so a line genuinely inside a tight region can
  measure 0.994 against it and exactly 1.000 against the roomier one — an
  exact-tie test misses this; use "essentially contained" (>= 0.95)
- ownership must be resolved **once**, by the tightest region

## Bidi and RTL

- **Do not apply rule L4 (bracket mirroring) unconditionally.** Some
  producers store the visual shape, some the logical character. Decide per
  line by which reading nests better; on a tie, leave the stored characters
  alone.
- Reorder **before** deshaping, so a ligature is still one codepoint.
- Any glyph reorder must also fix the glyph's **position**, or downstream
  word-gap measurement inserts a false space mid-word. Real case: reordering
  the lam-alef alef without moving it left produced `الأ نواع` for `الأنواع`.
- Uncased scripts return `False` from `str.islower()`. Case-based rules are
  therefore safe for Arabic by construction — but say so explicitly.

## Rotated text

Every step of a line rebuild assumes glyphs share a **y** and advance in
**x**. Rotated text is the transpose: shared x, advancing in y. Left
unhandled, each glyph becomes its own one-character "line" and interleaves
with real body text. Check `line["dir"]` — `(0,-1)` reads up the page.

## Verify like a scientist

1. Reproduce the user's exact case first and quote the actual bad output.
2. Find the lowest broken layer.
3. Count both populations; show the separation.
4. Fix; re-measure the **same** numbers.
5. Run the full corpus, not a sample (see the `release-gate` skill).
6. Report what is still broken. A partial fix reported honestly is worth
   more than a complete-sounding one that quietly moved the failure.
