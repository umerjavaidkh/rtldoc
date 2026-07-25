"""
Arabic text repair for PDF-extracted strings.

Three failure modes are handled, in this order:
  1. Presentation forms  (U+FB50-FDFF, U+FE70-FEFF) leaking into the text layer
     because the PDF's ToUnicode map points at shaped glyphs, not base letters.
  2. Visual-order strings (logically reversed) produced by extractors that do no
     bidi reordering, or partially-reversed strings caused by ligature glyphs
     (lam-alef) that survive reversal as a unit.
  3. Cosmetic noise: kashida/tatweel, non-breaking marks, digit systems.

Nothing here is lossy by default. Diacritics (harakat) are PRESERVED because
vowelled text is the whole point of a primary-school Arabic corpus.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --------------------------------------------------------------------------
# character classes
# --------------------------------------------------------------------------

PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
ARABIC_BLOCK = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
HARAKAT = re.compile(r"[\u064B-\u0652\u0670\u0640]")   # incl. tatweel
TATWEEL = "\u0640"
INVISIBLES = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u2066\u2067\u2068\u2069"), None)

ARABIC_INDIC = {ord(c): ord(d) for c, d in zip("٠١٢٣٤٥٦٧٨٩", "0123456789")}
EASTERN_INDIC = {ord(c): ord(d) for c, d in zip("۰۱۲۳۴۵۶۷۸۹", "0123456789")}

# High-frequency Arabic function words. Used as a language-model-free
# orientation test: a correctly-ordered string contains many of these,
# a reversed one contains almost none.
STOPWORDS = frozenset(
    (
        "في من على إلى عن أن إن التي الذي هذا هذه ذلك مع كان قد هو هي هم "
        "بعد قبل بين كل عند ثم أو حتى لكن كما حيث منذ نحو لدى وفي ومن والتي "
        "الهدف إجابات النص التلاميذ القراءة السؤال إجابة الصف الدرس"
    ).split()
)
_WORD = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
_STRIP_HARAKAT = re.compile(r"[\u064B-\u0652\u0670\u0640]")
AL = "\u0627\u0644"          # definite article, word-INITIAL in logical order
AL_REV = "\u0644\u0627"      # what it looks like at word-END when reversed


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def is_arabic(text: str, threshold: float = 0.25) -> bool:
    """True if a meaningful fraction of the letters are Arabic script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    hits = sum(1 for c in letters if ARABIC_BLOCK.match(c))
    return hits / len(letters) >= threshold


def deshape(text: str) -> str:
    """Map Arabic presentation forms back to base letters, char by char.

    We deliberately do NOT run NFKC over the whole string: NFKC would also
    fold full-width Latin, superscripts and some math symbols that carry
    meaning in a textbook. We only touch the two presentation blocks.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in PRESENTATION_RANGES):
            decomposed = unicodedata.normalize("NFKC", ch)
            out.append(decomposed)
        else:
            out.append(ch)
    return "".join(out)


def _score_orientation(text: str) -> int:
    """How 'logically-ordered' is this Arabic string?

    Two independent signals, both computed on whole tokens (substring matching
    is worthless here -- 'لا' and 'ما' occur inside half the words in the
    language and fire constantly on reversed text):

      * exact function-word hits, weighted x3
      * definite-article position: 'ال' opens roughly a third of tokens in
        written Arabic. Reversed, it closes them instead. The difference
        between prefix-count and suffix-count is a very strong, very cheap
        orientation signal that needs no dictionary.
    """
    tokens = [_STRIP_HARAKAT.sub("", t) for t in _WORD.findall(text)]
    if not tokens:
        return 0
    stops = sum(1 for t in tokens if t in STOPWORDS)
    prefix = sum(1 for t in tokens if len(t) > 3 and t.startswith(AL))
    suffix = sum(1 for t in tokens if len(t) > 3 and t.endswith(AL_REV))
    return 3 * stops + (prefix - suffix)


MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def _reverse_preserving_runs(text: str) -> str:
    """Reverse the string, but keep Latin/digit runs internally intact and
    mirror directional punctuation.

    Two things a naive `text[::-1]` gets wrong:
      * an embedded '2021' or 'PDF' was already laid out LTR inside the RTL
        flow, so reversing char-by-char corrupts it into '1202'
      * '(' in visual order is the *closing* paren in logical order
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.,:/\-]*|.", text, re.S)
    return "".join(reversed(tokens)).translate(MIRROR)


def fix_order(text: str) -> tuple[str, bool]:
    """Return (text_in_logical_order, was_reversed).

    Uses a stopword-frequency vote rather than trusting the extractor. This
    catches the nasty middle case: strings that are *mostly* reordered but
    where lam-alef ligatures stayed put, which no bidi library will detect
    for you because the string is already 'valid' Unicode.
    """
    if not is_arabic(text):
        return text, False
    forward = _score_orientation(text)
    flipped = _reverse_preserving_runs(text)
    backward = _score_orientation(flipped)
    if backward > forward:
        return flipped, True
    return text, False


_LEAD_PUNCT = re.compile(r"^([.،؛:!؟?]+)\s*(.+)$", re.S)
_LEAD_ORDINAL = re.compile(r"^([.\u0660-\u06690-9]{1,3})\s*[.\u061B]\s*", re.S)


def _repair_punctuation(text: str) -> str:
    """Move sentence-final punctuation that the reversal stranded at the head.

    Full stops, commas and question marks are Unicode-neutral characters: they
    take direction from context, so in a visual-order stream they sit at the
    physical left edge and land at string-start after reversal.
    """
    t = text.strip()
    m = _LEAD_PUNCT.match(t)
    if m and is_arabic(m.group(2)):
        t = f"{m.group(2).strip()}{m.group(1)}"
    # list ordinals: '.. في العبارة1' -> '1. في العبارة'
    m2 = re.match(r"^\.\s*(.*?)([\u0660-\u06690-9]{1,2})\.?$", t, re.S)
    if m2 and is_arabic(m2.group(1)):
        t = f"{m2.group(2)}. {m2.group(1).strip()}."
    return t


@dataclass
class NormalizeOptions:
    strip_tatweel: bool = True
    strip_harakat: bool = False        # keep vowels: this is a teaching corpus
    western_digits: bool = False       # keep ٢٢ as ٢٢ unless asked
    collapse_space: bool = True


def normalize(text: str, opts: NormalizeOptions | None = None) -> tuple[str, dict]:
    """Full repair pipeline. Returns (clean_text, diagnostics)."""
    opts = opts or NormalizeOptions()
    diag = {"had_presentation_forms": False, "was_reversed": False}
    text = text.translate(INVISIBLES)

    has_pf = any(any(lo <= ord(c) <= hi for lo, hi in PRESENTATION_RANGES) for c in text)
    diag["had_presentation_forms"] = has_pf

    # ORDER MATTERS. Reverse BEFORE deshaping.
    #
    # A lam-alef ligature is a single codepoint (U+FEFB..U+FEFC) in the
    # content stream. If we deshape first it becomes two codepoints, and the
    # subsequent reversal flips them: لا -> ال. Every word containing 'لا'
    # silently corrupts, which is the failure behind PyMuPDF #2199 and the
    # reason so much Arabic PDF text "looks almost right". Reversing while
    # the ligature is still atomic keeps it intact.
    probe = deshape(text) if has_pf else text          # score on readable text
    _, reversed_ = fix_order(probe)
    if reversed_:
        text = _reverse_preserving_runs(text)
    diag["was_reversed"] = reversed_

    if has_pf:
        text = deshape(text)
    text = _repair_punctuation(text)

    if opts.strip_tatweel:
        text = text.replace(TATWEEL, "")
    if opts.strip_harakat:
        text = HARAKAT.sub("", text)
    if opts.western_digits:
        text = text.translate(ARABIC_INDIC).translate(EASTERN_INDIC)
    if opts.collapse_space:
        text = re.sub(r"[ \t\u00a0]+", " ", text).strip()

    return text, diag
