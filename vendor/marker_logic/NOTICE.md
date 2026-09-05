# Vendored from Marker (datalab-to/marker) v2.0.0

Source: https://github.com/datalab-to/marker
Licence: Apache License 2.0 (see LICENSE in this directory). Retained verbatim
as required; these files are unmodified copies.

## Why these files and not others

Only Marker's MODEL-FREE logic is here. Marker's layout, line, OCR and equation
stages call surya models, whose WEIGHTS carry a separate licence ("AI Pubs Open
Rail-M": free only under $5M revenue/funding). None of that is vendored, and
nothing here imports surya, so this directory is Apache-2.0 throughout and
carries no commercial restriction.

Charts/figures are deliberately absent: Marker has no model-free chart logic --
figure detection IS the surya layout model.

## What each file is for

  table_recon.py    the reason this directory exists. Reconstructs a table from
                    the PDF text layer with no model: builds ~6 candidate grids
                    (whitespace projection at 3 row-frequency thresholds x two
                    bucketings) and picks a winner with a CONTENT-aware judge --
                    column type purity, a compound-cell penalty, fill, header
                    agreement, spans-per-cell. Its projection grid needs no
                    drawn rules, which is exactly rtldoc's blind spot on
                    borderless statistical tables.
  sectionheader.py  heading levels by clustering line heights, rather than by
                    absolute font size.
  structure.py      assembling blocks into a document structure/reading order.
  line_merge.py     joining lines split across a layout boundary.
  ignoretext.py     repeated running text (headers/footers) by cross-page
                    frequency.
  marginalia.py, page_header.py, list.py, blank_page.py, block_relabel.py,
  footnote.py, text.py   supporting block classification.

## Status

REFERENCE ONLY -- not imported by rtldoc. Nothing in rtldoc/ depends on this
directory. It is here to be read and selectively reimplemented or adapted; if
any of it is adapted into rtldoc, the Apache-2.0 notice must travel with it.
