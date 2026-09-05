# Vendored from Marker (datalab-to/marker) v2.0.0

Source:  https://github.com/datalab-to/marker
Licence: Apache License 2.0 -- see LICENSE here, retained verbatim as required.
Files under src/ are unmodified copies.

## Scope

The whole package is copied EXCEPT chart/figure block definitions, which were
excluded deliberately:

    schema/blocks/figure.py     schema/blocks/picture.py
    schema/groups/figure.py     schema/groups/picture.py

122 of Marker's 126 modules are here.

## Licence, precisely

Marker's own CODE is Apache-2.0 with no commercial restriction, including the
modules that call surya. The "$5M revenue/funding" limit belongs to the surya
MODEL WEIGHTS, which are not redistributed here and are not in this repository
at all. So everything under src/ may be read, adapted and shipped commercially,
provided the Apache-2.0 notice travels with anything adapted.

Practically: modules that import surya cannot RUN without those weights, even
though their source may be freely reused.

## What is worth reading first

  processors/table_recon.py   The reason this directory exists, and model-free
        (Marker's own docstring: "There is no dedicated table-structure
        model"). Reconstructs a table from the PDF text layer by building ~6
        candidate grids -- whitespace-gap projection at three row-frequency
        thresholds x two bucketings -- and choosing between them with a
        CONTENT-aware judge: column type purity, a compound-cell penalty,
        fill, header agreement, spans-per-cell at double weight.

        Two properties rtldoc lacks. Its projection needs no drawn rules,
        which is exactly where rtldoc is blind: the Statistical Yearbook
        draws zero vertical rules on pages whose tables we reach only by
        other means. And generating hypotheses and scoring them replaces
        hand-tuned tolerances -- rtldoc accumulated roughly fifteen this
        week, one pair of which had to be measured at 0.175 against 0.204
        to separate two documents. Its compound-cell penalty ("a text cell
        that is 40% digits means columns were merged") is rtldoc's welded-
        cell defect stated as a score instead of another special case.

  processors/sectionheader.py  Heading LEVELS by KMeans over line heights.
        Note this only levels blocks the surya layout model has ALREADY
        classified as headers -- no vendored file ever assigns
        BlockTypes.SectionHeader. Marker's heading DETECTION is the model,
        so it cannot be borrowed as logic. rtldoc's weakest axis (HEADING
        58.6%) gets no help from this directory.

  builders/structure.py        Assembling blocks into document structure.
  processors/line_merge.py     Joining lines split across a layout boundary.
  processors/ignoretext.py     Running heads/feet by cross-page frequency.
  providers/                   PDF text-layer extraction (pdftext).
  renderers/                   Block tree -> HTML / markdown / JSON.

## Status

REFERENCE ONLY. Nothing in rtldoc/ imports this directory, and it is not on
the package path. It exists to be read and selectively adapted.
