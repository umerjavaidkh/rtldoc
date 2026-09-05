# table_recon.py, vendored from Marker (datalab-to/marker) v2.0.0

Source:  https://github.com/datalab-to/marker  (marker/processors/table_recon.py)
Licence: Apache License 2.0 -- see LICENSE here, retained verbatim as required.
The file is an unmodified copy.

## Why only this file

Marker reconstructs digital tables with NO model -- its own docstring says
"There is no dedicated table-structure model". A model is reached only as a
fallback when the judge score falls below min_recon_score, and even then it is
OCR of the table crop (surya RecognitionPredictor), not table structure.

So the whole of Marker's table capability for born-digital PDFs is this one
file, and it is stdlib-only: no marker imports, no surya, no weights. That
makes it Apache-2.0 with no commercial restriction -- the "$5M revenue" limit
belongs to the surya model WEIGHTS, which this file never touches.

Marker's heading and chart handling were NOT taken: detection there IS the
layout model, so there is no logic to borrow (no Marker file ever assigns
BlockTypes.SectionHeader itself).

## What it does, and why it is worth merging

Builds ~6 candidate grids and picks between them with a content-aware judge:

    PROJ_FRACS = (0.01, 0.03, 0.10)   x   bucketings ("x0", "center")
    _score_grid = purity + no_compound + fill + hdr + 2*one_span  / 6

Two properties rtldoc does not have:

  * _grid_proj needs NO drawn rules -- a column exists where span coverage
    exceeds a fraction of the rows. That is precisely rtldoc's blind spot:
    the Statistical Yearbook draws zero vertical rules on pages whose tables
    we currently reach only by other means, and where our own benchmark
    reference finds nothing at all.

  * Candidates scored against each other replace hand-tuned tolerances.
    rtldoc accumulated ~15 of them in one week, and one pair had to be
    measured at 0.175 against 0.204 to tell two documents apart -- a gap no
    threshold could have been guessed into.

Its compound-cell penalty ("a text cell that is 40% digits means columns were
merged") is rtldoc's welded-cell defect -- "50%" split across a boundary --
expressed as a score rather than as another special case.

## Status

REFERENCE ONLY. Nothing in rtldoc/ imports it.

Next step when merging: port _score_grid alone and run it over the 43
hand-verified tables in eval/ragbench/gold/gold.json, to see whether
candidate-selection beats rtldoc's current 0.625 before changing the parser.
