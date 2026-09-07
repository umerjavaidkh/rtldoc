"""Generate HTML with SLANet+ supplying table geometry and rtldoc the text.

Monkeypatched in-process only; nothing in the repo changes.
"""
import sys; sys.path.insert(0,".")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from rtldoc import pipeline
import slanet

_orig = pipeline._table_grid
def _model_grid(region, owned, opts, fills=None, page=None):
    """Ask the model for the grid; fall back to the rules if it declines."""
    if page is not None:
        try:
            g = slanet.predict(page, tuple(region.bbox))
            if g and len(g) >= 2 and max(len(r) for r in g) >= 2:
                return g, {"reversed_lines": 0, "presentation_forms": 0}
        except Exception:
            pass
    return _orig(region, owned, opts, fills, page)
pipeline._table_grid = _model_grid

sys.argv = ["rtldoc", "parse", sys.argv[1], "--html", sys.argv[2]]
from rtldoc.cli import main
main()
