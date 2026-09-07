"""Full chain: layout model locates tables, SLANet+ builds the grid,
rtldoc supplies every character. Monkeypatched in-process; repo untouched."""
import sys, glob; sys.path.insert(0,".")
sys.path.insert(0,"/private/tmp/claude-501/-Users-umerjavaid-Documents-umerwork-AI-pdf-parser/999924f6-e6a0-4460-893a-bf2b496c73bd/scratchpad")
import numpy as np, fitz
from pathlib import Path
from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
import slanet
from rtldoc import pipeline
from rtldoc.layout import Region

W=glob.glob(str(Path.home())+"/.cache/huggingface/**/models/Layout/PP-DocLayoutV2", recursive=True)[0]
LAY=PPDocLayoutV2LayoutModel(W, device="cpu"); Z=1.6

def _table_boxes(page):
    pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z))
    img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
    out=[]
    for o in LAY.predict(img):
        lb=str(o.get("category") or o.get("label") or o.get("category_name") or "")
        if "table" not in lb.lower(): continue
        x0,y0,x1,y1=[float(v) for v in (o.get("bbox") or o.get("box"))]
        out.append((x0/Z,y0/Z,x1/Z,y1/Z))
    return out

def _inside(a,b,pad=8):
    return a[0]>=b[0]-pad and a[1]>=b[1]-pad and a[2]<=b[2]+pad and a[3]<=b[3]+pad

_adopt=pipeline._adopt_found_tables
def adopt(regions, page):
    """Replace PyMuPDF's find_tables with the layout model's table boxes."""
    if page is None: return regions
    try: boxes=_table_boxes(page)
    except Exception: return regions
    out=list(regions)
    for bb in boxes:
        ours=[r for r in out if r.kind=="table" and _inside(r.bbox, bb)]
        if len(ours)==1: continue          # we already have it, once
        for r in ours: out.remove(r)
        out.append(Region(bbox=bb, kind="table", cells=[]))
    return out
pipeline._adopt_found_tables = adopt

_grid=pipeline._table_grid
def grid(region, owned, opts, fills=None, page=None):
    if page is not None:
        try:
            g=slanet.predict(page, tuple(region.bbox))
            if g and len(g)>=2 and max(len(r) for r in g)>=2:
                return g, {"reversed_lines":0,"presentation_forms":0}
        except Exception: pass
    if not region.cells: return [], {}
    return _grid(region, owned, opts, fills, page)
pipeline._table_grid = grid

sys.argv=["rtldoc","parse",sys.argv[1],"--html",sys.argv[2]]
from rtldoc.cli import main
main()
