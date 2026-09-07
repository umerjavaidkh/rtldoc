"""Full chain on Arabic: layout model locates, SLANet+ structures, rtldoc reads.
Scored against the same RAGBench reference that gives TABLE 31.2%."""
import sys, glob, statistics; sys.path.insert(0,".")
sys.path.insert(0,"/private/tmp/claude-501/-Users-umerjavaid-Documents-umerwork-AI-pdf-parser/999924f6-e6a0-4460-893a-bf2b496c73bd/scratchpad")
import numpy as np, fitz
from pathlib import Path
from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
from slanet import predict
from rtldoc import pipeline
from eval.ragbench import rules, metrics

W=glob.glob(str(Path.home())+"/.cache/huggingface/**/models/Layout/PP-DocLayoutV2", recursive=True)[0]
lay=PPDocLayoutV2LayoutModel(W, device="cpu"); Z=1.6

def table_boxes(page):
    pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z))
    img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)[:,:,:3]
    out=[]
    for o in lay.predict(img):
        lb=str(o.get("category") or o.get("label") or o.get("category_name") or "")
        if "table" not in lb.lower(): continue
        x0,y0,x1,y1=[float(v) for v in (o.get("bbox") or o.get("box"))]
        out.append((x0/Z,y0/Z,x1/Z,y1/Z))
    return out

def to_records(g):
    if not g or len(g)<2: return []
    hdr=[h or f"col{i}" for i,h in enumerate(g[0])]; out=[]
    for row in g[1:]:
        v={hdr[i]: row[i] for i in range(min(len(hdr),len(row))) if row[i].strip()}
        if v: out.append(v)
    return out

ours=[]; chain=[]; pages=0
for path in sorted(glob.glob("corpus_gulf/*.pdf")):
    doc=fitz.open(path); ok=rules.has_reliable_ruled_tables(doc)
    for i in range(doc.page_count):
        page=doc[i]
        truth=rules.build_page_truth(page, ruled_tables=ok)
        if not truth.record_groups: continue
        try: res=pipeline.parse_page(page)
        except Exception: continue
        if not res.born_digital: continue
        pages+=1
        pred_ours=metrics.tables_from_markdown(pipeline.to_markdown(res)) or [[]]
        pred_chain=[]
        try:
            for bb in table_boxes(page):
                r=to_records(predict(page, bb))
                if r: pred_chain.append(r)
        except Exception: pass
        for grp in truth.record_groups:
            ours.append(max((metrics.table_record_fidelity(grp,c).value for c in pred_ours), default=0.0))
            chain.append(max((metrics.table_record_fidelity(grp,c).value for c in pred_chain), default=0.0))
    doc.close()
n=max(len(ours),1)
print(f"Gulf — {len(ours)} truth tables over {pages} pages")
print(f"  rtldoc rules only            : {100*sum(ours)/n:5.1f}%")
print(f"  layout model + SLANet+ + ours: {100*sum(chain)/n:5.1f}%")
print(f"  median   rules {100*statistics.median(ours):.1f}%   chain {100*statistics.median(chain):.1f}%")
