"""SLANet+ geometry + rtldoc text, scored on the Gulf corpus against the
same RAGBench reference that yields TABLE 31.2%."""
import sys, glob, os, statistics; sys.path.insert(0,".")
sys.path.insert(0,"/private/tmp/claude-501/-Users-umerjavaid-Documents-umerwork-AI-pdf-parser/999924f6-e6a0-4460-893a-bf2b496c73bd/scratchpad")
import fitz
from slanet import predict          # model wrapper, weights already pinned
from rtldoc import pipeline
from eval.ragbench import rules, metrics

def grid_to_records(g):
    if not g or len(g)<2: return []
    hdr=[h or f"col{i}" for i,h in enumerate(g[0])]
    out=[]
    for row in g[1:]:
        v={hdr[i]: row[i] for i in range(min(len(hdr),len(row))) if row[i].strip()}
        if v: out.append(v)
    return out

ours=[]; slan=[]; pages=0
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
        md=pipeline.to_markdown(res)
        pred_ours=metrics.tables_from_markdown(md) or [[]]
        regions=[b for b in res.blocks if b.table_grid]
        pred_sl=[]
        for b in regions:
            try:
                g=predict(page, tuple(b.bbox))
                r=grid_to_records(g)
                if r: pred_sl.append(r)
            except Exception: pass
        if not pred_sl:
            try:
                g=predict(page, tuple(page.rect))
                r=grid_to_records(g)
                if r: pred_sl.append(r)
            except Exception: pass
        for grp in truth.record_groups:
            ours.append(max((metrics.table_record_fidelity(grp,c).value for c in pred_ours), default=0.0))
            slan.append(max((metrics.table_record_fidelity(grp,c).value for c in pred_sl), default=0.0))
    doc.close()
n=max(len(ours),1)
print(f"Gulf corpus — {len(ours)} truth tables over {pages} pages")
print(f"  rtldoc (current)          : {100*sum(ours)/n:5.1f}%")
print(f"  SLANet+ geometry + our text: {100*sum(slan)/n:5.1f}%")
print(f"  median  rtldoc {100*statistics.median(ours):.1f}%   SLANet+ {100*statistics.median(slan):.1f}%")
