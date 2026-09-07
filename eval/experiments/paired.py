import sys, collections; sys.path.insert(0,".")
import fitz
from rtldoc import pipeline
from eval.ragbench import rules, metrics
DOCS=["600bb49d_user-manual-ar.pdf","325b88ed_Statistical-Yearbook-2023.pdf",
      "8a1fac72_book-Statistics-2018.pdf","ad90d515_Statistical-Yearbook-2022.pdf",
      "fd22e590_1Statistical-Yearbook-2021.pdf"]
def run():
    out={}
    for name in DOCS:
        doc=fitz.open("corpus_gulf/"+name); ok=rules.has_reliable_ruled_tables(doc)
        for i in range(min(doc.page_count,80)):
            page=doc[i]
            try: res=pipeline.parse_page(page)
            except Exception: continue
            if not res.born_digital: continue
            truth=rules.build_page_truth(page, ruled_tables=ok)
            if not truth.record_groups: continue
            md=pipeline.to_markdown(res); preds=metrics.tables_from_markdown(md) or [[]]
            for gi,g in enumerate(truth.record_groups):
                best=max((metrics.table_record_fidelity(g,c).value for c in preds), default=0.0)
                out[(name,i,gi)]=best
        doc.close()
    return out
after=run()
orig=pipeline._adopt_found_tables
pipeline._adopt_found_tables=lambda r,p: r
before=run()
keys=sorted(set(before)|set(after))
up=[k for k in keys if after.get(k,0)-before.get(k,0)>0.02]
dn=[k for k in keys if before.get(k,0)-after.get(k,0)>0.02]
same=len(keys)-len(up)-len(dn)
print(f"tables {len(keys)}   improved {len(up)}   worsened {len(dn)}   unchanged {same}")
print(f"mean before {sum(before.get(k,0) for k in keys)/len(keys):.3f}"
      f"   after {sum(after.get(k,0) for k in keys)/len(keys):.3f}")
def band(d):
    c=collections.Counter()
    for k in keys:
        v=d.get(k,0.0)
        c["0.0"]+= v==0; c["0-0.2"]+= 0<v<0.2; c["0.2-0.5"]+= 0.2<=v<0.5; c["0.5+"]+= v>=0.5
    return dict(c)
print("before bands:", band(before)); print("after  bands:", band(after))
print("biggest gains:", sorted(((round(after.get(k,0)-before.get(k,0),2), k[0][:12], k[1]+1) for k in up), reverse=True)[:6])
print("biggest losses:", sorted(((round(after.get(k,0)-before.get(k,0),2), k[0][:12], k[1]+1) for k in dn))[:6])
