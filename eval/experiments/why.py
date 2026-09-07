import sys, collections; sys.path.insert(0,".")
import fitz
from rtldoc import pipeline
from eval.ragbench import rules, metrics
DOCS=["600bb49d_user-manual-ar.pdf","325b88ed_Statistical-Yearbook-2023.pdf",
      "8a1fac72_book-Statistics-2018.pdf","ad90d515_Statistical-Yearbook-2022.pdf",
      "fd22e590_1Statistical-Yearbook-2021.pdf"]
c=collections.Counter()
for name in DOCS:
    doc=fitz.open("corpus_gulf/"+name); ok=rules.has_reliable_ruled_tables(doc)
    for i in range(min(doc.page_count,80)):
        page=doc[i]
        try: res=pipeline.parse_page(page)
        except Exception: continue
        if not res.born_digital: continue
        truth=rules.build_page_truth(page, ruled_tables=ok)
        if not truth.record_groups: continue
        rot = sum(1 for b in page.get_text("dict")["blocks"] if b.get("type")==0
                  for l in b["lines"] if abs(l["dir"][0])<0.7)
        hor = sum(1 for b in page.get_text("dict")["blocks"] if b.get("type")==0
                  for l in b["lines"]) or 1
        rotated = rot/hor > 0.5
        md=pipeline.to_markdown(res); preds=metrics.tables_from_markdown(md) or [[]]
        ntab_pred=len([b for b in res.blocks if b.table_grid])
        for g in truth.record_groups:
            best,bc=None,None
            for cand in preds:
                sc=metrics.table_record_fidelity(g,cand)
                if best is None or sc.value>best.value: best,bc=sc,cand
            if best is None: continue
            t=len(g[0].keys()) if g else 0; p=len(bc[0].keys()) if bc else 0
            if p>=t: c["ok"]+=1; continue
            c["short"]+=1
            if rotated: c["short_rotated"]+=1
            elif ntab_pred>len(truth.record_groups): c["short_table_split"]+=1
            else:
                # our grid wide enough but row has blanks?
                widest=max((len(b.table_grid[0]) for b in res.blocks if b.table_grid), default=0)
                if widest>=t: c["short_blank_cells"]+=1
                else: c["short_grid_narrow"]+=1
print(dict(c))
