import sys, json, collections
import fitz
sys.path.insert(0,".")
from rtldoc import pipeline
from eval.ragbench import rules, metrics

DOCS = ["600bb49d_user-manual-ar.pdf",
        "325b88ed_Statistical-Yearbook-2023.pdf",
        "8a1fac72_book-Statistics-2018.pdf",
        "ad90d515_Statistical-Yearbook-2022.pdf",
        "fd22e590_1Statistical-Yearbook-2021.pdf"]
MAXP = 80
buckets = collections.Counter(); rows=[]
for name in DOCS:
    doc = fitz.open("corpus_gulf/"+name)
    ruled_ok = rules.has_reliable_ruled_tables(doc)
    for i in range(min(doc.page_count, MAXP)):
        page = doc[i]
        try: res = pipeline.parse_page(page)
        except Exception: continue
        if not res.born_digital: continue
        truth = rules.build_page_truth(page, ruled_tables=ruled_ok)
        if not truth.record_groups: continue
        md = pipeline.to_markdown(res)
        preds = metrics.tables_from_markdown(md) or [[]]
        for group in truth.record_groups:
            best, bc = None, None
            for cand in preds:
                sc = metrics.table_record_fidelity(group, cand)
                if best is None or sc.value > best.value: best, bc = sc, cand
            if best is None: continue
            tk = list(group[0].keys()) if group else []
            pk = list(bc[0].keys()) if bc else []
            paired = bool(set(tk) & set(pk))
            buckets["tables"] += 1
            if best.value == 0: buckets["score_zero"] += 1
            if not bc: buckets["no_pred_table"] += 1
            if len(tk) != len(pk): buckets["ncols_differ"] += 1
            if not paired: buckets["header_unpaired"] += 1
            if paired and best.value < 0.5: buckets["paired_but_low"] += 1
            rows.append({"doc":name[:12],"page":i+1,"score":round(best.value,3),
                         "t_cols":len(tk),"p_cols":len(pk),"paired":paired,
                         "t_head":tk[:4],"p_head":pk[:4]})
    doc.close()
vals=[r["score"] for r in rows]
print(json.dumps(dict(buckets), indent=1))
print("mean score %.3f over %d tables" % (sum(vals)/max(1,len(vals)), len(vals)))
json.dump(rows, open("/private/tmp/claude-501/-Users-umerjavaid-Documents-umerwork-AI-pdf-parser/999924f6-e6a0-4460-893a-bf2b496c73bd/scratchpad/diag.json","w"), ensure_ascii=False)
