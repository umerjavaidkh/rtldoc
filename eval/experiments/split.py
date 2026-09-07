"""Report Arabic and Latin separately. Never pool them: the large corpus is
84% arXiv English, so a pooled mean is an English score with a trace of
Arabic in it."""
import json, sys, collections
NAMES=[("table_detection","TBL-FOUND"),("citability","CITATION"),
       ("reading_order","COLUMN"),("page_integrity","PAGE"),
       ("content_faithfulness","TEXT"),("structure","HEADING"),
       ("table_record_fidelity","TABLE")]

def script_of(stratum, path):
    s=(stratum or "")+" "+(path or "")
    if "corpus_gulf" in s or "corpus_hr" in s: return "ARABIC"
    if "arabic" in s or "mixed_ar" in s: return "ARABIC"
    if "latin" in s: return "LATIN"
    return "OTHER"

def rows(docs):
    out={}
    for key,label in NAMES:
        v=[d["dims"][key]["score"] for d in docs
           if (d.get("dims") or {}).get(key) and d["dims"][key].get("n",0)>0]
        out[label]=(100*sum(v)/len(v), len(v)) if v else (None,0)
    return out

def show(title, docs):
    if not docs: return
    pages=sum(d.get("pages",0) for d in docs)
    print(f"\n### {title} — {len(docs)} docs / {pages} pages")
    r=rows(docs)
    for label,(v,n) in r.items():
        print(f"   {label:11s} {'--' if v is None else f'{v:5.1f}%'}   (scored on {n} docs)")

for path in sys.argv[1:]:
    d=json.load(open(path)); docs=d["documents"]
    print(f"\n{'='*58}\n{path}   {len(docs)} documents")
    by=collections.defaultdict(list)
    for x in docs: by[script_of(x.get("stratum"), x.get("pdf"))].append(x)
    for k in ("ARABIC","LATIN","OTHER"): show(k, by[k])
