import sys, json, re, statistics; sys.path.insert(0,".")
import torch, fitz
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from rtldoc import pipeline, arabic
M="microsoft/table-transformer-structure-recognition"
proc=AutoImageProcessor.from_pretrained(M); model=TableTransformerForObjectDetection.from_pretrained(M).eval()
Z=150/72.0
def nm(t): return re.sub(r"[.…\s]+$","", " ".join(arabic.normalize(t or "")[0].split()).strip().lower())
def dedup(bx,ax,iou=0.4):
    bx=sorted(bx,key=lambda b:b[ax+2]-b[ax],reverse=True); k=[]
    for b in bx:
        if not any((min(b[ax+2],x[ax+2])-max(b[ax],x[ax]))>0 and
           (min(b[ax+2],x[ax+2])-max(b[ax],x[ax]))/max(min(b[ax+2]-b[ax],x[ax+2]-x[ax]),1e-6)>iou for x in k):
            k.append(b)
    return k
def tatr(page,bb):
    pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z), clip=fitz.Rect(*bb))
    img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    with torch.no_grad(): o=model(**proc(images=img,return_tensors="pt"))
    r=proc.post_process_object_detection(o,threshold=0.75,target_sizes=[(img.height,img.width)])[0]
    rows=[];cols=[]
    for lb,box in zip(r["labels"],r["boxes"]):
        n=model.config.id2label[int(lb)]; x0,y0,x1,y1=[float(v) for v in box]
        R=(bb[0]+x0/Z,bb[1]+y0/Z,bb[0]+x1/Z,bb[1]+y1/Z)
        if n=="table row": rows.append(R)
        elif n=="table column": cols.append(R)
    rows=sorted(dedup(rows,1),key=lambda x:x[1]); cols=sorted(dedup(cols,0),key=lambda x:x[0])
    if not rows or not cols: return []
    w=page.get_text("words"); g=[]
    for rr in rows:
        g.append([" ".join(x[4] for x in w
                  if min(x[2],cc[2])-max(x[0],cc[0])>0.5*(x[2]-x[0])
                  and min(x[3],rr[3])-max(x[1],rr[1])>0.5*(x[3]-x[1])) for cc in cols])
    return g
def sc(g,p):
    gt=[(r,c,nm(v)) for r,row in enumerate(g) for c,v in enumerate(row) if nm(v)]
    if not gt or not p: return 0.0
    return max(sum(1 for r,c,v in gt if 0<=r-dr<len(p) and c<len(p[r-dr]) and nm(p[r-dr][c])==v)
               for dr in range(-2,4))/len(gt)
def degenerate(p):
    """Runtime test -- no gold needed."""
    if not p: return True
    nc=max(len(r) for r in p)
    if nc<2 or len(p)<2: return True
    cells=[c for r in p for c in r]
    return sum(1 for c in cells if c and c.strip())/max(len(cells),1) < 0.30

A=[];B=[];R=[];routed_to_model=0; tot=0
for stem in sorted(p.stem for p in Path("eval/fintabnet/gold").glob("*.json"))[:120]:
    f=Path(f"eval/fintabnet/pdfs/{stem}.pdf")
    if not f.exists(): continue
    d=fitz.open(f); page=d[0]
    res=pipeline.parse_page(page); ours=[b.table_grid for b in res.blocks if b.table_grid]
    for t in json.loads(Path(f"eval/fintabnet/gold/{stem}.json").read_text()):
        cs=[c for c in (t.get("cells") or []) if c.get("pdf_bbox")]
        if not cs: continue
        nr=max(max(c["row_nums"]) for c in cs)+1; ncg=max(max(c["column_nums"]) for c in cs)+1
        g=[["" for _ in range(ncg)] for _ in range(nr)]
        for c in cs: g[min(c["row_nums"])][min(c["column_nums"])]=c.get("pdf_text_content") or ""
        if not any(nm(v) for row in g for v in row): continue
        tot+=1
        bb=(min(c["pdf_bbox"][0] for c in cs)-4, min(c["pdf_bbox"][1] for c in cs)-4,
            max(c["pdf_bbox"][2] for c in cs)+4, max(c["pdf_bbox"][3] for c in cs)+4)
        best=max(ours,key=lambda p:sc(g,p)) if ours else []
        a=sc(g,best); A.append(a)
        try: tg=tatr(page,bb)
        except Exception: tg=[]
        b=sc(g,tg); B.append(b)
        if degenerate(best):
            routed_to_model+=1; R.append(b)
        else: R.append(a)
    d.close()
n=max(len(A),1)
print(f"{tot} tables")
print(f"  rtldoc only        : {100*sum(A)/n:5.1f}%")
print(f"  TATR only          : {100*sum(B)/n:5.1f}%")
print(f"  ROUTED             : {100*sum(R)/n:5.1f}%   ({routed_to_model} of {tot} sent to the model)")
