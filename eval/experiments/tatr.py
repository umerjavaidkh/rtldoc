"""TATR for geometry, rtldoc for text."""
import sys, json, re, statistics; sys.path.insert(0,".")
import torch, fitz
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from rtldoc import arabic

MODEL="microsoft/table-transformer-structure-recognition"
proc=AutoImageProcessor.from_pretrained(MODEL)
model=TableTransformerForObjectDetection.from_pretrained(MODEL).eval()
DPI=150; Z=DPI/72.0
def nm(t): return re.sub(r"[.…\s]+$","", " ".join(arabic.normalize(t or "")[0].split()).strip().lower())

def structure(page, bb):
    pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z), clip=fitz.Rect(*bb))
    img=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    with torch.no_grad():
        out=model(**proc(images=img, return_tensors="pt"))
    res=proc.post_process_object_detection(out, threshold=0.75,
                                           target_sizes=[(img.height,img.width)])[0]
    rows=[]; cols=[]
    for sc,lb,box in zip(res["scores"],res["labels"],res["boxes"]):
        name=model.config.id2label[int(lb)]
        x0,y0,x1,y1=[float(v) for v in box]
        # image px -> pdf pts, offset by the crop origin
        r=(bb[0]+x0/Z, bb[1]+y0/Z, bb[0]+x1/Z, bb[1]+y1/Z)
        if name=="table row": rows.append(r)
        elif name=="table column": cols.append(r)
    # DETR proposes many overlapping boxes for the same column. Without
    # de-duplication a 6-column table comes back with 12 columns (and one
    # 2-column table with 35), which is what the reference implementation's
    # objects-to-cells step exists to prevent.
    def dedup(boxes, axis, iou=0.4):
        boxes=sorted(boxes, key=lambda b: b[axis+2]-b[axis], reverse=True)
        kept=[]
        for b in boxes:
            a0,a1=b[axis],b[axis+2]
            hit=False
            for k in kept:
                k0,k1=k[axis],k[axis+2]
                ov=min(a1,k1)-max(a0,k0)
                if ov>0 and ov/max(min(a1-a0,k1-k0),1e-6)>iou: hit=True; break
            if not hit: kept.append(b)
        return kept
    rows=dedup(rows,1); cols=dedup(cols,0)
    rows.sort(key=lambda r:r[1]); cols.sort(key=lambda c:c[0])
    return rows, cols

def grid_from(page, rows, cols):
    if not rows or not cols: return []
    words=page.get_text("words")
    g=[]
    for r in rows:
        line=[]
        for c in cols:
            cell=(c[0], r[1], c[2], r[3])
            txt=[w[4] for w in words
                 if min(w[2],cell[2])-max(w[0],cell[0])>0.5*(w[2]-w[0])
                 and min(w[3],cell[3])-max(w[1],cell[1])>0.5*(w[3]-w[1])]
            line.append(" ".join(txt))
        g.append(line)
    return g

found=shape=tot=0; acc=[]
stems=sorted(p.stem for p in Path("eval/fintabnet/gold").glob("*.json"))[:60]
for stem in stems:
    f=Path(f"eval/fintabnet/pdfs/{stem}.pdf")
    if not f.exists(): continue
    d=fitz.open(f); page=d[0]
    for t in json.loads(Path(f"eval/fintabnet/gold/{stem}.json").read_text()):
        cs=[c for c in (t.get("cells") or []) if c.get("pdf_bbox")]
        if not cs: continue
        nr=max(max(c["row_nums"]) for c in cs)+1; ncg=max(max(c["column_nums"]) for c in cs)+1
        gg=[["" for _ in range(ncg)] for _ in range(nr)]
        for c in cs: gg[min(c["row_nums"])][min(c["column_nums"])]=c.get("pdf_text_content") or ""
        gt=[(r,c,nm(v)) for r,row in enumerate(gg) for c,v in enumerate(row) if nm(v)]
        if not gt: continue
        tot+=1
        bb=(min(c["pdf_bbox"][0] for c in cs)-4, min(c["pdf_bbox"][1] for c in cs)-4,
            max(c["pdf_bbox"][2] for c in cs)+4, max(c["pdf_bbox"][3] for c in cs)+4)
        try:
            rows,cols=structure(page, bb); g=grid_from(page, rows, cols)
        except Exception as e:
            g=[]
        if not g: acc.append(0.0); continue
        found+=1
        best=max(sum(1 for r,c,v in gt if 0<=r-dr<len(g) and c<len(g[r-dr]) and nm(g[r-dr][c])==v)
                 for dr in range(-2,4))
        acc.append(best/len(gt))
        if len(g)==len(gg) and len(g[0])==len(gg[0]): shape+=1
    d.close()
n=max(len(acc),1)
print(f"TATR geometry + rtldoc text — {tot} tables")
print(f"  produced a grid  : {100*found/max(tot,1):5.1f}%")
print(f"  exact grid shape : {100*shape/max(tot,1):5.1f}%")
print(f"  cells recovered  : {100*sum(acc)/n:5.1f}%")
print(f"  median per-table : {100*statistics.median(acc):5.1f}%")
