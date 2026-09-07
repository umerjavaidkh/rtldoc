"""SLANet+ for cell geometry, the PDF's own text layer for content.

The model never sees a character: it is handed a picture of the table and
OUR word boxes, and returns the grid with our text matched into it.
"""
import sys, json, re, statistics; sys.path.insert(0,".")
import numpy as np, fitz
from pathlib import Path
from mineru.model.table.rec.slanet_plus.main import PaddleTable, PaddleTableInput
from rtldoc import arabic

Z = 2.0   # render scale, pdf pt -> px
WEIGHTS = "/Users/umerjavaid/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/snapshots/ed6b654c018d742e65a17671e379c5e6ecc87ec9/models/TabRec/SlanetPlus/slanet-plus.onnx"
tbl = PaddleTable(PaddleTableInput(model_type="slanet_plus", model_path=WEIGHTS, device="cpu"))

def nm(t): return re.sub(r"[.…\s]+$","", " ".join(arabic.normalize(t or "")[0].split()).strip().lower())

def html_to_grid(h):
    rows=re.findall(r"<tr>(.*?)</tr>", h or "", re.S); g=[]
    for r in rows:
        c=[re.sub(r"<[^>]+>","",x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        if c: g.append(c)
    if g:
        w=max(len(r) for r in g); g=[r+[""]*(w-len(r)) for r in g]
    return g

def predict(page, bb):
    pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z), clip=fitz.Rect(*bb))
    img=np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n==4: img=img[:,:,:3]
    img=img[:,:,::-1].copy()                      # RGB -> BGR for cv2 conventions
    ocr=[]
    for w in page.get_text("words"):
        if w[0]<bb[0] or w[2]>bb[2] or w[1]<bb[1] or w[3]>bb[3]: continue
        x0=(w[0]-bb[0])*Z; y0=(w[1]-bb[1])*Z; x1=(w[2]-bb[0])*Z; y1=(w[3]-bb[1])*Z
        ocr.append(([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], w[4], 1.0))
    if not ocr: return []
    out=tbl.predict(img, ocr)
    return html_to_grid(out.pred_html)

def score(g,p):
    gt=[(r,c,nm(v)) for r,row in enumerate(g) for c,v in enumerate(row) if nm(v)]
    if not gt or not p: return 0.0
    return max(sum(1 for r,c,v in gt if 0<=r-dr<len(p) and c<len(p[r-dr]) and nm(p[r-dr][c])==v)
               for dr in range(-2,4))/len(gt)

if __name__ == "__main__":
    lim=int(sys.argv[1]) if len(sys.argv)>1 else 40
    acc=[]; shape=0; tot=0
    for stem in sorted(p.stem for p in Path("eval/fintabnet/gold").glob("*.json"))[:lim]:
        f=Path(f"eval/fintabnet/pdfs/{stem}.pdf")
        if not f.exists(): continue
        d=fitz.open(f); page=d[0]
        for t in json.loads(Path(f"eval/fintabnet/gold/{stem}.json").read_text()):
            cs=[c for c in (t.get("cells") or []) if c.get("pdf_bbox")]
            if not cs: continue
            nr=max(max(c["row_nums"]) for c in cs)+1; nc=max(max(c["column_nums"]) for c in cs)+1
            g=[["" for _ in range(nc)] for _ in range(nr)]
            for c in cs: g[min(c["row_nums"])][min(c["column_nums"])]=c.get("pdf_text_content") or ""
            if not any(nm(v) for row in g for v in row): continue
            tot+=1
            bb=(min(c["pdf_bbox"][0] for c in cs)-4, min(c["pdf_bbox"][1] for c in cs)-4,
                max(c["pdf_bbox"][2] for c in cs)+4, max(c["pdf_bbox"][3] for c in cs)+4)
            try: p2=predict(page, bb)
            except Exception as e: p2=[]
            s=score(g,p2); acc.append(s)
            if p2 and len(p2)==len(g) and len(p2[0])==len(g[0]): shape+=1
        d.close()
    n=max(len(acc),1)
    print(f"SLANet+ geometry + PDF text — {tot} tables")
    print(f"  exact grid shape : {100*shape/max(tot,1):5.1f}%")
    print(f"  cells recovered  : {100*sum(acc)/n:5.1f}%")
    print(f"  median per-table : {100*statistics.median(acc):5.1f}%")
