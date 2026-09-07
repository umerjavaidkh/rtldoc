"""Grid from clustered word edges -- the property every table has.

A column exists where MANY ROWS independently begin or end text at the same
x. That is what makes text a table; prose has one left edge and a ragged
right. Unlike whitespace projection it survives a spanning cell, because a
single wide row cannot outvote the rows that agree.
"""
import collections

def rows_of(words, tol=3.0):
    ws=sorted(words, key=lambda w:(round((w[1]+w[3])/2,1), w[0]))
    rows=[]; cur=[]; last=None
    for w in ws:
        cy=(w[1]+w[3])/2
        if last is None or abs(cy-last)<=tol: cur.append(w)
        else: rows.append(cur); cur=[w]
        last=cy if last is None else (last+cy)/2 if abs(cy-last)<=tol else cy
    if cur: rows.append(cur)
    return [sorted(r, key=lambda w:w[0]) for r in rows]

def column_edges(rows, min_support=0.35, tol=4.0):
    """x positions where many rows start a word."""
    n=len(rows)
    if n<3: return []
    # Vote on BOTH edges. A numeric column is right-aligned, so its left
    # edges scatter with the number's width while its right edges agree
    # exactly -- counting only starts misses every money column there is.
    votes=collections.Counter()
    for r in rows:
        seen=set()
        for w in r:
            for x in (w[0], w[2]):
                k=round(x/tol)
                if k not in seen:
                    votes[k]+=1; seen.add(k)
    need=max(3, int(n*min_support))
    xs=sorted(k*tol for k,v in votes.items() if v>=need)
    # merge near-duplicates
    out=[]
    for x in xs:
        if not out or x-out[-1]>tol*1.5: out.append(x)
    return out

def grid(words, min_support=0.35):
    rows=rows_of(words)
    edges=column_edges(rows, min_support)
    if len(edges)<2: return []
    spans=[(edges[i], edges[i+1] if i+1 < len(edges) else float("inf"))
           for i in range(len(edges))]
    g=[]
    for r in rows:
        cells=[""]*len(spans)
        for w in r:
            # the column this word OVERLAPS most, not the one it starts in
            best, bi = 0.0, 0
            for i,(a,b) in enumerate(spans):
                ov=min(w[2], b)-max(w[0], a)
                if ov>best: best, bi = ov, i
            cells[bi]=(cells[bi]+" "+w[4]).strip()
        g.append(cells)
    return g
