"""rtldoc CLI:  python -m rtldoc.cli parse book.pdf --pages 1-5 --md out/"""
import argparse, json, os, sys
import fitz
from . import pipeline, primitives, arabic

def _pages(spec, n):
    if not spec: return list(range(n))
    out=[]
    for part in spec.split(","):
        if "-" in part:
            a,b=part.split("-"); out += list(range(int(a)-1,int(b)))
        else: out.append(int(part)-1)
    return [p for p in out if 0 <= p < n]

def main(argv=None):
    ap = argparse.ArgumentParser(prog="rtldoc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parse"); p.add_argument("pdf"); p.add_argument("--pages")
    p.add_argument("--style-map"); p.add_argument("--json"); p.add_argument("--md")
    p.add_argument("--html", help="write one HTML page per PDF page, plus styles.css and index.html")
    p.add_argument("--no-geo", action="store_true")
    p.add_argument("--strip-harakat", action="store_true")

    s = sub.add_parser("styles", help="style census -> label once per book series")
    s.add_argument("pdf"); s.add_argument("--pages"); s.add_argument("--out")

    a = sub.add_parser("audit"); a.add_argument("pdf"); a.add_argument("--pages")

    args = ap.parse_args(argv)
    doc = fitz.open(args.pdf); idx = _pages(getattr(args,"pages",None), doc.page_count)

    if args.cmd == "styles":
        prof = primitives.style_profile([primitives.extract_page(doc[i]) for i in idx])
        payload = {e["style"]: "" for e in prof}
        print(json.dumps(prof, ensure_ascii=False, indent=2)[:4000])
        if args.out:
            open(args.out,"w").write(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"\n-> blank style map written to {args.out}; fill in roles and pass with --style-map", file=sys.stderr)
        return

    style_map = json.load(open(args.style_map)) if getattr(args,"style_map",None) else None
    opts = arabic.NormalizeOptions(strip_harakat=getattr(args,"strip_harakat",False))
    results = [pipeline.parse_page(doc[i], style_map, opts,
                                   geometry_bidi=not getattr(args,"no_geo",False)) for i in idx]

    if args.cmd == "audit":
        print(json.dumps(pipeline.audit(results), ensure_ascii=False, indent=2)); return

    img_dir = args.html or args.md or (os.path.dirname(args.json) if args.json else None)
    if img_dir:
        n = pipeline.save_images(doc, results, os.path.join(img_dir, "images"))
        if n:
            print(f"images -> {os.path.join(img_dir, 'images')}/ ({n})", file=sys.stderr)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        open(args.json,"w").write(pipeline.to_json(results)); print(f"json -> {args.json}")
    if args.md:
        os.makedirs(args.md, exist_ok=True)
        for r in results:
            open(os.path.join(args.md, f"page_{r.page:04d}.md"),"w").write(pipeline.to_markdown(r))
        print(f"markdown -> {args.md}/")
    if args.html:
        title = os.path.splitext(os.path.basename(args.pdf))[0]
        pipeline.save_html(results, args.html, title_prefix=title)
        print(f"html -> {args.html}/ ({len(results)} pages, index.html)")
    if not args.json and not args.md and not args.html:
        for r in results: print(pipeline.to_markdown(r))

if __name__ == "__main__":
    main()
