"""Fetch the FinTabNet.c evaluation set: PDFs plus cell-level ground truth.

    python -m eval.fintabnet_fetch      # ~142 MB into eval/fintabnet/

545 financial-report pages carrying 723 annotated tables. The annotations
give row_nums and column_nums per cell, so SPANNING cells and multi-level
headers are ground truth -- structure this parser has no way to express
and, until this set existed, no way to measure.

Two sources, because neither half is complete on its own:
  * annotations  bsmock/FinTabNet.c        (CDLA-Permissive-2.0)
  * the PDFs     corvicai/FinTabNet_ComTQA (the only mirror that kept them)

PubTables-1M is the better-known set and is useless here: it ships cropped
IMAGES for training vision models, and this parser reads glyph coordinates.
IBM's original FinTabNet CDN no longer resolves.
"""
from __future__ import annotations
import json, os, shutil, tarfile
from pathlib import Path

ANN_REPO = "bsmock/FinTabNet.c"
PDF_REPO = "corvicai/FinTabNet_ComTQA"
OUT = Path("eval/fintabnet")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download
    (OUT / "gold").mkdir(parents=True, exist_ok=True)
    (OUT / "pdfs").mkdir(parents=True, exist_ok=True)

    api = HfApi()
    pdf_files = [s.rfilename for s in api.dataset_info(PDF_REPO).siblings
                 if s.rfilename.startswith("pdfs/")]
    have = {os.path.basename(n)[:-4] for n in pdf_files}

    tar = hf_hub_download(ANN_REPO, "FinTabNet.c-PDF_Annotations.tar.gz",
                          repo_type="dataset")
    with tarfile.open(tar) as t:
        members = {os.path.basename(n)[:-len("_tables.json")]: n
                   for n in t.getnames() if n.endswith("_tables.json")}
        wanted = sorted(have & set(members))
        print(f"{len(wanted)} pages have both a PDF and ground truth")
        for stem in wanted:
            dst = OUT / "gold" / f"{stem}.json"
            if not dst.exists():
                dst.write_text(json.dumps(json.load(t.extractfile(members[stem]))))

    for n in pdf_files:
        stem = os.path.basename(n)[:-4]
        dst = OUT / "pdfs" / f"{stem}.pdf"
        if stem in wanted and not dst.exists():
            shutil.copy(hf_hub_download(PDF_REPO, n, repo_type="dataset"), dst)
    print(f"pdfs {len(list((OUT/'pdfs').glob('*.pdf')))}, "
          f"gold {len(list((OUT/'gold').glob('*.json')))} -> {OUT}")


if __name__ == "__main__":
    main()
