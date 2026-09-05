"""Harvest Arabic PDFs from Gulf institutional sites.

Why these sources: rtldoc targets English and Arabic, and the Arabic half of
the corpus was 36 Chrome-printed Wikipedia articles plus one textbook -- a
monoculture as narrow as the arXiv one on the Latin side. Government and
university documents from Saudi Arabia, the UAE, Oman and Qatar are the real
target domain: born-digital, table-heavy, produced by Word/InDesign/Acrobat
rather than a browser print dialog.

Only files that pass every filter are kept, because an unusable download is
worse than none: it must be a real PDF, born-digital (a scan measures OCR, not
extraction), predominantly Arabic, and small enough to be worth parsing.

Usage:
    python eval/fetch_gulf.py --out corpus_gulf --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
import urllib.parse as up
from pathlib import Path

import fitz
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Seeds are listing pages likely to link PDFs, one or two per institution.
SEEDS = [
    # Saudi Ministry of Human Resources -- Drupal, so its PDFs sit on a plain
    # static path (/sites/default/files/) and are actually harvestable. Most
    # Gulf portals are SharePoint apps that build their lists in JavaScript,
    # which is why the reachable-but-empty ones below are still worth seeding:
    # they cost one request each and occasionally expose a document link.
    "https://www.hrsd.gov.sa/knowledge-centre/decisions-and-regulations/regulation-and-procedures",
    "https://www.hrsd.gov.sa/en/knowledge-centre/decisions-and-regulations/regulation-and-procedures",
    "https://www.hrsd.gov.sa/knowledge-centre/decisions-and-regulations",
    "https://www.hrsd.gov.sa/knowledge-centre",
    "https://www.hrsd.gov.sa/ar/services",
    "https://www.hrsd.gov.sa/en/media-center/publications",
    "https://www.hrsd.gov.sa/ar/media-center",
    # Saudi social insurance -- contribution and entitlement regulations
    "https://www.gosi.gov.sa/ar/Regulations",
    "https://www.gosi.gov.sa/ar",
    # UAE identity/visa authority
    "https://icp.gov.ae/en/services/",
    "https://icp.gov.ae/ar/",
    # Oman Ministry of Labour
    "https://www.mol.gov.om/",
    "https://www.mol.gov.om/Laws",
]

_AR = re.compile(r"[ء-ي]")
MAX_BYTES = 40 * 1024 * 1024
MIN_ARABIC_FRAC = 0.30


def links_on(session, url: str, timeout: int = 20) -> tuple[list[str], list[str]]:
    """(pdf links, same-host page links) found on one page."""
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return [], []
        html = r.text
    except Exception:
        return [], []
    host = up.urlparse(r.url).netloc
    pdfs, pages = [], []
    for m in re.finditer(r'href=["\']([^"\']+)["\']', html, re.I):
        href = up.urljoin(r.url, m.group(1))
        p = up.urlparse(href)
        if p.netloc != host:
            continue
        (pdfs if p.path.lower().endswith(".pdf") else pages).append(href.split("#")[0])
    return list(dict.fromkeys(pdfs)), list(dict.fromkeys(pages))


def usable(path: Path) -> tuple[bool, str]:
    """Is this a born-digital, predominantly Arabic PDF worth keeping?"""
    try:
        doc = fitz.open(path)
    except Exception as exc:
        return False, f"unreadable ({exc.__class__.__name__})"
    try:
        n = doc.page_count
        if n == 0:
            return False, "no pages"
        sample = sorted({0, n // 3, (2 * n) // 3, n - 1})
        text = "".join(doc[i].get_text() for i in sample)
        if len(text.strip()) < 200 * len(sample) / 2:
            return False, "scanned / no text layer"
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False, "no letters"
        frac = sum(1 for c in letters if _AR.match(c)) / len(letters)
        if frac < MIN_ARABIC_FRAC:
            return False, f"only {frac:.0%} Arabic"
        return True, f"{n}p, {frac:.0%} Arabic"
    finally:
        doc.close()


def harvest(out: Path, limit: int, per_site: int, delay: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    seen_sha: set[str] = set()
    kept = 0
    log = []

    for seed in SEEDS:
        if kept >= limit:
            break
        host = up.urlparse(seed).netloc
        pdfs, pages = links_on(session, seed)
        # one level deeper, to reach the actual publication listings
        for page in pages[:60]:
            if len(pdfs) >= per_site * 3:
                break
            more, _ = links_on(session, page)
            pdfs.extend(p for p in more if p not in pdfs)
            time.sleep(delay)
        if not pdfs:
            log.append(f"  {host}: no PDF links found")
            continue

        got = 0
        for url in pdfs:
            if kept >= limit or got >= per_site:
                break
            try:
                r = session.get(url, timeout=45, stream=True, allow_redirects=True)
                if r.status_code != 200:
                    continue
                if "pdf" not in r.headers.get("content-type", "").lower():
                    continue
                body = b""
                for chunk in r.iter_content(1 << 16):
                    body += chunk
                    if len(body) > MAX_BYTES:
                        body = b""
                        break
                if not body:
                    continue
            except Exception:
                continue

            sha = hashlib.sha256(body).hexdigest()
            if sha in seen_sha:
                continue
            name = re.sub(r"[^\w.\-]+", "_", up.unquote(Path(up.urlparse(url).path).name))[:70]
            dest = out / f"{sha[:8]}_{name}"
            dest.write_bytes(body)
            ok, why = usable(dest)
            if not ok:
                dest.unlink(missing_ok=True)
                continue
            seen_sha.add(sha)
            kept += 1
            got += 1
            log.append(f"  KEPT {dest.name[:52]:54} {why:22} {len(body)//1024:>6} KB  {host}")
            time.sleep(delay)
        log.append(f"  {host}: kept {got}")

    print("\n".join(log))
    print(f"\n{kept} Arabic born-digital PDFs in {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_gulf")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--per-site", type=int, default=25)
    ap.add_argument("--delay", type=float, default=0.7)
    a = ap.parse_args()
    harvest(Path(a.out), a.limit, a.per_site, a.delay)
