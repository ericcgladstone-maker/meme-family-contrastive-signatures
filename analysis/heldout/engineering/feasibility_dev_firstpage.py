#!/usr/bin/env python3
"""Engineering evidence (pre-freeze; DEV entries only; NO held-out candidate is touched).

Question: under robots-compliant retrieval, how many usable variants does a family yield?
For DEV entry slugs only:
  (a) first gallery page /memes/<slug>/photos: gallery items (data-is-photo anchors linked to the entry),
      original image URL (data-image), per-image NSFW class flag;
  (b) entry page /memes/<slug>: KYM photo URLs embedded in the entry body (option A in 14 decision D7);
download originals, apply D1 (sha256) and D2 (pHash <= 6, greedy, keep first) within family, and report
counts for (a) alone and (a) u (b). No embedding, no scoring.
Output: out/feasibility_dev_firstpage.json
"""
import hashlib
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import imagehash
from PIL import Image

OUT = Path(__file__).resolve().parent / "out"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
DEV = ["distracted-boyfriend", "roll-safe", "this-is-fine", "trade-offer", "galaxy-brain", "disaster-girl",
       "doge", "is-this-a-pigeon", "surprised-pikachu", "stonks", "hide-the-pain-harold", "woman-yelling-at-a-cat"]


def get(u, binary=False):
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=40)
    b = r.read()
    return b if binary else b.decode("utf-8", "ignore")


def gallery_items(html, slug):
    out = []
    for a, body in re.findall(r'(<a\b[^>]*data-is-photo="true"[^>]*>)(.*?)</a>', html, flags=re.S):
        if f'data-entry-link="/memes/{slug}"' not in a:
            continue
        imgs = re.findall(r"<img\b[^>]*>", body)
        url, nsfw = None, False
        for im in imgs:
            m = re.search(r'data-image="(https://i\.kym-cdn\.com/photos/images/original/[^"]+)"', im)
            if m and not url:
                url = m.group(1)
            if re.search(r'class="[^"]*\bnsfw-img\b', im):
                nsfw = True
        out.append({"url": url, "nsfw_flag": nsfw, "has_image": url is not None})
    return out


def entry_photo_urls(html):
    """KYM photo images inside the entry body (id='entry_body' ... up to the comments block).
    Body images are served at 'newsfeed' (or 'original') size; mapped to the 'original' path."""
    m = re.search(r"id=[\"']entry_body[\"']", html)
    if not m:
        return []
    end = html.find("id='comments'", m.end())
    body = html[m.end(): end if end > 0 else len(html)]
    urls = re.findall(r'https://i\.kym-cdn\.com/photos/images/(?:newsfeed|original)/([0-9/]+[^"\' )<]+)', body)
    return sorted(set("https://i.kym-cdn.com/photos/images/original/" + u for u in urls))


def dedup(images):
    seen_sha, reps, kept = set(), [], []
    d1 = d2 = 0
    for url, b in images:
        s = hashlib.sha256(b).hexdigest()
        if s in seen_sha:
            d1 += 1
            continue
        seen_sha.add(s)
        try:
            h = imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"), hash_size=8)
        except Exception:
            continue
        if any((h - r) <= 6 for r in reps):
            d2 += 1
            continue
        reps.append(h)
        kept.append(url)
    return len(kept), d1, d2


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    for slug in DEV:
        g = get(f"https://knowyourmeme.com/memes/{slug}/photos"); time.sleep(6)
        e = get(f"https://knowyourmeme.com/memes/{slug}"); time.sleep(6)
        items = gallery_items(g, slug)
        eurls = entry_photo_urls(e)
        imgs_a, imgs_b = [], []
        for it in items:
            if it["url"]:
                try:
                    imgs_a.append((it["url"], get(it["url"], True)))
                except Exception:
                    pass
                time.sleep(1)
        gal_set = {u for u, _ in imgs_a}
        for u in eurls:
            if u in gal_set:
                continue
            try:
                imgs_b.append((u, get(u, True)))
            except Exception:
                pass
            time.sleep(1)
        na, a1, a2 = dedup(imgs_a)
        nab, b1, b2 = dedup(imgs_a + imgs_b)
        res[slug] = dict(gallery_items=len(items), gallery_with_image=sum(i["has_image"] for i in items),
                         gallery_nsfw_flagged=sum(i["nsfw_flag"] for i in items), gallery_downloaded=len(imgs_a),
                         gallery_after_dedup=na, gallery_d1=a1, gallery_d2=a2,
                         entry_extra_urls=len(eurls), union_after_dedup=nab)
        print(slug, res[slug], flush=True)
    vals = [r["gallery_after_dedup"] for r in res.values()]
    uni = [r["union_after_dedup"] for r in res.values()]
    res["_summary"] = dict(gallery_after_dedup=sorted(vals), union_after_dedup=sorted(uni),
                           share_gallery_ge10=sum(v >= 10 for v in vals) / len(vals),
                           share_gallery_ge12=sum(v >= 12 for v in vals) / len(vals),
                           share_gallery_ge15=sum(v >= 15 for v in vals) / len(vals),
                           share_union_ge15=sum(v >= 15 for v in uni) / len(uni))
    (OUT / "feasibility_dev_firstpage.json").write_text(json.dumps(res, indent=2))
    print(res["_summary"])


if __name__ == "__main__":
    main()
