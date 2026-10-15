#!/usr/bin/env python3
"""Build the frozen offline reference sheet, the gold items and the DEV practice images (15 v0.5 §3–§4).

DEV gallery images only (no Reddit, no held-out). Deterministic: every selection uses the frozen seeds.
  - exemplars: 4 per eligible family via default_rng([20260934, family_index]) over members sorted by variant_id;
    both members for the two-member family
  - gold: default_rng(20260933), families in alphabetical order, 1 non-exemplar member each (30)
  - DEV practice: default_rng(20260938).choice over remaining eligible members -> 10 round-1 + 10 round-2
Outputs (to --out, default analysis/human_validation/reference/):
  reference_sheet.html (offline; images copied under img/; NO URLs/links), reference_manifest.csv (provenance),
  reference_assets.csv (every coder-visible img/ file with its SHA-256; verified by hv_common.freeze_ok),
  gold_items.csv, practice_dev.csv (answer keys; never shipped to coders except the practice feedback),
  reference_sheet_page1.png (a rendered page of the sheet for audit)
"""
import argparse
import html
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import hv_common as H


def select(dev, fams, eligible):
    ref, used = [], set()
    for fi, f in enumerate(fams):
        m = dev[dev.family_id == f].sort_values("variant_id").reset_index(drop=True)
        if f in eligible:
            idx = np.random.default_rng([H.SEED_EXEMPLAR, fi]).choice(len(m), H.N_EXEMPLARS, replace=False)
            pick = m.iloc[sorted(idx)]
        else:  # two-member family: both members
            pick = m
        for _, r in pick.iterrows():
            ref.append(dict(family_id=f, family_index=fi, display_name=H.display_name(f), variant_id=r.variant_id,
                            image_path=r.image_path, sha256=H.sha(r.image_path)))
            used.add(r.variant_id)
    ref = pd.DataFrame(ref)
    rng = np.random.default_rng(H.SEED_GOLD)
    gold = []
    for f in eligible:  # alphabetical (eligible is built from sorted fams)
        m = dev[(dev.family_id == f) & ~dev.variant_id.isin(used)].sort_values("variant_id").reset_index(drop=True)
        r = m.iloc[int(rng.choice(len(m)))]
        gold.append(dict(family_id=f, variant_id=r.variant_id, image_path=r.image_path, sha256=H.sha(r.image_path)))
        used.add(r.variant_id)
    gold = pd.DataFrame(gold)
    pool = dev[dev.family_id.isin(eligible) & ~dev.variant_id.isin(used)].sort_values("variant_id").reset_index(drop=True)
    idx = np.random.default_rng(H.SEED_PRACTICE_DEV).choice(len(pool), H.N_PRACTICE_DEV, replace=False)
    prac = pool.iloc[idx][["family_id", "variant_id", "image_path"]].reset_index(drop=True)
    prac["round"] = [1] * 10 + [2] * 10
    # disjointness (15 §4: exemplars and gold are disjoint; practice disjoint from both)
    assert not (set(ref.variant_id) & set(gold.variant_id)), "exemplar/gold overlap"
    assert not (set(prac.variant_id) & (set(ref.variant_id) | set(gold.variant_id))), "practice overlap"
    return ref, gold, prac


def render_html(ref, out):
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, g in ref.groupby("family_id", sort=True):
        cells = []
        for _, r in g.iterrows():
            ext = Path(r.image_path).suffix.lower() or ".img"
            name = f"{r.family_index:02d}_{r.sha256[:12]}{ext}"
            shutil.copy(r.image_path, img / name)
            cells.append(f'<img src="img/{name}" alt="">')
        dn = html.escape(g.display_name.iloc[0])
        rows.append(f'<section id="f{g.family_index.iloc[0]:02d}"><h2>{dn}</h2>'
                    f'<p class="n">{len(g)} examples</p><div class="row">{"".join(cells)}</div></section>')
    doc = ("<!doctype html><html><head><meta charset='utf-8'><title>Reference sheet</title><style>"
           "body{font-family:sans-serif;margin:16px}section{margin:0 0 24px}h2{font-size:18px;margin:4px 0}"
           ".n{color:#666;margin:0 0 6px;font-size:12px}.row{display:flex;gap:8px;flex-wrap:wrap}"
           ".row img{height:180px;border:1px solid #ccc}</style></head><body>"
           "<h1>Meme family reference sheet</h1><p>Families in alphabetical order. Use only this sheet.</p>"
           + "".join(rows) + "</body></html>")
    assert "http" not in doc and "href" not in doc, "URLs/links must not appear in the coder-facing sheet"
    (out / "reference_sheet.html").write_text(doc)


def write_asset_manifest(out):
    """The stimulus actually shown to coders: every img/ file and its SHA-256 (sorted by file name)."""
    rows = [dict(file=p.name, sha256=H.sha(p), bytes=p.stat().st_size)
            for p in sorted((out / "img").iterdir(), key=lambda q: q.name) if p.is_file()]
    pd.DataFrame(rows).to_csv(out / "reference_assets.csv", index=False)


def render_png(ref, out, n_fam=4):
    """A rendered page (first n_fam families) as PNG, for the implementation audit."""
    from PIL import Image, ImageDraw
    fams = sorted(ref.family_id.unique())[:n_fam]
    H_ROW, W = 220, 1100
    page = Image.new("RGB", (W, 40 + H_ROW * len(fams)), "white")
    d = ImageDraw.Draw(page)
    d.text((10, 10), "Meme family reference sheet (page 1)", fill="black")
    y = 40
    for f in fams:
        g = ref[ref.family_id == f]
        d.text((10, y), f"{g.display_name.iloc[0]}  ({len(g)} examples)", fill="black")
        x = 10
        for _, r in g.iterrows():
            im = Image.open(r.image_path).convert("RGB")
            im.thumbnail((250, 180))
            page.paste(im, (x, y + 20))
            x += 260
        y += H_ROW
    page.save(out / "reference_sheet_page1.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-index", default=str(H.DEV_INDEX))
    ap.add_argument("--out", default=str(H.HV / "reference"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev, fams, eligible = H.load_dev(a.dev_index)
    ref, gold, prac = select(dev, fams, eligible)
    ref.to_csv(out / "reference_manifest.csv", index=False)
    gold.to_csv(out / "gold_items.csv", index=False)
    prac.to_csv(out / "practice_dev.csv", index=False)
    render_html(ref, out)
    write_asset_manifest(out)
    render_png(ref, out)
    print(f"reference: {ref.family_id.nunique()} families, {len(ref)} exemplars; gold {len(gold)}; practice DEV {len(prac)}")


if __name__ == "__main__":
    main()
