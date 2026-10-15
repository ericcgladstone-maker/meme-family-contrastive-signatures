#!/usr/bin/env python3
"""Build the expanded-corpus feature bundle for the held-out validation (14 §7-§8, §14 E3).

Modes:
  --dev-modern      : compute SigLIP2 + DINOv2 embeddings for the 680 DEV images only (E2; pre-freeze allowed)
  --expanded        : (post-freeze only) DEV 678 (30 eligible families) + analyzable held-out families from
                      heldout/out/heldout_manifest.csv -> heldout/out/expanded_features.npz + expanded_texts.json
Every feature uses the frozen extractor: OpenAI CLIP ViT-L/14 (unnormalized) -> MSAE; LAION CLIP; pHash
(imagehash, 8); OCR (ocrmac defaults, fragments joined by ' '); SigLIP2 get_image_features; DINOv2 CLS.
DEV features are reused from the G3 bundle (identical extractors), never recomputed differently.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OUT = HERE / "out"
G3 = HERE.parent / "g3_recovery" / "out"
MANIFEST = json.loads((HERE / "engineering" / "out" / "checkpoint_manifest.json").read_text()) \
    if (HERE / "engineering" / "out" / "checkpoint_manifest.json").exists() else None


def modern_embed(paths):
    """SigLIP2 + DINOv2 per 14 §7 (CPU, float32, default processors at pinned revisions)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    torch.set_grad_enabled(False)
    sig = MANIFEST["siglip2"]
    din = MANIFEST["dinov2"]
    sp = AutoImageProcessor.from_pretrained(sig["repo"], revision=sig["revision"], use_fast=False)
    sm = AutoModel.from_pretrained(sig["repo"], revision=sig["revision"], torch_dtype=torch.float32).eval()
    dp = AutoImageProcessor.from_pretrained(din["repo"], revision=din["revision"], use_fast=False)
    dm = AutoModel.from_pretrained(din["repo"], revision=din["revision"], torch_dtype=torch.float32).eval()
    S, D = [], []
    for i in range(0, len(paths), 8):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
        s = sm.get_image_features(**sp(images=imgs, return_tensors="pt"))
        d = dm(**dp(images=imgs, return_tensors="pt")).last_hidden_state[:, 0, :]
        S.append(s.float().numpy())
        D.append(d.float().numpy())
    return np.vstack(S), np.vstack(D)


def dev_modern():
    z = np.load(G3 / "features.npz", allow_pickle=True)
    ids = list(z["ids"])
    idx = {r["variant_id"]: r for r in csv.DictReader(C.CFD_INDEX.open(newline=""))}
    paths = [idx[v]["image_path"] for v in ids]
    S, D = modern_embed(paths)
    # determinism check: re-embed the first 8 images
    S2, D2 = modern_embed(paths[:8])
    res = dict(n=len(ids), siglip2_dim=int(S.shape[1]), dinov2_dim=int(D.shape[1]),
               finite=bool(np.isfinite(S).all() and np.isfinite(D).all()),
               rerun_max_abs_diff_siglip2=float(np.abs(S[:8] - S2).max()),
               rerun_max_abs_diff_dinov2=float(np.abs(D[:8] - D2).max()),
               siglip2_norm_range=[float(np.linalg.norm(S, axis=1).min()), float(np.linalg.norm(S, axis=1).max())],
               dinov2_norm_range=[float(np.linalg.norm(D, axis=1).min()), float(np.linalg.norm(D, axis=1).max())])
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "dev_modern_embeddings.npz", ids=np.array(ids, object), siglip2=S, dinov2=D)
    (HERE / "engineering" / "out" / "e2_dev_modern.json").write_text(json.dumps(res, indent=2))
    print(res)


def ocr_texts(paths):
    from ocrmac import ocrmac as om
    out = []
    for pth in paths:
        try:
            res = om.OCR(pth).recognize()
            out.append(" ".join(t[0] for t in res).strip() if res else "")
        except Exception:  # noqa: BLE001  (as fr_08: errors -> empty text)
            out.append("")
    return out


def clip_embed(paths, key, l2):
    from src.analysis.clip_embed import embed_image_paths
    recs = [{"variant_id": str(i), "image_path": p} for i, p in enumerate(paths)]
    r = embed_image_paths(recs, pretrained_key=key, device="auto", l2_normalize=l2)
    assert [int(x) for x in r["ids"]] == list(range(len(paths))), "embedding dropped images"
    return r["vectors"]


def extractor_check():
    """Re-embed 8 DEV images with the frozen extractors and compare to stored G3 features (E3 test)."""
    z = np.load(G3 / "features.npz", allow_pickle=True)
    ids = list(z["ids"])[:8]
    idx = {r["variant_id"]: r for r in csv.DictReader(C.CFD_INDEX.open(newline=""))}
    paths = [idx[v]["image_path"] for v in ids]
    vo = clip_embed(paths, "openai", False)
    vl = clip_embed(paths, "laion", True)
    dense, _, _ = C.sae_dense(ids, vo)
    from src.analysis import surface_baselines as SB
    _, bits = SB.phash_vectors([{"variant_id": v, "image_path": p} for v, p in zip(ids, paths)])
    cos = lambda a, b: float(((a / np.linalg.norm(a, axis=1, keepdims=True)) * (b / np.linalg.norm(b, axis=1, keepdims=True))).sum(1).min())
    res = dict(openai_min_cos=cos(vo, z["clip_openai"][:8]), openai_max_abs=float(np.abs(vo - z["clip_openai"][:8]).max()),
               laion_min_cos=cos(vl, z["clip_laion"][:8]), dense_max_abs=float(np.abs(dense - z["dense"][:8]).max()),
               phash_identical=bool((bits.astype(np.uint8) == z["phash_bits"][:8]).all()))
    (HERE / "engineering" / "out" / "e3_extractor_check.json").write_text(json.dumps(res, indent=2))
    print(res)


def expanded():
    """POST-FREEZE ONLY: DEV (30 eligible families) + analyzable held-out families -> engine bundle."""
    import csv as _csv
    if not (HERE / "FREEZE_RECORD.json").exists():
        raise SystemExit("REFUSING: not frozen")
    z = np.load(G3 / "features.npz", allow_pickle=True)
    dev_ids, dev_fam = list(z["ids"]), list(z["fam"])
    order = []
    for f in dev_fam:  # G3 family enumeration order (k convention)
        if f not in order:
            order.append(f)
    sizes = {f: dev_fam.count(f) for f in order}
    keep = [i for i, f in enumerate(dev_fam) if sizes[f] >= 5]
    dm = np.load(OUT / "dev_modern_embeddings.npz", allow_pickle=True)
    assert list(dm["ids"]) == dev_ids
    texts_dev = json.loads((G3 / "ocr_texts.json").read_text())
    rows = list(_csv.DictReader(open(OUT / "heldout_manifest.csv")))
    paths = [r["image_path"] for r in rows]
    new_fam = ["ho_" + r["family_slug"] for r in rows]
    vo = clip_embed(paths, "openai", False)
    vl = clip_embed(paths, "laion", True)
    dense, _, _ = C.sae_dense([f"ho{i}" for i in range(len(paths))], vo)
    from src.analysis import surface_baselines as SB
    pid, bits = SB.phash_vectors([{"variant_id": str(i), "image_path": p} for i, p in enumerate(paths)])
    assert [int(x) for x in pid] == list(range(len(paths)))
    S, D = modern_embed(paths)
    tx = ocr_texts(paths)
    k_keys, k_vals, tgt, fset = [], [], [], []
    for k, f in enumerate(order):
        if sizes[f] >= 5:
            k_keys.append(f); k_vals.append(k); tgt.append(1); fset.append("DEV")
    for r in rows:
        f = "ho_" + r["family_slug"]
        if f not in k_keys:
            k_keys.append(f); k_vals.append(1000 + int(r["queue_rank"])); tgt.append(1); fset.append(r["stratum"])
    np.savez_compressed(
        OUT / "expanded_features.npz",
        ids=np.array([dev_ids[i] for i in keep] + [f"ho{i}" for i in range(len(paths))], object),
        fam=np.array([dev_fam[i] for i in keep] + new_fam, object),
        dense=np.vstack([z["dense"][keep], dense]).astype(np.float32),
        clip_openai=np.vstack([z["clip_openai"][keep], vo]), clip_laion=np.vstack([z["clip_laion"][keep], vl]),
        phash_bits=np.vstack([z["phash_bits"][keep], bits.astype(np.uint8)]),
        siglip2=np.vstack([dm["siglip2"][keep], S]), dinov2=np.vstack([dm["dinov2"][keep], D]),
        fam_k_keys=np.array(k_keys, object), fam_k_vals=np.array(k_vals), fam_target=np.array(tgt),
        fam_set=np.array(fset, object))
    (OUT / "expanded_texts.json").write_text(json.dumps([texts_dev[i] for i in keep] + tx))
    print("expanded bundle:", len(keep) + len(paths), "variants;", len(k_keys), "families")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-modern", action="store_true")
    ap.add_argument("--expanded", action="store_true")
    ap.add_argument("--extractor-check", action="store_true")
    a = ap.parse_args()
    if a.extractor_check:
        extractor_check()
    elif a.dev_modern:
        dev_modern()
    elif a.expanded:
        expanded()
