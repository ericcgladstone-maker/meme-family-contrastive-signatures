#!/usr/bin/env python3
"""
Embed the 4,263 denominator Reddit images (CLIP -> MSAE), then compute four
family-comparable per-post scores for the engagement comparison:
  raw_max_fit        : max contrastive PH4 fit (must reproduce the stored column -> validation)
  calib_max_fit      : max over families of the family-null-standardized fit (cross-family calibration)
  clf_max_prob       : max over families of a per-family L2-logistic P(member) on sparse features
  sae_centroid_max   : max over families of sparse-centroid cosine
Checkpoints rvec/rdense to npz so the ~50-min embed is not lost if scoring fails.
Run from build/.  Saves downstream_scores.csv next to this script.
"""
from __future__ import annotations
import csv, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path("/Users/ericgladstone/My Drive (eric.c.gladstone@gmail.com)/Projects/Paper PH4 and Memes Computational/build")
sys.path.insert(0, str(REPO))
from src.analysis.clip_embed import load_npz, embed_image_paths      # noqa
from src.analysis import sae_msae                                     # noqa
from sklearn.linear_model import LogisticRegression
HERE = REPO / "first_results/ph4_manuscript_natcomms/SSCR Submission/07_Analysis_Audit"
CFD = REPO / "first_results/cross_family_dedup/cross_family_corrected_index.csv"
OUTC = REPO / "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
MANI = REPO / "first_results/ph4_q2_horsepower/q2_denom_images_manifest.csv"
CKPT = HERE / "reddit_denom_embeddings.npz"
K = 15

def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

def main():
    # ---- cohort signatures / nulls / centroids / classifiers ----
    cohort = {r["variant_id"]: r for r in csv.DictReader(CFD.open(newline=""))}
    emb = load_npz(REPO / "first_results/embeddings_openai.npz")
    canon = [v for v in emb["ids"] if v in cohort]
    cvec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
    cfam = [cohort[v]["family_id"] for v in canon]
    sae = sae_msae.MSAE(device="cpu"); csr = sae_msae.activation_records(canon, cvec, sae)
    cdense = np.zeros((len(canon), sae.latent_dim), np.float32)
    for i, r in enumerate(csr): cdense[i, r["active_indices"]] = r["active_values"]
    cdn = cdense / np.clip(np.linalg.norm(cdense, axis=1, keepdims=True), 1e-12, None)
    fam_idx = defaultdict(list)
    for i, f in enumerate(cfam): fam_idx[f].append(i)
    fams = list(fam_idx)
    sig, null, cent, clf = {}, {}, {}, {}
    for f, idxs in fam_idx.items():
        nm = [i for i in range(len(canon)) if cfam[i] != f]
        diff = cdense[idxs].mean(0) - cdense[nm].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
        sig[f] = (top, w)
        nmfit = np.array([cosv(cdense[j][top], w) for j in nm])
        null[f] = (nmfit.mean(), nmfit.std() + 1e-9)
        c = cdn[idxs].mean(0); cent[f] = c / (np.linalg.norm(c) + 1e-12)
        y = np.array([1 if cfam[i] == f else 0 for i in range(len(canon))])
        clf[f] = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(cdense, y)
    print(f"[cohort] {len(fams)} families, signatures/nulls/centroids/classifiers built", flush=True)

    # ---- denominator posts + image paths ----
    outc = [r for r in csv.DictReader(OUTC.open(newline="")) if r["listing_group"] == "denominator"]
    path = {r["post_id"]: r["local_path"] for r in csv.DictReader(MANI.open(newline="")) if r.get("status") == "cached"}
    recs = [{"variant_id": r["post_id"], "image_path": path[r["post_id"]]}
            for r in outc if r["post_id"] in path and Path(path[r["post_id"]]).exists()]
    print(f"[denom] {len(recs)} posts to embed", flush=True)

    # ---- embed (checkpointed) ----
    if CKPT.exists():
        z = np.load(CKPT, allow_pickle=True); rid = list(z["ids"]); rvec = z["vectors"]
        print("[embed] loaded checkpoint", flush=True)
    else:
        t = time.time()
        res = embed_image_paths(recs, pretrained_key="openai", device="auto", l2_normalize=False)
        rid, rvec = list(res["ids"]), res["vectors"]
        np.savez(CKPT, ids=np.array(rid, object), vectors=rvec)
        print(f"[embed] {len(rid)} in {(time.time()-t)/60:.1f} min -> checkpoint saved", flush=True)
    rsr = sae_msae.activation_records(rid, rvec, sae)
    rdense = np.zeros((len(rid), sae.latent_dim), np.float32)
    for i, r in enumerate(rsr): rdense[i, r["active_indices"]] = r["active_values"]
    rdn = rdense / np.clip(np.linalg.norm(rdense, axis=1, keepdims=True), 1e-12, None)

    # ---- score every post against every family ----
    tops = {f: sig[f][0] for f in fams}; ws = {f: sig[f][1] for f in fams}
    clf_P = {f: clf[f].predict_proba(rdense)[:, 1] for f in fams}   # vectorized per family
    rows = []
    for i, pid in enumerate(rid):
        raw = {f: cosv(rdense[i][tops[f]], ws[f]) for f in fams}
        cal = {f: (raw[f] - null[f][0]) / null[f][1] for f in fams}
        cen = {f: float(rdn[i] @ cent[f]) for f in fams}
        clp = {f: float(clf_P[f][i]) for f in fams}
        rows.append(dict(
            post_id=pid,
            raw_max_fit=max(raw.values()), raw_argmax=max(raw, key=raw.get),
            calib_max_fit=max(cal.values()), calib_argmax=max(cal, key=cal.get),
            clf_max_prob=max(clp.values()), clf_argmax=max(clp, key=clp.get),
            sae_centroid_max=max(cen.values()),
        ))
    with open(HERE / "downstream_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- VALIDATION: raw_max_fit must reproduce stored max_contrastive_ph4_fit ----
    stored = {r["post_id"]: float(r["max_contrastive_ph4_fit"]) for r in outc}
    pairs = [(stored[r["post_id"]], r["raw_max_fit"]) for r in rows if r["post_id"] in stored]
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    print(f"[VALIDATE] n={len(pairs)} corr(stored,recon)={np.corrcoef(a,b)[0,1]:.4f} "
          f"mean_abs_diff={np.mean(np.abs(a-b)):.4f} (want corr~1, diff~0)", flush=True)
    print(f"wrote {HERE/'downstream_scores.csv'}", flush=True)

if __name__ == "__main__":
    main()
