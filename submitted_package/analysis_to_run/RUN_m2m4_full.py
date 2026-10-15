#!/usr/bin/env python3
"""
Full same-representation baselines (M2) + non-CLIP adversarial decoys (M4), reconstructed
faithfully from fr_25's pipeline. VALIDATES by reproducing the known contrastive (0.895)
and sparse-centroid (0.822) recovery AUROCs before trusting the new baselines.

Run from build/:  python3 first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run/RUN_m2m4_full.py
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO_ROOT = Path("/Users/ericgladstone/My Drive (eric.c.gladstone@gmail.com)/Projects/Paper PH4 and Memes Computational/build")
sys.path.insert(0, str(REPO_ROOT))
from src.analysis.clip_embed import load_npz            # noqa
from src.analysis import diagnostics as D               # noqa
from src.analysis import sae_msae                        # noqa
from src.analysis import surface_baselines as SB         # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CFD = REPO_ROOT / "first_results" / "cross_family_dedup"
K, NSAMP, SEED = 15, 200, 20260512


def auc(pos, neg):
    if not pos or not neg: return None
    p = np.array(pos)[:, None]; n = np.array(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))
def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
def reorder(canon, ids, sim):
    pos = {v: i for i, v in enumerate(ids)}; idx = [pos[v] for v in canon]; return sim[np.ix_(idx, idx)]


def main():
    rng = np.random.default_rng(SEED)
    cohort = {r["variant_id"]: r for r in csv.DictReader((CFD / "cross_family_corrected_index.csv").open(newline=""))}
    emb = load_npz(REPO_ROOT / "first_results" / "embeddings_openai.npz")
    canon = [v for v in emb["ids"] if v in cohort]
    vec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
    fam = [cohort[v]["family_id"] for v in canon]
    n = len(canon)
    print(f"cohort n={n} families={len(set(fam))}")
    clip = D.cosine_matrix(vec)
    sae = sae_msae.MSAE(device="cpu"); srecs = sae_msae.activation_records(canon, vec, sae)
    dense = np.zeros((n, sae.latent_dim), np.float32)
    for i, r in enumerate(srecs): dense[i, r["active_indices"]] = r["active_values"]
    pid, ps = SB.phash_similarity([{"variant_id": v, "image_path": cohort[v]["image_path"]} for v in canon])
    phash = reorder(canon, pid, ps)
    vn = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12, None)
    dn = dense / np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
    fam_idx = defaultdict(list)
    for i, f in enumerate(fam): fam_idx[f].append(i)

    rows = []
    for f, idxs in fam_idx.items():
        if len(idxs) < 5: continue
        nonfam = [i for i in range(n) if fam[i] != f]
        samp = list(rng.choice(nonfam, size=min(NSAMP, len(nonfam)), replace=False))
        train_neg = [i for i in nonfam if i not in set(samp)]  # for the classifier: eval-neg held out
        con_m, con_nm, cen_m, cen_nm, knn_m, knn_nm = [], [], [], [], [], []
        # LOVO for the no-training baselines (contrastive/centroid/knn) -- parallels fr_25
        for q in idxs:
            train = [i for i in idxs if i != q]
            diff = dense[train].mean(0) - dense[nonfam].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
            sc = dn[train].mean(0); sc /= (np.linalg.norm(sc) + 1e-12)
            def con(c): return cosv(dense[c][top], w)
            def cen(c): return float(dn[c] @ sc)
            def knn(c): return max(float(dn[c] @ dn[t]) for t in train)
            con_m.append(con(q)); cen_m.append(cen(q)); knn_m.append(knn(q))
            for c in samp:
                con_nm.append(con(c)); cen_nm.append(cen(c)); knn_nm.append(knn(c))
        # regularized logistic on sparse features: stratified 5-fold CV over members + samp negatives (clean, no train/eval overlap)
        Xpos = dense[idxs]; Xneg = dense[samp]
        Xc = np.vstack([Xpos, Xneg]); yc = np.r_[np.ones(len(idxs)), np.zeros(len(samp))]
        oof = np.zeros(len(yc))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xc, yc):
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xc[tr], yc[tr])
            oof[te] = m.predict_proba(Xc[te])[:, 1]
        log_auc = roc_auc_score(yc, oof)
        # non-CLIP (pHash-selected) vs CLIP-selected adversarial decoys (separate member/decoy lists)
        near_p = sorted(nonfam, key=lambda i: -max(phash[i, j] for j in idxs))[:12]
        near_c = sorted(nonfam, key=lambda i: -max(clip[i, j] for j in idxs))[:12]
        cpm, cpd, lpm, lpd = [], [], [], []   # contrastive/clip member & pHash-decoy
        ccm, ccd, lcm, lcd = [], [], [], []   # contrastive/clip member & CLIP-decoy
        for q in idxs:
            train = [i for i in idxs if i != q]
            diff = dense[train].mean(0) - dense[nonfam].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
            cc = vn[train].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
            cpm.append(cosv(dense[q][top], w)); lpm.append(float(vn[q] @ cc))
            ccm.append(cosv(dense[q][top], w)); lcm.append(float(vn[q] @ cc))
            for c in near_p:
                cpd.append(cosv(dense[c][top], w)); lpd.append(float(vn[c] @ cc))
            for c in near_c:
                ccd.append(cosv(dense[c][top], w)); lcd.append(float(vn[c] @ cc))
        rows.append(dict(
            family=f.replace("pool_", ""), n=len(idxs),
            contrastive=auc(con_m, con_nm), sae_centroid=auc(cen_m, cen_nm),
            sae_knn=auc(knn_m, knn_nm), logistic_sparse=log_auc,
            adv_con_phash=auc(cpm, cpd), adv_clip_phash=auc(lpm, lpd),
            adv_con_clip=auc(ccm, ccd), adv_clip_clip=auc(lcm, lcd),
        ))
    df = _tab(rows)
    print("\n=== VALIDATION (should reproduce fr_25 means 0.895 contrastive / 0.822 centroid) ===")
    print(f"  contrastive mean = {np.mean([r['contrastive'] for r in rows]):.3f}  (target 0.895)")
    print(f"  sae_centroid mean = {np.mean([r['sae_centroid'] for r in rows]):.3f}  (target 0.822)")
    print("\n=== M2 same-representation baselines (mean recovery AUROC) ===")
    for k in ["contrastive", "sae_centroid", "sae_knn", "logistic_sparse"]:
        print(f"  {k:16s} {np.mean([r[k] for r in rows]):.3f}")
    for base in ["sae_centroid", "sae_knn", "logistic_sparse"]:
        w02 = sum(1 for r in rows if r["contrastive"] - r[base] > 0.02)
        gain = np.mean([r["contrastive"] - r[base] for r in rows])
        print(f"  contrastive - {base:16s} gain {gain:+.3f}  wins(>0.02) {w02}/{len(rows)}")
    print("\n=== M4 non-CLIP (pHash-selected) adversarial decoys (mean AUROC) ===")
    print(f"  contrastive vs pHash-decoys = {np.mean([r['adv_con_phash'] for r in rows]):.3f}")
    print(f"  CLIP        vs pHash-decoys = {np.mean([r['adv_clip_phash'] for r in rows]):.3f}")
    print(f"  (ref) contrastive vs CLIP-decoys = {np.mean([r['adv_con_clip'] for r in rows]):.3f}")
    print(f"  (ref) CLIP        vs CLIP-decoys = {np.mean([r['adv_clip_clip'] for r in rows]):.3f}")
    out = REPO_ROOT / "first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run/m2m4_results.csv"
    _wcsv(out, rows); print(f"\nwrote {out}")


def _tab(rows): return rows
def _wcsv(path, rows):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)


if __name__ == "__main__":
    main()
