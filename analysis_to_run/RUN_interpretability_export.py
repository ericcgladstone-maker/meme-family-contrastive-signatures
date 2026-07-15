#!/usr/bin/env python3
"""Interpretability inspectability artifact: per family, the top-15 contrastive feature
indices and the top-5 activating member and nonmember exemplars. Run from build/."""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO_ROOT = Path("/Users/ericgladstone/My Drive (eric.c.gladstone@gmail.com)/Projects/Paper PH4 and Memes Computational/build")
sys.path.insert(0, str(REPO_ROOT))
from src.analysis.clip_embed import load_npz            # noqa
from src.analysis import sae_msae                        # noqa
K = 15
CFD = REPO_ROOT / "first_results" / "cross_family_dedup"
OUT = REPO_ROOT / "first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run/interpretability_signatures.csv"

def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

def main():
    cohort = {r["variant_id"]: r for r in csv.DictReader((CFD / "cross_family_corrected_index.csv").open(newline=""))}
    emb = load_npz(REPO_ROOT / "first_results" / "embeddings_openai.npz")
    canon = [v for v in emb["ids"] if v in cohort]
    vec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
    fam = [cohort[v]["family_id"] for v in canon]; n = len(canon)
    sae = sae_msae.MSAE(device="cpu"); srecs = sae_msae.activation_records(canon, vec, sae)
    dense = np.zeros((n, sae.latent_dim), np.float32)
    for i, r in enumerate(srecs): dense[i, r["active_indices"]] = r["active_values"]
    fam_idx = defaultdict(list)
    for i, f in enumerate(fam): fam_idx[f].append(i)
    rows = []
    for f, idxs in fam_idx.items():
        if len(idxs) < 5: continue
        nm = [i for i in range(n) if fam[i] != f]
        diff = dense[idxs].mean(0) - dense[nm].mean(0)
        top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
        fits = np.array([cosv(dense[i][top], w) for i in range(n)])
        mem_sorted = sorted(idxs, key=lambda i: -fits[i])[:5]
        non_sorted = sorted(nm, key=lambda i: -fits[i])[:5]
        rows.append(dict(
            family=f.replace("pool_", ""), n_members=len(idxs),
            top_feature_indices="|".join(map(str, top.tolist())),
            top_feature_weights="|".join(f"{x:.3f}" for x in w.tolist()),
            top_member_exemplars="|".join(canon[i] for i in mem_sorted),
            top_nonmember_exemplars="|".join(canon[i] for i in non_sorted),
            top_nonmember_fit=round(float(fits[non_sorted[0]]), 3) if non_sorted else None,
        ))
    with open(OUT, "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader()
        for r in rows: w_.writerow(r)
    print(f"wrote {OUT} ({len(rows)} families)")
    print("example (Loss):")
    for r in rows:
        if "loss" in r["family"]:
            print("  top features:", r["top_feature_indices"])
            print("  top members :", r["top_member_exemplars"])

if __name__ == "__main__":
    main()
