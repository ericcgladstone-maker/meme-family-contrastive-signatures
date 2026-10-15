#!/usr/bin/env python3
"""G1 steps 1-3: signature bank (as submitted), per-post scores, identity check B.

Bank reproduces RUN_downstream_embed_score.py / fr_36 exactly: one contrastive signature per corpus
family (ALL 31 families, incl. the 2-variant family; AI-25), background = all corpus nonmembers,
family-null (mean, sd) of nonmember fits, sparse centroid, one-vs-rest L2 logistic (C=1, balanced)
on all corpus variants.

Scores per post: raw_max_fit (+argmax, spec margin), calib_max_fit, clf_max_prob (as submitted),
clf_max_decision (AI-23), sae_centroid_max, sparse_norm, clip_norm, rand_sig_max (seed 0, as
RUN_typicality_controls).

Writes scores for (a) the submitted 2,543 subset embeddings [validation] and (b) all 4,263.
Identity check B: reconstructed raw_max_fit / argmax / spec vs stored fr_36 columns, by status.
Output: out/scores_subset.csv, out/scores_all.csv, out/identity_check.json
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def build_bank():
    cohort, canon, cvec, cfam = C.load_cohort()
    cdense, _, sae = C.sae_dense(canon, cvec)
    cdn = cdense / np.clip(np.linalg.norm(cdense, axis=1, keepdims=True), 1e-12, None)
    fi = C.fam_index(cfam)
    n = len(canon)
    bank = {}
    for f, idxs in fi.items():
        nm = [i for i in range(n) if cfam[i] != f]
        top, w = C.contrastive_signature(cdense, idxs, nm)
        nmfit = C.sig_scores(cdense[nm], top, w)
        c = cdn[idxs].mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        y = np.array([1 if cfam[i] == f else 0 for i in range(n)])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(cdense, y)
        bank[f] = dict(top=top, w=w, mu=nmfit.mean(), sd=nmfit.std() + 1e-9, cent=c, clf=clf, n=len(idxs))
    return bank, sae


def score_posts(ids, rvec, bank, sae):
    rdense, _, _ = C.sae_dense(ids, rvec, sae)
    rdn = rdense / np.clip(np.linalg.norm(rdense, axis=1, keepdims=True), 1e-12, None)
    fams = list(bank)
    raw = np.column_stack([C.sig_scores(rdense, bank[f]["top"], bank[f]["w"]) for f in fams])
    cal = np.column_stack([(raw[:, j] - bank[f]["mu"]) / bank[f]["sd"] for j, f in enumerate(fams)])
    cen = np.column_stack([rdn @ bank[f]["cent"] for f in fams])
    prob = np.column_stack([bank[f]["clf"].predict_proba(rdense)[:, 1] for f in fams])
    dec = np.column_stack([bank[f]["clf"].decision_function(rdense) for f in fams])
    srt = np.sort(raw, axis=1)
    rng = np.random.default_rng(0)  # as RUN_typicality_controls
    sigs = [(rng.choice(sae.latent_dim, C.K, replace=False), np.abs(rng.normal(size=C.K))) for _ in range(30)]
    rand = np.column_stack([C.sig_scores(rdense, t, w) for t, w in sigs]).max(1)
    df = pd.DataFrame(dict(
        post_id=ids,
        raw_max_fit=raw.max(1), raw_argmax=[fams[j].replace("pool_", "") for j in raw.argmax(1)],
        raw_spec_margin=srt[:, -1] - srt[:, -2],
        calib_max_fit=cal.max(1), clf_max_prob=prob.max(1), clf_max_decision=dec.max(1),
        sae_centroid_max=cen.max(1), sparse_norm=np.linalg.norm(rdense, axis=1),
        clip_norm=np.linalg.norm(rvec, axis=1), rand_sig_max=rand))
    return df


def main():
    OUT.mkdir(exist_ok=True)
    bank, sae = build_bank()
    print(f"[bank] {len(bank)} signatures (sizes: min {min(b['n'] for b in bank.values())})", flush=True)
    z = np.load(C.SUBSET_CKPT, allow_pickle=True)
    sub = score_posts(list(z["ids"]), z["vectors"], bank, sae)
    sub.to_csv(OUT / "scores_subset.csv", index=False)
    z = np.load(OUT / "reddit_denom_embeddings_all.npz", allow_pickle=True)
    allp = score_posts(list(z["ids"]), z["vectors"], bank, sae)
    allp.to_csv(OUT / "scores_all.csv", index=False)

    # identity check B vs stored fr_36 columns
    o = pd.read_csv(C.OUTCOMES)
    o = o[o.listing_group == "denominator"]
    mani = pd.read_csv(C.MANIFEST)[["post_id", "status"]]
    m = o.merge(allp, on="post_id").merge(mani, on="post_id")
    chk = {}
    for st, g in [("all", m)] + list(m.groupby("status")):
        d = np.abs(g["raw_max_fit"] - g["max_contrastive_ph4_fit"])
        chk[st] = dict(n=len(g), corr=float(np.corrcoef(g["raw_max_fit"], g["max_contrastive_ph4_fit"])[0, 1]),
                       mean_abs_diff=float(d.mean()), max_abs_diff=float(d.max()),
                       share_within_rounding=float((d <= 0.00005 + 1e-9).mean()),
                       argmax_agreement=float((g["raw_argmax"] == g["ph4_fit_family"]).mean()),
                       spec_max_abs_diff=float(np.abs(g["raw_spec_margin"] - g["ph4_spec_margin"]).max()))
    # submitted subset validation of stored score file
    ss = pd.read_csv(C.AUDIT / "downstream_scores.csv")
    v = ss.merge(sub, on="post_id", suffixes=("_submitted", ""))
    val = {c: float(np.abs(v[f"{c}_submitted"] - v[c]).max()) for c in
           ["raw_max_fit", "calib_max_fit", "clf_max_prob", "sae_centroid_max"]}
    res = dict(identity_check_B=chk, subset_scores_vs_submitted_max_abs_diff=val, n_bank_signatures=len(bank),
               bank_families=sorted(f.replace("pool_", "") for f in bank),
               argmax_never_small_family=bool(not (allp.raw_argmax.isin([f.replace("pool_", "") for f, b in bank.items() if b["n"] < 5])).any()))
    C.write_json(OUT / "identity_check.json", res)
    print("[identity B]", {k: (round(v["corr"], 6), round(v["max_abs_diff"], 6), round(v["argmax_agreement"], 4)) for k, v in chk.items()}, flush=True)
    print("[subset vs submitted scores]", val, flush=True)


if __name__ == "__main__":
    main()
