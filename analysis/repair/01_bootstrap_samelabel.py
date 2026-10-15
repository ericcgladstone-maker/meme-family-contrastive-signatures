#!/usr/bin/env python3
"""Audit repair A-1: cluster bootstraps with duplicate draws kept in ONE CV group.

For each bootstrap specification used in G1 and G2, the picks are drawn exactly as before
(same seed, same sequential rng.choice). Only the CV grouping of the replicate changes: every copy
of an original group keeps the original group's label, so identical rows can never straddle a
train/test split. GroupKFold(min(5, number of unique sampled groups)).

Specs:
  G1 (full 4,263; groups = stored ph4_fit_family; control D):
    raw_fit seeds 0-3, calibrated_fit, classifier_prob, classifier_decision, sae_centroid (seed 0),
    raw_over_magnitude seeds 0-3 (D + magnitude block), plus the submitted-chain adjusted bootstrap
    (reviewer_numerics picks) on subset and full
  G2 (full 4,263; control D; full = raw + spec): bestfit_family seed 0; template cc10 seed 0
Output: out/bootstrap_samelabel.json and CSVs
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

E = C.submitted_engine()
OUT = HERE / "out"
G1 = HERE.parent / "g1_full_sample"
G2OUT = HERE.parent / "g2_template_cluster" / "out"
D = C.CTRL_D
GEN = ["sparse_norm", "clip_norm", "ocr_len"]
NJOBS, NBOOT = 11, 2000


def _chunk(df, gcol, picks, ctrl, add):
    rows_of = {g: np.where(df[gcol].to_numpy() == g)[0] for g in df[gcol].unique()}
    E.CTRL, E.PH4 = ctrl, add
    out = []
    for pick in picks:
        idx = np.concatenate([rows_of[g] for g in pick])
        lab = np.concatenate([np.full(len(rows_of[g]), str(g), dtype=object) for g in pick])  # SAME label for copies
        bt = df.iloc[idx].reset_index(drop=True)
        g = pd.Categorical(lab).codes
        k = min(5, len(np.unique(g)))
        try:
            E.build_dummies(bt)
            r = E.run(bt, GroupKFold(k), groups=g, infold=True)
            out.append((r["dR2"], r["dAUC"], len(np.unique(g))))
        except Exception:
            out.append((np.nan, np.nan, len(np.unique(g))))
    return out


def boot(df, gcol, picks, ctrl, add):
    k = max(1, len(picks) // (NJOBS * 4))
    res = Parallel(n_jobs=NJOBS)(delayed(_chunk)(df, gcol, picks[i:i + k], ctrl, add) for i in range(0, len(picks), k))
    v = np.array([x for r in res for x in r], dtype=float)
    return v[~np.isnan(v[:, 0])]


def summ(v):
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(np.mean(v)),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], share_gt0=float((v > 0).mean()))


def picks_for(ug, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(ug, len(ug), replace=True) for _ in range(NBOOT)]


def load_g1_module():
    spec = importlib.util.spec_from_file_location("g1d", G1 / "03_decomposition.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    g1 = load_g1_module()
    # ---------------- G1
    full = g1.frame(G1 / "out" / "scores_all.csv")
    ug = full.ph4_fit_family.unique()
    specs = {"raw_fit": (D, ["raw_max_fit"], [0, 1, 2, 3]), "calibrated_fit": (D, ["calib_max_fit"], [0]),
             "classifier_prob": (D, ["clf_max_prob"], [0]), "classifier_decision": (D, ["clf_max_decision"], [0]),
             "sae_centroid": (D, ["sae_centroid_max"], [0]), "raw_over_magnitude": (D + GEN, ["raw_max_fit"], [0, 1, 2, 3])}
    for name, (ctrl, add, seeds) in specs.items():
        for s in seeds:
            v = boot(full, "ph4_fit_family", picks_for(ug, s), ctrl, add)
            pd.DataFrame(v, columns=["dR2", "dAUC", "n_unique_groups"]).to_csv(OUT / f"g1_{name}_seed{s}_samelabel.csv", index=False)
            res[f"g1_full_{name}_seed{s}"] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]),
                                                   min_unique_groups=int(v[:, 2].min()))
            print(f"[G1 {name} s{s}] same-label dR2 CI {[round(x, 4) for x in res[f'g1_full_{name}_seed{s}']['dR2']['ci95']]}", flush=True)
    # submitted-chain adjusted bootstrap (subset reproduces the MS interval under replicate IDs)
    cohort, canon, cvec, cfam = C.load_cohort()
    cdense, _, sae = C.sae_dense(canon, cvec)
    fams_full = pd.read_csv(C.OUTCOMES).query("listing_group == 'denominator'").ph4_fit_family.unique()
    for label, csvp in [("subset_2543", G1 / "out" / "scores_subset.csv"), ("full_4263", G1 / "out" / "scores_all.csv")]:
        df = g1.frame(csvp)
        _, _, _, adj = g1.reviewer_numerics_chain(0, sae.latent_dim, cdense, np.array(cfam), fams_full, df.ph4_fit_family.unique())
        v = boot(df, "ph4_fit_family", adj, D + GEN, ["max_contrastive_ph4_fit"])
        res[f"g1_{label}_adjusted_submitted_chain"] = dict(dR2=summ(v[:, 0]))
        print(f"[G1 {label} adjusted submitted-chain] same-label CI {[round(x, 4) for x in res[f'g1_{label}_adjusted_submitted_chain']['dR2']['ci95']]}", flush=True)
    # ---------------- G2
    df = pd.read_csv(C.OUTCOMES)
    df = df[df.listing_group == "denominator"].reset_index(drop=True)
    df = df.merge(pd.read_csv(G2OUT / "clusters.csv", dtype={"post_id": str}), on="post_id", how="left", validate="one_to_one")
    for gcol, label in [("ph4_fit_family", "bestfit_family"), ("cc10_as_planned", "template_cc10_as_planned")]:
        v = boot(df, gcol, picks_for(df[gcol].unique(), 0), D, C.PH4_F)
        pd.DataFrame(v, columns=["dR2", "dAUC", "n_unique_groups"]).to_csv(OUT / f"g2_{label}_samelabel.csv", index=False)
        res[f"g2_{label}"] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]), min_unique_groups=int(v[:, 2].min()))
        print(f"[G2 {label}] same-label dR2 CI {[round(x, 4) for x in res[f'g2_{label}']['dR2']['ci95']]}", flush=True)
    C.write_json(OUT / "bootstrap_samelabel.json", res)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
