#!/usr/bin/env python3
"""G2 steps 5-9: engagement increment under the internally planned template-cluster grouping,
side by side with every other scheme, all in one engine.

Model = the submitted engine (RUN_e1_m2_reconcile): control D + community dummies; full adds
max_contrastive_ph4_fit + ph4_spec_margin; ridge(1) for R^2, logistic with in-fold labels for AUC.
Only the grouping changes across schemes.

  0. pipeline validation: reproduce submitted random (+0.046), family (+0.072), community (+0.052)
     via E.run; and the submitted family-cluster bootstrap (seed 0, 2,000) [0.017, 0.123]
  1. schemes: random5 (seed 0), best-fit family GroupKFold(5), community GroupKFold(7) [= LOCO],
     template clusters (cc10 as planned; cc6, cc8, greedy12 sensitivities) GroupKFold(5)
  2. per-fold increments for every scheme (R1-8)
  3. template-cluster bootstrap (as planned): resample clusters, 2,000, replicate-level ids, seed 0
  4. repeated CV: random5 seeds 1-20; grouped schemes: 20 random balanced group->fold assignments
Output: out/g2_results.json, out/g2_per_fold.csv, out/g2_bootstrap_*.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

E = C.submitted_engine()
OUT = Path(__file__).resolve().parent / "out"
NBOOT = 2000


def load():
    df = pd.read_csv(C.OUTCOMES)
    df = df[df.listing_group == "denominator"].reset_index(drop=True)
    cl = pd.read_csv(OUT / "clusters.csv", dtype={"post_id": str})
    df = df.merge(cl, on="post_id", how="left", validate="one_to_one")
    assert df[["cc10_as_planned", "cc6", "cc8", "greedy12"]].notna().all().all()
    return df


def run_folds(df, splits):
    """Same computation as E.run, but also returns per-fold metrics. splits: list of (tr, te)."""
    yc = df["log_score"].to_numpy(float)
    pCD, pCF, pBD, pBF, yb = (np.zeros(len(df)) for _ in range(5))
    folds = []
    for k, (tr, te) in enumerate(splits):
        dtr, dte = df.iloc[tr], df.iloc[te]
        for cols, dst in [(E.CTRL, pCD), (E.CTRL + E.PH4, pCF)]:
            m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(E.design(dtr, cols), yc[tr])
            dst[te] = m.predict(E.design(dte, cols))
        ytr, yte = E.infold_binary(dtr, dte)
        yb[te] = yte
        for cols, dst in [(E.CTRL, pBD), (E.CTRL + E.PH4, pBF)]:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(E.design(dtr, cols), ytr)
            dst[te] = m.predict_proba(E.design(dte, cols))[:, 1]
        f = dict(fold=k, n_test=len(te), R2_D=r2_score(yc[te], pCD[te]), R2_F=r2_score(yc[te], pCF[te]))
        f["dR2"] = f["R2_F"] - f["R2_D"]
        if len(set(yte)) == 2:
            f.update(AUC_D=roc_auc_score(yte, pBD[te]), AUC_F=roc_auc_score(yte, pBF[te]))
            f["dAUC"] = f["AUC_F"] - f["AUC_D"]
        folds.append(f)
    pooled = dict(R2_D=r2_score(yc, pCD), R2_F=r2_score(yc, pCF), AUC_D=roc_auc_score(yb, pBD),
                  AUC_F=roc_auc_score(yb, pBF))
    pooled["dR2"] = pooled["R2_F"] - pooled["R2_D"]
    pooled["dAUC"] = pooled["AUC_F"] - pooled["AUC_D"]
    return pooled, folds


def random_group_folds(groups, k, seed):
    """Balanced random assignment of whole groups to k folds (greedy by size, random tie order)."""
    rng = np.random.default_rng(seed)
    ug, counts = np.unique(groups, return_counts=True)
    order = rng.permutation(len(ug))
    order = order[np.argsort(-counts[order], kind="stable")]
    load, assign = np.zeros(k), {}
    for gi in order:
        f = int(np.argmin(load + rng.random(k) * 1e-9))
        assign[ug[gi]] = f
        load[f] += counts[gi]
    fid = np.array([assign[g] for g in groups])
    return [(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)]


def cluster_bootstrap(df, gcol, seed, nboot=NBOOT):
    """Resample whole groups with replacement; duplicate draws become distinct replicate groups;
    refit GroupKFold(5) on replicate ids. Mirrors RUN_reviewer_numerics.boot_inc."""
    rng = np.random.default_rng(seed)
    ug = df[gcol].unique()
    by = {g: df[df[gcol] == g] for g in ug}
    out = []
    for b in range(nboot):
        pick = rng.choice(ug, len(ug), replace=True)
        bt = pd.concat([by[g].assign(__rep=f"{g}_{i}") for i, g in enumerate(pick)], ignore_index=True)
        g = bt.__rep.astype("category").cat.codes.to_numpy()
        try:
            E.build_dummies(bt)
            r = E.run(bt, GroupKFold(min(5, len(np.unique(g)))), groups=g, infold=True)
            out.append((r["dR2"], r["dAUC"]))
        except Exception:
            pass
        if b % 250 == 0:
            print(f"    boot[{gcol}] {b}", flush=True)
    return np.array(out)


def summ(v):
    return dict(n_ok=int(len(v)), median=float(np.median(v)), mean=float(np.mean(v)),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], share_gt0=float((v > 0).mean()))


def main():
    df = load()
    E.CTRL, E.PH4 = C.CTRL_D, C.PH4_F
    E.build_dummies(df)
    res = {"n": len(df), "control": E.CTRL, "full_adds": E.PH4}

    # ---- 0. pipeline validation against submitted values ----
    val = {}
    val["random5_seed0"] = E.run(df, KFold(5, shuffle=True, random_state=0), infold=True)
    gfam = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    val["bestfit_family"] = E.run(df, GroupKFold(5), groups=gfam, infold=True)
    gcom = df.subreddit.astype("category").cat.codes.to_numpy()
    val["community_gkf7"] = E.run(df, GroupKFold(7), groups=gcom, infold=True)
    res["validation_E_run"] = val
    res["validation_submitted"] = {"random5_seed0_dR2": 0.046, "random5_seed0_dAUC": 0.083,
                                   "bestfit_family_dR2": 0.072, "bestfit_family_dAUC": 0.094,
                                   "community_dR2": 0.052, "community_dAUC": 0.070}
    print("[validation]", {k: (round(v["dR2"], 4), round(v["dAUC"], 4)) for k, v in val.items()}, flush=True)

    # ---- 1-2. all schemes with per-fold output (fold splits identical to E.run's) ----
    schemes = {
        "random5": list(KFold(5, shuffle=True, random_state=0).split(df)),
        "bestfit_family": list(GroupKFold(5).split(df, groups=gfam)),
        "community_gkf7_eq_LOCO": list(GroupKFold(7).split(df, groups=gcom)),
    }
    for c in ["cc10_as_planned", "cc6", "cc8", "greedy12"]:
        schemes[f"template_{c}"] = list(GroupKFold(5).split(df, groups=df[c].to_numpy()))
    rows, pooled = [], {}
    for name, sp in schemes.items():
        E.build_dummies(df)
        p, folds = run_folds(df, sp)
        pooled[name] = p
        for f in folds:
            rows.append(dict(scheme=name, **f))
        print(f"[{name}] dR2={p['dR2']:+.4f} dAUC={p['dAUC']:+.4f}  "
              f"R2_D={p['R2_D']:.3f} R2_F={p['R2_F']:.3f}", flush=True)
    res["pooled"] = pooled
    # consistency: run_folds pooled must equal E.run for the three validation schemes
    res["run_folds_matches_E_run"] = {
        k: abs(pooled[s]["dR2"] - val[k]["dR2"]) < 1e-9
        for k, s in [("random5_seed0", "random5"), ("bestfit_family", "bestfit_family"), ("community_gkf7", "community_gkf7_eq_LOCO")]}
    pd.DataFrame(rows).to_csv(OUT / "g2_per_fold.csv", index=False)

    # ---- 4. repeated CV (seeds 1-20) ----
    rep = {}
    for name in ["random5", "bestfit_family", "template_cc10_as_planned"]:
        vals = []
        for s in range(1, 21):
            if name == "random5":
                sp = list(KFold(5, shuffle=True, random_state=s).split(df))
            else:
                g = gfam if name == "bestfit_family" else df["cc10_as_planned"].to_numpy()
                sp = random_group_folds(g, 5, s)
            E.build_dummies(df)
            p, _ = run_folds(df, sp)
            vals.append((p["dR2"], p["dAUC"]))
        v = np.array(vals)
        rep[name] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]), values=v.tolist())
        print(f"[repeat {name}] dR2 median {np.median(v[:,0]):+.4f} range [{v[:,0].min():+.4f}, {v[:,0].max():+.4f}]", flush=True)
    res["repeated_cv_seeds_1_20"] = rep
    C.write_json(OUT / "g2_results.json", res)

    # ---- 3. bootstraps: template clusters (as planned) + reproduce submitted family bootstrap ----
    boots = {}
    for gcol, seed, label in [("cc10_as_planned", 0, "template_cc10_as_planned"), ("ph4_fit_family", 0, "bestfit_family_repro")]:
        v = cluster_bootstrap(df, gcol, seed)
        pd.DataFrame(v, columns=["dR2", "dAUC"]).to_csv(OUT / f"g2_bootstrap_{label}.csv", index=False)
        boots[label] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]))
        print(f"[boot {label}] dR2 {boots[label]['dR2']}", flush=True)
    res["cluster_bootstrap"] = boots
    res["submitted_family_bootstrap"] = {"median": 0.052, "ci95": [0.017, 0.123], "share_gt0": 0.999}
    C.write_json(OUT / "g2_results.json", res)
    print("[done] wrote g2_results.json", flush=True)


if __name__ == "__main__":
    main()
