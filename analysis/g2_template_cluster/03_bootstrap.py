#!/usr/bin/env python3
"""G2 step 6 (as planned): cluster bootstraps, index-based and parallel.

Same resampling semantics as RUN_reviewer_numerics.boot_inc: draw whole groups with replacement
(rng = default_rng(seed); pick = rng.choice(unique_groups, len, replace=True)); duplicate draws become
distinct replicate groups labelled f"{g}_{i}" (category codes as in the original); refit GroupKFold(5).
Picks are drawn sequentially in the main process, so results equal the sequential implementation.

  template_cc10_as_planned : resample template clusters (the 2026-07-13 plan), seed 0
  bestfit_family_repro     : resample best-fit families, seed 0 -> must reproduce submitted [0.017, 0.123]
Output: out/g2_bootstrap_*.csv and a 'cluster_bootstrap' block merged into out/g2_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

E = C.submitted_engine()
OUT = Path(__file__).resolve().parent / "out"
NBOOT, NJOBS = 2000, 11


def load():
    df = pd.read_csv(C.OUTCOMES)
    df = df[df.listing_group == "denominator"].reset_index(drop=True)
    cl = pd.read_csv(OUT / "clusters.csv", dtype={"post_id": str})
    return df.merge(cl, on="post_id", how="left", validate="one_to_one")


def _chunk(df, gcol, picks):
    rows_of = {g: np.where(df[gcol].to_numpy() == g)[0] for g in df[gcol].unique()}
    E.CTRL, E.PH4 = C.CTRL_D, C.PH4_F
    out = []
    for pick in picks:
        idx = np.concatenate([rows_of[g] for g in pick])
        lab = np.concatenate([np.full(len(rows_of[g]), f"{g}_{i}", dtype=object) for i, g in enumerate(pick)])
        bt = df.iloc[idx].reset_index(drop=True)
        g = pd.Categorical(lab).codes
        try:
            E.build_dummies(bt)
            r = E.run(bt, GroupKFold(min(5, len(np.unique(g)))), groups=g, infold=True)
            out.append((r["dR2"], r["dAUC"]))
        except Exception:
            out.append((np.nan, np.nan))
    return out


def boot(df, gcol, seed):
    rng = np.random.default_rng(seed)
    ug = df[gcol].unique()
    picks = [rng.choice(ug, len(ug), replace=True) for _ in range(NBOOT)]
    k = max(1, len(picks) // (NJOBS * 4))
    res = Parallel(n_jobs=NJOBS)(delayed(_chunk)(df, gcol, picks[i:i + k]) for i in range(0, len(picks), k))
    v = np.array([x for r in res for x in r])
    return v[~np.isnan(v[:, 0])]


def summ(v):
    return dict(n_ok=int(len(v)), median=float(np.median(v)), mean=float(np.mean(v)),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], share_gt0=float((v > 0).mean()))


def main():
    df = load()
    boots = {}
    for gcol, label in [("ph4_fit_family", "bestfit_family_repro"), ("cc10_as_planned", "template_cc10_as_planned")]:
        v = boot(df, gcol, 0)
        pd.DataFrame(v, columns=["dR2", "dAUC"]).to_csv(OUT / f"g2_bootstrap_{label}.csv", index=False)
        boots[label] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]))
        print(f"[boot {label}] dR2 {boots[label]['dR2']['ci95']} median {boots[label]['dR2']['median']:+.4f} "
              f"| dAUC {boots[label]['dAUC']['ci95']}", flush=True)
    res = json.loads((OUT / "g2_results.json").read_text())
    res["cluster_bootstrap"] = boots
    res["submitted_family_bootstrap"] = {"median": 0.052, "ci95": [0.017, 0.123], "share_gt0": 0.999}
    C.write_json(OUT / "g2_results.json", res)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
