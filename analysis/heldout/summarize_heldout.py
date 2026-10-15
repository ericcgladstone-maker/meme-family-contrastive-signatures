#!/usr/bin/env python3
"""Held-out validation summary and mechanical §11 verdicts (14 §9-§11). POST-FREEZE ONLY.

Sets: H (primary), C, DEV (expanded universe), ALL. For each unsupervised baseline b separately:
mean family paired difference contrastive - b, 95% family bootstrap CI (5,000; seed 0), and the
descriptive share of families > 0.02. (No bootstrap p-values: 14 v0.5 item 3.)
Classifier: logistic - contrastive. RQ2: contrastive - CLIP centroid on LAION decoys (family-level primary,
fold-local sensitivity); pHash decoys descriptive. Size: bands, OLS of paired difference on set + log(n),
15-capped sensitivity. Seed sweep ranges. Verdicts per 14 §11, computed from the rules, not judgment.
Output: heldout/out/heldout_results.json
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("HELDOUT_OUT", str(HERE / "out")))
FIXTURE = OUT != HERE / "out"  # synthetic-fixture test directory (engineering only)
BASELINES = ["recurrence", "clip_centroid", "sparse_centroid", "sparse_knn", "surf_composite", "oracle_surface",
             "fused_centroid"]
CORE_THREE = ["sparse_centroid", "sparse_knn", "clip_centroid"]


def paired(a, b, seed=0, nboot=5000):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                share_gt_0p02=float((d > 0.02).mean()), n=int(len(d)))


def with_sets(df):
    z = np.load(OUT / "expanded_features.npz", allow_pickle=True)
    kset = dict(zip([int(x) for x in z["fam_k_vals"]], list(z["fam_set"])))
    df = df.copy()
    df["set"] = df["k"].map(kset)
    df["oracle_surface__E_local"] = df[["surf_phash__E_local", "surf_ocr__E_local", "surf_template__E_local"]].max(1)
    return df


def block(df):
    r = {"n_families": int(len(df)), "means": {}}
    for m in ["contrastive", "logistic"] + BASELINES:
        r["means"][m] = float(df[f"{m}__E_local"].mean())
    r["rq1"] = {b: paired(df["contrastive__E_local"], df[f"{b}__E_local"]) for b in BASELINES}
    r["classifier_minus_contrastive"] = paired(df["logistic__E_local"], df["contrastive__E_local"])
    r["rq2_laion_familylevel"] = paired(df["contrastive__DL_local"], df["clip_centroid__DL_local"])
    r["phash_decoys_descriptive"] = paired(df["contrastive__DP_local"], df["clip_centroid__DP_local"])
    return r


def verdicts(r):
    ok = {b: r["rq1"][b]["ci95"][0] > 0 for b in BASELINES}
    unsup = [b for b in BASELINES if b != "fused_centroid"]
    if all(ok[b] for b in BASELINES):
        l1 = "L1 validated on held-out families"
    elif not any(ok[b] for b in CORE_THREE):
        l1 = ("L1 becomes 'compact and inspectable, without a clear recovery advantage over the simple "
              "same-representation and whole-image baselines'")
    else:
        l1 = "L1 weakened baseline-by-baseline: advantage (interval excludes 0) over " + \
             ", ".join(b for b in BASELINES if ok[b]) + "; no clear contrastive advantage detected over " + \
             ", ".join(b for b in BASELINES if not ok[b])
    ci = r["rq2_laion_familylevel"]["ci95"]
    m = r["rq2_laion_familylevel"]["mean"]
    l2 = "validated" if ci[0] > 0 else ("weakened to suggestive" if m > 0 else "not replicated; restricted to DEV")
    return dict(L1=l1, L2_laion=l2, baseline_ci_excludes_zero=ok,
                classifier_note="expected logistic > contrastive; " +
                ("interval includes 0" if r["classifier_minus_contrastive"]["ci95"][0] <= 0 else "interval excludes 0"),
                unsupervised_excluding_fused=unsup)


def main():
    if not FIXTURE and not (HERE / "FREEZE_RECORD.json").exists():
        raise SystemExit("REFUSING: not frozen")
    if FIXTURE and "fixture" not in OUT.name:
        raise SystemExit("REFUSING: HELDOUT_OUT override must be a *fixture* directory")
    res = {}
    p = with_sets(pd.read_csv(OUT / "ho_primary.csv"))
    for s in ["H", "C", "DEV"]:
        sub = p[p.set == s]
        if len(sub):
            res[s] = block(sub)
    res["ALL"] = block(p)
    res["verdicts_H"] = verdicts(res["H"])
    if "C" in res:
        res["verdicts_C"] = verdicts(res["C"])
    # size analyses
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location("sr", HERE / "screen_and_retrieve.py"); _m = _iu.module_from_spec(_sp)
    _sp.loader.exec_module(_m)
    mv = _m.FREEZE_PARAMS["MIN_VARIANTS"]
    p["size_band"] = pd.cut(p["n"], [mv - 1, 14, 19, 10 ** 6], labels=[f"{mv}-14", "15-19", ">=20"])
    res["size_bands"] = {str(b): {"n": int(len(g)), **{f"contrastive-{x}": float((g["contrastive__E_local"] - g[f"{x}__E_local"]).mean())
                                                       for x in BASELINES}} for b, g in p.groupby("size_band", observed=True)}
    reg = {}
    for x in BASELINES:
        y = (p["contrastive__E_local"] - p[f"{x}__E_local"]).to_numpy()
        X = np.column_stack([np.ones(len(p)), (p.set == "H").astype(float), (p.set == "C").astype(float), np.log(p["n"])])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        reg[x] = dict(intercept_DEV=float(beta[0]), H_minus_DEV=float(beta[1]), C_minus_DEV=float(beta[2]), log_n=float(beta[3]))
    res["size_regression"] = reg
    for cap in (10, 15):
        if (OUT / f"ho_capped{cap}.csv").exists():
            cp = with_sets(pd.read_csv(OUT / f"ho_capped{cap}.csv"))
            res[f"H_capped{cap}"] = block(cp[cp.set == "H"])
            res[f"verdicts_H_capped{cap}"] = verdicts(res[f"H_capped{cap}"])
    if (OUT / "ho_sweep.csv").exists():
        sw = with_sets(pd.read_csv(OUT / "ho_sweep.csv"))
        h = sw[sw.set == "H"]
        res["H_seed_sweep_ranges"] = {x: [float(v) for v in (h.groupby("seed").apply(
            lambda g: (g["contrastive__E_local"] - g[f"{x}__E_local"]).mean())).agg(["min", "max"])] for x in BASELINES}
    if (OUT / "ho_foldlocal.csv").exists():
        fl = pd.read_csv(OUT / "ho_foldlocal.csv").merge(p[["k", "set"]], on="k")
        h = fl[fl.set == "H"]
        res["H_rq2_laion_foldlocal"] = paired(h["contrastive__DL_foldlocal"], h["clip_centroid__DL_foldlocal"])
    (OUT / "heldout_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("verdicts")}, indent=1))


if __name__ == "__main__":
    main()
