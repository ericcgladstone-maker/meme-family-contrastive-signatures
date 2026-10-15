#!/usr/bin/env python3
"""Audit repairs A-2, A-3, A-4 (cheap; no refits of nulls).

A-2  Monte Carlo p = (b+1)/(B+1) from the saved G1 null arrays (subset & full, seeds 0-3).
A-3  pHash cluster statistics: share of POSTS in singleton clusters and share of CLUSTERS that are singletons.
A-4  Matched 30-signature downstream bank (drop the 2-variant family): how often it is runner-up or the
     argmax of any score; increments for raw, raw+spec, calibrated, classifier, centroid with the 30-bank.
Output: out/repair_cheap.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OUT = HERE / "out"
G1 = HERE.parent / "g1_full_sample"
G2OUT = HERE.parent / "g2_template_cluster" / "out"


def mc_p(null, obs):
    null = np.asarray(null)
    b = int((null >= obs).sum())
    return dict(b=b, B=int(len(null)), p_corrected=(b + 1) / (len(null) + 1), p_raw=b / len(null))


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    g1 = json.loads((G1 / "out" / "g1_results.json").read_text())
    # ---------------- A-2
    pv = {}
    for label in ["subset_2543", "full_4263"]:
        obs = g1[label]["increments"]["raw_fit"]["dR2"]
        for s in range(4):
            rf = pd.read_csv(G1 / "out" / f"g1_nulls_random_feature_{label}_seed{s}.csv")["random_feature"]
            lp = pd.read_csv(G1 / "out" / f"g1_nulls_label_perm_{label}_seed{s}.csv")["label_perm"]
            pv[f"{label}_seed{s}"] = dict(observed=obs, random_feature=mc_p(rf, obs), label_perm=mc_p(lp, obs))
    res["mc_pvalues"] = pv
    # ---------------- A-3
    cl = pd.read_csv(G2OUT / "clusters.csv")
    cs = {}
    for c in ["cc10_as_planned", "cc6", "cc8", "greedy12"]:
        sizes = cl[c].value_counts()
        cs[c] = dict(n_posts=int(len(cl)), n_clusters=int(len(sizes)), n_singleton_clusters=int((sizes == 1).sum()),
                     share_posts_in_singletons=float((sizes == 1).sum() / len(cl)),
                     share_clusters_singleton=float((sizes == 1).mean()), largest=int(sizes.max()))
    res["cluster_singletons"] = cs
    # ---------------- A-4
    spec = importlib.util.spec_from_file_location("g1s", G1 / "02_scores.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    bank, sae = m.build_bank()
    small = [f for f, b in bank.items() if b["n"] < 5]
    z = np.load(G1 / "out" / "reddit_denom_embeddings_all.npz", allow_pickle=True)
    ids, vec = list(z["ids"]), z["vectors"]
    rdense, _, _ = C.sae_dense(ids, vec, sae)
    rdn = rdense / np.clip(np.linalg.norm(rdense, axis=1, keepdims=True), 1e-12, None)
    fams = list(bank)
    si = [fams.index(f) for f in small]
    raw = np.column_stack([C.sig_scores(rdense, bank[f]["top"], bank[f]["w"]) for f in fams])
    cal = np.column_stack([(raw[:, j] - bank[f]["mu"]) / bank[f]["sd"] for j, f in enumerate(fams)])
    cen = np.column_stack([rdn @ bank[f]["cent"] for f in fams])
    prob = np.column_stack([bank[f]["clf"].predict_proba(rdense)[:, 1] for f in fams])
    dec = np.column_stack([bank[f]["clf"].decision_function(rdense) for f in fams])
    keep = [j for j in range(len(fams)) if j not in si]
    order = np.argsort(-raw, axis=1)
    affect = dict(
        small_family=[f.replace("pool_", "") for f in small],
        n_posts=len(ids),
        small_is_raw_argmax=int(np.isin(order[:, 0], si).sum()),
        small_is_raw_runner_up=int(np.isin(order[:, 1], si).sum()),
        small_is_calibrated_argmax=int(np.isin(cal.argmax(1), si).sum()),
        small_is_classifier_prob_argmax=int(np.isin(prob.argmax(1), si).sum()),
        small_is_classifier_decision_argmax=int(np.isin(dec.argmax(1), si).sum()),
        small_is_centroid_argmax=int(np.isin(cen.argmax(1), si).sum()))
    r30 = raw[:, keep]
    s30 = np.sort(r30, axis=1)
    b30 = pd.DataFrame(dict(post_id=ids, raw30=r30.max(1), spec30=s30[:, -1] - s30[:, -2], cal30=cal[:, keep].max(1),
                            prob30=prob[:, keep].max(1), dec30=dec[:, keep].max(1), cen30=cen[:, keep].max(1)))
    g1d_spec = importlib.util.spec_from_file_location("g1d", G1 / "03_decomposition.py")
    g1d = importlib.util.module_from_spec(g1d_spec)
    g1d_spec.loader.exec_module(g1d)
    df = g1d.frame(G1 / "out" / "scores_all.csv").merge(b30, on="post_id")
    grp = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    inc = {}
    for name, cols in {"raw31_submitted": ["raw_max_fit"], "raw30": ["raw30"],
                       "raw_plus_spec31_stored": C.PH4_F, "raw_plus_spec30": ["raw30", "spec30"],
                       "calibrated31": ["calib_max_fit"], "calibrated30": ["cal30"],
                       "classifier_prob31": ["clf_max_prob"], "classifier_prob30": ["prob30"],
                       "classifier_decision31": ["clf_max_decision"], "classifier_decision30": ["dec30"],
                       "centroid31": ["sae_centroid_max"], "centroid30": ["cen30"]}.items():
        inc[name] = g1d._inc(df, grp, C.CTRL_D, cols)["dR2"]
    inc["raw30_over_magnitude"] = g1d._inc(df, grp, C.CTRL_D + g1d.GEN, ["raw30"])["dR2"]
    res["bank30"] = dict(affect=affect, max_abs_diff_raw31_vs_raw30=float(np.abs(df.raw_max_fit - df.raw30).max()),
                         max_abs_diff_spec=float(np.abs(df.ph4_spec_margin - df.spec30).max()),
                         n_spec_changed=int((np.abs(df.raw_spec_margin - df.spec30) > 1e-12).sum()),
                         increments_dR2=inc)
    C.write_json(OUT / "repair_cheap.json", res)
    print(json.dumps({"pv_full_seed0": pv["full_4263_seed0"], "singletons_cc10": cs["cc10_as_planned"],
                      "bank30": res["bank30"]}, indent=1))


if __name__ == "__main__":
    main()
