#!/usr/bin/env python3
"""G3 summary: validation of the replications, clean-protocol RQ1/RQ2 results, attribution factorial,
paired family bootstraps (5,000; seed 0), win counts, LOFO range, seed sweep, K-curve.
Writes out/g3_results.json and out/g3_family_table.csv. All numbers come from engine outputs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

OUT = HERE / "out"
PROF = C.FR / "ph4_contrastive_consolidation" / "contrastive_family_profiles.csv"
M2M4 = C.REPRO / "data" / "m2m4_results.csv"
SUBMITTED = {  # Table 2 / SI S7-S9 means as submitted
    "recurrence": 0.774, "clip_centroid": 0.844, "oracle_surface": 0.813, "sparse_centroid": 0.821,
    "sparse_knn": 0.814, "contrastive": 0.895, "logistic": 0.956,
    "RQ2_laion": {"contrastive": 0.838, "clip_centroid": 0.655},
    "RQ2_phash": {"contrastive": 0.888, "clip_centroid": 0.854},
    "RQ2_openai_dependent_ref": {"clip_centroid": 0.45},
    "paired": {"contrastive-sparse_centroid": [0.077, 0.052, 0.104], "contrastive-sparse_knn": [0.083, 0.055, 0.111],
               "logistic-contrastive": [0.059, 0.042, 0.077]},
    "wins_margin_0.02": {"recurrence": 27, "clip_centroid": 20, "oracle_surface": 19, "sparse_centroid": 23, "sparse_knn": 24},
}


def paired(a, b, seed=0, nboot=5000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_pos=int((d > 0).sum()), n_gt_0p02=int((d > 0.02).sum()), n_lt_m0p02=int((d < -0.02).sum()), n=len(d))


def main():
    res = {"submitted": SUBMITTED}
    # ---- validation of replications
    val = {}
    if (OUT / "repro_fr25.csv").exists():
        r = pd.read_csv(OUT / "repro_fr25.csv")
        p = pd.read_csv(PROF)
        p["family"] = p["family_name"]
        m = r.merge(p, on="family", suffixes=("", "_sub"))
        pairs = {"rec_auc": "recurrence_recovery_auc", "con_auc": "contrastive_recovery_auc", "clip_auc": "clip_auc_sub",
                 "sae_auc": "sae_auc_sub", "phash_auc": "phash_auc_sub", "ocr_auc": "ocr_auc_sub", "tmpl_auc": "template_auc",
                 "best_surface_auc": "best_surface_auc_sub", "adv_con_laion": "adv_contrastive_laion",
                 "adv_rec_laion": "adv_recurrence_laion", "adv_clip_laion": "adv_clip_laion_sub"}
        val["fr25"] = {k: dict(max_abs_diff_vs_submitted_3dp=float((m[k].round(3) - m[v]).abs().max()),
                               mean_repro=float(m[k].mean()), mean_submitted=float(m[v].mean()))
                       for k, v in pairs.items() if v in m.columns}
    if (OUT / "repro_m2m4.csv").exists():
        r = pd.read_csv(OUT / "repro_m2m4.csv")
        s = pd.read_csv(M2M4)
        m = r.merge(s, on="family", suffixes=("", "_sub"))
        val["m2m4"] = {k: dict(max_abs_diff_vs_submitted=float((m[k] - m[f"{k}_sub"]).abs().max()),
                               mean_repro=float(m[k].mean()), mean_submitted=float(m[f"{k}_sub"].mean()))
                       for k in ["contrastive", "sae_centroid", "sae_knn", "logistic_sparse", "adv_con_phash",
                                 "adv_clip_phash", "adv_con_clip", "adv_clip_clip"]}
    res["validation"] = val

    # ---- clean primary
    c = pd.read_csv(OUT / "clean_primary.csv")
    methods = ["contrastive", "recurrence", "clip_centroid", "sparse_centroid", "sparse_knn", "logistic",
               "surf_phash", "surf_ocr", "surf_template", "surf_template_transductive", "surf_composite"]
    c["oracle_surface__E_local"] = c[["surf_phash__E_local", "surf_ocr__E_local", "surf_template__E_local"]].max(1)
    allm = methods + ["oracle_surface"]
    rq1 = {m: dict(mean_local=float(c[f"{m}__E_local"].mean()),
                   mean_pooled_clean=(float(c[f"{m}__E_pooled"].mean()) if f"{m}__E_pooled" in c else None)) for m in allm}
    res["RQ1_means"] = rq1
    res["RQ1_paired_contrastive_minus"] = {m: paired(c["contrastive__E_local"], c[f"{m}__E_local"])
                                           for m in allm if m != "contrastive"}
    res["RQ1_logistic_minus_contrastive"] = paired(c["logistic__E_local"], c["contrastive__E_local"])
    lofo = [c.drop(i)["contrastive__E_local"].mean() for i in c.index]
    res["RQ1_contrastive_LOFO_range"] = [float(min(lofo)), float(max(lofo))]
    res["RQ1_corr_family_size_vs_gain_over_sparse_centroid"] = float(
        np.corrcoef(c["n"], c["contrastive__E_local"] - c["sparse_centroid__E_local"])[0, 1])
    # ---- RQ2 (separately)
    rq2 = {}
    for sname, lab in [("DL", "laion_encoder_selected"), ("DP", "phash_selected"), ("DO", "openai_dependent_reference")]:
        rq2[lab] = dict(means={m: float(c[f"{m}__{sname}_local"].mean()) for m in methods},
                        contrastive_minus_clip=paired(c[f"contrastive__{sname}_local"], c[f"clip_centroid__{sname}_local"]),
                        contrastive_minus_recurrence=paired(c[f"contrastive__{sname}_local"], c[f"recurrence__{sname}_local"]),
                        logistic_minus_clip=paired(c[f"logistic__{sname}_local"], c[f"clip_centroid__{sname}_local"]))
    res["RQ2"] = rq2
    # ---- attribution factorial (contrastive)
    res["attribution_factorial_contrastive"] = {k: float(c[k].mean()) for k in
                                                ["fact_contam_pooled", "fact_contam_local", "fact_clean_pooled", "fact_clean_local"]}
    res["design"] = dict(n_families=len(c), mean_background=float(c.n_background.mean()),
                         min_background=int(c.n_background.min()), mean_E_decoy_overlap=float(c.overlap_E_decoys.mean()))
    # ---- seed sweep
    if (OUT / "clean_seed_sweep.csv").exists():
        s = pd.read_csv(OUT / "clean_seed_sweep.csv")
        s["oracle_surface__E_local"] = s[["surf_phash__E_local", "surf_ocr__E_local", "surf_template__E_local"]].max(1)
        per_seed = s.groupby("seed")[[f"{m}__E_local" for m in allm] + [f"{m}__DL_local" for m in ["contrastive", "clip_centroid"]]].mean()
        diffs = pd.DataFrame({f"contrastive-{m}": s.groupby("seed").apply(lambda g: (g["contrastive__E_local"] - g[f"{m}__E_local"]).mean())
                              for m in allm if m != "contrastive"})
        diffs["RQ2_laion_contrastive-clip"] = s.groupby("seed").apply(lambda g: (g["contrastive__DL_local"] - g["clip_centroid__DL_local"]).mean())
        res["seed_sweep_1_10"] = dict(method_means={k: [float(per_seed[k].min()), float(per_seed[k].max())] for k in per_seed},
                                      paired_diff_range={k: [float(diffs[k].min()), float(diffs[k].max())] for k in diffs})
    if (OUT / "clean_kcurve.csv").exists():
        kc = pd.read_csv(OUT / "clean_kcurve.csv")
        res["K_curve_clean_local"] = {f"K{kk}": float(kc[f"K{kk}__E_local"].mean()) for kk in (5, 10, 25, 50)}
        res["K_curve_clean_local"]["K15"] = float(kc["contrastive__E_local"].mean())
        res["K_curve_submitted"] = {"K5": 0.877, "K10": 0.890, "K15": 0.897, "K25": 0.903, "K50": 0.905}
    c.to_csv(OUT / "g3_family_table.csv", index=False)
    C.write_json(OUT / "g3_results.json", res)
    print("[g3] wrote g3_results.json")


if __name__ == "__main__":
    main()
