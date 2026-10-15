#!/usr/bin/env python3
"""Independent reconstruction of Table 2 (Panels 2A and 2B) and the RQ2 decoy differences from family-level outputs
(numerical audit, 09 F-67). Written from the Method definitions, NOT from the summarizers (g3 03_summarize.py,
heldout summarize_heldout.py), to catch a shared implementation error.

Definitions used (MS Method):
  recovery AUROC per family: the fold-local column `<method>__E_local`
  oracle surface: per-family maximum of the surface measures (pHash, OCR, template)
  mean recovery: mean over families; paired advantage: mean over families of contrastive - method
  interval: family bootstrap, 5,000 resamples, percentile 95%, numpy default_rng(0)
  families > 0.02: count of families with contrastive - method > 0.02
  RQ2: contrastive - CLIP centroid on `__DL_local` (encoder-selected) and `__DP_local` (pHash-selected)
  H / C: analyzable held-out families by screening queue (dispositions.csv), independent of the summarizer.
Output: out/table2_reconstruction.json and a comparison against the stored results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AN = Path(__file__).resolve().parents[1]
METHODS = ["recurrence", "clip_centroid", "surf_composite", "oracle_surface", "sparse_centroid", "sparse_knn", "logistic",
           "fused_centroid"]
SURF = ["surf_phash", "surf_ocr", "surf_template"]


def boot_ci(x, B=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def block(df):
    df = df.copy()
    df["oracle_surface__E_local"] = df[[f"{s}__E_local" for s in SURF]].max(axis=1)
    out = {"n_families": int(len(df)), "means": {}, "paired_contrastive_minus": {}, "rq2": {}}
    out["means"]["contrastive"] = float(df["contrastive__E_local"].mean())
    for m in METHODS:
        col = f"{m}__E_local"
        if col not in df:
            continue
        out["means"][m] = float(df[col].mean())
        d = df["contrastive__E_local"] - df[col]
        out["paired_contrastive_minus"][m] = {"mean": float(d.mean()), "ci95": boot_ci(d), "n_gt_0p02": int((d > 0.02).sum())}
    for key, sfx in [("laion", "DL"), ("phash", "DP")]:
        a, b = f"contrastive__{sfx}_local", f"clip_centroid__{sfx}_local"
        if a in df and b in df:
            d = (df[a] - df[b]).dropna()
            out["rq2"][key] = {"mean": float(d.mean()), "ci95": boot_ci(d), "n": int(len(d))}
    return out


def main():
    dev = pd.read_csv(AN / "g3_recovery/out/clean_primary.csv")
    ho = pd.read_csv(AN / "heldout/out/ho_primary.csv")
    disp = pd.read_csv(AN / "heldout/out/dispositions.csv")
    ana = disp[disp.disposition == "analyzable"]
    sets = {"H": set("ho_" + ana[ana.queue == "H"].slug), "C": set("ho_" + ana[ana.queue == "C"].slug)}
    res = {"2A_dev": block(dev),
           "2B_H": block(ho[ho.family.isin(sets["H"])]), "2B_C": block(ho[ho.family.isin(sets["C"])])}
    (Path(__file__).parent / "out").mkdir(exist_ok=True)
    (Path(__file__).parent / "out" / "table2_reconstruction.json").write_text(json.dumps(res, indent=2) + "\n")
    # ---- compare with stored results
    g3 = json.loads((AN / "g3_recovery/out/g3_results.json").read_text())
    hr = json.loads((AN / "heldout/out/heldout_results.json").read_text())
    rows = []

    def cmp(label, mine, stored, tol):
        ok = stored is not None and abs(mine - stored) <= tol
        rows.append((label, round(mine, 4), None if stored is None else round(stored, 4), "OK" if ok else "DIFF"))
    A = res["2A_dev"]
    for m in ["recurrence", "clip_centroid", "surf_composite", "oracle_surface", "sparse_centroid", "sparse_knn", "logistic", "contrastive"]:
        cmp(f"2A mean {m}", A["means"][m], g3["RQ1_means"][m]["mean_local"], 5e-4)
    for m in ["recurrence", "clip_centroid", "surf_composite", "oracle_surface", "sparse_centroid", "sparse_knn", "logistic"]:
        p, s = A["paired_contrastive_minus"][m], g3["RQ1_paired_contrastive_minus"][m]
        cmp(f"2A diff {m}", p["mean"], s["mean"], 5e-4)
        cmp(f"2A ci_lo {m}", p["ci95"][0], s["ci95"][0], 4e-3)
        cmp(f"2A ci_hi {m}", p["ci95"][1], s["ci95"][1], 4e-3)
        if "n_gt_0p02" in s:
            cmp(f"2A n>0.02 {m}", p["n_gt_0p02"], s["n_gt_0p02"], 0)
    for st in ["H", "C"]:
        B, S = res[f"2B_{st}"], hr[st]
        cmp(f"2B {st} n_families", B["n_families"], S.get("n_families"), 0)
        for m in ["recurrence", "clip_centroid", "surf_composite", "oracle_surface", "sparse_centroid", "sparse_knn", "logistic", "fused_centroid", "contrastive"]:
            if m in B["means"]:
                cmp(f"2B {st} mean {m}", B["means"][m], S["means"].get(m), 5e-4)
        for m in ["recurrence", "clip_centroid", "surf_composite", "oracle_surface", "sparse_centroid", "sparse_knn", "fused_centroid"]:
            if m in B["paired_contrastive_minus"] and m in S["rq1"]:
                p, s = B["paired_contrastive_minus"][m], S["rq1"][m]
                cmp(f"2B {st} diff {m}", p["mean"], s["mean"], 5e-4)
                cmp(f"2B {st} ci_lo {m}", p["ci95"][0], s["ci95"][0], 4e-3)
                cmp(f"2B {st} ci_hi {m}", p["ci95"][1], s["ci95"][1], 4e-3)
        p = B["paired_contrastive_minus"]["logistic"]
        s = S["classifier_minus_contrastive"]
        cmp(f"2B {st} logistic-contrastive", -p["mean"], s["mean"], 5e-4)
        cmp(f"2B {st} logistic ci_lo", -p["ci95"][1], s["ci95"][0], 4e-3)
        cmp(f"2B {st} logistic ci_hi", -p["ci95"][0], s["ci95"][1], 4e-3)
        cmp(f"2B {st} rq2 laion", B["rq2"]["laion"]["mean"], S["rq2_laion_familylevel"]["mean"], 5e-4)
        cmp(f"2B {st} rq2 laion ci_lo", B["rq2"]["laion"]["ci95"][0], S["rq2_laion_familylevel"]["ci95"][0], 4e-3)
        cmp(f"2B {st} rq2 laion ci_hi", B["rq2"]["laion"]["ci95"][1], S["rq2_laion_familylevel"]["ci95"][1], 4e-3)
    cmp("2B H rq2 phash", res["2B_H"]["rq2"]["phash"]["mean"], hr["H"]["phash_decoys_descriptive"]["mean"], 5e-4)
    for key, gk in [("laion", "laion_encoder_selected"), ("phash", "phash_selected")]:
        s = g3["RQ2"][gk]["contrastive_minus_clip"]
        cmp(f"2A rq2 {key}", A["rq2"][key]["mean"], s["mean"], 5e-4)
        cmp(f"2A rq2 {key} ci_lo", A["rq2"][key]["ci95"][0], s["ci95"][0], 4e-3)
        cmp(f"2A rq2 {key} ci_hi", A["rq2"][key]["ci95"][1], s["ci95"][1], 4e-3)
    bad = [r for r in rows if r[3] != "OK"]
    out = {"n_checks": len(rows), "n_ok": len(rows) - len(bad), "diffs": bad, "all": rows}
    (Path(__file__).parent / "out" / "table2_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{out['n_ok']}/{out['n_checks']} checks agree")
    for r in bad:
        print("  DIFF", r)


if __name__ == "__main__":
    main()
