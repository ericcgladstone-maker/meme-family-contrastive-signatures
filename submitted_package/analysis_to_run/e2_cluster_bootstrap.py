"""
e2_cluster_bootstrap.py  --  Repairs issue E2 (understated uncertainty).

Replaces the post-level, R2-only, 300-draw bootstrap with a CLUSTER bootstrap that
resamples the dependence unit, refits the FULL grouped-CV pipeline inside each resample,
and reports intervals for BOTH increments (continuous dR2 and binary dAUC).

Two clustering units are reported (run both):
  * community x collection-context blocks
  * template / perceptual clusters (the axis PH4 fit rides on)

Depends on e1_grouped_cv.py (imports its pipeline). Inputs as in e1.
Output: cluster_bootstrap_ci.csv + printed intervals.
"""
import numpy as np, pandas as pd
from e1_grouped_cv import (INPUT, RENAME, CONTROL, PH4, perceptual_clusters,
                           grouped_increment, PHASH_GROUP_HAMMING)

N_BOOT = 2000
SEED = 0

def cluster_bootstrap(df, groups, cluster_key, n_boot=N_BOOT):
    rng = np.random.default_rng(SEED)
    clusters = df[cluster_key].to_numpy() if cluster_key in df else groups
    uniq = np.unique(clusters)
    dR2, dAUC = [], []
    for b in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)      # resample CLUSTERS
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        dfb = df.iloc[idx].reset_index(drop=True)
        gb = perceptual_clusters(dfb["phash"].values, PHASH_GROUP_HAMMING)
        try:
            r = grouped_increment(dfb, gb)
            dR2.append(r["dR2"]); dAUC.append(r["dAUC"])
        except Exception:
            continue
    def ci(x):
        x = np.array(x); return (np.percentile(x, 2.5), np.percentile(x, 97.5),
                                 float((x > 0).mean()))
    return ci(dR2), ci(dAUC)

def main():
    df = pd.read_csv(INPUT).rename(columns=RENAME).reset_index(drop=True)
    groups = perceptual_clusters(df["phash"].values, PHASH_GROUP_HAMMING)
    out = []
    for key in ["subreddit", "__template__"]:
        ck = key if key == "subreddit" else "__template__"
        if ck == "__template__":
            df["__template__"] = groups
        (lo2, hi2, p2), (loa, hia, pa) = cluster_bootstrap(df, groups, ck)
        print(f"[{key}] dR2 95% CI [{lo2:+.3f}, {hi2:+.3f}] frac>0={p2:.2f} | "
              f"dAUC 95% CI [{loa:+.3f}, {hia:+.3f}] frac>0={pa:.2f}")
        out.append(dict(cluster=key, dR2_lo=lo2, dR2_hi=hi2, dR2_frac=p2,
                        dAUC_lo=loa, dAUC_hi=hia, dAUC_frac=pa))
    pd.DataFrame(out).to_csv("cluster_bootstrap_ci.csv", index=False)
    print("wrote cluster_bootstrap_ci.csv -> paste the template-cluster intervals into R6.")

if __name__ == "__main__":
    main()
