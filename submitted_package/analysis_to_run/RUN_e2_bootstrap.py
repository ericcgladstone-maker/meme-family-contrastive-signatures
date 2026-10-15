"""e2: cluster bootstrap (resample families) of the family-grouped increment. Run from build/."""
import numpy as np, pandas as pd
import RUN_e1_m2_reconcile as E
from sklearn.model_selection import GroupKFold

OUT = "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
NBOOT = 300; SEED = 0

def grouped_increment(df):
    df = df.reset_index(drop=True); E.build_dummies(df)
    g = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    k = min(5, df.ph4_fit_family.nunique())
    return E.run(df, GroupKFold(k), groups=g, infold=True)

def main():
    df = pd.read_csv(OUT); df = df[df.listing_group == "denominator"].reset_index(drop=True)
    fams = df.ph4_fit_family.unique()
    by = {f: df[df.ph4_fit_family == f] for f in fams}
    rng = np.random.default_rng(SEED)
    dR2, dAUC = [], []
    for b in range(NBOOT):
        pick = rng.choice(fams, len(fams), replace=True)
        boot = pd.concat([by[f] for f in pick], ignore_index=True)
        try:
            r = grouped_increment(boot); dR2.append(r["dR2"]); dAUC.append(r["dAUC"])
        except Exception:
            continue
    def ci(x):
        x = np.array(x); return np.percentile(x, 2.5), np.percentile(x, 97.5), float((x > 0).mean())
    l2, h2, p2 = ci(dR2); la, ha, pa = ci(dAUC)
    print(f"n_boot_ok={len(dR2)}")
    print(f"family-cluster bootstrap, family-grouped increment:")
    print(f"  dR2  median {np.median(dR2):+.3f}  95% CI [{l2:+.3f}, {h2:+.3f}]  frac>0={p2:.3f}")
    print(f"  dAUC median {np.median(dAUC):+.3f}  95% CI [{la:+.3f}, {ha:+.3f}]  frac>0={pa:.3f}")

if __name__ == "__main__":
    main()
