"""
Executed repair analyses (e1 grouped CV + E3 in-fold labels, m2 sparse-centroid, win-count
reconciliation) against the real on-disk artifacts. Run from build/.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score, r2_score

OUT = "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
PROF = "first_results/ph4_contrastive_consolidation/contrastive_family_profiles.csv"
SEED = 0
CTRL = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]
PH4 = ["max_contrastive_ph4_fit", "ph4_spec_margin"]

_DUM = None  # fixed subreddit-dummy matrix over the FULL df (consistent columns across folds)
def build_dummies(df):
    global _DUM
    _DUM = pd.get_dummies(df["subreddit"], prefix="s").astype(float).reset_index(drop=True)

def design(df, cols):
    X = df[cols].to_numpy(float)
    d = _DUM.loc[df.index].to_numpy(float)
    return np.hstack([X, d])

def infold_binary(dtr, dte):
    def thr_table(d):
        t = {}
        for (s, a), g in d.groupby(["subreddit", "age_window"]):
            if len(g) >= 4:
                t[(s, a)] = np.percentile(g["log_score"].values, 75)
        return t, np.percentile(d["log_score"].values, 75)
    t, gl = thr_table(dtr)
    def lab(d):
        return np.array([int(r.log_score >= t.get((r.subreddit, r.age_window), gl))
                         for r in d.itertuples()])
    return lab(dtr), lab(dte)

def run(df, splitter, groups=None, infold=True):
    yc = df["log_score"].to_numpy(float)
    predCD = np.zeros(len(df)); predCF = np.zeros(len(df))
    predBD = np.zeros(len(df)); predBF = np.zeros(len(df)); yb = np.zeros(len(df))
    it = splitter.split(df, yc, groups) if groups is not None else splitter.split(df, yc)
    for tr, te in it:
        dtr, dte = df.iloc[tr], df.iloc[te]
        for cols, dst in [(CTRL, predCD), (CTRL + PH4, predCF)]:
            m = make_pipeline(StandardScaler(), Ridge(1.0)).fit(design(dtr, cols), yc[tr])
            dst[te] = m.predict(design(dte, cols))
        if infold:
            ytr, yte = infold_binary(dtr, dte)
        else:
            ytr, yte = df["high_engagement"].to_numpy()[tr], df["high_engagement"].to_numpy()[te]
        yb[te] = yte
        for cols, dst in [(CTRL, predBD), (CTRL + PH4, predBF)]:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(design(dtr, cols), ytr)
            dst[te] = m.predict_proba(design(dte, cols))[:, 1]
    return dict(R2_D=r2_score(yc, predCD), R2_F=r2_score(yc, predCF),
                dR2=r2_score(yc, predCF) - r2_score(yc, predCD),
                AUC_D=roc_auc_score(yb, predBD), AUC_F=roc_auc_score(yb, predBF),
                dAUC=roc_auc_score(yb, predBF) - roc_auc_score(yb, predBD))

def main():
    df = pd.read_csv(OUT)
    df = df[df.listing_group == "denominator"].reset_index(drop=True)
    build_dummies(df)
    print(f"denominator n={len(df)}  families={df.ph4_fit_family.nunique()}  subs={df.subreddit.nunique()}")

    print("\n[A] RANDOM 5-fold, GLOBAL labels (reproduce published +0.046 as pipeline check):")
    r = run(df, KFold(5, shuffle=True, random_state=SEED), infold=False)
    print("   " + "  ".join(f"{k}={v:+.4f}" for k, v in r.items()))

    print("\n[B] RANDOM 5-fold, IN-FOLD labels (E3 fix only):")
    r = run(df, KFold(5, shuffle=True, random_state=SEED), infold=True)
    print("   " + "  ".join(f"{k}={v:+.4f}" for k, v in r.items()))

    print("\n[C] GROUPED by subreddit (community leakage blocked):")
    g = df.subreddit.astype("category").cat.codes.to_numpy()
    r = run(df, GroupKFold(7), groups=g, infold=True)
    print("   " + "  ".join(f"{k}={v:+.4f}" for k, v in r.items()))

    print("\n[D] GROUPED by best-fit family (E1 template/family leak blocked, conservative):")
    g = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    nfold = min(5, df.ph4_fit_family.nunique())
    r = run(df, GroupKFold(nfold), groups=g, infold=True)
    print("   " + "  ".join(f"{k}={v:+.4f}" for k, v in r.items()))
    df.to_parquet("/tmp/ph4_denom.parquet") if False else None

    # ---- m2: sparse-centroid same-representation baseline (already computed in profiles) ----
    print("\n[m2] SAME-REPRESENTATION BASELINE (sparse centroid), from profiles:")
    p = pd.read_csv(PROF)
    con, sae = p.contrastive_recovery_auc, p.sae_auc
    print(f"   mean recovery: contrastive={con.mean():.3f}  sparse_centroid={sae.mean():.3f}  "
          f"CLIP={p.clip_auc.mean():.3f}  best_surface={p.best_surface_auc.mean():.3f}  "
          f"recurrence={p.recurrence_recovery_auc.mean():.3f}")
    print(f"   contrastive - sparse_centroid gain = {(con-sae).mean():+.3f}")
    print(f"   contrastive beats sparse_centroid (margin>0.02): {int(((con-sae)>0.02).sum())}/{len(p)}")
    print(f"   contrastive beats sparse_centroid (strict >):     {int((con>sae).sum())}/{len(p)}")

    # ---- win-count reconciliation (M5) ----
    print("\n[M5] win-count reconciliation vs recurrence / CLIP / best-surface:")
    for margin, lab in [(0.0, "strict>"), (0.02, "margin>0.02")]:
        wr = int(((p.contrastive_recovery_auc - p.recurrence_recovery_auc) > margin).sum())
        wc = int(((p.contrastive_recovery_auc - p.clip_auc) > margin).sum())
        ws = int(((p.contrastive_recovery_auc - p.best_surface_auc) > margin).sum())
        print(f"   {lab:12s}: recurrence {wr}/30  CLIP {wc}/30  best_surface {ws}/30")

if __name__ == "__main__":
    np.random.seed(SEED)
    main()
