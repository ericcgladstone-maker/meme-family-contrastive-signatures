"""
e1_grouped_cv.py  --  Repairs issues E1 (ungrouped CV) and E3 (global outcome bins).

Primary engagement estimand, corrected:
  * cross-validation folds GROUPED by template / perceptual cluster, so template
    siblings do not straddle train and test (E1);
  * top-quartile-within-bin label and the continuous outcome recomputed INSIDE each
    training fold, so a held-out label is never defined against the full-sample
    distribution (E3);
  * AUC (binary) and R^2 (continuous) reported as two separate results (E5).

Inputs (CONFIRM paths/columns against the real artifacts):
  ../../ph4_q2_horsepower/q2_denom_outcomes.csv   # per-post table written by fr_36
Required columns (rename map below if they differ):
  post_id, subreddit, log_score, age_hours, phash (uint64 or hex),
  ph4_fit (max contrastive fit), ph4_spec (specificity margin),
  clip_sim (max whole-image family similarity), phash_sim, tmpl_sim, text_sim
Outputs:
  grouped_cv_results.csv  and printed increments to paste into the manuscript.

Run:  python e1_grouped_cv.py
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, r2_score

SEED = 0
INPUT = "../../ph4_q2_horsepower/q2_denom_outcomes.csv"   # CONFIRM
PHASH_GROUP_HAMMING = 10   # looser than the <=6 DEDUP threshold: groups same-template siblings
CONTROL = ["clip_sim", "phash_sim", "tmpl_sim", "text_sim", "age_hours"]   # + subreddit dummies
PH4 = ["ph4_fit", "ph4_spec"]

# ---- rename map: set to the real column names if they differ ----
RENAME = {}   # e.g. {"score_log": "log_score", "max_fit": "ph4_fit"}


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


def perceptual_clusters(phash, thr):
    """Connected components over the pHash Hamming graph => template-sibling clusters."""
    ph = [int(x, 16) if isinstance(x, str) else int(x) for x in phash]
    n = len(ph)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    # O(n^2); n=4263 is fine. For larger n use a BK-tree.
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(ph[i], ph[j]) <= thr:
                union(i, j)
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])


def infold_binary_label(df_train, df_eval):
    """Top-quartile-within (subreddit x age-window) label, thresholds fit on TRAIN only."""
    def age_bin(a):
        return np.digitize(a, [6, 12, 24, 48])   # posting-age windows; adjust to fr_36
    tr = df_train.copy(); tr["ab"] = age_bin(tr["age_hours"].values)
    ev = df_eval.copy();  ev["ab"] = age_bin(ev["age_hours"].values)
    thr = {}
    for (s, ab), g in tr.groupby(["subreddit", "ab"]):
        if len(g) >= 4:
            thr[(s, ab)] = np.percentile(g["log_score"].values, 75)
    glob = np.percentile(tr["log_score"].values, 75)
    def lab(row):
        return int(row["log_score"] >= thr.get((row["subreddit"], row["ab"]), glob))
    return tr.apply(lab, axis=1).values, ev.apply(lab, axis=1).values


def design(df, cols):
    X = df[cols].to_numpy(float)
    d = pd.get_dummies(df["subreddit"], prefix="sub").to_numpy(float)
    return np.hstack([X, d])


def grouped_increment(df, groups):
    gkf = GroupKFold(n_splits=5)
    yc = df["log_score"].to_numpy(float)
    predC_D = np.zeros(len(df)); predC_F = np.zeros(len(df))
    predB_D = np.zeros(len(df)); predB_F = np.zeros(len(df)); yb = np.zeros(len(df))
    for tr, te in gkf.split(df, yc, groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        # continuous (ridge), in-fold scaling via pipeline
        for cols, pc in [(CONTROL, "D"), (CONTROL + PH4, "F")]:
            m = make_pipeline(StandardScaler(), Ridge(1.0))
            m.fit(design(dtr, cols), yc[tr])
            p = m.predict(design(dte, cols))
            (predC_D if pc == "D" else predC_F)[te] = p
        # binary (logistic), labels recomputed in-fold
        ytr, yte = infold_binary_label(dtr, dte); yb[te] = yte
        for cols, pc in [(CONTROL, "D"), (CONTROL + PH4, "F")]:
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(design(dtr, cols), ytr)
            p = m.predict_proba(design(dte, cols))[:, 1]
            (predB_D if pc == "D" else predB_F)[te] = p
    r2D, r2F = r2_score(yc, predC_D), r2_score(yc, predC_F)
    aD, aF = roc_auc_score(yb, predB_D), roc_auc_score(yb, predB_F)
    return dict(R2_D=r2D, R2_F=r2F, dR2=r2F - r2D, AUC_D=aD, AUC_F=aF, dAUC=aF - aD)


def main():
    df = pd.read_csv(INPUT).rename(columns=RENAME).reset_index(drop=True)
    groups = perceptual_clusters(df["phash"].values, PHASH_GROUP_HAMMING)
    print(f"n={len(df)}  n_template_clusters={groups.max()+1}")
    res = grouped_increment(df, groups)
    print("GROUPED-CV (template-cluster) increments, in-fold labels:")
    for k, v in res.items():
        print(f"  {k:6s} {v:+.4f}")
    print("\nPaste into manuscript R5/E1:")
    print(f"  continuous dR2 (PRIMARY) = {res['dR2']:+.3f};  binary dAUC = {res['dAUC']:+.3f}")
    pd.DataFrame([res]).to_csv("grouped_cv_results.csv", index=False)
    print("wrote grouped_cv_results.csv")
    # For comparison, run the OLD random-fold estimate here too if desired (KFold shuffle).


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
