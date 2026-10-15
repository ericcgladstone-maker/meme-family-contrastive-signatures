"""Engagement comparison: does calibrated fit or the classifier-derived score predict
Reddit engagement better than raw family-fit? Family-grouped CV, in-fold labels.
Run from build/ with the e1 machinery on path."""
import sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, "first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run")
import RUN_e1_m2_reconcile as E
from sklearn.model_selection import GroupKFold

OUTC = "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
SCOR = "first_results/ph4_manuscript_natcomms/SSCR Submission/07_Analysis_Audit/downstream_scores.csv"

def main():
    o = pd.read_csv(OUTC); o = o[o.listing_group == "denominator"]
    s = pd.read_csv(SCOR)
    df = o.merge(s, on="post_id", how="inner").reset_index(drop=True)
    print(f"merged n={len(df)} (subset of 4263 with surviving images)")
    E.build_dummies(df)
    E.CTRL = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]
    g = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    gkf = lambda: GroupKFold(min(5, df.ph4_fit_family.nunique()))
    variants = {
        "raw_fit (subset)": ["raw_max_fit"],
        "raw_fit+spec (paper F)": ["max_contrastive_ph4_fit", "ph4_spec_margin"],
        "calibrated_fit": ["calib_max_fit"],
        "classifier_score": ["clf_max_prob"],
        "sae_centroid_score": ["sae_centroid_max"],
        "clip_similarity": ["max_clip_family_sim"],
    }
    print(f"{'variant':26s} {'dR2':>8s} {'dAUC':>8s}")
    res = {}
    for name, ph4 in variants.items():
        E.PH4 = ph4
        r = E.run(df, gkf(), groups=g, infold=True)
        res[name] = r
        print(f"{name:26s} {r['dR2']:+8.4f} {r['dAUC']:+8.4f}")
    # correlations with engagement for context
    print("\ncorr with log_score:")
    for c in ["raw_max_fit", "calib_max_fit", "clf_max_prob", "sae_centroid_max", "max_clip_family_sim"]:
        print(f"  {c:20s} {df[c].corr(df['log_score']):+.3f}")
    pd.DataFrame(res).T.to_csv("first_results/ph4_manuscript_natcomms/SSCR Submission/07_Analysis_Audit/downstream_engagement_results.csv")

if __name__ == "__main__":
    main()
