"""
Concern 4 / random-signature control (reviewer): is the engagement signal generic magnitude
rather than family fit? Adds sparse-activation norm, active-feature count, CLIP embedding
norm, OCR length, and a RANDOM-SIGNATURE maximum fit to the engagement models, and tests
whether raw family-fit survives after controlling for generic magnitude. Run from build/.
"""
import sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, "first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run")
sys.path.insert(0, ".")
import RUN_e1_m2_reconcile as E
from sklearn.model_selection import GroupKFold
from src.analysis import sae_msae

SS = "/Users/ericgladstone/My Drive (eric.c.gladstone@gmail.com)/Projects/Paper PH4 and Memes Computational/SSCR Submission/07_Analysis_Audit"
CKPT = f"{SS}/reddit_denom_embeddings.npz"
OUTC = "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
K = 15

def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

def main():
    z = np.load(CKPT, allow_pickle=True); rid = list(z["ids"]); rvec = z["vectors"]
    sae = sae_msae.MSAE(device="cpu"); rsr = sae_msae.activation_records(rid, rvec, sae)
    rdense = np.zeros((len(rid), sae.latent_dim), np.float32)
    for i, r in enumerate(rsr): rdense[i, r["active_indices"]] = r["active_values"]
    # generic-magnitude features
    sparse_norm = np.linalg.norm(rdense, axis=1)
    n_active = (rdense > 0).sum(1).astype(float)
    clip_norm = np.linalg.norm(rvec, axis=1)
    # random-signature maximum fit (30 random top-15 signatures, positive weights)
    rng = np.random.default_rng(0)
    sigs = [(rng.choice(sae.latent_dim, K, replace=False), np.abs(rng.normal(size=K))) for _ in range(30)]
    rand_max = np.array([max(cosv(rdense[i][idx], w) for idx, w in sigs) for i in range(len(rid))])
    feat = pd.DataFrame({"post_id": rid, "sparse_norm": sparse_norm, "n_active": n_active,
                         "clip_norm": clip_norm, "rand_sig_max": rand_max})
    o = pd.read_csv(OUTC); o = o[o.listing_group == "denominator"]
    df = o.merge(feat, on="post_id").reset_index(drop=True)
    df["ocr_len"] = df["ocr_len"].astype(float)
    print(f"n={len(df)}")
    print("\ncorr with log_score:")
    for c in ["max_contrastive_ph4_fit", "sparse_norm", "n_active", "clip_norm", "rand_sig_max", "ocr_len"]:
        print(f"  {c:22s} {df[c].corr(df['log_score']):+.3f}")

    E.build_dummies(df)
    g = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    gkf = lambda: GroupKFold(min(5, df.ph4_fit_family.nunique()))
    def inc(ctrl, add, label):
        E.CTRL = ctrl; E.PH4 = add
        r = E.run(df, gkf(), groups=g, infold=True)
        print(f"  {label:52s} dR2={r['dR2']:+.4f}  dAUC={r['dAUC']:+.4f}")
        return r
    D = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]
    GEN = ["sparse_norm", "clip_norm", "ocr_len"]  # n_active dropped: constant (top-k=64)
    print("\nincrements over control D (family-grouped, in-fold labels):")
    inc(D, ["max_contrastive_ph4_fit"], "raw family-fit added to D (reference)")
    inc(D, ["sparse_norm"], "sparse_norm added to D")
    inc(D, ["rand_sig_max"], "RANDOM-signature max added to D")
    inc(D, ["clip_norm"], "clip_norm added to D")
    print("\ndoes raw family-fit survive generic-magnitude controls?")
    inc(D + GEN, ["max_contrastive_ph4_fit"], "raw fit added to D + generic-magnitude controls")
    inc(D, GEN, "generic-magnitude block added to D (no family-fit)")

if __name__ == "__main__":
    main()
