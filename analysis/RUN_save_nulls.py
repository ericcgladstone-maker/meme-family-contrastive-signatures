"""Re-run the two randomization nulls, SAVING the full increment arrays so the SI/main
null-distribution histograms can be plotted. Run from build/. ~10-15 min."""
import sys, csv
import numpy as np, pandas as pd
sys.path.insert(0, "first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run")
sys.path.insert(0, ".")
import RUN_e1_m2_reconcile as E
from sklearn.model_selection import GroupKFold
from src.analysis import sae_msae
from src.analysis.clip_embed import load_npz

SS = "/Users/ericgladstone/My Drive (eric.c.gladstone@gmail.com)/Projects/Paper PH4 and Memes Computational/SSCR Submission/07_Analysis_Audit"
OUTC = "first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"
CFD = "first_results/cross_family_dedup/cross_family_corrected_index.csv"
K = 15; rng = np.random.default_rng(0)
D = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]

def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

z = np.load(f"{SS}/reddit_denom_embeddings.npz", allow_pickle=True); rid = list(z["ids"]); rvec = z["vectors"]
sae = sae_msae.MSAE(device="cpu"); rsr = sae_msae.activation_records(rid, rvec, sae)
rdense = np.zeros((len(rid), sae.latent_dim), np.float32)
for i, r in enumerate(rsr): rdense[i, r["active_indices"]] = r["active_values"]
sub = pd.read_csv(OUTC); sub = sub[sub.listing_group == "denominator"]
sub = sub.merge(pd.DataFrame({"post_id": rid, "_ix": range(len(rid))}), on="post_id").reset_index(drop=True)
gsub = sub.ph4_fit_family.astype("category").cat.codes.to_numpy(); ix = sub["_ix"].to_numpy()
cohort = {r["variant_id"]: r for r in csv.DictReader(open(CFD, newline=""))}
emb = load_npz("first_results/embeddings_openai.npz")
canon = [v for v in emb["ids"] if v in cohort]; cvec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
cfam = np.array([cohort[v]["family_id"] for v in canon])
csr = sae_msae.activation_records(canon, cvec, sae)
cdense = np.zeros((len(canon), sae.latent_dim), np.float32)
for i, r in enumerate(csr): cdense[i, r["active_indices"]] = r["active_values"]

def score_max(sigs):
    return np.array([max(cosv(rdense[j][t], w) for t, w in sigs) for j in ix])
def inc(scores):
    d = sub.copy(); d["_s"] = scores; E.build_dummies(d); E.CTRL = D; E.PH4 = ["_s"]
    return E.run(d, GroupKFold(5), groups=gsub, infold=True)["dR2"]

rf = []
for b in range(500):
    sigs = [(rng.choice(sae.latent_dim, K, replace=False), np.abs(rng.normal(size=K))) for _ in range(30)]
    rf.append(inc(score_max(sigs)))
    if b % 100 == 0: print("random-feature", b, flush=True)
lp = []
for b in range(200):
    perm = cfam.copy(); rng.shuffle(perm); sigs = []
    for f in set(perm):
        idx = np.where(perm == f)[0]
        if len(idx) < 5: continue
        nm = np.where(perm != f)[0]
        diff = cdense[idx].mean(0) - cdense[nm].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
        sigs.append((top, w))
    lp.append(inc(score_max(sigs)))
    if b % 50 == 0: print("label-perm", b, flush=True)

pd.DataFrame({"increment": rf}).to_csv(f"{SS}/null_random_feature.csv", index=False)
pd.DataFrame({"increment": lp}).to_csv(f"{SS}/null_label_perm.csv", index=False)
print(f"SAVED nulls. random-feature: n={len(rf)} mean={np.mean(rf):.4f} max={np.max(rf):.4f} p(>=.080)={np.mean(np.array(rf)>=0.080):.4f}")
print(f"label-perm: n={len(lp)} mean={np.mean(lp):.4f} max={np.max(lp):.4f} p(>=.080)={np.mean(np.array(lp)>=0.080):.4f}")
