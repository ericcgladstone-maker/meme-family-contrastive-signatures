"""
Reviewer-gated numerical tasks (engagement side). Run from build/.
1. Primary increment (full 4,263): family-cluster bootstrap at 2,000 iters (BCa-style percentile).
2. Random-signature randomization test (subset 2,543): 500 random-feature banks + 200 label-
   permutation banks; locate the observed raw-fit increment (0.080) in each null.
3. Uncertainty on the generic-magnitude-adjusted increment (0.026): 2,000 family-cluster bootstrap.
Writes reviewer_numerics_results.md.
"""
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
K = 15
rng = np.random.default_rng(0)
out = []
def log(s): print(s, flush=True); out.append(s)

def gkf_inc(df, groups, ctrl, ph4):
    E.build_dummies(df); E.CTRL = ctrl; E.PH4 = ph4
    k = min(5, len(np.unique(groups)))
    return E.run(df, GroupKFold(k), groups=groups, infold=True)

def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

D = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]

# ---------- 1. primary increment bootstrap @2000 (full sample) ----------
full = pd.read_csv(OUTC); full = full[full.listing_group == "denominator"].reset_index(drop=True)
fams = full.ph4_fit_family.unique(); by = {f: full[full.ph4_fit_family == f] for f in fams}
def boot_inc(dframe_fn, nboot, ph4, ctrl=D):
    vals = []
    for b in range(nboot):
        pick = rng.choice(fams, len(fams), replace=True)
        bt = pd.concat([by[f].assign(__rep=f"{f}_{i}") for i, f in enumerate(pick)], ignore_index=True)
        g = bt.__rep.astype("category").cat.codes.to_numpy()  # replicate-level id: duplicate draws are distinct groups
        try:
            r = gkf_inc(bt, g, ctrl, ph4); vals.append(r["dR2"])
        except Exception:
            pass
    return np.array(vals)
log("## 1. Primary family-grouped continuous increment, family-cluster bootstrap @2000 (full n=4,263)")
v = boot_inc(None, 2000, ["max_contrastive_ph4_fit", "ph4_spec_margin"])
log(f"n_ok={len(v)} median={np.median(v):+.3f} mean={v.mean():+.3f} "
    f"95% percentile CI [{np.percentile(v,2.5):+.3f}, {np.percentile(v,97.5):+.3f}] "
    f"n>0={int((v>0).sum())}/{len(v)} ({(v>0).mean()*100:.1f}%)")

# ---------- subset embeddings + cohort dense (for randomization tests) ----------
z = np.load(f"{SS}/reddit_denom_embeddings.npz", allow_pickle=True); rid = list(z["ids"]); rvec = z["vectors"]
sae = sae_msae.MSAE(device="cpu"); rsr = sae_msae.activation_records(rid, rvec, sae)
rdense = np.zeros((len(rid), sae.latent_dim), np.float32)
for i, r in enumerate(rsr): rdense[i, r["active_indices"]] = r["active_values"]
sub = pd.read_csv(OUTC); sub = sub[sub.listing_group == "denominator"]
sub = sub.merge(pd.DataFrame({"post_id": rid, "_ix": range(len(rid))}), on="post_id").reset_index(drop=True)
gsub = sub.ph4_fit_family.astype("category").cat.codes.to_numpy()
ix = sub["_ix"].to_numpy()
# cohort dense for label-permutation signatures
cohort = {r["variant_id"]: r for r in csv.DictReader(open(CFD, newline=""))}
emb = load_npz("first_results/embeddings_openai.npz")
canon = [v for v in emb["ids"] if v in cohort]; cvec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
cfam = np.array([cohort[v]["family_id"] for v in canon])
csr = sae_msae.activation_records(canon, cvec, sae)
cdense = np.zeros((len(canon), sae.latent_dim), np.float32)
for i, r in enumerate(csr): cdense[i, r["active_indices"]] = r["active_values"]

def score_max(sigs):  # sigs: list of (topidx, w); returns per-subset-post max fit
    return np.array([max(cosv(rdense[j][t], w) for t, w in sigs) for j in ix])

def inc_of_scores(scores):
    d = sub.copy(); d["_s"] = scores
    return gkf_inc(d, gsub, D, ["_s"])["dR2"]

OBS = 0.080  # observed raw-fit increment on subset
# ---------- 2a. random-feature-signature null (500 banks) ----------
log("\n## 2a. Random-feature-signature null (500 banks); observed raw-fit increment = 0.080")
rf = []
for b in range(500):
    sigs = [(rng.choice(sae.latent_dim, K, replace=False), np.abs(rng.normal(size=K))) for _ in range(30)]
    rf.append(inc_of_scores(score_max(sigs)))
rf = np.array(rf)
log(f"null: mean={rf.mean():+.4f} 95th pct={np.percentile(rf,95):+.4f} max={rf.max():+.4f}; "
    f"p(null>=obs)={(rf>=OBS).mean():.4f}")

# ---------- 2b. label-permutation null (200 banks): rebuild real-structure signatures on shuffled labels ----------
log("\n## 2b. Label-permutation null (200 banks): real sparse structure, family identity removed")
fam_sizes = pd.Series(cfam).value_counts().to_dict()
lp = []
n = len(canon)
for b in range(200):
    perm = cfam.copy(); rng.shuffle(perm)
    sigs = []
    for f in set(perm):
        idx = np.where(perm == f)[0]
        if len(idx) < 5: continue
        nm = np.where(perm != f)[0]
        diff = cdense[idx].mean(0) - cdense[nm].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
        sigs.append((top, w))
    lp.append(inc_of_scores(score_max(sigs)))
lp = np.array(lp)
log(f"null: mean={lp.mean():+.4f} 95th pct={np.percentile(lp,95):+.4f} max={lp.max():+.4f}; "
    f"p(null>=obs)={(lp>=OBS).mean():.4f}")

# ---------- 3. uncertainty on generic-magnitude-adjusted increment (~0.026) ----------
log("\n## 3. Adjusted increment (raw fit over generic-magnitude block), family-cluster bootstrap @2000 (subset)")
sparse_norm = np.linalg.norm(rdense[ix], axis=1); clip_norm = np.linalg.norm(rvec[ix], axis=1)
sub2 = sub.copy(); sub2["sparse_norm"] = sparse_norm; sub2["clip_norm"] = clip_norm; sub2["ocr_len"] = sub2["ocr_len"].astype(float)
GEN = ["sparse_norm", "clip_norm", "ocr_len"]
fams2 = sub2.ph4_fit_family.unique(); by2 = {f: sub2[sub2.ph4_fit_family == f] for f in fams2}
adj = []
for b in range(2000):
    pick = rng.choice(fams2, len(fams2), replace=True)
    bt = pd.concat([by2[f].assign(__rep=f"{f}_{i}") for i, f in enumerate(pick)], ignore_index=True)
    g = bt.__rep.astype("category").cat.codes.to_numpy()
    try:
        adj.append(gkf_inc(bt, g, D + GEN, ["max_contrastive_ph4_fit"])["dR2"])
    except Exception:
        pass
adj = np.array(adj)
log(f"adjusted increment: median={np.median(adj):+.3f} mean={adj.mean():+.3f} "
    f"95% CI [{np.percentile(adj,2.5):+.3f}, {np.percentile(adj,97.5):+.3f}] n>0={(adj>0).mean()*100:.1f}%")

open(f"{SS}/reviewer_numerics_results.md", "w").write("# Reviewer-gated numerics\n\n" + "\n".join(out) + "\n")
print("WROTE reviewer_numerics_results.md")
