"""SI recovery-side numerics: K-sweep, negative-pool sensitivity, classifier sparsity.
Reconstructs cohort sparse matrix. Run from build/. Writes si_recovery_results.md."""
import sys, csv
from collections import defaultdict
import numpy as np
sys.path.insert(0, ".")
from src.analysis.clip_embed import load_npz
from src.analysis import sae_msae
from sklearn.linear_model import LogisticRegression
CFD = "first_results/cross_family_dedup/cross_family_corrected_index.csv"
NSAMP = 200
out = []
def log(s): print(s, flush=True); out.append(s)

def auc(pos, neg):
    if not pos or not neg: return None
    p = np.array(pos)[:, None]; n = np.array(neg)[None, :]
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))
def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

cohort = {r["variant_id"]: r for r in csv.DictReader(open(CFD, newline=""))}
emb = load_npz("first_results/embeddings_openai.npz")
canon = [v for v in emb["ids"] if v in cohort]; vec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
fam = [cohort[v]["family_id"] for v in canon]; n = len(canon)
sae = sae_msae.MSAE(device="cpu"); srecs = sae_msae.activation_records(canon, vec, sae)
dense = np.zeros((n, sae.latent_dim), np.float32)
for i, r in enumerate(srecs): dense[i, r["active_indices"]] = r["active_values"]
fam_idx = defaultdict(list)
for i, f in enumerate(fam): fam_idx[f].append(i)
fams = [f for f in fam_idx if len(fam_idx[f]) >= 5]

def recovery_mean(K, seed):
    rng = np.random.default_rng(seed); accs = []
    for f in fams:
        idxs = fam_idx[f]; nonfam = [i for i in range(n) if fam[i] != f]
        samp = list(rng.choice(nonfam, min(NSAMP, len(nonfam)), replace=False))
        m, nm = [], []
        for q in idxs:
            train = [i for i in idxs if i != q]
            diff = dense[train].mean(0) - dense[nonfam].mean(0); top = np.argsort(-diff)[:K]; w = np.clip(diff[top], 0, None)
            m.append(cosv(dense[q][top], w))
            for c in samp: nm.append(cosv(dense[c][top], w))
        a = auc(m, nm)
        if a: accs.append(a)
    return float(np.mean(accs))

log("## S5 K-sensitivity (contrastive recovery mean AUROC, seed 20260512)")
for K in [5, 10, 15, 25, 50]:
    log(f"  K={K:2d}: {recovery_mean(K, 20260512):.3f}")
log("\n## S5 Negative-pool sensitivity (K=15, 5 seeds)")
vals = [recovery_mean(15, s) for s in range(5)]
log(f"  means: {[round(v,3) for v in vals]}  mean={np.mean(vals):.3f} sd={np.std(vals):.4f}")

log("\n## S6 Classifier specification (30 one-vs-rest L2 logistic, C=1.0, balanced)")
nz = []
for f in fams:
    y = np.array([1 if fam[i] == f else 0 for i in range(n)])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(dense, y)
    nz.append(int((np.abs(clf.coef_[0]) > 1e-6).sum()))
log(f"  latent dim 6144; mean nonzero coefficients per family = {np.mean(nz):.0f} "
    f"(min {min(nz)}, max {max(nz)}); L2 penalty does not zero coefficients, so effective "
    f"sparsity is low relative to the 15-feature contrastive signature.")
open("../SSCR Submission/07_Analysis_Audit/si_recovery_results.md", "w").write("\n".join(out) + "\n")
print("WROTE si_recovery_results.md")
