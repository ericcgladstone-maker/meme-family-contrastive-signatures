#!/usr/bin/env python3
"""Audit sensitivity A-5: fold-local decoy selection for RQ2.

Same clean protocol as g3_recovery/02_engine.py (seed 20260512, per-family RNG, same E_f), except
that in each LOVO fold the decoy sets are re-selected by max similarity to the n-1 TRAINING members
(LAION-CLIP, pHash, OpenAI-CLIP; 12 each). The training background B_f excludes E_f and the UNION of
all fold-level decoy sets, so it is fixed across folds and disjoint from every evaluation item.
Methods: contrastive, CLIP centroid, recurrence, sparse centroid, logistic (fold-local ranking).
Output: out/decoys_foldlocal.csv, out/decoys_foldlocal.json
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

spec = importlib.util.spec_from_file_location("eng", HERE.parent / "g3_recovery" / "02_engine.py")
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)
OUT = HERE / "out"
METH = ["contrastive", "clip_centroid", "recurrence", "sparse_centroid", "logistic"]


def fam(k, f, idxs, seed=eng.SEED):
    from sklearn.linear_model import LogisticRegression
    F = eng.data()
    rng = np.random.default_rng([seed, k])
    nonfam = [i for i in range(F["n"]) if F["fam"][i] != f]
    Eneg = np.array(sorted(rng.choice(nonfam, size=min(eng.NSAMP, len(nonfam)), replace=False)))
    dec = {}
    for q in idxs:
        tr = [i for i in idxs if i != q]
        dec[q] = {nm: sorted(nonfam, key=lambda i: -max(M[i, j] for j in tr))[:12]
                  for nm, M in [("DL", F["clipL"]), ("DP", F["phash"]), ("DO", F["clipO"])]}
    union = set().union(*[set(v) for d in dec.values() for v in d.values()])
    B = np.array([i for i in nonfam if i not in (set(Eneg.tolist()) | union)])
    thrB = eng.thresholds_from(B.tolist())
    loc = {m: {s: [] for s in ["DL", "DP", "DO"]} for m in METH}
    for q in idxs:
        tr = [i for i in idxs if i != q]
        for s in ["DL", "DP", "DO"]:
            cands = np.array([q] + dec[q][s])
            sc = {}
            top, w = C.contrastive_signature(F["dense"], tr, B)
            sc["contrastive"] = C.sig_scores(F["dense"][cands], top, w)
            cc = F["vn"][tr].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
            sc["clip_centroid"] = F["vn"][cands] @ cc
            ridx, rload, _ = eng.recurrence_vec(f, tr, thrB)
            sc["recurrence"] = eng.recurrence_score(ridx, rload, cands)
            scn = F["dn"][tr].mean(0); scn /= (np.linalg.norm(scn) + 1e-12)
            sc["sparse_centroid"] = F["dn"][cands] @ scn
            if s == "DL":  # fit the classifier once per fold, reuse for all decoy sets
                X = np.vstack([F["dense"][tr], F["dense"][B]])
                y = np.r_[np.ones(len(tr)), np.zeros(len(B))]
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
            sc["logistic"] = clf.decision_function(F["dense"][cands])
            for m in METH:
                loc[m][s].append(eng.rank_local(sc[m][0], sc[m][1:]))
    out = {"family": f.replace("pool_", ""), "n": len(idxs), "n_background": len(B), "n_union_decoys": len(union)}
    for m in METH:
        for s in ["DL", "DP", "DO"]:
            out[f"{m}__{s}_foldlocal"] = float(np.mean(loc[m][s]))
    return out


def paired(a, b, seed=0, nboot=5000):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nboot)])
    return dict(mean=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_pos=int((d > 0).sum()), n=len(d))


def main():
    OUT.mkdir(exist_ok=True)
    rows = Parallel(n_jobs=11)(delayed(fam)(k, f, idxs) for (k, f, idxs) in eng.fams_eval())
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decoys_foldlocal.csv", index=False)
    fl = pd.read_csv(HERE.parent / "g3_recovery" / "out" / "clean_primary.csv")
    res = {"design": dict(mean_background=float(df.n_background.mean()), min_background=int(df.n_background.min()),
                          mean_union_decoys=float(df.n_union_decoys.mean()))}
    for s, lab in [("DL", "laion_encoder_selected"), ("DP", "phash_selected"), ("DO", "openai_dependent_reference")]:
        res[lab] = dict(foldlocal_means={m: float(df[f"{m}__{s}_foldlocal"].mean()) for m in METH},
                        familylevel_means={m: float(fl[f"{m}__{s}_local"].mean()) for m in METH},
                        foldlocal_contrastive_minus_clip=paired(df[f"contrastive__{s}_foldlocal"], df[f"clip_centroid__{s}_foldlocal"]))
        print(lab, {m: round(v, 3) for m, v in res[lab]["foldlocal_means"].items()},
              "c-clip", round(res[lab]["foldlocal_contrastive_minus_clip"]["mean"], 4),
              [round(x, 4) for x in res[lab]["foldlocal_contrastive_minus_clip"]["ci95"]], flush=True)
    C.write_json(OUT / "decoys_foldlocal.json", res)


if __name__ == "__main__":
    main()
