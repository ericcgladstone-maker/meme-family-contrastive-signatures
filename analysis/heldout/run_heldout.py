#!/usr/bin/env python3
"""Held-out validation runs (14 §8-§9). POST-FREEZE ONLY (requires FREEZE_RECORD.json and the expanded bundle).

  primary   : clean protocol, seed 20260512, all target families (DEV + H + C) in the expanded universe
  sweep     : E_f redrawn with seeds 1-10 (no factorial)
  capped10  : every family subsampled without replacement to 10 members (rng [20260929, k]); clean protocol
  capped15  : same, capped at 15 (families with fewer members kept whole)
  foldlocal : fold-local decoy selection sensitivity (LAION / pHash / OpenAI), methods incl. fused centroid
Outputs: heldout/out/ho_primary.csv, ho_sweep.csv, ho_capped.csv, ho_foldlocal.csv
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("HELDOUT_OUT", str(HERE / "out")))
FIXTURE = OUT != HERE / "out"  # synthetic-fixture test directory (engineering only)
os.environ["ENGINE_BUNDLE"] = str(OUT / "expanded_features.npz")
os.environ["ENGINE_TEXTS"] = str(OUT / "expanded_texts.json")


def load_engine(bundle=None, texts=None):
    if bundle:
        os.environ["ENGINE_BUNDLE"] = str(bundle)
    if texts:
        os.environ["ENGINE_TEXTS"] = str(texts)
    spec = importlib.util.spec_from_file_location("engine", HERE / "engine.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def capped_bundle(cap):
    z = np.load(OUT / "expanded_features.npz", allow_pickle=True)
    fam = np.array(list(z["fam"]))
    kmap = dict(zip(list(z["fam_k_keys"]), [int(x) for x in z["fam_k_vals"]]))
    keep = []
    for f in dict.fromkeys(fam):
        idx = np.where(fam == f)[0]
        if len(idx) > cap:
            idx = np.sort(np.random.default_rng([20260929, kmap[f]]).choice(idx, cap, replace=False))
        keep.extend(idx.tolist())
    keep = np.array(sorted(keep))
    arrs = {k: (z[k][keep] if (z[k].ndim >= 1 and len(z[k]) == len(fam)) else z[k]) for k in z.files}
    np.savez_compressed(OUT / f"expanded_features_capped{cap}.npz", **arrs)
    texts = json.loads((OUT / "expanded_texts.json").read_text())
    (OUT / f"expanded_texts_capped{cap}.json").write_text(json.dumps([texts[i] for i in keep]))


def foldlocal(eng):
    from sklearn.linear_model import LogisticRegression
    C = sys.modules.get("common") or __import__("common")

    def fam_run(k, f, idxs):
        F = eng.data()
        rng = np.random.default_rng([eng.SEED, k])
        nonfam = [i for i in range(F["n"]) if F["fam"][i] != f]
        Eneg = np.array(sorted(rng.choice(nonfam, size=min(eng.NSAMP, len(nonfam)), replace=False)))
        dec = {q: {nm: sorted(nonfam, key=lambda i: -max(M[i, j] for j in [t for t in idxs if t != q]))[:12]
                   for nm, M in [("DL", F["clipL"]), ("DP", F["phash"]), ("DO", F["clipO"])]} for q in idxs}
        union = set().union(*[set(v) for d in dec.values() for v in d.values()])
        B = np.array([i for i in nonfam if i not in (set(Eneg.tolist()) | union)])
        loc = {m: {s: [] for s in ["DL", "DP", "DO"]} for m in ["contrastive", "clip_centroid", "fused_centroid", "logistic"]}
        for q in idxs:
            tr = [i for i in idxs if i != q]
            top, w = C.contrastive_signature(F["dense"], tr, B)
            cc = F["vn"][tr].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
            V = F["fused"]["fused"]; fc = V[tr].mean(0); fc /= (np.linalg.norm(fc) + 1e-12)
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(
                np.vstack([F["dense"][tr], F["dense"][B]]), np.r_[np.ones(len(tr)), np.zeros(len(B))])
            for s in ["DL", "DP", "DO"]:
                cands = np.array([q] + dec[q][s])
                sc = {"contrastive": C.sig_scores(F["dense"][cands], top, w), "clip_centroid": F["vn"][cands] @ cc,
                      "fused_centroid": V[cands] @ fc, "logistic": clf.decision_function(F["dense"][cands])}
                for m in sc:
                    loc[m][s].append(eng.rank_local(sc[m][0], sc[m][1:]))
        out = {"family": f, "k": k, "n": len(idxs)}
        for m in loc:
            for s in loc[m]:
                out[f"{m}__{s}_foldlocal"] = float(np.mean(loc[m][s]))
        return out
    rows = Parallel(n_jobs=11)(delayed(fam_run)(k, f, idxs) for (k, f, idxs) in eng.fams_eval())
    return pd.DataFrame(rows)


def main():
    if not FIXTURE and not (HERE / "FREEZE_RECORD.json").exists():
        raise SystemExit("REFUSING: not frozen")
    if FIXTURE and "fixture" not in OUT.name:
        raise SystemExit("REFUSING: HELDOUT_OUT override must be a *fixture* directory")
    mode = sys.argv[1]
    if mode == "primary":
        eng = load_engine()
        eng.run_clean([eng.SEED]).to_csv(OUT / "ho_primary.csv", index=False)
    elif mode == "sweep":
        eng = load_engine()
        eng.run_clean(list(range(1, 11)), with_factorial=False).to_csv(OUT / "ho_sweep.csv", index=False)
    elif mode in ("capped10", "capped15"):
        cap = int(mode[-2:])
        capped_bundle(cap)
        eng = load_engine(OUT / f"expanded_features_capped{cap}.npz", OUT / f"expanded_texts_capped{cap}.json")
        eng.run_clean([eng.SEED], with_factorial=False).to_csv(OUT / f"ho_capped{cap}.csv", index=False)
    elif mode == "foldlocal":
        eng = load_engine()
        foldlocal(eng).to_csv(OUT / "ho_foldlocal.csv", index=False)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    main()
