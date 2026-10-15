#!/usr/bin/env python3
"""PRE-LABEL PLANNING DIAGNOSTIC for 15 v0.3 (no labels exist; the frozen draw seed 20260930 and the frozen
replicate seeds 1..10,000 are NOT used). Simulates the 15 §3 capped sequential draw (cap 12, stepwise relaxation)
with planning seeds 900001.., estimates inclusion probabilities, and reports the Kish effective n that the
design implies for each weighted endpoint domain. Also reports the same for a no-cap (simple random within bin)
alternative, for comparison only.
Output: human_validation/prelabel/planning_neff.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AN = Path(__file__).resolve().parents[2]
s = pd.read_csv(AN / "g1_full_sample/out/scores_all.csv")
CUTS = [0.7422, 0.7962, 0.8323, 0.8682, 0.9110]
s["bin"] = np.searchsorted(CUTS, s.raw_max_fit.to_numpy(), side="right") + 1
fam = s.raw_argmax.to_numpy()
bins = s.bin.to_numpy()
N, R = len(s), 2000


def draw(rng, cap0=12, n=70, use_cap=True):
    chosen = []
    for b in range(1, 7):
        idx = np.where(bins == b)[0]
        perm = idx[rng.permutation(len(idx))]
        cap = cap0 if use_cap else 10 ** 9
        while True:
            cnt, pick = {}, []
            for i in perm:
                if cnt.get(fam[i], 0) < cap:
                    pick.append(i); cnt[fam[i]] = cnt.get(fam[i], 0) + 1
                    if len(pick) == n:
                        break
            if len(pick) == n:
                break
            cap += 1
        chosen.extend(pick)
    return np.array(chosen)


def kish(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


out = {"note": "planning diagnostic; planning seeds 900001+; not the frozen replicate procedure", "replicates": R}
for label, use_cap in [("capped_design_15v0.2", True), ("no_cap_alternative", False)]:
    freq = np.zeros(N)
    for r in range(R):
        freq[draw(np.random.default_rng(900001 + r), use_cap=use_cap)] += 1
    pi = freq / R
    smp = draw(np.random.default_rng(999999), use_cap=use_cap)  # one illustrative planning sample
    w = 1 / np.maximum(pi[smp], 1 / R)
    b = bins[smp]
    dom = {f"B{k}": b == k for k in range(1, 7)}
    dom["B5_B6"] = (b >= 5)
    dom["B1_B2"] = (b <= 2)
    dom["all_420"] = np.ones(len(smp), bool)
    res = {k: dict(n=int(m.sum()), kish_neff=round(kish(w[m]), 1)) for k, m in dom.items()}
    res["min_pi_sampled"] = float(pi[smp].min())
    res["top_family_share_in_sample"] = {f"B{k}": round(float(pd.Series(fam[smp][b == k]).value_counts(normalize=True).iloc[0]), 2)
                                         for k in range(1, 7)}
    out[label] = res
(Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
