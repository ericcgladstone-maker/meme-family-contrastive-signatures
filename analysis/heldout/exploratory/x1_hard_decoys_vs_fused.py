#!/usr/bin/env python3
"""EXPLORATORY (not pre-specified in 14; run 2026-09-30 after the held-out results were read).

Question: is the hard-decoy advantage (L2, pre-specified vs the OpenAI CLIP centroid) also present against the
fused SigLIP2 + DINOv2 centroid? Uses the existing frozen outputs only (ho_primary.csv, ho_foldlocal.csv);
paired family differences with the same bootstrap as summarize_heldout.paired (5,000; seed 0).
Result may be reported only as exploratory; it cannot validate a claim.
Output: heldout/out/exploratory_x1_decoys_vs_fused.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sh", HO / "summarize_heldout.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)
z = np.load(HO / "out/expanded_features.npz", allow_pickle=True)
kset = dict(zip([int(x) for x in z["fam_k_vals"]], list(z["fam_set"])))
p = pd.read_csv(HO / "out/ho_primary.csv")
p["set"] = p.k.map(kset)
fl = pd.read_csv(HO / "out/ho_foldlocal.csv")
fl["set"] = fl.k.map(kset)
res = {"status": "EXPLORATORY - not pre-specified; report only as exploratory"}
for s in ["H", "C", "DEV"]:
    g, f = p[p.set == s], fl[fl.set == s]
    res[s] = dict(
        laion_contrastive_minus_fused=sh.paired(g.contrastive__DL_local, g.fused_centroid__DL_local),
        laion_contrastive_minus_fused_foldlocal=sh.paired(f.contrastive__DL_foldlocal, f.fused_centroid__DL_foldlocal),
        openai_contrastive_minus_fused=sh.paired(g.contrastive__DO_local, g.fused_centroid__DO_local),
        phash_contrastive_minus_fused=sh.paired(g.contrastive__DP_local, g.fused_centroid__DP_local))
(HO / "out/exploratory_x1_decoys_vs_fused.json").write_text(json.dumps(res, indent=2))
print(json.dumps({s: {k: [round(v["mean"], 4)] + [round(x, 4) for x in v["ci95"]] for k, v in res[s].items()} for s in ["H", "C", "DEV"]}, indent=1))
