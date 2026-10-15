#!/usr/bin/env python3
"""Synthetic fixture for end-to-end code testing of run_heldout.py / summarize_heldout.py (no real results).

Takes the 30 DEV families' structure (family sizes only) and fills every feature with ROW-PERMUTED or random
values, so no DEV signal is present and no held-out data is involved. Sets: 10 DEV, 14 H, 6 C (labels only).
Writes engineering/out/fixture/expanded_features.npz + expanded_texts.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G3 = HERE.parents[1] / "g3_recovery" / "out"
FX = HERE / "out" / "fixture"


def main():
    FX.mkdir(parents=True, exist_ok=True)
    z = np.load(G3 / "features.npz", allow_pickle=True)
    fam = np.array(list(z["fam"]))
    fams = [f for f in dict.fromkeys(fam) if (fam == f).sum() >= 5]
    rows = np.concatenate([np.where(fam == f)[0] for f in fams])
    rng = np.random.default_rng(12345)
    perm = rng.permutation(len(z["ids"]))[: len(rows)]  # features from random rows -> no family signal
    n = len(rows)
    sets = ["DEV"] * 10 + ["H"] * 14 + ["C"] * 6
    k_vals = list(range(10)) + [1000 + i for i in range(20)]
    np.savez_compressed(FX / "expanded_features.npz",
                        ids=np.array([f"fx{i}" for i in range(n)], object), fam=fam[rows],
                        dense=z["dense"][perm], clip_openai=z["clip_openai"][perm], clip_laion=z["clip_laion"][perm],
                        phash_bits=z["phash_bits"][perm], siglip2=rng.normal(size=(n, 1152)),
                        dinov2=rng.normal(size=(n, 1024)), fam_k_keys=np.array(fams, object),
                        fam_k_vals=np.array(k_vals), fam_target=np.ones(30, int), fam_set=np.array(sets, object))
    texts = json.loads((G3 / "ocr_texts.json").read_text())
    (FX / "expanded_texts.json").write_text(json.dumps([texts[i] for i in perm]))
    print("fixture", n, "rows,", len(fams), "families")


if __name__ == "__main__":
    main()
