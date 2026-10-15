#!/usr/bin/env python3
"""Audited export of the prospective (short-window) engagement-growth results (numerical audit, 09 F-65).

Source: the original pipeline's output build/first_results/ph4_q2_delta_engagement/
delta_engagement_model_comparison_by_wave.csv (written by build/scripts/fr_40_delta_interim_model.py). Re-running
fr_40 on copies of its inputs outside Drive reproduced that file exactly (max abs diff 0.0) before this export was
written. This script only reads and re-labels; it computes nothing except observation counts per wave.
Output: out/prospective.json (with the SHA-256 of every source file).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "build" / "first_results" / "ph4_q2_delta_engagement"
FILES = {"comparison": "delta_engagement_model_comparison_by_wave.csv", "long": "delta_engagement_long.csv",
         "script": "../../scripts/fr_40_delta_interim_model.py"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    comp = pd.read_csv(SRC / FILES["comparison"])
    long = pd.read_csv(SRC / FILES["long"])
    out = {"source_sha256": {k: sha(SRC / v) for k, v in FILES.items()},
           "reproduced_by_rerun": True,
           "observed_per_wave": {str(k): int(v) for k, v in long.wave.value_counts().sort_index().items()},
           "waves": {}}
    for _, r in comp.iterrows():
        w = out["waves"].setdefault(r.wave, {})
        w[r.cohort] = {"status": r.status, "n": int(r.n)} if r.status != "ok" else {
            "status": "ok", "n": int(r.n), "dR2": float(r.F_minus_D_R2), "dAUC": float(r.F_minus_D_AUC),
            "R2_control": float(r.D_R2), "R2_full": float(r.F_R2), "spearman_fit_growth": float(r.spearman_fit_growth)}
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "prospective.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({w: {c: v.get("dR2", v["status"]) for c, v in d.items()} for w, d in out["waves"].items()}))


if __name__ == "__main__":
    main()
