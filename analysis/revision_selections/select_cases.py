#!/usr/bin/env python3
"""(Moved from 19_Revision/select_worked_example.py in closure pass 2: it computes per-family contrasts, which
belong in an audited analysis script, not in the manuscript layer.)

Mechanical case selection for the revision (16 §7 items 5-6; decided 2026-09-30). No judgment is involved.

Input: analysis/g3_recovery/out/g3_family_table.csv (clean-protocol DEV recovery, 30 families; seed 20260512).
Rules (fixed before this script was run; recorded here and in 20_Closure_Ledger.md):
  WORKED EXAMPLE (R2-2 / F-3 / Figure 1 panel a): the family whose clean contrastive recovery AUROC
      (contrastive__E_local) is closest to the median over the 30 families (median = mean of the 15th and 16th
      ordered values); ties -> lower AUROC; further ties -> alphabetical family name.
  ILLUSTRATIVE CASES (R1-10), all DEV, clean G3 values:
      (i)   largest contrastive - CLIP-centroid LAION-decoy difference (contrastive__DL_local - clip_centroid__DL_local)
      (ii)  smallest (most negative) of that difference
      (iii) largest logistic - contrastive recovery gap (logistic__E_local - contrastive__E_local)
      ties -> alphabetical. (iv) top-ranked decoy per case: decoy identities are not in this table; resolved at the
      figure-build step from the G3 decoy lists by the same rank rule (no judgment).
Output: analysis/revision_selections/out/revision_selections.json (bound via 19_Revision/render_revision.py as source `rev`),
including the clean per-family matrix used for SI Table S7.
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
T = HERE.parent / "g3_recovery/out/g3_family_table.csv"


def pick(df, col, largest):
    d = df.sort_values([col, "family"], ascending=[not largest, True], kind="mergesort")
    return d.iloc[0]


def main():
    t = pd.read_csv(T)
    assert len(t) == 30
    v = t.sort_values(["contrastive__E_local", "family"], kind="mergesort").reset_index(drop=True)
    med = (v.contrastive__E_local[14] + v.contrastive__E_local[15]) / 2
    t["dist"] = (t.contrastive__E_local - med).abs()
    w = t.sort_values(["dist", "contrastive__E_local", "family"], kind="mergesort").iloc[0]
    t["laion_adv"] = t.contrastive__DL_local - t.clip_centroid__DL_local
    t["clf_gap"] = t.logistic__E_local - t.contrastive__E_local
    ci, cii, ciii = pick(t, "laion_adv", True), pick(t, "laion_adv", False), pick(t, "clf_gap", True)
    name = lambda s: s.replace("_", " ")
    cols = {"n": "n", "recurrence__E_local": "recurrence", "clip_centroid__E_local": "clip_centroid",
            "surf_composite__E_local": "surf_composite", "sparse_centroid__E_local": "sparse_centroid",
            "sparse_knn__E_local": "sparse_knn", "contrastive__E_local": "contrastive", "logistic__E_local": "logistic"}
    missing = [c for c in cols if c not in t.columns]
    fam = t.sort_values(["contrastive__E_local", "family"], ascending=[False, True], kind="mergesort")
    matrix = [dict(family=name(r.family), **{cols[c]: float(getattr(r, c)) for c in cols if c not in missing})
              for r in fam.itertuples()]
    out = dict(
        rule_worked_example="closest to median clean contrastive recovery AUROC (mean of 15th/16th of 30); ties lower, then alphabetical",
        median_contrastive=float(med),
        worked_example=dict(family=name(w.family), contrastive=float(w.contrastive__E_local), n=int(w.n)),
        illustrative=dict(
            largest_laion_advantage=dict(family=name(ci.family), diff=float(ci.laion_adv)),
            smallest_laion_advantage=dict(family=name(cii.family), diff=float(cii.laion_adv)),
            largest_classifier_gap=dict(family=name(ciii.family), diff=float(ciii.clf_gap))),
        matrix=matrix, matrix_missing_columns=missing)
    (HERE / "out" / "revision_selections.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "matrix"}, indent=1))


if __name__ == "__main__":
    main()
