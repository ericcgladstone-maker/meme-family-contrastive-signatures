#!/usr/bin/env python3
"""E1 regression: the generalized held-out engine (heldout/engine.py), run on the 680-variant G3 bundle
with the clean protocol (seed 20260512), must reproduce g3_recovery/out/clean_primary.csv.
Acceptance: identical row set/order and identical columns; numeric columns equal within 1e-12 (absolute).
Output: out/e1_regression.json, out/e1_regression_clean_primary.csv
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("engine", HERE.parent / "engine.py")
eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eng)
REF = HERE.parents[1] / "g3_recovery" / "out" / "clean_primary.csv"
TOL = 1e-12


def main():
    new = eng.run_clean([eng.SEED])
    new.to_csv(HERE / "out" / "e1_regression_clean_primary.csv", index=False)
    ref = pd.read_csv(REF)
    new = pd.read_csv(HERE / "out" / "e1_regression_clean_primary.csv")  # same I/O path as the reference
    res = dict(rows_ref=len(ref), rows_new=len(new), same_columns=list(ref.columns) == list(new.columns),
               same_family_order=list(ref.family) == list(new.family))
    diffs = {}
    for c in ref.columns:
        if ref[c].dtype == object:
            diffs[c] = bool((ref[c] == new[c]).all())
        else:
            diffs[c] = float(np.nanmax(np.abs(ref[c].to_numpy(float) - new[c].to_numpy(float))))
    res["max_abs_diff_by_column"] = diffs
    num = [v for v in diffs.values() if not isinstance(v, bool)]
    res["max_abs_diff_overall"] = float(max(num))
    res["passed"] = bool(res["same_columns"] and res["same_family_order"] and res["max_abs_diff_overall"] <= TOL
                         and all(v for v in diffs.values() if isinstance(v, bool)))
    (HERE / "out" / "e1_regression.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k != "max_abs_diff_by_column"})


if __name__ == "__main__":
    main()
