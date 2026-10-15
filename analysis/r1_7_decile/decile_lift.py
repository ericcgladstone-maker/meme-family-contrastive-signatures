#!/usr/bin/env python3
"""R1-7 anchor (b): top-decile hit-rate lift. Implements the spec FROZEN in 09 F-31 before any computation.

Frame: the 4,263 posts in analysis/g1_full_sample/out/scores_all.csv.
Top decile: raw_max_fit >= numpy.percentile(raw_max_fit, 90) (ties included).
Outcome: the paper's binary engagement outcome = `high_engagement` in q2_denom_outcomes.csv
  (build/scripts/fr_36_denominator_model.py: log_score >= 75th percentile within (subreddit, age_window);
   the submitted engine RUN_e1_m2_reconcile.run(infold=False) uses this column directly as the full-sample label).
Reported ONLY: top-decile rate, observed 4,263-frame reference rate, percentage-point difference, ratio.
AMENDED 09 F-34 (logged before rerun): the [0.24, 0.26] rate assertion is replaced by a provenance guard: the
outcome is the existing high_engagement column exactly as used in the models (thresholds set over all collected
listings, not recomputed), and its observed rate on the 4,263-post frame must equal 0.2252 (4 dp; the value recorded
in out/decile_lift_STOPPED.json). Nothing else changed.
No p-value, no other cutoffs, no search, no sensitivity branch. Scale interpretation only (L3 inference rests on R2).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
AN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AN))
import common as C  # noqa: E402

s = pd.read_csv(AN / "g1_full_sample/out/scores_all.csv")[["post_id", "raw_max_fit"]]
o = pd.read_csv(C.OUTCOMES, dtype={"post_id": str})[["post_id", "high_engagement"]]
d = s.merge(o, on="post_id", how="left", validate="one_to_one")
assert len(d) == 4263, len(d)
assert d.high_engagement.notna().all(), "posts without an outcome"
ref = float(d.high_engagement.mean())
assert round(ref, 4) == 0.2252, f"provenance guard failed: observed rate {ref:.4f} != 0.2252 (09 F-34)"
cut = float(np.percentile(d.raw_max_fit, 90))
top = d[d.raw_max_fit >= cut]
rate = float(top.high_engagement.mean())
res = dict(spec="09 F-31 (frozen before computation)", n_frame=int(len(d)), decile_cutoff_raw_max_fit=cut,
           n_top_decile=int(len(top)), top_decile_rate=rate, reference_rate=ref, spec_amendment="09 F-34 (provenance guard; logged before rerun)",
           diff_pp=100 * (rate - ref), ratio=rate / ref,
           outcome_provenance="high_engagement: build/scripts/fr_36_denominator_model.py L139-145; "
                              "used directly by RUN_e1_m2_reconcile.run(infold=False)")
out = Path(__file__).resolve().parent / "out" / "decile_lift.json"
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
