#!/usr/bin/env python3
"""Pre-label check for 15 v0.3 (uses only scores_all.csv and clusters.csv; no labels exist).

1. Recompute the §3 bin cutpoints (20/40/60/80/95th percentiles of raw_max_fit) and compare them with the
   values recorded in 15 §3.
2. Tabulate cc10_as_planned clusters x score bins: does any multi-post cluster span more than one bin?
3. Bin sizes and per-bin raw_argmax concentration (to anticipate the family-cap relaxation rule).
Output: human_validation/prelabel/cluster_bin_check.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AN = Path(__file__).resolve().parents[2]
s = pd.read_csv(AN / "g1_full_sample/out/scores_all.csv")
c = pd.read_csv(AN / "g2_template_cluster/out/clusters.csv")
d = s.merge(c[["post_id", "cc10_as_planned"]], on="post_id", validate="one_to_one")
assert len(d) == len(s) == 4263
cuts_pct = [20, 40, 60, 80, 95]
cuts = np.percentile(d.raw_max_fit, cuts_pct)
recorded = [0.7422, 0.7962, 0.8323, 0.8682, 0.9110]
d["bin"] = np.searchsorted(np.round(cuts, 4), d.raw_max_fit.to_numpy(), side="right") + 1  # B1..B6 with 4-dp cutpoints
g = d.groupby("cc10_as_planned")
nb = g["bin"].nunique()
size = g.size()
multi = size[size > 1]
cross = nb[nb > 1]
res = dict(
    n_posts=int(len(d)),
    cutpoints_recomputed=[float(x) for x in cuts],
    cutpoints_recorded_15=recorded,
    cutpoints_match_4dp=bool(np.allclose(np.round(cuts, 4), recorded)),
    bin_sizes={f"B{b}": int((d.bin == b).sum()) for b in range(1, 7)},
    n_clusters=int(len(size)), n_multi_post_clusters=int(len(multi)),
    posts_in_multi_post_clusters=int(multi.sum()),
    n_clusters_crossing_bins=int(len(cross)),
    posts_in_crossing_clusters=int(size[cross.index].sum()),
    crossing_cluster_sizes=sorted([int(x) for x in size[cross.index]], reverse=True)[:20],
    crossing_cluster_nbins=sorted([int(x) for x in cross], reverse=True)[:20],
    largest_argmax_share_per_bin={f"B{b}": round(float(d[d.bin == b].raw_argmax.value_counts(normalize=True).iloc[0]), 3)
                                  for b in range(1, 7)},
    n_argmax_families_per_bin={f"B{b}": int(d[d.bin == b].raw_argmax.nunique()) for b in range(1, 7)},
    eligible_under_cap12_per_bin={f"B{b}": int(d[d.bin == b].raw_argmax.value_counts().clip(upper=12).sum())
                                  for b in range(1, 7)},
)
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
