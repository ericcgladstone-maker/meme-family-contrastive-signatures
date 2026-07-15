"""Leakage-safe pharmacophore evaluation.

`lovo_family_recovery` is the honest version of the first-run pharmacophore
recovery: when scoring a query variant against its OWN family's pharmacophore,
that pharmacophore is rebuilt with the query held out (leave-one-variant-out).
Other families' pharmacophores never contained the query, so they are used in
full. This removes the leakage flagged in the first run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pharmacophore import construct_pharmacophore
from .score import candidate_to_pharmacophore_score


def build_full_pharmacophores(records_by_family, thresholds, p_subset, max_features,
                              latent_dim, min_family=3, config_version="stress"):
    pharms = {}
    for fam, recs in records_by_family.items():
        if len(recs) < min_family:
            continue
        pharms[fam] = construct_pharmacophore(
            fam, recs, thresholds, p_subset, max_features, latent_dim,
            scope="retrospective_full", cutoff_timestamp=None, config_version=config_version)
    return pharms


def lovo_family_recovery(records, families, thresholds, p_subset, max_features,
                         latent_dim, min_family=3, config_version="stress") -> dict[str, Any]:
    by_family: dict[str, list] = {}
    for r, f in zip(records, families):
        by_family.setdefault(f, []).append(r)
    full = build_full_pharmacophores(by_family, thresholds, p_subset, max_features,
                                     latent_dim, min_family, config_version)
    eligible = set(full)  # families large enough to have a pharmacophore

    per_family_hits: dict[str, list[float]] = {}
    for fam, recs in by_family.items():
        if fam not in eligible:
            continue
        for i, q in enumerate(recs):
            held = recs[:i] + recs[i + 1:]
            if len(held) < 1:
                continue
            own = construct_pharmacophore(fam, held, thresholds, p_subset, max_features,
                                          latent_dim, scope="retrospective_full",
                                          cutoff_timestamp=None, config_version=config_version)
            scores = {f: candidate_to_pharmacophore_score(full[f], q) for f in eligible if f != fam}
            scores[fam] = candidate_to_pharmacophore_score(own, q)
            best = max(scores, key=scores.get)
            per_family_hits.setdefault(fam, []).append(1.0 if best == fam else 0.0)

    all_hits = [h for v in per_family_hits.values() for h in v]
    per_family = [{"family_id": f, "n": len(v), "lovo_p_at_1": round(float(np.mean(v)), 4)}
                  for f, v in per_family_hits.items()]
    return {
        "lovo_family_p_at_1": round(float(np.mean(all_hits)), 4) if all_hits else None,
        "n_scored": len(all_hits),
        "n_family_pharmacophores": len(eligible),
        "chance_p_at_1": round(1.0 / len(eligible), 4) if eligible else None,
        "per_family": sorted(per_family, key=lambda r: -r["lovo_p_at_1"]),
    }
