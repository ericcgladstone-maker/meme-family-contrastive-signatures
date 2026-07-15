"""Scoring procedures.

candidate_to_pharmacophore: normalized capped weighted dot product, bounded in [0, 1].
lineage_to_lineage: composite Jaccard-cosine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def _activation_at_index(rec: dict[str, Any], idx: int) -> float:
    for j, k in enumerate(rec["active_indices"]):
        if k == idx:
            return float(rec["active_values"][j])
    return 0.0


def candidate_to_pharmacophore_score(
    pharmacophore: dict[str, Any],
    candidate_activation: dict[str, Any],
) -> float:
    retained = pharmacophore["retained_feature_indices"]
    loadings = pharmacophore["lineage_typical_loadings"]
    if not retained:
        return 0.0
    num = 0.0
    denom = sum(loadings) or 1.0
    for idx, w in zip(retained, loadings):
        act = _activation_at_index(candidate_activation, idx)
        # Capped: do not reward overactivation beyond the lineage-typical loading.
        num += w * min(act, w)
    return float(num / denom)


def score_candidates(
    pharmacophore: dict[str, Any],
    candidate_activations: list[dict[str, Any]],
    pool_id: str,
    true_positive_ids: set[str],
    scoring_rule: str = "normalized_capped_weighted_dot_product",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in candidate_activations:
        s = candidate_to_pharmacophore_score(pharmacophore, c)
        out.append({
            "pharmacophore_id": pharmacophore["pharmacophore_id"],
            "candidate_variant_id": c["variant_id"],
            "scoring_rule": scoring_rule,
            "score": float(max(0.0, min(1.0, s))),
            "is_true_positive": c["variant_id"] in true_positive_ids,
            "pool_id": pool_id,
            "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        })
    return out


def lineage_to_lineage_distance(
    pharma_a: dict[str, Any],
    pharma_b: dict[str, Any],
) -> float:
    a_idx = set(pharma_a["retained_feature_indices"])
    b_idx = set(pharma_b["retained_feature_indices"])
    inter = a_idx & b_idx
    union = a_idx | b_idx
    jaccard = len(inter) / len(union) if union else 0.0
    if not inter:
        return 1.0
    # Cosine on loadings restricted to the intersection.
    a_load = {i: w for i, w in zip(pharma_a["retained_feature_indices"], pharma_a["lineage_typical_loadings"])}
    b_load = {i: w for i, w in zip(pharma_b["retained_feature_indices"], pharma_b["lineage_typical_loadings"])}
    va = np.array([a_load[i] for i in inter], dtype=np.float32)
    vb = np.array([b_load[i] for i in inter], dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    cos = float(np.dot(va, vb) / denom)
    # Distance is high when overlap or alignment is low.
    return float(1.0 - 0.5 * (jaccard + max(0.0, cos)))
