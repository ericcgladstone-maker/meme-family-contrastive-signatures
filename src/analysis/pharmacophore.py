"""Real common-feature extraction.

Implements the three-step pharmacophore construction from Build Spec §13:
1. Feature-specific background threshold at the 95th percentile of activation
   across a held-out background corpus of non-lineage variants.
2. Lineage-level common features by 80% subset-pharmacophore rule.
3. Pharmacophore = retained features plus lineage-typical loadings, capped at
   15 features by strongest loading.

The pharmacophore construction must:
- estimate background thresholds from non-lineage variants only,
- accept either full-family activations (retrospective scope) or pre-cutoff
  activations only (prospective scope),
- never see post-cutoff variants when scope == prospective_pre_cutoff_only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..curation.audit import sha256_bytes, sha256_json


@dataclass
class BackgroundThresholds:
    """Per-feature activation threshold derived from background corpus only."""
    n_background: int
    percentile: int
    thresholds: np.ndarray   # shape: (latent_dim,)
    background_signature: str  # hash of the input background corpus

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_background": self.n_background,
            "percentile": self.percentile,
            "background_signature": self.background_signature,
        }


def _activations_to_dense(act_records: list[dict[str, Any]], latent_dim: int) -> tuple[np.ndarray, list[str]]:
    ids = [r["variant_id"] for r in act_records]
    mat = np.zeros((len(act_records), latent_dim), dtype=np.float32)
    for i, r in enumerate(act_records):
        idxs = r["active_indices"]
        vals = r["active_values"]
        mat[i, idxs] = vals
    return mat, ids


def estimate_background_thresholds(
    background_activations: list[dict[str, Any]],
    latent_dim: int,
    percentile: int,
) -> BackgroundThresholds:
    """Background-only threshold estimation. Used by the
    `feature_thresholds_estimated_from_background_only` leakage test."""
    if not background_activations:
        raise ValueError("Background corpus is empty.")
    for r in background_activations:
        if not r.get("is_background_corpus_sample"):
            raise ValueError(
                f"estimate_background_thresholds received a non-background record: {r['variant_id']}"
            )
    mat, _ = _activations_to_dense(background_activations, latent_dim)
    thresholds = np.percentile(mat, percentile, axis=0).astype(np.float32)
    sig_payload = {
        "n": len(background_activations),
        "ids_prefix_hash": sha256_json([r["variant_id"] for r in background_activations]),
        "percentile": percentile,
    }
    return BackgroundThresholds(
        n_background=len(background_activations),
        percentile=percentile,
        thresholds=thresholds,
        background_signature=sha256_json(sig_payload),
    )


def _active_mask(rec: dict[str, Any], thresholds: np.ndarray, latent_dim: int) -> np.ndarray:
    mask = np.zeros(latent_dim, dtype=bool)
    for idx, val in zip(rec["active_indices"], rec["active_values"]):
        if val > thresholds[idx]:
            mask[idx] = True
    return mask


def construct_pharmacophore(
    family_id: str,
    lineage_activations: list[dict[str, Any]],
    thresholds: BackgroundThresholds,
    p_subset: float,
    max_features: int,
    latent_dim: int,
    scope: str,
    cutoff_timestamp: str | None,
    config_version: str,
) -> dict[str, Any]:
    """Three-step common-feature extraction with the canonical defaults."""
    if scope == "prospective_pre_cutoff_only":
        for r in lineage_activations:
            if not r.get("pre_cutoff", True):
                raise ValueError(
                    f"Prospective pharmacophore for {family_id} received a post-cutoff record: {r['variant_id']}"
                )
    n = len(lineage_activations)
    if n == 0:
        raise ValueError(f"Lineage {family_id} has no activations.")
    # Compute per-feature presence count using background thresholds.
    presence = np.zeros(latent_dim, dtype=np.int32)
    activations = np.zeros((n, latent_dim), dtype=np.float32)
    for i, rec in enumerate(lineage_activations):
        mask = _active_mask(rec, thresholds.thresholds, latent_dim)
        presence[mask] += 1
        for idx, val in zip(rec["active_indices"], rec["active_values"]):
            activations[i, idx] = val
    presence_share = presence / float(n)
    retained_mask = presence_share >= p_subset
    candidate_indices = np.where(retained_mask)[0]
    if candidate_indices.size == 0:
        # Fall back to top-k by mean activation across the lineage, which can
        # happen on tiny toy fixtures.
        mean_act = activations.mean(axis=0)
        candidate_indices = np.argsort(-mean_act)[:max_features]
    # Compute lineage-typical loadings (mean activation among active members
    # for each retained feature).
    loadings: list[float] = []
    for idx in candidate_indices:
        active_members = activations[:, idx] > thresholds.thresholds[idx]
        if active_members.sum() == 0:
            loadings.append(float(activations[:, idx].mean()))
        else:
            loadings.append(float(activations[active_members, idx].mean()))
    pairs = sorted(
        zip(candidate_indices.tolist(), loadings),
        key=lambda kv: kv[1],
        reverse=True,
    )[:max_features]
    retained_idx = [int(p[0]) for p in pairs]
    retained_loadings = [float(p[1]) for p in pairs]
    return {
        "pharmacophore_id": f"{family_id}__{scope}",
        "family_id": family_id,
        "scope": scope,
        "retained_feature_indices": retained_idx,
        "lineage_typical_loadings": retained_loadings,
        "p_subset": p_subset,
        "tau_active_percentile": thresholds.percentile,
        "n_background": thresholds.n_background,
        "background_corpus_size": thresholds.n_background,
        "background_threshold_sha256": sha256_bytes(thresholds.thresholds.tobytes()),
        "pre_cutoff_only": scope == "prospective_pre_cutoff_only",
        "cutoff_timestamp": cutoff_timestamp,
        "y_randomization_chance_distribution": None,
        "applicability_domain_centroid_sha256": sha256_bytes(activations.mean(axis=0).tobytes()),
        "computed_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_version": config_version,
    }
