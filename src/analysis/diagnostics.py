"""First-results diagnostics over a variant x feature matrix.

These are provisional, exploratory measurements, not confirmatory hypothesis
tests. They operate on any L2-normalized representation (raw CLIP embeddings now,
SAE pharmacophore features later), so the same code answers "does the
representation do X" for both, enabling a direct baseline-vs-SAE comparison.

Diagnostics:
  1. within_family_recovery   (~H1) do same-family variants retrieve each other?
  2. family_discrimination    (~H1') are families separable?
  3. clustering_recovery      (~H1') does unsupervised clustering recover families?
  4. productivity_association (~H4) does pre-cutoff compactness track post-cutoff
                                    productivity?
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---- similarity ----------------------------------------------------------

def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Assumes rows are (approximately) L2-normalized; renormalizes defensively."""
    v = vectors.astype(np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.clip(norms, 1e-12, None)
    return v @ v.T


# ---- 1. within-family recovery ------------------------------------------

def within_family_recovery(ids, families, sim) -> dict[str, Any]:
    n = len(ids)
    fam = np.array(families)
    p_at_1, rr, has_mate = [], [], 0
    for i in range(n):
        same = np.where(fam == fam[i])[0]
        same = same[same != i]
        if same.size == 0:
            continue
        has_mate += 1
        order = np.argsort(-sim[i])
        order = order[order != i]
        ranks_of_same = np.where(np.isin(order, same))[0]  # 0-based positions
        first = ranks_of_same.min()
        p_at_1.append(1.0 if first == 0 else 0.0)
        rr.append(1.0 / (first + 1))
    # Random expectation for P@1: chance that a uniformly random other variant
    # shares the family, averaged over query variants with >=1 mate.
    fam_sizes = {f: int((fam == f).sum()) for f in set(families)}
    exp_p1 = float(np.mean([(fam_sizes[fam[i]] - 1) / (n - 1)
                            for i in range(n) if fam_sizes[fam[i]] > 1]))
    return {
        "n_queries": has_mate,
        "precision_at_1": round(float(np.mean(p_at_1)), 4) if p_at_1 else None,
        "precision_at_1_random_expectation": round(exp_p1, 4),
        "mean_reciprocal_rank": round(float(np.mean(rr)), 4) if rr else None,
    }


# ---- 2. family discrimination -------------------------------------------

def family_discrimination(ids, families, sim) -> dict[str, Any]:
    n = len(ids)
    fam = np.array(families)
    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if fam[i] == fam[j] else between).append(sim[i, j])
    within = np.array(within)
    between = np.array(between)
    # leave-one-out 1-NN family accuracy
    correct = 0
    for i in range(n):
        order = np.argsort(-sim[i])
        order = order[order != i]
        if fam[order[0]] == fam[i]:
            correct += 1
    # silhouette on cosine distance, family labels
    sil = None
    try:
        from sklearn.metrics import silhouette_score
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        # silhouette needs >=2 labels and each label >=2 members
        labels = fam
        if len(set(labels)) >= 2:
            sil = float(silhouette_score(dist, labels, metric="precomputed"))
    except Exception:
        sil = None
    return {
        "within_family_mean_cosine": round(float(within.mean()), 4) if within.size else None,
        "between_family_mean_cosine": round(float(between.mean()), 4) if between.size else None,
        "separation": round(float(within.mean() - between.mean()), 4) if within.size and between.size else None,
        "loo_1nn_family_accuracy": round(correct / n, 4),
        "silhouette_cosine": round(sil, 4) if sil is not None else None,
    }


# ---- 3. clustering recovery ---------------------------------------------

def clustering_recovery(ids, families, sim) -> dict[str, Any]:
    fam = np.array(families)
    n_families = len(set(families))
    out: dict[str, Any] = {"n_families": n_families}
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        cl = AgglomerativeClustering(n_clusters=n_families, metric="precomputed", linkage="average")
        pred = cl.fit_predict(dist)
        out["agglomerative_ari"] = round(float(adjusted_rand_score(fam, pred)), 4)
        out["agglomerative_nmi"] = round(float(normalized_mutual_info_score(fam, pred)), 4)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


# ---- 4. productivity association ----------------------------------------

def _mean_pairwise_cos(idxs, sim) -> float | None:
    if len(idxs) < 2:
        return None
    vals = [sim[a, b] for k, a in enumerate(idxs) for b in idxs[k + 1:]]
    return float(np.mean(vals)) if vals else None


def productivity_association(ids, families, splits, sim,
                             min_pre: int = 2, min_post: int = 1) -> dict[str, Any]:
    """Per family: pre-cutoff compactness (mean pairwise cosine of pre variants)
    vs post-cutoff productivity (post count) and surface diversity
    (1 - mean pairwise cosine of post variants)."""
    fam = np.array(families)
    spl = np.array(splits)
    rows = []
    for f in sorted(set(families)):
        idx = np.where(fam == f)[0]
        pre = [int(i) for i in idx if spl[i] == "pre"]
        post = [int(i) for i in idx if spl[i] == "post"]
        if len(pre) < min_pre or len(post) < min_post:
            continue
        compactness = _mean_pairwise_cos(pre, sim)
        post_div = _mean_pairwise_cos(post, sim)
        rows.append({
            "family_id": f,
            "n_pre": len(pre),
            "n_post": len(post),
            "pre_compactness": round(compactness, 4) if compactness is not None else None,
            "post_count": len(post),
            "post_surface_diversity": round(1.0 - post_div, 4) if post_div is not None else None,
        })
    corr = {}
    if len(rows) >= 4:
        from scipy.stats import spearmanr
        comp = [r["pre_compactness"] for r in rows]
        cnt = [r["post_count"] for r in rows]
        div = [r["post_surface_diversity"] if r["post_surface_diversity"] is not None else np.nan for r in rows]
        rho_c, p_c = spearmanr(comp, cnt)
        mask = ~np.isnan(div)
        if mask.sum() >= 4:
            rho_d, p_d = spearmanr(np.array(comp)[mask], np.array(div)[mask])
        else:
            rho_d, p_d = (float("nan"), float("nan"))
        corr = {
            "n_families": len(rows),
            "spearman_compactness_vs_post_count": round(float(rho_c), 4),
            "p_compactness_vs_post_count": round(float(p_c), 4),
            "spearman_compactness_vs_surface_diversity": (round(float(rho_d), 4) if not np.isnan(rho_d) else None),
            "p_compactness_vs_surface_diversity": (round(float(p_d), 4) if not np.isnan(p_d) else None),
        }
    return {"per_family": rows, "correlations": corr,
            "note": "Exploratory. n is small and KYM upload timestamps are noisy; "
                    "treat directionally, not as an H4 test."}


def per_family_within_p_at_1(ids, families, sim) -> list[dict[str, Any]]:
    """Per-family within-family P@1, to see which families drive the aggregate."""
    n = len(ids)
    fam = np.array(families)
    hits: dict[str, list[float]] = {}
    for i in range(n):
        same = np.where(fam == fam[i])[0]
        same = same[same != i]
        if same.size == 0:
            continue
        order = np.argsort(-sim[i])
        order = order[order != i]
        hits.setdefault(fam[i], []).append(1.0 if order[0] in same else 0.0)
    rows = [{"family_id": f, "n": len(v), "within_p_at_1": round(float(np.mean(v)), 4)}
            for f, v in hits.items()]
    return sorted(rows, key=lambda r: -r["within_p_at_1"])


def per_family_recovery(ids, families, sim) -> dict[str, dict]:
    """Per-family within-family P@1 and MRR for a similarity matrix.
    Returns {family_id: {n, p_at_1, mrr}}."""
    n = len(ids)
    fam = np.array(families)
    acc: dict[str, dict] = {}
    for i in range(n):
        same = np.where(fam == fam[i])[0]
        same = same[same != i]
        if same.size == 0:
            continue
        order = np.argsort(-sim[i])
        order = order[order != i]
        first = np.where(np.isin(order, same))[0].min()
        d = acc.setdefault(fam[i], {"p1": [], "rr": []})
        d["p1"].append(1.0 if first == 0 else 0.0)
        d["rr"].append(1.0 / (first + 1))
    return {f: {"n": len(v["p1"]),
                "p_at_1": round(float(np.mean(v["p1"])), 4),
                "mrr": round(float(np.mean(v["rr"])), 4)}
            for f, v in acc.items()}


def diagnostics_on_sim(ids, families, splits, sim) -> dict[str, Any]:
    """All sim-based diagnostics for a precomputed similarity matrix."""
    return {
        "n_variants": len(ids),
        "n_families": len(set(families)),
        "within_family_recovery": within_family_recovery(ids, families, sim),
        "family_discrimination": family_discrimination(ids, families, sim),
        "clustering_recovery": clustering_recovery(ids, families, sim),
        "productivity_association": productivity_association(ids, families, splits, sim),
    }


def run_all(ids, families, splits, vectors) -> dict[str, Any]:
    sim = cosine_matrix(vectors)
    return {
        "n_variants": len(ids),
        "n_families": len(set(families)),
        "within_family_recovery": within_family_recovery(ids, families, sim),
        "family_discrimination": family_discrimination(ids, families, sim),
        "clustering_recovery": clustering_recovery(ids, families, sim),
        "productivity_association": productivity_association(ids, families, splits, sim),
    }
