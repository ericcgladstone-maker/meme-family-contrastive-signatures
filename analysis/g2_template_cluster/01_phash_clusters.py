#!/usr/bin/env python3
"""G2 steps 1-4: per-post pHash for the 4,263 denominator posts and template clusters.

pHash: the project's surface_baselines.phash_vectors (imagehash.phash, hash_size=8, RGB), the
same function fr_36 used for deduplication (PHASH_DUP = 6).
Clusters:
  - AS PLANNED (e1_grouped_cv.py, 2026-07-13 plan): connected components at Hamming <= 10
  - sensitivities (pre-specified in 08): components at <= 6 and <= 8; greedy leader clustering at
    <= 12 (fr_25 template-proxy rule; non-transitive, cannot chain)
Diagnostics: n clusters, size distribution, singleton share, largest-component share, and the
Hamming diameter (max pairwise distance) inside the largest components (chaining check).
Output: out/phash.csv, out/clusters.csv, out/cluster_diagnostics.json
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from src.analysis import surface_baselines as SB  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def components(D, thr):
    n = len(D)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    ii, jj = np.where(np.triu(D <= thr, 1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def greedy_leader(D, thr):
    """fr_25 phash_template_proxy rule: assign to the first representative within thr."""
    reps, lab = [], np.full(len(D), -1)
    for i in range(len(D)):
        for r_idx, (rep, l) in enumerate(reps):
            if D[i, rep] <= thr:
                lab[i] = l
                break
        else:
            lab[i] = len(reps)
            reps.append((i, len(reps)))
    return lab


def diag(lab, D, n):
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    top = []
    for c in order[:5]:
        idx = np.where(lab == c)[0]
        diam = int(D[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0
        top.append(dict(size=int(sizes[c]), share=float(sizes[c] / n), hamming_diameter=diam))
    q = np.percentile(sizes, [50, 90, 99])
    return dict(n_clusters=int(len(sizes)), singleton_share_of_posts=float((sizes == 1).sum() / n),
                size_median=float(q[0]), size_p90=float(q[1]), size_p99=float(q[2]), size_max=int(sizes.max()),
                largest_share=float(sizes.max() / n), degenerate_for_5fold=bool(sizes.max() / n > 0.20),
                largest_components=top)


def main():
    OUT.mkdir(exist_ok=True)
    outc = [r for r in csv.DictReader(C.OUTCOMES.open(newline="")) if r["listing_group"] == "denominator"]
    mani = {r["post_id"]: r for r in csv.DictReader(C.MANIFEST.open(newline=""))}
    recs = [{"variant_id": r["post_id"], "image_path": mani[r["post_id"]]["local_path"]} for r in outc]
    ids, bits = SB.phash_vectors(recs)
    print(f"[phash] computed {len(ids)}/{len(recs)}", flush=True)
    B = bits.astype(np.int16)
    D = (B[:, None, :] != B[None, :, :]).sum(2).astype(np.int16)  # Hamming distance matrix
    n = len(ids)
    off = D[np.triu_indices(n, 1)]
    sanity = dict(min_pairwise_hamming=int(off.min()), n_pairs_le_6=int((off <= 6).sum()),
                  note="fr_36 dedup kept posts differing by > 6 bits from every earlier kept post")
    print(f"[sanity] {sanity}", flush=True)
    specs = {"cc10_as_planned": ("cc", 10), "cc6": ("cc", 6), "cc8": ("cc", 8), "greedy12": ("greedy", 12)}
    labs, diags = {}, {}
    for name, (kind, thr) in specs.items():
        lab = components(D, thr) if kind == "cc" else greedy_leader(D, thr)
        labs[name] = lab
        diags[name] = diag(lab, D, n)
        print(f"[{name}] {diags[name]['n_clusters']} clusters; largest share {diags[name]['largest_share']:.3f}; "
              f"singleton share {diags[name]['singleton_share_of_posts']:.3f}", flush=True)
    with open(OUT / "phash.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id", "phash_bits"])
        for pid, b in zip(ids, bits.astype(int)):
            w.writerow([pid, "".join(map(str, b))])
    with open(OUT / "clusters.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["post_id"] + list(specs))
        for i, pid in enumerate(ids):
            w.writerow([pid] + [int(labs[k][i]) for k in specs])
    C.write_json(OUT / "cluster_diagnostics.json", dict(n_posts=n, sanity=sanity, specs={k: list(v) for k, v in specs.items()},
                                                        diagnostics=diags))


if __name__ == "__main__":
    main()
