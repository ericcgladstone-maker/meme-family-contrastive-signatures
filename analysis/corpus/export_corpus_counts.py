#!/usr/bin/env python3
"""Audited development-corpus counts (numerical audit, 09 F-66). Reads the deduplicated index and the dedup records;
computes counts and the family-size distribution only. Output: out/corpus_counts.json (with source hashes)."""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DD = HERE.parents[2] / "build" / "first_results" / "cross_family_dedup"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    idx = DD / "cross_family_corrected_index.csv"
    exact = DD / "cross_family_exact_duplicates.csv"
    d = pd.read_csv(idx)
    s = d.family_id.value_counts()
    rec = s[s >= 5]
    out = {"source_sha256": {"index": sha(idx), "exact_duplicates": sha(exact)},
           "n_variants": int(len(d)), "n_families": int(s.size),
           "n_recovery_families": int(rec.size), "n_recovery_variants": int(rec.sum()),
           "recovery_size_min": int(rec.min()), "recovery_size_max": int(rec.max()),
           "recovery_size_median": float(rec.median()),
           "excluded_family_size": int(s[s < 5].iloc[0]) if (s < 5).any() else None,
           "cross_family_exact_duplicate_sets": int(len(pd.read_csv(exact))),
           "family_sizes": {k.removeprefix("pool_"): int(v) for k, v in s.sort_index().items()}}
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "corpus_counts.json").write_text(json.dumps(out, indent=2) + "\n")
    print({k: out[k] for k in out if k not in ("family_sizes", "source_sha256")})


if __name__ == "__main__":
    main()
