#!/usr/bin/env python3
"""Figure input (audited copy, no computation): the submitted family-signature export used for Figure 2(b) and
Figure S7. Reads the read-only submitted file `SSCR Submission/_Working/07_Analysis_Audit/interpretability_signatures.csv`
(the 15 retained (feature index, weight) pairs per development family, with top member exemplars and the
highest-scoring nonmember variants under each signature), records its SHA-256, validates the structure
(15 distinct indices, positive weights in non-increasing order), and writes it unchanged as JSON.
Output: analysis/figure_inputs/out/signatures_submitted_export.json
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "SSCR Submission/_Working/07_Analysis_Audit/interpretability_signatures.csv"


def main():
    raw = SRC.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    out = {"source": str(SRC.relative_to(HERE.parents[2])), "source_sha256": hashlib.sha256(raw).hexdigest(),
           "families": {}}
    for r in rows:
        idx = [int(x) for x in r["top_feature_indices"].split("|")]
        w = [float(x) for x in r["top_feature_weights"].split("|")]
        assert len(idx) == 15 == len(set(idx)) == len(w), r["family"]
        assert all(x > 0 for x in w) and all(a >= b for a, b in zip(w, w[1:])), r["family"]
        out["families"][r["family"]] = dict(
            n_members=int(r["n_members"]), feature_indices=idx, weights=w,
            top_member_exemplars=r["top_member_exemplars"].split("|"),
            top_nonmember_exemplars=r["top_nonmember_exemplars"].split("|"),
            top_nonmember_fit=[float(x) for x in r["top_nonmember_fit"].split("|")])
    (HERE / "out" / "signatures_submitted_export.json").write_text(json.dumps(out, indent=1))
    print(len(out["families"]), "families;", out["source_sha256"][:12])


if __name__ == "__main__":
    main()
