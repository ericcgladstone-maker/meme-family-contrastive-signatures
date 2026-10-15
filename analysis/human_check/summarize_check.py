#!/usr/bin/env python3
"""Derived totals for reporting the 15b human check (read-only over the frozen analysis output).

Reads analysis/human_check/out/hc_results.json (never modified) and writes out/hc_summary.json with the totals the
manuscript, supplement and letter bind to: n unique posts, category totals summed over B1-B6, repeat agreement, and
the SHA-256 of the source results file. No new analysis.
"""
import hashlib
import json
from pathlib import Path

HC = Path(__file__).resolve().parent
SRC = HC / "out" / "hc_results.json"
r = json.loads(SRC.read_text())
cats = ["confirm", "other_family", "none", "uncertain", "any_family"]
tot = {c: sum(v["counts"][c] for v in r["per_bin"].values()) for c in cats}
assert sum(tot[c] for c in ["confirm", "other_family", "none", "uncertain"]) == r["n_unique"]
out = {"n_unique": r["n_unique"], "n_bins": len(r["per_bin"]), "n_per_bin": r["per_bin"]["B1"]["n"], "totals": tot,
       "repeat_pairs": r["repeat_agreement"]["n_pairs"], "repeat_agree": r["repeat_agreement"]["agree"],
       "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest()}
(HC / "out" / "hc_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
