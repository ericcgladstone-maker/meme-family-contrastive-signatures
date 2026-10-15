#!/usr/bin/env python3
"""Mechanical coder-qualification gate (15 v0.6 §6). Frozen with the other scripts.

Scores one coder's practice-round export (the coding form's CSV: item_id, primary, flag, ...) against the known
families of that round's 10 DEV practice images (key/key.csv rows with kind = practice_dev and practice_round = r).
  - correct = primary label equals the image's family display name AND flag = 0; flagged, "uncertain", "none",
    wrong-family and missing responses all count as incorrect;
  - PASS if correct >= 8 of 10;
  - round 1 fail -> FAIL_ROUND1_TAKE_ROUND2; round 2 fail -> REPLACE (no third attempt); round 2 is only allowed
    after a recorded round-1 fail (--prior).
Output: JSON verdict to stdout (and to --out if given).

Usage: python3 qualify.py --key DRAW/key/key.csv --responses coder_B_practice1.csv --round 1 [--prior verdict1.json]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

import hv_common as H

PASS_MIN, N_ITEMS = 8, 10


def score(key, responses, rnd):
    k = key[(key.kind == "practice_dev") & (pd.to_numeric(key.practice_round, errors="coerce") == rnd)]
    if len(k) != N_ITEMS:
        raise SystemExit(f"round {rnd}: expected {N_ITEMS} practice DEV items in the key, found {len(k)}")
    r = responses.copy()
    r["primary"] = r["primary"].fillna("").astype(str).str.strip()
    r["flag"] = pd.to_numeric(r.get("flag", 0), errors="coerce").fillna(0).astype(int)
    r = r.set_index("item_id")
    detail = []
    for iid, fam in zip(k.item_id, k.gold_family):
        want = H.display_name(fam)
        got = r.primary.get(iid, "")
        flag = int(r.flag.get(iid, 0)) if iid in r.index else 0
        ok = (iid in r.index) and flag == 0 and got == want
        detail.append(dict(item_id=iid, correct=bool(ok), answered=iid in r.index, flagged=bool(flag)))
    return sum(d["correct"] for d in detail), detail


def verdict(n_correct, rnd, prior=None):
    if rnd == 2 and (prior is None or prior.get("verdict") != "FAIL_ROUND1_TAKE_ROUND2"):
        raise SystemExit("round 2 is allowed only after a recorded round-1 fail (--prior)")
    if n_correct >= PASS_MIN:
        return "PASS"
    return "FAIL_ROUND1_TAKE_ROUND2" if rnd == 1 else "REPLACE"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--responses", required=True)
    ap.add_argument("--round", type=int, choices=[1, 2], required=True)
    ap.add_argument("--prior")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    key = pd.read_csv(a.key, dtype={"item_id": str})
    resp = pd.read_csv(a.responses, dtype={"item_id": str, "primary": str}, keep_default_na=False)
    prior = json.loads(Path(a.prior).read_text()) if a.prior else None
    n, detail = score(key, resp, a.round)
    res = dict(round=a.round, n_correct=int(n), n_items=N_ITEMS, threshold=PASS_MIN, verdict=verdict(n, a.round, prior),
               detail=detail)
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
    return res


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
