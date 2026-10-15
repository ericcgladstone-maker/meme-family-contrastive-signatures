#!/usr/bin/env python3
"""Verify that the frozen R1 resubmission package is unchanged since the R1 freeze (2026-10-01; `09` F-70).

Re-hashes every entry in r1_freeze_manifest.csv (paths relative to `SSCR Revise and Resubmit/`) and reports drift.
Run from anywhere:  python3 verify_r1_freeze.py      Exit code 0 = no drift; 1 = drift detected.
"""
import csv, hashlib, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RR = HERE.parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(HERE / "r1_freeze_manifest.csv")))
    bad = [r["path"] for r in rows if not (RR / r["path"]).exists() or sha(RR / r["path"]) != r["sha256"]]
    print(f"R1 freeze: {len(rows)} files checked")
    if bad:
        print("DRIFT:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("NO DRIFT")


if __name__ == "__main__":
    main()
