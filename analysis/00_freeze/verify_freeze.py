#!/usr/bin/env python3
"""Verify that submitted materials are unchanged since the R&R freeze (2026-09-29).

Re-hashes every entry in submission_freeze_manifest.csv and reports any drift.
Run from anywhere:  python3 verify_freeze.py
Exit code 0 = no drift; 1 = drift detected.
"""
import csv, hashlib, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]  # .../Paper PH4 and Memes Computational
IMG_DIR = "build/first_results/ph4_q2_horsepower/reddit_images"


def sha(p):
    s = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            s.update(b)
    return s.hexdigest()


def listing_digest(d):
    fs = sorted(os.listdir(d))
    return hashlib.sha256("\n".join(f"{f}\t{os.path.getsize(os.path.join(d, f))}" for f in fs).encode()).hexdigest()


def main():
    drift = 0
    os.chdir(PROJECT)
    for r in csv.DictReader(open(HERE / "submission_freeze_manifest.csv", newline="")):
        p = r["path"]
        if p.startswith(IMG_DIR + "/ [listing"):
            now = listing_digest(IMG_DIR)
        elif not os.path.exists(p):
            print(f"MISSING  {p}"); drift += 1; continue
        else:
            now = sha(p)
        if now != r["sha256"]:
            print(f"CHANGED  {p}"); drift += 1
    print(f"{'NO DRIFT' if drift == 0 else f'{drift} DRIFTED ENTRIES'}")
    sys.exit(1 if drift else 0)


if __name__ == "__main__":
    main()
