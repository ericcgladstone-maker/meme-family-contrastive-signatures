#!/usr/bin/env python3
"""Freeze Eric's exported labels BEFORE unblinding (15b §6).

Validates the exported CSV against the presentation list in the coding package (the key is NOT read here):
every one of the 70 presented IDs exactly once, no unknown IDs, coder = E, package_id = check, every label one of the
31 reference-sheet families, 'none' or 'uncertain'. Then copies the file to analysis/human_check/labels/labels_E.csv
(real mode) and writes LABELS_FREEZE.json with its SHA-256. analyze_check.py refuses to run without a matching record.

Usage: python3 freeze_labels.py RESPONSES.csv [--package-dir DIR] [--out-dir DIR] [--record PATH]
"""
import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

import pandas as pd

import hc_common as C


def presented_ids(pkg_dir):
    page = (Path(pkg_dir) / "index.html").read_text()
    m = re.search(r"const PACKAGE = (\{.*?\});\n", page, re.S)
    return [it["id"] for it in json.loads(m.group(1))["items"]]


def validate(path, expected_ids):
    r = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"coder", "package_id", "item_id", "primary"}
    if not need <= set(r.columns):
        raise SystemExit(f"{Path(path).name}: missing column(s) {sorted(need - set(r.columns))}")
    if (r.coder.str.strip() != C.CODER).any():
        raise SystemExit(f"{Path(path).name}: coder must be {C.CODER!r} on every row")
    if (r.package_id.str.strip() != "check").any():
        raise SystemExit(f"{Path(path).name}: package_id must be 'check' on every row")
    if r.item_id.duplicated().any():
        raise SystemExit(f"{Path(path).name}: duplicate item_id rows")
    got, exp = set(r.item_id), set(expected_ids)
    if got - exp:
        raise SystemExit(f"{Path(path).name}: {len(got - exp)} unknown item_id(s)")
    if exp - got:
        raise SystemExit(f"{Path(path).name}: incomplete: {len(exp - got)} of {len(exp)} presented items have no label")
    bad = r[~r.primary.str.strip().isin(C.valid_labels())]
    if len(bad):
        raise SystemExit(f"{Path(path).name}: {len(bad)} invalid label(s), e.g. {bad.primary.iloc[0]!r}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("responses")
    ap.add_argument("--package-dir")  # default: the real package in the non-Drive folder
    ap.add_argument("--out-dir", default=str(C.HC / "labels"))
    ap.add_argument("--record", default=str(C.LABELS_RECORD))
    a = ap.parse_args()
    pkg = Path(a.package_dir) if a.package_dir else C.scratch_dir("human_check") / "package_E"
    rec_path = Path(a.record)
    if rec_path.exists():
        raise SystemExit(f"REFUSING: {rec_path.name} already exists (labels are frozen once)")
    validate(a.responses, presented_ids(pkg))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "labels_E.csv"
    shutil.copy(a.responses, dst)
    rec = {"labels_file": str(dst.relative_to(C.RR)) if C.RR in dst.parents else str(dst), "sha256": C.sha(dst),
           "n_rows": int(len(pd.read_csv(dst))),
           "frozen_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
           "note": "Frozen before the key was opened (15b §6). Any later change to the labels file is refused."}
    rec_path.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
