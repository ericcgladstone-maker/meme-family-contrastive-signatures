#!/usr/bin/env python3
"""E4: OCR functional reproducibility check on 50 DEV images (criterion fixed in 14 v0.3 before running).

Selection: default_rng(20260929).choice(sorted DEV variant ids, 50, replace=False)
Call: ocrmac.OCR(path).recognize(); text = " ".join(fragment text).strip()   (as build/scripts/fr_08_ocr.py)
Compare to stored build/first_results/ocr_text.csv: lower-case, collapse whitespace; difflib ratio
(empty vs empty = 1). PASS iff median >= 0.90 AND >= 80% of images >= 0.80.
Output: out/e4_ocr_check.json
"""
import csv
import difflib
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import common as C  # noqa: E402


def norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def main():
    import ocrmac
    from ocrmac import ocrmac as om
    idx = {r["variant_id"]: r for r in csv.DictReader(C.CFD_INDEX.open(newline=""))}
    fams = {}
    for r in idx.values():
        fams.setdefault(r["family_id"], 0)
        fams[r["family_id"]] += 1
    dev_ids = sorted(v for v, r in idx.items() if fams[r["family_id"]] >= 5)
    pick = list(np.random.default_rng(20260929).choice(dev_ids, 50, replace=False))
    stored = {r["variant_id"]: r.get("ocr_text", "") for r in csv.DictReader(C.OCR_TEXT.open(newline=""))}
    rows = []
    for v in pick:
        try:
            res = om.OCR(idx[v]["image_path"]).recognize()
            text = " ".join(t[0] for t in res).strip() if res else ""
        except Exception as e:  # noqa: BLE001
            text = f"__error__{type(e).__name__}"
        a, b = norm(stored.get(v, "")), norm(text)
        ratio = 1.0 if (a == "" and b == "") else difflib.SequenceMatcher(None, a, b).ratio()
        rows.append(dict(variant_id=v, stored=a, new=b, ratio=ratio))
    r = np.array([x["ratio"] for x in rows])
    res = dict(n=len(rows), median=float(np.median(r)), share_ge_0p80=float((r >= 0.80).mean()),
               exact_match_share=float(np.mean([x["stored"] == x["new"] for x in rows])),
               passed=bool(np.median(r) >= 0.90 and (r >= 0.80).mean() >= 0.80),
               ocrmac_version=getattr(ocrmac, "__version__", "1.0.1 (pip)"), macos=platform.mac_ver()[0],
               rows=rows)
    (HERE / "out" / "e4_ocr_check.json").write_text(json.dumps(res, indent=2))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
