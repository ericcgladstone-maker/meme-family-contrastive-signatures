#!/usr/bin/env python3
"""Unblinded descriptive analysis of the author-coded human check (15b §8). Runs only after freeze_labels.py.

Per unique post (the 60 first presentations; repeat copies are used only for within-coder consistency):
  confirm       the label equals the model-nominated family (display name of raw_argmax)
  other_family  a listed family other than the nomination
  none          'none'
  uncertain     'uncertain'
  any_family  = confirm + other_family
Reported (all unweighted and descriptive; no population calibration, no thresholds, no tests):
  1. per bin B1..B6 (n = 10 each): counts and proportions of each category; mean raw score per category
  2. main comparison, lower-score B1-B2 vs higher-score B5-B6 (n = 20 each): proportions with Wilson 95% intervals and
     the high-minus-low difference with a Newcombe (hybrid score) 95% interval
  3. repeat agreement: identical labels on the two presentations of each repeated post (n pairs, Wilson 95% interval)
  4. nomination composition per bin (which families the model nominated), for reading 1-2

Usage: python3 analyze_check.py [--key-dir DIR] [--labels PATH] [--record PATH] [--out PATH]
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

import hc_common as C

CATS = ["confirm", "other_family", "none", "uncertain"]
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def newcombe(k1, n1, k2, n2):
    """95% interval for p1 - p2 (Newcombe 1998, method 10: hybrid score from Wilson limits)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return [d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)]


def category(label, nominated):
    if label == C.LABEL_NONE:
        return "none"
    if label == C.LABEL_UNCERTAIN:
        return "uncertain"
    return "confirm" if label == nominated else "other_family"


def run(key_dir, labels, record):
    rec = json.loads(Path(record).read_text()) if Path(record).exists() else None
    if rec is None:
        raise SystemExit("REFUSING: labels are not frozen (run freeze_labels.py first)")
    if C.sha(labels) != rec["sha256"]:
        raise SystemExit("REFUSING: labels file differs from the frozen labels (LABELS_FREEZE.json)")
    key = pd.read_csv(Path(key_dir) / "key.csv", keep_default_na=False)
    lab = pd.read_csv(labels, dtype=str, keep_default_na=False).set_index("item_id").primary.str.strip()
    if set(lab.index) != set(key.item_id):
        raise SystemExit("labels and key do not cover the same items")
    st = key[key.kind == "study"].copy()
    st["label"] = st.item_id.map(lab)
    st["nominated"] = st.raw_argmax.map(C.display_name)
    st["cat"] = [category(l, n) for l, n in zip(st.label, st.nominated)]
    out = {"n_unique": int(len(st)), "definitions": __doc__.split("Reported")[0].strip()}

    per_bin = {}
    for b in C.BINS:
        g = st[st.bin == b]
        cnt = {c: int((g.cat == c).sum()) for c in CATS}
        cnt["any_family"] = cnt["confirm"] + cnt["other_family"]
        per_bin[b] = {"n": int(len(g)), "counts": cnt,
                      "proportions": {c: (v / len(g) if len(g) else None) for c, v in cnt.items()},
                      "mean_raw_score_by_category": {c: float(g.raw_max_fit[g.cat == c].mean())
                                                     for c in CATS if (g.cat == c).any()},
                      "nominations": g.nominated.value_counts().to_dict()}
    out["per_bin"] = per_bin

    def pooled(bins):
        g = st[st.bin.isin(bins)]
        cnt = {c: int((g.cat == c).sum()) for c in CATS}
        cnt["any_family"] = cnt["confirm"] + cnt["other_family"]
        return g, cnt
    gl, cl = pooled(C.LOW)
    gh, ch = pooled(C.HIGH)
    comp = {}
    for c in CATS + ["any_family"]:
        comp[c] = {"low_B1_B2": {"k": cl[c], "n": int(len(gl)), "p": cl[c] / len(gl), "wilson95": wilson(cl[c], len(gl))},
                   "high_B5_B6": {"k": ch[c], "n": int(len(gh)), "p": ch[c] / len(gh), "wilson95": wilson(ch[c], len(gh))},
                   "high_minus_low": ch[c] / len(gh) - cl[c] / len(gl),
                   "newcombe95": newcombe(ch[c], len(gh), cl[c], len(gl))}
    out["main_comparison"] = comp

    rp = key[key.kind == "repeat"]
    agree = [lab[r.repeat_of] == lab[r.item_id] for r in rp.itertuples()]
    k, n = int(sum(agree)), int(len(agree))
    out["repeat_agreement"] = {"n_pairs": n, "agree": k, "p": (k / n if n else None), "wilson95": wilson(k, n),
                               "definition": "identical label on both presentations of a repeated post"}
    out["labels_freeze"] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-dir", default=str(C.HC / "draw"))
    ap.add_argument("--labels", default=str(C.HC / "labels" / "labels_E.csv"))
    ap.add_argument("--record", default=str(C.LABELS_RECORD))
    ap.add_argument("--out", default=str(C.HC / "out" / "hc_results.json"))
    a = ap.parse_args()
    res = run(a.key_dir, a.labels, a.record)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
