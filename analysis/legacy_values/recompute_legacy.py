#!/usr/bin/env python3
"""Recompute the legacy (submitted, hand-typed) sensitivity values so no typed number survives into the revision
(GPT audit, 2026-09-30; 09 F-31/F-32). Uses the audited revision machinery only:
  frame / variants: analysis/g1_full_sample/03_decomposition.py (4,263 denominator posts; best-fit-family groups =
  stored ph4_fit_family; control D = CTRL_D); engine: submitted RUN_e1_m2_reconcile.run (GroupKFold(5), in-fold binary);
  interval: analysis/repair/01_bootstrap_samelabel.py (repaired same-label cluster bootstrap, 2,000 draws, seed 0).
Values:
  excl_whenthe / excl_top1 / excl_top5  raw_fit increment after excluding r/whenthe, or posts above the 99th / 95th
                                        percentile of log_score (full-sample percentiles)
  spec_only                             increment from adding ph4_spec_margin alone
  outcome_*                             raw_fit_plus_spec_stored increment with the continuous outcome replaced by
                                        log10(score+1) [check vs g1], log10(comments+1), upvote ratio
  negctrl_permute_within_bins           raw_fit increment after permuting raw_max_fit within (subreddit, age_window), seed 0
  negctrl_shuffled_outcome              raw_fit increment (dR2) and full-model AUC with log_score permuted, seed 0
  selected_listing                      listing_group == 'selected' (n = 348); stored raw + spec (the only scores that
                                        exist for these posts), best-fit-family GroupKFold(5), repaired bootstrap
Output: legacy_values/out/legacy_values.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.dont_write_bytecode = True
AN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AN))
import common as C  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G1 = load("g1d", AN / "g1_full_sample/03_decomposition.py")
RB = load("rb", AN / "repair/01_bootstrap_samelabel.py")
E = G1.E
D = G1.D


def inc(df, add, gcol="ph4_fit_family"):
    d = df.reset_index(drop=True)
    g = pd.Categorical(d[gcol]).codes
    E.CTRL, E.PH4 = D, add
    E.build_dummies(d)
    return E.run(d, GroupKFold(min(5, len(np.unique(g)))), groups=g, infold=True)


def main():
    df = G1.frame(AN / "g1_full_sample/out/scores_all.csv")
    assert len(df) == 4263
    raw, both = G1.VARIANTS["raw_fit"], G1.VARIANTS["raw_fit_plus_spec_stored"]
    res = {"n_full": int(len(df))}
    res["excl_whenthe"] = inc(df[df.subreddit != "whenthe"], raw) | {"n": int((df.subreddit != "whenthe").sum())}
    for q, name in [(99, "excl_top1"), (95, "excl_top5")]:
        keep = df.log_score <= np.percentile(df.log_score, q)
        res[name] = inc(df[keep], raw) | {"n": int(keep.sum())}
    res["spec_only"] = inc(df, ["ph4_spec_margin"])
    for col, name in [("log_score", "outcome_log_score"), ("log_comments", "outcome_log_comments"),
                      ("upvote_ratio", "outcome_upvote_ratio")]:
        d = df.copy()
        d["log_score"] = d[col].astype(float)
        res[name] = inc(d, both)
    rng = np.random.default_rng(0)
    d = df.copy()
    for _, ix in d.groupby(["subreddit", "age_window"]).groups.items():
        ix = np.asarray(list(ix))
        d.loc[ix, "raw_max_fit"] = rng.permutation(d.loc[ix, "raw_max_fit"].to_numpy())
    res["negctrl_permute_within_bins"] = inc(d, raw)
    d = df.copy()
    d["log_score"] = np.random.default_rng(0).permutation(d.log_score.to_numpy())
    res["negctrl_shuffled_outcome"] = inc(d, raw)
    o = pd.read_csv(C.OUTCOMES)
    sel = o[o.listing_group == "selected"].reset_index(drop=True)
    sel["ocr_len"] = sel["ocr_len"].astype(float)
    stored = ["max_contrastive_ph4_fit", "ph4_spec_margin"]
    point = inc(sel, stored)
    ug = np.array(sorted(sel.ph4_fit_family.unique()))
    v = RB.boot(sel, "ph4_fit_family", RB.picks_for(ug, 0), D, stored)
    res["selected_listing"] = point | {"n": int(len(sel)), "boot_dR2": RB.summ(v[:, 0])}
    res["check_outcome_log_score_vs_g1"] = abs(res["outcome_log_score"]["dR2"] -
        json.load(open(AN / "g1_full_sample/out/g1_results.json"))["full_4263"]["increments"]["raw_fit_plus_spec_stored"]["dR2"]) < 1e-9
    out = Path(__file__).resolve().parent / "out" / "legacy_values.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: (v.get("dR2") if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
