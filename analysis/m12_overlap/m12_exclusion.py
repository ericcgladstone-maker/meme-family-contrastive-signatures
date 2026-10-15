#!/usr/bin/env python3
"""M12 exclusion sensitivity (SPEC_exclusion.md; 09 F-62). Reuses the existing engines verbatim; refuses unless
FREEZE_EXCLUSION.json verifies and the full-sample values reproduce the stored ones."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).resolve().parent
AN = HERE.parent
sys.path.insert(0, str(AN))
import common as C  # noqa: E402

OUT = HERE / "out" / "m12_exclusion_sensitivity.json"
MODS = {"g2": AN / "g2_template_cluster/02_grouped_cv.py", "g1": AN / "g1_full_sample/03_decomposition.py",
        "rep": AN / "repair/01_bootstrap_samelabel.py"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify():
    rec = json.loads((HERE / "FREEZE_EXCLUSION.json").read_text())
    for rel, h in rec["hashes"].items():
        if sha(AN.parent / rel) != h:
            raise SystemExit(f"REFUSING: hash mismatch {rel}")


def g2_schemes(df):
    gfam = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
    gcom = df.subreddit.astype("category").cat.codes.to_numpy()
    return {"random5": list(KFold(5, shuffle=True, random_state=0).split(df)),
            "template_cc10_as_planned": list(GroupKFold(5).split(df, groups=df["cc10_as_planned"].to_numpy())),
            "community_gkf7_eq_LOCO": list(GroupKFold(7).split(df, groups=gcom)),
            "bestfit_family": list(GroupKFold(5).split(df, groups=gfam))}


def run_g2(g2, df):
    g2.E.CTRL, g2.E.PH4 = C.CTRL_D, C.PH4_F
    out = {}
    for name, sp in g2_schemes(df).items():
        g2.E.build_dummies(df)
        p, _ = g2.run_folds(df, sp)
        out[name] = {"dR2": p["dR2"], "dAUC": p["dAUC"], "R2_D": p["R2_D"], "R2_F": p["R2_F"]}
    return out


def main():
    verify()
    if OUT.exists():
        raise SystemExit("REFUSING: output exists (run once)")
    excl = sorted({p["reddit_post_id"] for p in json.loads((HERE / "out/m12_overlap.json").read_text())["pairs_for_traceability_not_inspected"]})
    assert len(excl) == 14, len(excl)
    g2, g1, rep = (load(k, p) for k, p in MODS.items())
    stored_g2 = json.loads((AN / "g2_template_cluster/out/g2_results.json").read_text())["pooled"]
    stored_g1 = json.loads((AN / "g1_full_sample/out/g1_results.json").read_text())["full_4263"]["raw_over_D_plus_magnitude"]["dR2"]
    stored_ci = json.loads((AN / "repair/out/bootstrap_samelabel.json").read_text())["g1_full_raw_over_magnitude_seed0"]["dR2"]["ci95"]
    # ---- reproduction gate (full 4,263)
    d2 = g2.load()
    assert len(d2) == 4263
    full = run_g2(g2, d2)
    for k, v in full.items():
        if abs(v["dR2"] - stored_g2[k]["dR2"]) > 1e-9:
            raise SystemExit(f"REFUSING: {k} does not reproduce ({v['dR2']} vs {stored_g2[k]['dR2']})")
    f1 = g1.frame(AN / "g1_full_sample/out/scores_all.csv")
    assert len(f1) == 4263
    gf = f1.ph4_fit_family.astype("category").cat.codes.to_numpy()
    adj_full = g1._inc(f1, gf, g1.D + g1.GEN, ["raw_max_fit"])["dR2"]
    if abs(adj_full - stored_g1) > 1e-9:
        raise SystemExit(f"REFUSING: magnitude-adjusted residual does not reproduce ({adj_full} vs {stored_g1})")
    # ---- exclusion (4,249)
    d2x = d2[~d2.post_id.astype(str).isin(excl)].reset_index(drop=True)
    f1x = f1[~f1.post_id.astype(str).isin(excl)].reset_index(drop=True)
    assert len(d2x) == 4249 and len(f1x) == 4249, (len(d2x), len(f1x))
    exc = run_g2(g2, d2x)
    gfx = f1x.ph4_fit_family.astype("category").cat.codes.to_numpy()
    adj_x = g1._inc(f1x, gfx, g1.D + g1.GEN, ["raw_max_fit"])["dR2"]
    v = rep.boot(f1x, "ph4_fit_family", rep.picks_for(f1x.ph4_fit_family.unique(), 0), g1.D + g1.GEN, ["raw_max_fit"])
    s = rep.summ(v[:, 0])
    res = {"n_full": 4263, "n_excluded": len(excl), "n_after": 4249, "excluded_post_ids": excl,
           "reproduction_gate": "passed (all four schemes and the magnitude-adjusted point reproduce stored values to 1e-9)",
           "continuous_dR2": {k: {"original": stored_g2[k]["dR2"], "exclusion": exc[k]["dR2"]} for k in exc},
           "binary_dAUC_audit_only": {k: {"original": stored_g2[k]["dAUC"], "exclusion": exc[k]["dAUC"]} for k in exc},
           "magnitude_adjusted_residual": {"original": {"dR2": stored_g1, "ci95": stored_ci},
                                           "exclusion": {"dR2": adj_x, "ci95": s["ci95"], "boot_n": s["n"]}},
           "families_in_excluded_sample": int(f1x.ph4_fit_family.nunique()), "families_full": int(f1.ph4_fit_family.nunique())}
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ["continuous_dR2", "magnitude_adjusted_residual", "binary_dAUC_audit_only"]}, indent=1))


if __name__ == "__main__":
    main()
