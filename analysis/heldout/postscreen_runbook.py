#!/usr/bin/env python3
"""Post-screening execution runbook: integrity checks only (NOT part of the frozen bundle; reads, never writes,
frozen/live artifacts). Prints PASS/FAIL plus operational counts. It never prints a score, a difference or a verdict.

  --fixture : build a synthetic screening fixture (engineering/out/runbook_fixture/) next to the synthetic feature
              fixture (engineering/out/fixture/), run every check (expect all PASS), then run negative tests
              (injected faults; expect each targeted check to FAIL). No held-out or DEV outcome is used.
  --live    : run every check on heldout/out/ (use only after screening has formally closed).

Checks
  C1 freeze chain     verify_freeze_record passes; original_freeze kept; amendments ordered after the freeze and
                      each has a row in 14 section 17; current bundle differs from the original iff amendments exist
  C2 closure          screening_summary.json satisfies 14 section 5 (H >= 60; C == 20 or C queue exhausted;
                      H + C >= 70; screened <= 1,500; shortfall False) and agrees with a recount of dispositions.csv
  C3 dispositions /   unique slugs; no 'pending'; no queue gap before the last H rank (live: slug at each rank is
     manifest         the frozen queue's); manifest families == analyzable; each >= MIN_VARIANTS; files exist and
                      match SHA-256; no gated-flag rows; n_after_d4 == manifest count; per-anchor C cap; no DEV/excluded
  C4 aborted run      no manifest/feature path points into _aborted*; aborted-run hash manifest re-verifies
  C5 features         required arrays; finite; no zero rows; rows = DEV rows + manifest rows; held-out family
                      sequence == manifest order; family counts per set == closure counts; texts aligned
  C6 output schemas   ho_primary / ho_capped10 / ho_capped15 / ho_foldlocal: one row per family, required columns,
                      finite ranks in [0, 1]; ho_sweep: seeds 1-10 x families; capped n == min(n, cap)
  C7 verdict logic    summarize_heldout.verdicts() reproduces every 14 section 11 row on synthetic inputs; stored
                      verdicts in heldout_results.json equal verdicts recomputed from the stored numbers
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HO = Path(__file__).resolve().parent
RR = HO.parents[1]
FX_FEAT = HO / "engineering" / "out" / "fixture"
FX_SCREEN = HO / "engineering" / "out" / "runbook_fixture"
METHODS = ["contrastive", "recurrence", "clip_centroid", "sparse_centroid", "sparse_knn", "logistic", "surf_phash",
           "surf_ocr", "surf_template", "surf_composite", "fused_centroid"]
REQ_PRIMARY = ["family", "k", "n", "seed"] + [f"{m}__{s}" for m in METHODS for s in ["E_local", "DL_local", "DP_local"]]
REQ_FOLD = ["family", "k", "n"] + [f"{m}__{s}_foldlocal" for m in ["contrastive", "clip_centroid", "fused_centroid",
                                                                    "logistic"] for s in ["DL", "DP", "DO"]]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SR = load("sr", HO / "screen_and_retrieve.py")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Report:
    def __init__(self):
        self.res = {}

    def check(self, cid, cond, note=""):
        self.res.setdefault(cid, []).append((bool(cond), note))

    def ok(self, cid):
        return all(c for c, _ in self.res.get(cid, [(False, "not run")]))

    def show(self):
        for cid in sorted(self.res):
            fails = [n for c, n in self.res[cid] if not c]
            print(f"{cid}: {'PASS' if not fails else 'FAIL'} ({len(self.res[cid])} assertions)"
                  + ("" if not fails else " -> " + "; ".join(fails)))
        return all(self.ok(c) for c in self.res)


# ---------------------------------------------------------------- C1
def c1_freeze(rep, live):
    if not live:
        rep.check("C1", True, "fixture mode: freeze chain checked only in --live")
        return
    import subprocess
    r = subprocess.run([sys.executable, str(HO / "verify_freeze_record.py")], capture_output=True, text=True)
    rep.check("C1", r.returncode == 0 and "FREEZE VERIFIED" in r.stdout, "verify_freeze_record failed")
    rec = json.loads((HO / "FREEZE_RECORD.json").read_text())
    am = rec.get("amendments", [])
    if am:
        of = rec.get("original_freeze")
        rep.check("C1", of is not None, "original_freeze missing although amendments exist")
        if of:
            rep.check("C1", of["bundle_sha256"] != rec["bundle_sha256"], "bundle unchanged despite amendment")
            ts = [of["freeze_timestamp"]] + [a["timestamp"] for a in am]
            rep.check("C1", ts == sorted(ts), "amendment timestamps not after the freeze / not ordered")
        s17 = (RR / rec["protocol_file"]).read_text().split("## 17")[-1]
        for a in am:
            rep.check("C1", f"| {a['id']} |" in s17, f"amendment {a['id']} has no row in 14 section 17")
    else:
        rep.check("C1", "original_freeze" not in rec, "original_freeze present without amendments")


# ---------------------------------------------------------------- C2 / C3 / C4
def c2_c4_screening(rep, out, live, queue=None):
    S = json.loads((out / "screening_summary.json").read_text())
    D = list(csv.DictReader(open(out / "dispositions.csv")))
    M = list(csv.DictReader(open(out / "heldout_manifest.csv")))
    an = [d for d in D if d["disposition"] == "analyzable"]
    h = sum(d["queue"] == "H" for d in an)
    c = sum(d["queue"] == "C" for d in an)
    c_left = [d for d in D if d["queue"] == "C"]  # C exhaustion is not recorded; accept C == cap only, else flag
    rep.check("C2", S["H"] == h and S["C"] == c, "summary H/C differ from recount")
    rep.check("C2", S["screened"] == len(D) <= SR.BUDGET, "screened count / budget")
    rep.check("C2", S["dispositions"] == dict(Counter(d["disposition"] for d in D)), "summary dispositions != recount")
    rep.check("C2", h >= SR.H_TARGET, "H below target")
    rep.check("C2", c == SR.C_CAP, "C below cap (C-queue exhaustion must then be shown by hand)")
    rep.check("C2", h + c >= SR.NEW_TARGET and not S["shortfall"], "H + C below target / shortfall")
    rep.check("C2", S["freeze_params"] == SR.FREEZE_PARAMS, "freeze params differ")
    # C3
    slugs = [d["slug"] for d in D]
    rep.check("C3", len(slugs) == len(set(slugs)), "duplicate slugs")
    rep.check("C3", not any(d["disposition"] == "pending" for d in D), "pending rows")
    ranks = {int(d["queue_rank"]): d["slug"] for d in D}
    mh = max(int(d["queue_rank"]) for d in D if d["queue"] == "H")
    rep.check("C3", all(r in ranks for r in range(mh + 1)), "queue gap before last H rank")
    if queue is not None:
        rep.check("C3", all(queue[r] == s for r, s in ranks.items()), "slug at rank differs from frozen queue")
    per = Counter(m["family_slug"] for m in M)
    rep.check("C3", set(per) == {d["slug"] for d in an}, "manifest families != analyzable set")
    rep.check("C3", min(per.values()) >= SR.FREEZE_PARAMS["MIN_VARIANTS"], "family below MIN_VARIANTS")
    bad = [m for m in M if not os.path.exists(m["image_path"]) or sha(m["image_path"]) != m["sha256"]]
    rep.check("C3", not bad, f"{len(bad)} missing/hash-mismatched images")
    rep.check("C3", not any(m["nsfw_flag"] == "1" for m in M), "gated-flag rows in manifest")
    rep.check("C3", all(int(d["n_after_d4"]) == per[d["slug"]] for d in an), "n_after_d4 != manifest count")
    capc = Counter(d["anchor"] for d in an if d["queue"] == "C")
    rep.check("C3", max(capc.values(), default=0) <= SR.ANCHOR_CAP, "per-anchor C cap exceeded")
    rep.check("C3", not ({d["slug"] for d in an} & (set(SR.DEV_SLUGS) | SR.EXCLUDED_SLUGS)), "DEV/excluded analyzable")
    # C4
    rep.check("C4", not any("_aborted" in m["image_path"] for m in M), "manifest points into aborted run")
    for am in out.glob("_aborted_run*_MANIFEST.json"):
        A = json.loads(am.read_text())
        d = out / am.name.replace("_MANIFEST.json", "_2026-09-29") if live else out / am.name.replace("_MANIFEST.json", "")
        cur = {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file()}
        rep.check("C4", cur == A["files"], f"aborted run {d.name} changed")
    return S, M


# ---------------------------------------------------------------- C5
def c5_features(rep, feat, S, M, n_dev_fams):
    z = np.load(feat / "expanded_features.npz", allow_pickle=True)
    need = ["ids", "fam", "dense", "clip_openai", "clip_laion", "phash_bits", "siglip2", "dinov2", "fam_k_keys",
            "fam_k_vals", "fam_target", "fam_set"]
    rep.check("C5", all(k in z.files for k in need), "missing arrays")
    n = len(z["ids"])
    for k in ["dense", "clip_openai", "clip_laion", "siglip2", "dinov2"]:
        a = z[k]
        rep.check("C5", len(a) == n and np.isfinite(a).all(), f"{k}: length/non-finite")
        rep.check("C5", (np.abs(a).sum(1) > 0).all(), f"{k}: zero rows")
    rep.check("C5", len(z["phash_bits"]) == n, "phash length")
    sets = dict(zip(list(z["fam_k_keys"]), list(z["fam_set"])))
    cnt = Counter(sets.values())
    rep.check("C5", cnt.get("DEV", 0) == n_dev_fams, "DEV family count")
    rep.check("C5", cnt.get("H", 0) == S["H"] and cnt.get("C", 0) == S["C"], "H/C family counts != closure")
    fam = list(z["fam"])
    ho_seq = [f for f in fam if sets[f] != "DEV"]
    # build_features labels held-out families "ho_<slug>" (DEV labels are unprefixed)
    rep.check("C5", ho_seq == ["ho_" + m["family_slug"] for m in M], "held-out rows != manifest order")
    rep.check("C5", list(z["ids"])[n - len(M):] == [f"ho{i}" for i in range(len(M))], "held-out ids != ho0..")
    rep.check("C5", sum(sets[f] == "DEV" for f in fam) + len(M) == n, "row count != DEV + manifest")
    texts = json.loads((feat / "expanded_texts.json").read_text())
    rep.check("C5", len(texts) == n, "texts misaligned")
    return z


# ---------------------------------------------------------------- C6
def c6_outputs(rep, feat, z):
    kvals = set(int(x) for x in z["fam_k_vals"])
    fam = np.array(list(z["fam"]))
    size = {int(k): int((fam == f).sum()) for f, k in zip(list(z["fam_k_keys"]), z["fam_k_vals"])}

    def rng01(df, cols):
        v = df[cols].to_numpy(float)
        return np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1

    for name, cap in [("ho_primary", None), ("ho_capped10", 10), ("ho_capped15", 15)]:
        p = feat / f"{name}.csv"
        rep.check("C6", p.exists(), f"{name} missing")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rep.check("C6", all(c in df.columns for c in REQ_PRIMARY), f"{name}: required columns")
        rep.check("C6", set(df.k) == kvals and len(df) == len(kvals), f"{name}: one row per family")
        rep.check("C6", rng01(df, [c for c in REQ_PRIMARY if "__" in c]), f"{name}: ranks not finite in [0,1]")
        rep.check("C6", all(int(r.n) == (size[int(r.k)] if cap is None else min(size[int(r.k)], cap))
                            for r in df.itertuples()), f"{name}: n != family size")
    p = feat / "ho_sweep.csv"
    rep.check("C6", p.exists(), "ho_sweep missing")
    if p.exists():
        df = pd.read_csv(p)
        rep.check("C6", sorted(df.seed.unique()) == list(range(1, 11)), "sweep seeds != 1..10")
        rep.check("C6", len(df) == 10 * len(kvals) and all(set(g.k) == kvals for _, g in df.groupby("seed")),
                  "sweep: families x seeds")
        rep.check("C6", rng01(df, [c for c in REQ_PRIMARY if "__E_local" in c]), "sweep ranks")
    p = feat / "ho_foldlocal.csv"
    rep.check("C6", p.exists(), "ho_foldlocal missing")
    if p.exists():
        df = pd.read_csv(p)
        rep.check("C6", all(c in df.columns for c in REQ_FOLD) and set(df.k) == kvals and len(df) == len(kvals),
                  "foldlocal schema")
        rep.check("C6", rng01(df, [c for c in REQ_FOLD if "__" in c]), "foldlocal ranks")


# ---------------------------------------------------------------- C7
def synth(ci_low):
    """Synthetic result block: ci_low maps baseline -> lower CI bound (upper = low + 0.1)."""
    SH = load("sh_syn", HO / "summarize_heldout.py")
    r = {"rq1": {b: {"mean": ci_low.get(b, 0.1) + 0.05, "ci95": [ci_low.get(b, 0.1), ci_low.get(b, 0.1) + 0.1]}
                 for b in SH.BASELINES},
         "classifier_minus_contrastive": {"mean": 0.02, "ci95": [-0.01, 0.05]},
         "rq2_laion_familylevel": {"mean": 0.05, "ci95": [0.01, 0.09]}}
    return SH, r


def c7_verdicts(rep, feat):
    SH, r = synth({})
    v = SH.verdicts(r)
    rep.check("C7", v["L1"] == "L1 validated on held-out families", "row 1 (all exclude 0)")
    SH, r = synth({"fused_centroid": -0.02})
    v = SH.verdicts(r)
    rep.check("C7", v["L1"].startswith("L1 weakened") and "fused_centroid" in v["L1"].split("detected over")[1]
              and "compact" not in v["L1"], "row 4 (fused alone fails -> weakened, not compact)")
    SH, r = synth({"recurrence": -0.05, "surf_composite": -0.01})
    v = SH.verdicts(r)
    tail = v["L1"].split("detected over")[1]
    rep.check("C7", v["L1"].startswith("L1 weakened") and "recurrence" in tail and "surf_composite" in tail
              and "sparse_centroid" not in tail, "row 2 (some fail)")
    SH, r = synth({"sparse_centroid": -0.01, "sparse_knn": -0.02, "clip_centroid": -0.03})
    rep.check("C7", "compact and inspectable" in SH.verdicts(r)["L1"], "row 3 (core three fail)")
    SH, r = synth({b: -0.05 for b in SH.BASELINES})
    rep.check("C7", "compact and inspectable" in SH.verdicts(r)["L1"], "row 3 (all fail)")
    SH, r = synth({"sparse_centroid": -0.01, "sparse_knn": -0.02})
    rep.check("C7", SH.verdicts(r)["L1"].startswith("L1 weakened"), "two of core three fail -> weakened")
    SH, r = synth({"sparse_centroid": 0.0})
    rep.check("C7", SH.verdicts(r)["L1"].startswith("L1 weakened"), "lower bound exactly 0 counts as includes 0")
    for m, lo, want in [(0.05, 0.01, "validated"), (0.03, -0.01, "weakened to suggestive"),
                        (0.0, -0.04, "not replicated; restricted to DEV"), (-0.02, -0.06, "not replicated; restricted to DEV")]:
        SH, r = synth({})
        r["rq2_laion_familylevel"] = {"mean": m, "ci95": [lo, lo + 0.08]}
        rep.check("C7", SH.verdicts(r)["L2_laion"] == want, f"L2 row mean={m}")
    SH, r = synth({})
    r["classifier_minus_contrastive"] = {"mean": 0.05, "ci95": [0.01, 0.09]}
    rep.check("C7", SH.verdicts(r)["classifier_note"].endswith("interval excludes 0"), "classifier note")
    # stored verdicts == recomputed from stored numbers (no values printed)
    p = feat / "heldout_results.json"
    rep.check("C7", p.exists(), "heldout_results.json missing")
    if p.exists():
        R = json.loads(p.read_text())
        for key in [k for k in R if k.startswith("verdicts_")]:
            blk = R[key.replace("verdicts_", "")]
            rep.check("C7", SH.verdicts(blk) == R[key], f"{key} not reproducible from stored numbers")


# ---------------------------------------------------------------- fixture construction
def build_screen_fixture():
    """Synthetic screening artifacts consistent with the synthetic feature fixture (sizes/sets only)."""
    if FX_SCREEN.exists():
        shutil.rmtree(FX_SCREEN)
    (FX_SCREEN / "images").mkdir(parents=True)
    z = np.load(FX_FEAT / "expanded_features.npz", allow_pickle=True)
    sets = dict(zip(list(z["fam_k_keys"]), list(z["fam_set"])))
    fam = list(z["fam"])
    hofams = [f for f in dict.fromkeys(fam) if sets[f] != "DEV"]
    # rename synthetic held-out families so no DEV slug appears (fixture only)
    ren = {f: f"fx-{sets[f].lower()}-{i:02d}" for i, f in enumerate(hofams)}
    lab = {f: "ho_" + ren[f] for f in hofams}  # feature labels mirror build_features ("ho_" + slug)
    D, M, rank = [], [], 0
    anchors = list(SR.ANCHORS)
    for i, f in enumerate(hofams):
        s = sets[f]
        n = sum(x == f for x in fam)
        D.append(dict(slug=ren[f], queue=s, queue_rank=rank, disposition="analyzable", n_after_d4=n,
                      anchor=anchors[i % len(anchors)] if s == "C" else ""))
        rank += 1
        for j in range(n):
            p = FX_SCREEN / "images" / f"{ren[f]}_{j:03d}.bin"
            p.write_bytes(f"{ren[f]}-{j}".encode())
            M.append(dict(family_slug=ren[f], stratum=s, anchor="", queue_rank=rank - 1, image_path=str(p),
                          sha256=sha(p), source="gallery", nsfw_flag=0))
    D.append(dict(slug="fx-too-small", queue="H", queue_rank=rank, disposition="too_small", n_after_d4=4, anchor=""))
    pd.DataFrame(D).to_csv(FX_SCREEN / "dispositions.csv", index=False)
    pd.DataFrame(M).to_csv(FX_SCREEN / "heldout_manifest.csv", index=False)
    ab = FX_SCREEN / "_aborted_runX"
    ab.mkdir()
    (ab / "x.txt").write_text("aborted")
    (FX_SCREEN / "_aborted_runX_MANIFEST.json").write_text(json.dumps({"files": {"x.txt": sha(ab / "x.txt")}}))
    h = sum(d["queue"] == "H" and d["disposition"] == "analyzable" for d in D)
    c = sum(d["queue"] == "C" for d in D)
    S = dict(screened=len(D), H=h, C=c, budget=SR.BUDGET, shortfall=False,
             dispositions=dict(Counter(d["disposition"] for d in D)), freeze_params=SR.FREEZE_PARAMS)
    (FX_SCREEN / "screening_summary.json").write_text(json.dumps(S))
    # features: same arrays, family labels renamed; outputs copied (synthetic)
    feat = FX_SCREEN / "feat_fixture"
    feat.mkdir()
    arrs = {k: z[k] for k in z.files}
    arrs["fam"] = np.array([lab.get(f, f) for f in fam], object)
    arrs["fam_k_keys"] = np.array([lab.get(f, f) for f in z["fam_k_keys"]], object)
    nd = sum(sets[f] == "DEV" for f in fam)
    arrs["ids"] = np.array(list(z["ids"])[:nd] + [f"ho{i}" for i in range(len(fam) - nd)], object)
    np.savez_compressed(feat / "expanded_features.npz", **arrs)
    for f in ["expanded_texts.json", "ho_primary.csv", "ho_sweep.csv", "ho_capped10.csv", "ho_capped15.csv",
              "ho_foldlocal.csv", "heldout_results.json"]:
        shutil.copy(FX_FEAT / f, feat / f)
    return h, c, feat


def run_all(out, feat, live, n_dev, queue=None, targets=None):
    rep = Report()
    saved = (SR.H_TARGET, SR.C_CAP, SR.NEW_TARGET)
    if targets:  # fixture only: scale 14 section 5 targets to the fixture's size
        SR.H_TARGET, SR.C_CAP, SR.NEW_TARGET = targets
    try:
        c1_freeze(rep, live)
        S, M = c2_c4_screening(rep, out, live, queue)
        z = c5_features(rep, feat, S, M, n_dev)
        c6_outputs(rep, feat, z)
        c7_verdicts(rep, feat)
    finally:
        SR.H_TARGET, SR.C_CAP, SR.NEW_TARGET = saved
    return rep


def fixture_mode():
    h, c, feat = build_screen_fixture()
    tg = (h, c, h + c)
    print(f"== fixture (synthetic): H={h} C={c}")
    ok = run_all(FX_SCREEN, feat, False, 10, targets=tg).show()
    print("ALL PASS" if ok else "UNEXPECTED FAILURES")
    # negative tests: each injected fault must turn its target check to FAIL
    neg = []

    def inject(desc, target, mutate, restore):
        mutate()
        try:
            rep = run_all(FX_SCREEN, feat, False, 10, targets=tg)
            neg.append((desc, not rep.ok(target)))
        except Exception as e:  # noqa: BLE001  (a crash also counts as detection)
            neg.append((desc + f" [raised {type(e).__name__}]", True))
        restore()

    dpath, mpath, spath = FX_SCREEN / "dispositions.csv", FX_SCREEN / "heldout_manifest.csv", FX_SCREEN / "screening_summary.json"
    D0, M0, S0 = dpath.read_text(), mpath.read_text(), spath.read_text()
    img0 = pd.read_csv(mpath).image_path.iloc[0]
    b0 = Path(img0).read_bytes()
    R0 = (feat / "heldout_results.json").read_text()
    P0 = (feat / "ho_primary.csv").read_text()

    def rw(p, txt):
        return lambda: p.write_text(txt)
    df = pd.read_csv(dpath)
    inject("duplicate slug", "C3", rw(dpath, D0 + D0.splitlines()[1] + "\n"), rw(dpath, D0))
    inject("pending row", "C3", rw(dpath, df.assign(disposition=["pending"] + list(df.disposition[1:])).to_csv(index=False)),
           rw(dpath, D0))
    inject("image hash mismatch", "C3", lambda: Path(img0).write_bytes(b"tampered"), lambda: Path(img0).write_bytes(b0))
    mdf = pd.read_csv(mpath)
    inject("aborted path in manifest", "C4",
           rw(mpath, mdf.assign(image_path=[str(FX_SCREEN / "_aborted_runX" / "x.txt")] + list(mdf.image_path[1:]),
                                sha256=[sha(FX_SCREEN / "_aborted_runX" / "x.txt")] + list(mdf.sha256[1:])).to_csv(index=False)),
           rw(mpath, M0))
    inject("aborted run modified", "C4", lambda: (FX_SCREEN / "_aborted_runX" / "x.txt").write_text("changed"),
           lambda: (FX_SCREEN / "_aborted_runX" / "x.txt").write_text("aborted"))
    S1 = json.loads(S0)
    S1["shortfall"] = True
    inject("shortfall flagged", "C2", rw(spath, json.dumps(S1)), rw(spath, S0))
    S2 = json.loads(S0)
    S2["H"] -= 1
    inject("summary/recount mismatch", "C2", rw(spath, json.dumps(S2)), rw(spath, S0))
    inject("manifest row dropped", "C5", rw(mpath, "\n".join(M0.splitlines()[:-1]) + "\n"), rw(mpath, M0))
    pdf = pd.read_csv(feat / "ho_primary.csv")
    inject("primary missing a family", "C6", rw(feat / "ho_primary.csv", pdf.iloc[1:].to_csv(index=False)),
           rw(feat / "ho_primary.csv", P0))
    inject("rank out of [0,1]", "C6", rw(feat / "ho_primary.csv", pdf.assign(contrastive__E_local=1.5).to_csv(index=False)),
           rw(feat / "ho_primary.csv", P0))
    R1 = json.loads(R0)
    R1["verdicts_H"] = copy.deepcopy(R1["verdicts_H"])
    R1["verdicts_H"]["L1"] = "L1 validated on held-out families" if "validated" not in R1["verdicts_H"]["L1"] else "x"
    inject("stored verdict tampered", "C7", rw(feat / "heldout_results.json", json.dumps(R1)),
           rw(feat / "heldout_results.json", R0))
    print("== negative tests (each must be detected)")
    for d, hit in neg:
        print(f"  {'DETECTED' if hit else 'MISSED  '}  {d}")
    allneg = all(h for _, h in neg)
    print("NEGATIVE TESTS: ALL DETECTED" if allneg else "NEGATIVE TESTS: SOME MISSED")
    res = dict(positive_all_pass=ok, negative_all_detected=allneg, negatives=neg)
    (HO / "engineering" / "out" / "runbook_fixture_test.json").write_text(json.dumps(res, indent=2))
    return ok and allneg


def live_mode():
    out = HO / "out"
    if not (HO / "FREEZE_RECORD.json").exists() or not (out / "screening_summary.json").exists():
        raise SystemExit("REFUSING: no freeze record or screening not closed")
    q = SR.candidate_queue([r["slug"] for r in csv.DictReader(open(SR.FRAME / "frame_slugs.csv"))])
    ok = run_all(out, out, True, 30, queue=q).show()
    print("RUNBOOK: ALL PASS" if ok else "RUNBOOK: FAILURES (see above) - stop and report")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixture", action="store_true")
    g.add_argument("--live", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if (fixture_mode() if a.fixture else live_mode()) else 1)
