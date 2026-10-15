#!/usr/bin/env python3
"""Fixture tests for the author-coded human check (15b). Synthetic data only; all output outside Drive.
Writes analysis/human_check/fixture_test_report.json."""
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import hc_common as C
import draw_check as D
import freeze_labels as F
import analyze_check as A
H = C.H

OUT = C.scratch_dir("human_check") / "test_run"
FX = OUT / "fixture_hc"
RESULTS = []


def check(name, cond, note=""):
    RESULTS.append({"name": name, "pass": bool(cond), "note": str(note)[:300]})
    print(("PASS " if cond else "FAIL ") + name + (f" ({note})" if note and not cond else ""))


def make_fixture():
    if OUT.exists():
        shutil.rmtree(OUT)
    (FX / "img").mkdir(parents=True)
    rng = np.random.default_rng(424242)
    sizes = [853, 853, 852, 852, 640, 213]
    edges = [0.60] + H.CUTPOINTS + [0.99]
    slugs = [f.removeprefix("pool_") for f in sorted(pd.read_csv(C.REF_DIR / "reference_manifest.csv").family_id.unique())]
    rows = []
    for b, (n, lo, hi) in enumerate(zip(sizes, edges[:-1], edges[1:])):
        x = rng.uniform(lo, hi - 1e-6, n)
        am = rng.choice(slugs, n)
        rows += [dict(post_id=f"fx{b}{i:04d}", raw_max_fit=float(v), raw_argmax=a, subreddit="fixturesub")
                 for i, (v, a) in enumerate(zip(x, am))]
    s = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    s.to_csv(FX / "scores_all.csv", index=False)
    pd.DataFrame({"post_id": s.post_id, H.CLUSTER_COL: np.arange(len(s))}).to_csv(FX / "clusters.csv", index=False)
    paths = []
    for pid in s.post_id:
        p = FX / "img" / f"{pid}.png"
        Image.new("RGB", (4, 4), (10, 20, 30)).save(p)
        paths.append(str(p))
    pd.DataFrame({"post_id": s.post_id, "local_path": paths}).to_csv(FX / "manifest.csv", index=False)
    return H.load_frame(FX / "scores_all.csv", FX / "clusters.csv", FX / "manifest.csv")


def export(pkg, labels_by_id, path, coder="E", package_id="check", drop=0, dup=False, extra=False):
    ids = F.presented_ids(pkg)
    rows = [dict(coder=coder, package_id=package_id, item_id=i, primary=labels_by_id[i], ts="t") for i in ids]
    rows = rows[drop:]
    if dup:
        rows.append(rows[0])
    if extra:
        rows.append(dict(rows[0], item_id="zzzzzzzzzz"))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def refused(fn, token):
    try:
        fn()
    except SystemExit as e:
        return token in str(e), str(e)
    return False, "not refused"


LIVE_DRAW_EXISTED_AT_START = (C.HC / "draw").exists()


def main():
    frame = make_fixture()
    key_dir, pkg = FX / "draw", FX / "package_E"
    summ = D.build(frame, C.REF_DIR, key_dir, pkg, salt=b"fixture-salt")
    key = pd.read_csv(key_dir / "key.csv", keep_default_na=False)
    st, rp = key[key.kind == "study"], key[key.kind == "repeat"]

    # draw
    check("draw_60_unique_10_per_bin", len(st) == 60 and st.post_id.nunique() == 60
          and st.bin.value_counts().to_dict() == {b: 10 for b in C.BINS}, summ)
    check("draw_10_repeats_of_sampled_posts", len(rp) == 10 and set(rp.post_id) <= set(st.post_id)
          and set(rp.repeat_of) <= set(st.item_id))
    check("draw_70_presentations", summ["n_presentations"] == 70 and len(F.presented_ids(pkg)) == 70)
    pos = dict(zip(key.item_id, key.position))
    gaps = [pos[r.item_id] - pos[r.repeat_of] for r in rp.itertuples()]
    check("repeat_copies_at_least_20_after_original", min(gaps) >= C.REPEAT_MIN_GAP, gaps)
    s2 = D.draw(frame)
    check("draw_deterministic", sorted(s2.post_id) == sorted(st.post_id))
    check("draw_N_h_recorded", st.groupby("bin").N_h.first().to_dict() == {"B1": 853, "B2": 853, "B3": 852, "B4": 852,
                                                                           "B5": 640, "B6": 213})
    ok, msg = refused(lambda: D.build(frame, C.REF_DIR, key_dir, FX / "pkg2"), "already exists")
    check("second_draw_refused", ok, msg)

    # blinding of the package
    page = (pkg / "index.html").read_text()
    pj = json.loads(re.search(r"const PACKAGE = (\{.*?\});\n", page, re.S).group(1))
    check("package_items_carry_only_id_and_file", all(set(it) == {"id", "file"} for it in pj["items"]))
    body = re.sub(r"<!--.*?-->", "", page, flags=re.S)  # the comment only lists what is excluded
    scores = [f"{v:.4f}"[:6] for v in st.raw_max_fit]
    leaks = [t for t in list(st.post_id) + ["fixturesub", "raw_max_fit", "raw_argmax", "nominat", "score", "subreddit"]
             + [f"B{k}" for k in range(1, 7)] + scores if re.search(rf"\b{re.escape(t)}", body)]
    check("package_page_has_no_post_ids_scores_bins_or_community", not leaks, leaks)
    check("package_has_no_key_or_salt", not any(p.name in ("key.csv", "salt.hex", "draw_summary.json")
                                                 for p in pkg.rglob("*")))
    check("package_image_names_are_opaque_ids", sorted(p.stem for p in (pkg / "images").iterdir())
          == sorted(set(key.item_id)))
    check("package_reference_sheet_verified_copy", H.verify_reference_assets(pkg / "reference")[0]
          if (pkg / "reference" / "reference_assets.csv").exists() else
          sorted(p.name for p in (pkg / "reference" / "img").iterdir())
          == sorted(p.name for p in (C.REF_DIR / "img").iterdir()))
    check("instructions_on_every_item_page", all(__import__("html").escape(t) in page for t in C.CODER_INSTRUCTIONS))
    check("form_offers_31_families_plus_none_uncertain", len(pj["families"]) == 31)

    # label freeze
    nom = dict(zip(key.item_id, key.raw_argmax.map(C.display_name)))
    lab_ok = {i: nom[i] for i in key.item_id}
    rec = OUT / "LABELS_FREEZE.json"
    for nm, kw, tok in [("incomplete", dict(drop=1), "incomplete"), ("duplicate", dict(dup=True), "duplicate"),
                        ("unknown", dict(extra=True), "unknown"), ("wrong_coder", dict(coder="A"), "coder"),
                        ("wrong_package", dict(package_id="study"), "package_id")]:
        p = export(pkg, lab_ok, OUT / f"bad_{nm}.csv", **kw)
        ok, msg = refused(lambda: F.validate(p, F.presented_ids(pkg)), tok)
        check(f"label_file_{nm}_refused", ok, msg)
    bad = dict(lab_ok)
    bad[next(iter(bad))] = "Not A Family"
    p = export(pkg, bad, OUT / "bad_label.csv")
    ok, msg = refused(lambda: F.validate(p, F.presented_ids(pkg)), "invalid label")
    check("label_file_invalid_label_refused", ok, msg)

    def freeze(path):
        sys.argv = ["freeze_labels.py", str(path), "--package-dir", str(pkg), "--out-dir", str(OUT / "labels"),
                    "--record", str(rec)]
        F.main()
    ok, msg = refused(lambda: A.run(key_dir, OUT / "labels" / "labels_E.csv", rec), "not frozen")
    check("analysis_refused_before_label_freeze", ok, msg)
    good = export(pkg, lab_ok, OUT / "good.csv")
    freeze(good)
    check("label_freeze_record_written", rec.exists() and json.loads(rec.read_text())["n_rows"] == 70)
    ok, msg = refused(lambda: freeze(good), "already exists")
    check("second_label_freeze_refused", ok, msg)

    # analysis on known truths
    res = A.run(key_dir, OUT / "labels" / "labels_E.csv", rec)
    check("all_confirm_gives_confirm_1_everywhere", all(v["proportions"]["confirm"] == 1 for v in res["per_bin"].values())
          and res["main_comparison"]["confirm"]["high_minus_low"] == 0)
    check("per_bin_n_10_and_categories_sum", all(v["n"] == 10 and sum(v["counts"][c] for c in A.CATS) == 10
                                                 for v in res["per_bin"].values()))
    check("main_comparison_n_20_each", all(v["low_B1_B2"]["n"] == 20 and v["high_B5_B6"]["n"] == 20
                                           for v in res["main_comparison"].values()))
    check("repeat_agreement_10_of_10", res["repeat_agreement"]["n_pairs"] == 10 and res["repeat_agreement"]["agree"] == 10)
    shutil.copy(OUT / "bad_label.csv", OUT / "labels" / "labels_E.csv")
    ok, msg = refused(lambda: A.run(key_dir, OUT / "labels" / "labels_E.csv", rec), "differs")
    check("tampered_labels_refused", ok, msg)

    # mixed truth: low bins none, high bins confirm, one repeat changed
    rec2, ld2 = OUT / "LABELS2.json", OUT / "labels2"
    lab_mix = {i: (nom[i] if b in C.HIGH else ("none" if b in C.LOW else "uncertain"))
               for i, b in zip(key.item_id, key.bin)}
    flip = rp.item_id.iloc[0]
    lab_mix[flip] = "uncertain" if lab_mix[flip] != "uncertain" else "none"
    p = export(pkg, lab_mix, OUT / "mix.csv")
    sys.argv = ["freeze_labels.py", str(p), "--package-dir", str(pkg), "--out-dir", str(ld2), "--record", str(rec2)]
    F.main()
    r2 = A.run(key_dir, ld2 / "labels_E.csv", rec2)
    mc = r2["main_comparison"]
    check("mixed_low_none_high_confirm", mc["none"]["low_B1_B2"]["p"] == 1 and mc["confirm"]["high_B5_B6"]["p"] == 1
          and mc["confirm"]["high_minus_low"] == 1)
    check("mixed_middle_bins_uncertain", r2["per_bin"]["B3"]["proportions"]["uncertain"] == 1)
    check("repeat_agreement_counts_the_flip", r2["repeat_agreement"]["agree"] == 9 and r2["repeat_agreement"]["n_pairs"] == 10)

    # interval arithmetic
    w = A.wilson(10, 20)
    check("wilson_10_of_20", abs(w[0] - 0.2993) < 1e-3 and abs(w[1] - 0.7007) < 1e-3, w)
    w0 = A.wilson(0, 20)
    check("wilson_0_of_20", w0[0] == 0 and abs(w0[1] - 0.1611) < 1e-3, w0)
    nc = A.newcombe(15, 20, 5, 20)  # Newcombe (1998) method 10 style check: d = 0.5
    check("newcombe_contains_d_and_is_bounded", nc[0] < 0.5 < nc[1] and -1 <= nc[0] and nc[1] <= 1, nc)

    # gates
    ok, why = C.freeze_ok(record=OUT / "no_record.json")
    check("real_freeze_absent_refuses", not ok and "no 15b" in why, why)
    # The harness never invokes the real-mode draw (GPT pre-draw check, 09 F-52): once the 15b freeze exists, real mode
    # would proceed. Instead verify statically that real mode checks the freeze before touching the real frame.
    src = (C.HC / "draw_check.py").read_text()
    real = src[src.index("    else:\n        ok, why = C.freeze_ok()"):]
    check("real_mode_checks_freeze_before_loading_frame",
          real.index("C.freeze_ok()") < real.index("H.load_frame(") and "REFUSING real draw" in real)
    check("live_draw_not_created_by_harness", (C.HC / "draw").exists() == LIVE_DRAW_EXISTED_AT_START)
    ok_old, why_old = H.freeze_ok()
    check("superseded_15_freeze_record_untouched_and_still_verifies_hashes",
          (not ok_old) and "H-1" in why_old, why_old)

    n_pass = sum(r["pass"] for r in RESULTS)
    rep = {"all_pass": n_pass == len(RESULTS), "n": len(RESULTS), "tests": RESULTS}
    (C.HC / "fixture_test_report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(f"{n_pass}/{len(RESULTS)} PASS")


if __name__ == "__main__":
    main()
