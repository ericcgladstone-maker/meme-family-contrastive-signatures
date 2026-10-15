#!/usr/bin/env python3
"""Implementation tests for the human-validation machinery (15 v0.5). Synthetic Reddit frame + synthetic labels only;
the reference sheet is built from DEV gallery images (pre-label instrument material). No real post is drawn.

All test/fixture output goes to PH4_NONDRIVE_DIR (default ~/Desktop/PH4_Computational_Revise_Resubmit_Non_Drive_Materials),
outside Google Drive. Inside Drive it writes only analysis/human_validation/fixture_test_report.json (the deliverable), with one PASS/FAIL per assertion (GPT implementation-audit list + extras).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True  # no __pycache__ inside Drive
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import analyze_labels as A
import draw_sample as D
import hv_common as H

OUT = H.scratch_dir("human_validation") / "test_run"  # outside Drive (PH4_NONDRIVE_DIR)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
REPORT_DELIVERABLE = H.HV / "fixture_test_report.json"  # the only file this test writes inside Drive
FX = OUT / "fixture_hv"
REPORT = {}


def check(name, cond, note=""):
    REPORT[name] = {"pass": bool(cond), "note": note}
    print(("PASS " if cond else "FAIL ") + name + (f" ({note})" if note else ""))


# ------------------------------------------------------------------ fixture
def make_fixture():
    if FX.exists():
        shutil.rmtree(FX)
    (FX / "img").mkdir(parents=True)
    rng = np.random.default_rng(424242)
    sizes = [853, 853, 852, 852, 640, 213]  # same bin sizes as the real frame (weights identical)
    edges = [0.60] + H.CUTPOINTS + [0.99]
    fams = sorted(pd.read_csv(H.DEV_INDEX).family_id.unique())
    slugs = [f.removeprefix("pool_") for f in fams]
    dom = "nobody_literally_no_one"
    rows = []
    for b, (n, lo, hi) in enumerate(zip(sizes, edges[:-1], edges[1:])):
        x = rng.uniform(lo, hi - 1e-6, n)
        p_dom = [0.25, 0.45, 0.45, 0.55, 0.84, 0.98][b]
        am = np.where(rng.random(n) < p_dom, dom, rng.choice(slugs, n))
        subs = rng.choice(["memes", "dankmemes", "me_irl", "wholesomememes", "historymemes", "shitposting", "whenthe"], n)
        rows += [dict(post_id=f"fx{b}{i:04d}", raw_max_fit=float(v), raw_argmax=a, subreddit=sr)
                 for i, (v, a, sr) in enumerate(zip(x, am, subs))]
    s = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    s.to_csv(FX / "scores_all.csv", index=False)
    cl = np.arange(len(s))
    for k in range(60):  # multi-post clusters, many crossing bins
        members = rng.choice(len(s), rng.integers(2, 6), replace=False)
        cl[members] = cl[members[0]]
    pd.DataFrame({"post_id": s.post_id, H.CLUSTER_COL: cl}).to_csv(FX / "clusters.csv", index=False)
    paths = []
    for i, pid in enumerate(s.post_id):
        p = FX / "img" / f"{pid}.png"
        Image.new("RGB", (8, 8), tuple(int(v) for v in rng.integers(0, 255, 3))).save(p)
        paths.append(str(p))
    pd.DataFrame({"post_id": s.post_id, "local_path": paths}).to_csv(FX / "manifest.csv", index=False)
    return s


def synth_labels(d, p_member_by_bin, p_conf_given_member=0.8, noise=0.1, seed=7, p_flag=0.03):
    """Coder labels simulated from a latent truth; B = A with independent noise. Each coder flags a random ~3% of
    items with a BLANK primary label (flag = 1), exactly as the form exports them."""
    key = pd.read_csv(d / "key" / "key.csv")
    rng = np.random.default_rng(seed)
    fams = [H.display_name(f) for f in sorted(pd.read_csv(H.DEV_INDEX).family_id.unique())]
    truth = {}
    for _, r in key.iterrows():
        if r.kind in ("study",):
            if rng.random() < p_member_by_bin[r.bin]:
                truth[r.item_id] = H.display_name(r.raw_argmax) if rng.random() < p_conf_given_member else rng.choice(fams)
            else:
                truth[r.item_id] = "none"
        elif r.kind == "gold":
            truth[r.item_id] = H.display_name(r.gold_family)
    for r in key[key.kind == "repeat"].itertuples():
        truth[r.item_id] = truth[r.repeat_of]
    (d / "labels").mkdir(exist_ok=True)
    for c, s in [("A", 11), ("B", 12)]:
        rr = np.random.default_rng(s)
        rows = []
        for iid, t in truth.items():
            if rr.random() < p_flag:
                rows.append(dict(coder=c, package_id="study", item_id=iid, primary="", second="", confidence="", flag=1, ts=""))
                continue
            lab = t if rr.random() > noise else rr.choice(["none", "uncertain"] + fams)
            rows.append(dict(coder=c, package_id="study", item_id=iid, primary=lab, second="", confidence=2, flag=0, ts=""))
        pd.DataFrame(rows).to_csv(d / "labels" / f"coder_{c}.csv", index=False)
    a = pd.read_csv(d / "labels" / "coder_A.csv", keep_default_na=False, dtype=str).set_index("item_id")
    b = pd.read_csv(d / "labels" / "coder_B.csv", keep_default_na=False, dtype=str).set_index("item_id")
    dis = [i for i in a.index if a.flag[i] == "0" and b.flag[i] == "0" and a.primary[i] != b.primary[i]
           and i in key[key.kind == "study"].item_id.values]
    pd.DataFrame({"item_id": dis, "primary": ["uncertain"] * len(dis)}).to_csv(d / "labels" / "adjudicated.csv", index=False)
    return truth


# ------------------------------------------------------------------ tests
def main():
    OUT.mkdir(exist_ok=True)
    # reference sheet from real DEV (instrument material; no Reddit post involved), built OUTSIDE Drive;
    # it must reproduce the Drive deliverable (analysis/human_validation/reference/) exactly
    ref_dir = OUT / "reference_build"
    subprocess.run([sys.executable, str(H.HV / "make_reference_sheet.py"), "--out", str(ref_dir)], check=True,
                   cwd=OUT, env=ENV)
    drv = H.HV / "reference"
    check("reference_build_matches_drive_deliverable", all(
        (drv / f).exists() and (ref_dir / f).read_bytes() == (drv / f).read_bytes()
        for f in ["reference_manifest.csv", "gold_items.csv", "practice_dev.csv", "reference_sheet.html",
                  "reference_assets.csv"]))
    ref = pd.read_csv(ref_dir / "reference_manifest.csv")
    gold = pd.read_csv(ref_dir / "gold_items.csv")
    prac = pd.read_csv(ref_dir / "practice_dev.csv")
    dev, fams, eligible = H.load_dev()
    two = [f for f in fams if f not in eligible]
    check("ref_31_families", ref.family_id.nunique() == 31)
    check("ref_4_per_eligible_family", all((ref.family_id == f).sum() == 4 for f in eligible))
    check("two_member_family_both_shown", len(two) == 1 and (ref.family_id == two[0]).sum() == 2
          and set(ref[ref.family_id == two[0]].variant_id) == set(dev[dev.family_id == two[0]].variant_id))
    html = (ref_dir / "reference_sheet.html").read_text()
    check("two_member_family_rendered", html.count('<section id="f') == 31 and "2 examples" in html)
    check("reference_no_urls_or_links", "http" not in html and "href" not in html)
    check("ref_gold_disjoint", not (set(ref.variant_id) & set(gold.variant_id)))
    check("practice_disjoint_from_ref_gold", not (set(prac.variant_id) & (set(ref.variant_id) | set(gold.variant_id))))
    check("gold_one_per_eligible_family", sorted(gold.family_id) == sorted(eligible))
    r2 = subprocess.run([sys.executable, str(H.HV / "make_reference_sheet.py"), "--out", str(OUT / "ref_repro")],
                        capture_output=True, cwd=OUT, env=ENV)
    same = all(pd.read_csv(OUT / "ref_repro" / f).equals(pd.read_csv(ref_dir / f))
               for f in ["reference_manifest.csv", "gold_items.csv", "practice_dev.csv"])
    check("reference_selection_deterministic", r2.returncode == 0 and same)

    # synthetic frame -> draw
    frame_raw = make_fixture()
    shutil.copytree(ref_dir, FX / "reference")
    frame = H.load_frame(FX / "scores_all.csv", FX / "clusters.csv", FX / "manifest.csv")
    salt = b"\x01" * 32
    s1, summ = D.build(frame, FX / "reference", FX / "draw", salt=salt)
    s2, _ = D.build(frame, FX / "reference", OUT / "fixture_hv_repro", salt=salt)
    check("six_bin_counts_exact", summ["per_bin"] == {b: 70 for b in H.BINS})
    check("draw_deterministic", s1.post_id.tolist() == s2.post_id.tolist())
    k1 = pd.read_csv(FX / "draw" / "key" / "key.csv")
    k2 = pd.read_csv(OUT / "fixture_hv_repro" / "key" / "key.csv")
    check("keys_and_orders_reproducible_given_salt", k1.equals(k2) and
          (FX / "draw/coder_A/study/index.html").read_text() == (OUT / "fixture_hv_repro/coder_A/study/index.html").read_text())
    check("no_duplicate_post_ids", s1.post_id.is_unique and k1.item_id.is_unique)
    Nh = frame.bin.value_counts()
    check("exact_pi_and_weights", all(np.isclose(r.pi, 70 / Nh[r.bin]) and np.isclose(r.w, Nh[r.bin] / 70)
                                      for r in s1.itertuples()))
    check("kish_exact", np.isclose(summ["kish_neff"]["B5_B6"], 111.95, atol=0.01)
          and np.isclose(summ["kish_neff"]["all"], 378.38, atol=0.01), json.dumps(summ["kish_neff"]))
    # B5-B6 weighted aggregation = population-size weighting, not 50:50
    y = np.r_[np.ones(70) * 0.0, np.ones(70)]  # B5 all 0, B6 all 1
    w = np.r_[np.full(70, 640 / 70), np.full(70, 213 / 70)]
    check("B5B6_weighted_aggregation", np.isclose(A.hajek(y, w), 213 / 853), f"{A.hajek(y, w):.4f} vs 213/853")
    # global cluster resampling never splits a cluster; clusters crossing bins stay whole
    st = k1[k1.kind == "study"].reset_index(drop=True)
    crossing = st.groupby("cluster").bin.nunique()
    rng = np.random.default_rng(0)
    whole = True
    for _ in range(200):
        ix = A.resample_clusters(st.cluster, rng)
        cnt = pd.Series(ix).value_counts()
        for c, g in st.groupby("cluster"):
            m = [cnt.get(i, 0) for i in g.index]
            whole &= len(set(m)) == 1
    check("global_cluster_resampling_no_split", whole, f"{int((crossing > 1).sum())} sampled clusters cross bins")
    # blinding
    pid_set = set(frame_raw.post_id)
    leaks = []
    for c in ["A", "B"]:
        for pk in ["study", "practice1", "practice2"]:
            dd = FX / "draw" / f"coder_{c}" / pk
            txt = (dd / "index.html").read_text()
            pkg = json.loads(txt.split("const PACKAGE = ")[1].split(";\nconst KEY")[0])
            if any(set(it) != {"id", "file"} for it in pkg["items"]):
                leaks.append("extra item fields")
            names = [p.name for p in (dd / "images").iterdir()]
            for tok in list(pid_set)[:0] + ["raw_", "argmax", "bin", "score", "gold", "repeat", "post_id", "fx"]:
                if tok in json.dumps(pkg["items"]) or any(tok in n for n in names):
                    leaks.append(f"{c}/{pk}:{tok}")
            if any(p in txt for p in list(pid_set)):
                leaks.append(f"{c}/{pk}:post_id")
            if any(sl in txt for sl in ["nobody_literally_no_one", "pool_"]):
                leaks.append(f"{c}/{pk}:slug")
    check("opaque_id_blinding", not leaks, "; ".join(leaks[:5]))
    check("key_not_in_packages", not any(p.name in ("key.csv", "salt.hex") for p in (FX / "draw").rglob("coder_*/**/*")))
    # repeat placement
    ok_rep = True
    for c in ["A", "B"]:
        txt = (FX / "draw" / f"coder_{c}" / "study" / "index.html").read_text()
        pkg = json.loads(txt.split("const PACKAGE = ")[1].split(";\nconst KEY")[0])
        pos = {it["id"]: i for i, it in enumerate(pkg["items"])}
        ok_rep &= len(pkg["items"]) == 470
        for r in k1[k1.kind == "repeat"].itertuples():
            gap = pos[r.item_id] - pos[r.repeat_of]
            ok_rep &= gap >= H.REPEAT_MIN_GAP
    check("repeat_items_placed_after_original_min_gap_50", ok_rep)
    check("coder_orders_differ", (FX / "draw/coder_A/study/index.html").read_text() !=
          (FX / "draw/coder_B/study/index.html").read_text())

    # negative freeze-guard tests
    r = subprocess.run([sys.executable, str(H.HV / "draw_sample.py")], capture_output=True, text=True, cwd=OUT, env=ENV)
    check("freeze_guard_refuses_real_draw_without_record", r.returncode != 0 and "REFUSING" in (r.stderr + r.stdout))
    with tempfile.TemporaryDirectory() as td:
        rec = Path(td) / "rec.json"
        items = {rel: H.sha(H.RR / rel) if (H.RR / rel).exists() else "0" * 64 for rel in H.FROZEN_ITEMS}
        rec.write_text(json.dumps({"items": items, "h1_coder_b_recorded": True, "h7_determination_recorded": True}))
        ok_full, why_full = H.freeze_ok(rec)
        items_bad = dict(items)
        items_bad["15_Human_Validation_Protocol.md"] = "f" * 64
        rec.write_text(json.dumps({"items": items_bad, "h1_coder_b_recorded": True, "h7_determination_recorded": True}))
        ok_bad, why_bad = H.freeze_ok(rec)
        rec.write_text(json.dumps({"items": items, "h1_coder_b_recorded": True, "h7_determination_recorded": False}))
        ok_h7, why_h7 = H.freeze_ok(rec)
    check("freeze_guard_refuses_tampered_hash", not ok_bad and "mismatch" in why_bad, why_bad)
    check("freeze_guard_refuses_without_H1_H7", not ok_h7 and "H-7" in why_h7, why_h7)
    check("freeze_guard_accepts_complete_record_(temp)", ok_full, why_full)
    r = subprocess.run([sys.executable, str(H.HV / "draw_sample.py"), "--fixture", str(OUT / "not_a_fx_dir")],
                       capture_output=True, text=True, cwd=OUT, env=ENV)
    check("fixture_mode_refuses_non_fixture_dir", r.returncode != 0 and "REFUSING" in (r.stderr + r.stdout))
    check("real_freeze_record_absent", not H.FREEZE_RECORD.exists())

    # estimator unit checks
    check("kappa_alpha_identity", np.isclose(A.cohen_kappa(list("aabbc"), list("aabbc")), 1)
          and np.isclose(A.kripp_alpha_nominal(list("aabbc"), list("aabbc")), 1))
    check("wilson_known_value", np.allclose(A.wilson(21, 30), [0.5211, 0.8334], atol=1e-3), str(A.wilson(21, 30)))
    xs = np.linspace(0, 1, 400)
    ys = (np.random.default_rng(3).random(400) < 1 / (1 + np.exp(-(-2 + 4 * xs)))).astype(float)
    b = A.weighted_logit(xs, ys, np.ones(400))
    check("weighted_logit_recovers_slope", 2.5 < b[1] < 5.5, str(b))
    check("weighted_logit_weight_scale_invariant", np.allclose(A.weighted_logit(xs, ys, np.full(400, 7.0)), b))

    # §10 rule engine (order R4 -> R2 -> R1 -> R3; R0 global; conservative combination)
    def ep(b6, member, none56):
        return {"per_bin": {"B6": {"confirmed": b6}}, "member_share_w": member, "B5B6_none_share_w": none56}
    cases = [(ep(0.2, 0.1, 0.6), {"slope": [-0.1, 0.5], "B6_conf": [0.1, 0.3]}, "R4"),
             (ep(0.8, 0.5, 0.1), {"slope": [0.5, 2.0], "B6_conf": [0.6, 0.9]}, "R2"),
             (ep(0.8, 0.5, 0.1), {"slope": [0.5, 2.0], "B6_conf": [0.45, 0.9]}, "R3"),
             (ep(0.3, 0.2, 0.3), {"slope": [0.1, 1.0], "B6_conf": [0.2, 0.4]}, "R1"),
             (ep(0.6, 0.4, 0.3), {"slope": [0.1, 1.0], "B6_conf": [0.4, 0.7]}, "R3"),
             (ep(0.8, 0.1, 0.55), {"slope": [-0.2, 1.0], "B6_conf": [0.6, 0.9]}, "R4")]
    check("rule_engine_cases", all(A.rule_for(e, c) == want for e, c, want in cases),
          str([A.rule_for(e, c) for e, c, _ in cases]))
    check("R0_gate", A.r0_gate(0.39, 0.9, 0.9) and A.r0_gate(0.6, 0.69, 0.9) and not A.r0_gate(0.4, 0.7, 0.7))
    check("conservative_combination", A.combine("R2", "R3") == "R3" and A.combine("R4", "R1") == "R4"
          and A.combine("R1", "R2") == "R1")

    # end-to-end on synthetic labels
    synth_labels(FX / "draw", {"B1": 0.1, "B2": 0.15, "B3": 0.2, "B4": 0.3, "B5": 0.5, "B6": 0.8})
    res = A.run(FX / "draw", B=200)
    check("end_to_end_runs", res["rule_final"] in {"R0", "R1", "R2", "R3", "R4"}, res["rule_final"])
    ep_adj = res["label_sets"]["adj"]["endpoints"]
    _k = pd.read_csv(FX / "draw" / "key" / "key.csv", dtype={"item_id": str})
    _a = pd.read_csv(FX / "draw" / "labels" / "coder_A.csv", keep_default_na=False, dtype=str).set_index("item_id")
    _b = pd.read_csv(FX / "draw" / "labels" / "coder_B.csv", keep_default_na=False, dtype=str).set_index("item_id")
    _s = _k[(_k.kind == "study") & _k.bin.isin(["B5", "B6"])]
    _keep = [(_a.flag[i] == "0") and (_b.flag[i] == "0") for i in _s.item_id]
    _exp = A.kish(_s.w.to_numpy(float)[np.array(_keep)])
    check("end_to_end_kish_matches_design_on_nonflagged_items", np.isclose(ep_adj["kish"]["B5_B6"], _exp)
          , f'{ep_adj["kish"]["B5_B6"]:.2f} vs expected {_exp:.2f} (design 111.95 before flags)')
    adj = FX / "draw" / "labels" / "adjudicated.csv"
    bak = adj.read_text()
    pd.read_csv(adj).iloc[1:].to_csv(adj, index=False)
    try:
        A.run(FX / "draw", B=10)
        refused = False
    except SystemExit as e:
        refused = "adjudicated" in str(e)
    adj.write_text(bak)
    check("missing_adjudication_refused", refused)
    uncertain_not_none = A.classify("uncertain", 0, "X") == "uncertain" and A.classify("none", 0, "X") == "none"
    check("uncertain_never_counted_as_none", uncertain_not_none)
    check("flagged_is_missing", A.classify("X", 1, "X") is None)

    # ---------------------------------------------------------- v0.6 repairs (GPT implementation audit)
    D_ = FX / "draw"
    key = pd.read_csv(D_ / "key" / "key.csv", dtype={"item_id": str, "repeat_of": str})
    la = pd.read_csv(D_ / "labels" / "coder_A.csv", keep_default_na=False, dtype=str).set_index("item_id")
    lb = pd.read_csv(D_ / "labels" / "coder_B.csv", keep_default_na=False, dtype=str).set_index("item_id")
    stud = key[key.kind == "study"].item_id
    fa, fb = (la.flag.reindex(stud) == "1"), (lb.flag.reindex(stud) == "1")
    blank_flagged = int(((la.primary == "") & (la.flag == "1")).sum() + ((lb.primary == "") & (lb.flag == "1")).sum())
    ls_ = res["label_sets"]
    check("flag_roundtrip_blank_primaries_in_csv", blank_flagged > 0, f"{blank_flagged} flagged rows with blank primary")
    check("flag_roundtrip_adjudicated_missing_if_either_flagged",
          ls_["adj"]["endpoints"]["n_missing_flagged"] == int((fa.values | fb.values).sum()),
          f'{ls_["adj"]["endpoints"]["n_missing_flagged"]} vs {int((fa.values | fb.values).sum())}')
    check("flag_roundtrip_coderB_missing_if_B_flagged", ls_["B"]["endpoints"]["n_missing_flagged"] == int(fb.sum()))
    check("reliability_excludes_flagged_items", res["reliability"]["n_flagged_excluded"] == int((fa.values | fb.values).sum()))
    rep_ok = all(res["controls"][c]["repeat_n"] + res["controls"][c]["repeat_excluded_flagged"] == 20 for c in "AB")
    check("repeat_agreement_treats_flagged_as_missing", rep_ok)
    # blank primary without a flag is refused
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        bad = Path(td) / "coder_X.csv"
        pd.DataFrame([dict(item_id="abc", primary="", flag=0)]).to_csv(bad, index=False)
        try:
            A.read_coder_csv(bad, A.valid_label_set())
            refused_blank = False
        except SystemExit as e:
            refused_blank = "blank primary" in str(e)
        good = Path(td) / "coder_Y.csv"
        pd.DataFrame([dict(item_id="abc", primary="", flag=1)]).to_csv(good, index=False)
        accepted_flagged = A.read_coder_csv(good, A.valid_label_set()).primary.iloc[0] == ""
    check("blank_primary_without_flag_refused", refused_blank)
    check("blank_primary_with_flag_accepted", accepted_flagged)
    # every §8 endpoint present, with intervals, for both label sets
    need_ci = [f"bin_{h}_{c}" for h in H.BINS for c in A.CATS] + [f"bin_{h}_confirmed_excl_uncertain" for h in H.BINS] + \
              [f"pop_{c}_w" for c in A.CATS] + ["pop_member_w", "B5B6_none_w", "B5B6_none_unw", "B5B6_other_among_members_w",
                                                "B1B2_member_w", "B1B2_member_unw", "slope_w"]
    for v in ["adj", "B"]:
        ci, epv = ls_[v]["ci"], ls_[v]["endpoints"]
        ok_ci = all(k in ci and len(ci[k]) == 2 for k in need_ci)
        ok_ep = all(k in epv for k in ["per_bin", "population_w", "B5B6_none_share_w", "B5B6_none_share_unw",
                                       "B5B6_other_family_among_members_w", "B5B6_other_family_among_members_unw",
                                       "B1B2_member_share_w", "B1B2_member_share_unw", "B1B2_member_families",
                                       "none_score_distribution", "nomination_family_composition",
                                       "community_prevalence_w_descriptive", "kish", "trend_beta", "trend_beta_unweighted"])
        ok_w = all("wilson" in epv["per_bin"][h] and all(c in epv["per_bin"][h]["wilson"] for c in A.CATS) for h in H.BINS)
        ok_band = len(ls_[v]["calibration_band"]["lo"]) == 50
        check(f"all_section8_endpoints_with_intervals_{v}", ok_ci and ok_ep and ok_w and ok_band)
    comp_flag = all(isinstance(r["sparse_n_lt_10"], bool) for h in H.BINS
                    for r in ls_["adj"]["endpoints"]["nomination_family_composition"][h].values())
    check("nomination_family_composition_flags_sparse", comp_flag)
    check("community_prevalence_present", len(ls_["adj"]["endpoints"]["community_prevalence_w_descriptive"]) >= 5)
    relk = ["kappa_full", "kappa_binary", "alpha_nominal", "raw_agreement", "specific_positive", "specific_negative"]
    check("reliability_items_1to5_with_cluster_bootstrap_CIs",
          all(k in res["reliability"]["point"] and len(res["reliability"]["ci"][k]) == 2 for k in relk))
    # instructions and the frozen reference sheet ship in every package
    frozen_ref = (H.HV / "reference" / "reference_sheet.html").read_bytes()
    n_assets = len(pd.read_csv(H.HV / "reference" / "reference_assets.csv"))
    import html as _h
    ok_instr, ok_ref = True, True
    for c in ["A", "B"]:
        for pk in ["study", "practice1", "practice2"]:
            dd = D_ / f"coder_{c}" / pk
            txt = (dd / "index.html").read_text()
            ok_instr &= all(_h.escape(t) in txt for t in H.CODER_INSTRUCTIONS) and "__INSTRUCTIONS__" not in txt
            ok_ref &= (dd / "reference" / "reference_sheet.html").read_bytes() == frozen_ref and \
                len(list((dd / "reference" / "img").iterdir())) == n_assets
    check("frozen_instructions_in_every_package", ok_instr)
    check("frozen_reference_sheet_shipped_in_every_package", ok_ref)
    # freeze guard verifies the actual stimulus (tampered reference asset -> refused), on a temp copy of the tree
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        root = Path(td) / "RR"
        for rel in H.FROZEN_ITEMS:
            dst = (root / rel).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy((H.RR / rel).resolve(), dst)
        shutil.copytree(H.HV / "reference" / "img", root / H.REF_DIR_REL / "img")
        items = {rel: H.sha((root / rel).resolve()) for rel in H.FROZEN_ITEMS}
        rec = Path(td) / "rec.json"
        rec.write_text(json.dumps({"items": items, "h1_coder_b_recorded": True, "h7_determination_recorded": True}))
        ok_copy, _ = H.freeze_ok(rec, root=root)
        victim = sorted((root / H.REF_DIR_REL / "img").iterdir())[0]
        victim.write_bytes(victim.read_bytes() + b"x")
        ok_t, why_t = H.freeze_ok(rec, root=root)
        victim.unlink()
        ok_m, why_m = H.freeze_ok(rec, root=root)
    check("freeze_guard_accepts_untampered_stimulus_copy", ok_copy)
    check("freeze_guard_refuses_tampered_reference_asset", (not ok_t) and "reference asset" in why_t, why_t)
    check("freeze_guard_refuses_missing_reference_asset", (not ok_m) and "differ" in why_m, why_m)
    check("qualify_and_stimulus_in_frozen_items", all(x in H.FROZEN_ITEMS for x in [
        "analysis/human_validation/qualify.py", "analysis/human_validation/reference/reference_sheet.html",
        "analysis/human_validation/reference/reference_assets.csv"]))
    # qualification gate
    import qualify as Q
    pk1 = key[(key.kind == "practice_dev") & (key.practice_round == 1)]
    pk2 = key[(key.kind == "practice_dev") & (key.practice_round == 2)]

    def resp(pk, n_right, flag_first=False):
        rows = []
        for j, r in enumerate(pk.itertuples()):
            right = j < n_right
            if flag_first and j == 0:
                rows.append(dict(item_id=r.item_id, primary="", flag=1))
            else:
                rows.append(dict(item_id=r.item_id, primary=H.display_name(r.gold_family) if right else "uncertain", flag=0))
        return pd.DataFrame(rows)
    n9, _ = Q.score(key, resp(pk1, 9), 1)
    n9f, _ = Q.score(key, resp(pk1, 9, flag_first=True), 1)
    n5, _ = Q.score(key, resp(pk1, 5), 1)
    v1 = Q.verdict(n5, 1)
    n8b, _ = Q.score(key, resp(pk2, 8), 2)
    n6b, _ = Q.score(key, resp(pk2, 6), 2)
    prior = {"verdict": v1}
    try:
        Q.verdict(n8b, 2, None)
        r2_without_prior = False
    except SystemExit:
        r2_without_prior = True
    check("qualify_pass_8plus", n9 == 9 and Q.verdict(n9, 1) == "PASS")
    check("qualify_flagged_counts_incorrect", n9f == 8)
    check("qualify_fail_then_pass", v1 == "FAIL_ROUND1_TAKE_ROUND2" and Q.verdict(n8b, 2, prior) == "PASS")
    check("qualify_fail_then_fail_replace", Q.verdict(n6b, 2, prior) == "REPLACE")
    check("qualify_round2_requires_round1_fail", r2_without_prior)

    # ---------------- 15 v0.7 guards (GPT final implementation audit)
    lab_dir = FX / "draw" / "labels"
    keyv = pd.read_csv(FX / "draw" / "key" / "key.csv", dtype=str)

    def refused_with(mutate_path, mutate, token):
        bakx = mutate_path.read_text()
        try:
            mutate()
            try:
                A.run(FX / "draw", B=5)
                return False, "accepted"
            except SystemExit as e:
                return (token in str(e)), str(e)[:120]
        finally:
            mutate_path.write_text(bakx)

    adjp = lab_dir / "adjudicated.csv"
    adjdf = pd.read_csv(adjp, dtype=str, keep_default_na=False)
    ca = pd.read_csv(lab_dir / "coder_A.csv", dtype=str, keep_default_na=False).set_index("item_id")
    cb = pd.read_csv(lab_dir / "coder_B.csv", dtype=str, keep_default_na=False).set_index("item_id")
    stud_ids = keyv[keyv.kind == "study"].item_id.tolist()
    agreed_id = next(i for i in stud_ids if ca.flag[i] == "0" and cb.flag[i] == "0" and ca.primary[i] == cb.primary[i])
    flagged_id = next(i for i in stud_ids if ca.flag[i] == "1" or cb.flag[i] == "1")
    cases = {
        "missing": (lambda: adjdf.iloc[1:].to_csv(adjp, index=False), "lack"),
        "extra_agreed": (lambda: pd.concat([adjdf, pd.DataFrame([{"item_id": agreed_id, "primary": "none"}])])
                         .to_csv(adjp, index=False), "extra"),
        "duplicate": (lambda: pd.concat([adjdf, adjdf.iloc[:1]]).to_csv(adjp, index=False), "duplicate"),
        "invalid_label": (lambda: adjdf.assign(primary=["Not A Family"] + list(adjdf.primary[1:])).to_csv(adjp, index=False),
                          "invalid"),
        "flagged_included": (lambda: pd.concat([adjdf, pd.DataFrame([{"item_id": flagged_id, "primary": "none"}])])
                             .to_csv(adjp, index=False), "extra"),
    }
    for nm, (fn, tok) in cases.items():
        ok, msg = refused_with(adjp, fn, tok)
        check(f"guard1_adjudication_{nm}_refused", ok, msg)
    cap = lab_dir / "coder_A.csv"
    cadf = pd.read_csv(cap, dtype=str, keep_default_na=False)
    ccases = {
        "469_rows": (lambda: cadf.iloc[1:].to_csv(cap, index=False), "incomplete"),
        "duplicate": (lambda: pd.concat([cadf, cadf.iloc[:1]]).to_csv(cap, index=False), "duplicate"),
        "unknown_id": (lambda: pd.concat([cadf, cadf.iloc[:1].assign(item_id="zzzzzzzzzz")]).to_csv(cap, index=False),
                       "unknown"),
        "wrong_package": (lambda: cadf.assign(package_id=["practice1"] + list(cadf.package_id[1:])).to_csv(cap, index=False),
                          "package_id"),
        # GPT pre-freeze schema guards: required coder column; binary flag
        "missing_coder_column": (lambda: cadf.drop(columns=["coder"]).to_csv(cap, index=False), "missing coder column"),
        "wrong_coder_value": (lambda: cadf.assign(coder=["B"] + list(cadf.coder[1:])).to_csv(cap, index=False),
                              "does not match coder"),
        "flag_2": (lambda: cadf.assign(flag=["2"] + list(cadf.flag[1:])).to_csv(cap, index=False), "flag must be 0 or 1"),
        "flag_minus1": (lambda: cadf.assign(flag=["-1"] + list(cadf.flag[1:])).to_csv(cap, index=False), "flag must be 0 or 1"),
        "missing_flag_column": (lambda: cadf.drop(columns=["flag"]).to_csv(cap, index=False), "missing flag column"),
    }
    for nm, (fn, tok) in ccases.items():
        ok, msg = refused_with(cap, fn, tok)
        check(f"guard2_coder_file_{nm}_refused", ok, msg)
    check("guard2_complete_file_has_470_rows", len(cadf) == 470 and set(cadf.item_id) ==
          set(keyv[keyv.kind.isin(["study", "gold", "repeat"])].item_id))
    k_nan = A.cohen_kappa([True] * 6, [True] * 6)
    why = A.r0_reasons(k_nan, 0.95, 0.95)
    check("guard3_undefined_kappa_fails_R0", (not np.isfinite(k_nan)) and A.r0_gate(k_nan, 0.95, 0.95)
          and any("undefined" in w for w in why), str(why))
    check("guard3_R0_reasons_recorded_in_results", "R0_reasons" in A.run(FX / "draw", B=5))
    rk = pd.DataFrame([dict(item_id=f"r{i}", kind="repeat", repeat_of=f"o{i}", gold_family="") for i in range(3)])
    la_ = pd.DataFrame(dict(item_id=["o0", "o1", "o2", "r0", "r1", "r2"], primary=["none", "", "none", "none", "none", "uncertain"],
                            flag=[0, 1, 0, 0, 0, 0])).set_index("item_id")
    lb_ = la_.assign(primary=["none", "none", "none", "none", "none", "none"], flag=0)
    ctl_ = A.controls(rk, {"A": la_, "B": lb_})
    check("guard4_repeat_denominator_excludes_flagged_pairs",
          ctl_["A"]["repeat_n"] == 2 and ctl_["A"]["repeat_same"] == 1 and ctl_["B"]["repeat_n"] == 3
          and np.allclose(ctl_["A"]["repeat_wilson"], A.wilson(1, 2)), json.dumps({c: [ctl_[c]["repeat_same"], ctl_[c]["repeat_n"]] for c in "AB"}))
    allpass = all(v["pass"] for v in REPORT.values())
    rep = json.dumps(dict(all_pass=allpass, n=len(REPORT), tests=REPORT), indent=2)
    (OUT / "fixture_test_report.json").write_text(rep)
    REPORT_DELIVERABLE.write_text(rep)
    print(f"{sum(v['pass'] for v in REPORT.values())}/{len(REPORT)} PASS")
    sys.exit(0 if allpass else 1)


if __name__ == "__main__":
    main()
