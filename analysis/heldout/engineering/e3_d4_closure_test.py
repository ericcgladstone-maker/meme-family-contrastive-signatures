#!/usr/bin/env python3
"""Synthetic tests (no network, no held-out data) for 14 v0.5 D4 and dynamic closure.

T1  A/B/C share image x: x removed from ALL THREE (the v0.4 mutable-set bug kept it in C).
T2  Order invariance: every insertion order of 5 families yields identical statuses and retained sets.
T3  >50% rule on immutable pre-D4 denominators -> label_conflict for both families of the pair.
T4  Near-duplicate (pHash Hamming <= 6, different sha) counts as a duplicate relation; Hamming 7 does not.
T5  Dynamic closure: a later C family knocks an H family below MIN via D4 -> H screening resumes;
    final state satisfies H >= H_TARGET, C == C_CAP (or C-queue exhausted), H + C >= NEW_TARGET.
Output: out/e3_d4_closure_test.json
"""
import importlib.util
import itertools
import json
import random
from pathlib import Path

import imagehash
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sr", HERE.parent / "screen_and_retrieve.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)
rng = random.Random(7)


def rand_hash():
    return str(imagehash.ImageHash(np.array([rng.random() < 0.5 for _ in range(64)]).reshape(8, 8)))


def far_hashes(n):
    """n pHashes pairwise > 20 bits apart (so only intended relations exist)."""
    out = []
    while len(out) < n:
        h = rand_hash()
        if all((imagehash.hex_to_hash(h) - imagehash.hex_to_hash(o)) > 20 for o in out):
            out.append(h)
    return out


def flip(hexh, k):
    b = imagehash.hex_to_hash(hexh).hash.flatten().copy()
    for i in range(k):
        b[i] = not b[i]
    return str(imagehash.ImageHash(b.reshape(8, 8)))


def img(tag, ph):
    return (f"sha_{tag}", ph)


def main():
    res = {}
    H = far_hashes(80)
    # T1: families of 12 images each; image x shared (identical sha) by A, B, C
    x = img("x", H[0])
    A = [x] + [img(f"a{i}", H[1 + i]) for i in range(11)]
    B = [x] + [img(f"b{i}", H[12 + i]) for i in range(11)]
    Cf = [x] + [img(f"c{i}", H[23 + i]) for i in range(11)]
    r = sr.d4_resolve({"A": A, "B": B, "C": Cf}, 10)
    t1 = all(0 not in r[f]["retained"] and 0 in r[f]["removed"] for f in "ABC") and all(
        len(r[f]["retained"]) == 11 and r[f]["status"] == "analyzable" for f in "ABC")
    # the v0.4 behaviour (mutable sets, sequential) for contrast
    kept = {"A": list(A), "B": list(B)}
    kept["A"] = [a for a in kept["A"] if a != x]; kept["B"] = [b for b in kept["B"] if b != x]  # A-B pass
    c_keeps_x_under_v04 = x in [c for c in Cf if all(c != a for a in kept["A"] + kept["B"])]
    res["T1_ABC_shared_removed_from_all"] = bool(t1)
    res["T1_v04_mutable_logic_would_keep_x_in_C"] = bool(c_keeps_x_under_v04)
    # T2: order invariance over all 120 orders of 5 families with overlapping relations
    D = [img(f"d{i}", H[34 + i]) for i in range(11)] + [img("xx", H[0])]
    E = [img(f"e{i}", H[45 + i]) for i in range(10)] + [img("near_a1", flip(H[2], 5)), img("b3dup", H[15])]
    fams = {"A": A, "B": B, "C": Cf, "D": D, "E": E}
    ref = None
    ok = True
    for order in itertools.permutations(fams):
        rr = sr.d4_resolve({k: fams[k] for k in order}, 10)
        sig = {k: (rr[k]["status"], tuple(rr[k]["retained"]), tuple(rr[k]["conflicts"])) for k in fams}
        ref = ref or sig
        ok &= (sig == ref)
    res["T2_order_invariant_over_120_orders"] = bool(ok)
    res["T2_statuses"] = {k: v[0] for k, v in ref.items()}
    # T3: >50% rule (F shares 7 of its 12 pre-D4 images with G)
    G = [img(f"g{i}", H[56 + i]) for i in range(12)]
    F = G[:7] + [img(f"f{i}", H[68 + i]) for i in range(5)]
    r3 = sr.d4_resolve({"F": F, "G": G}, 10)
    res["T3_label_conflict_both"] = bool(r3["F"]["status"] == "label_conflict" and r3["G"]["status"] == "label_conflict")
    # T4: Hamming 6 relates, Hamming 7 does not
    P6 = {"P": [img("p0", H[0])] + [img(f"p{i}", H[1 + i]) for i in range(10)],
          "Q": [img("q0", flip(H[0], 6))] + [img(f"q{i}", H[12 + i]) for i in range(10)]}
    P7 = {"P": P6["P"], "Q": [img("q0", flip(H[0], 7))] + P6["Q"][1:]}
    res["T4_hamming6_related"] = bool(0 in sr.d4_resolve(P6, 10)["P"]["removed"])
    res["T4_hamming7_not_related"] = bool(0 not in sr.d4_resolve(P7, 10)["P"]["removed"])
    # T5: dynamic closure with a stubbed screen(); small targets
    sr.H_TARGET, sr.C_CAP, sr.NEW_TARGET, sr.BUDGET = 3, 2, 5, 100
    sr.FREEZE_PARAMS["MIN_VARIANTS"] = 10
    S = object.__new__(sr.Screener)
    S.disp, S.fam, S.screened = [], {}, set()
    S.save = lambda final=False: None
    hh = far_hashes(200)
    pool = iter(range(200))
    fam_imgs = {}

    def make(n):
        return [img(f"u{next(pool)}", None) for _ in range(n)]
    # H1..H4 plain; C1 duplicates 3 images of H1 (knocking H1 from 12 to 9 < 10) -> H must resume
    for name in ["h1", "h2", "h3", "h4", "h5", "c1", "c2"]:
        fam_imgs[name] = [img(f"{name}_{i}", hh[next(pool)]) for i in range(12)]
    fam_imgs["c1"][:3] = fam_imgs["h1"][:3]
    queue = ["h1", "h2", "h3", "c1", "c2", "h4", "h5"]
    token = {"c1": "doge", "c2": "doge"}
    sr.candidate_queue = lambda slugs: queue
    sr.anchor_of = lambda sl: token.get(sl)
    sr.csv.DictReader = lambda fh: [{"slug": q} for q in queue]
    sr.open = lambda *a, **k: None

    def fake_screen(rank, slug, stratum, anchor):
        S.screened.add(slug)
        row = S.record(rank, slug, stratum, anchor, "pending")
        S.fam[slug] = dict(stratum=stratum, anchor=anchor, parent="none",
                           P=tuple({"sha": s, "ph": p} for s, p in fam_imgs[slug]), kept=[], status="analyzable", row=row)
        S.resolve()
        return row
    S.screen = fake_screen
    S.run()
    st = {k: (v["stratum"], v["status"]) for k, v in S.fam.items()}
    h = sum(1 for v in S.fam.values() if v["stratum"] == "H" and v["status"] == "analyzable")
    c = sum(1 for v in S.fam.values() if v["stratum"] == "C" and v["status"] == "analyzable")
    res["T5_final_statuses"] = st
    res["T5_h1_knocked_out_by_c1"] = st.get("h1", ("", ""))[1] == "too_small"
    c_queue_exhausted = all(q in S.screened for q in queue if token.get(q))
    res["T5_c2_reached_in_H_sequence_stays_H"] = S.fam["c2"]["stratum"] == "H"
    res["T5_H_resumed_and_closed"] = bool(h >= 3 and (c == 2 or c_queue_exhausted) and h + c >= 5 and "h4" in S.fam)
    res["ALL_PASSED"] = bool(t1 and ok and res["T3_label_conflict_both"] and res["T4_hamming6_related"]
                             and res["T4_hamming7_not_related"] and res["T5_h1_knocked_out_by_c1"]
                             and res["T5_c2_reached_in_H_sequence_stays_H"]
                             and res["T5_H_resumed_and_closed"])
    (HERE / "out" / "e3_d4_closure_test.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
