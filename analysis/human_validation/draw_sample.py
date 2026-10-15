#!/usr/bin/env python3
"""Draw the 420-post sample and build the blinded coding packages (15 v0.5 §3, §5, §6).

REAL MODE refuses to run unless the joint freeze record verifies and H-1/H-7 are recorded (hv_common.freeze_ok).
FIXTURE MODE (--fixture DIR, DIR name must contain 'fixture') runs the identical code on a synthetic frame.

Outputs under --out:
  key/key.csv            item_id -> source (post_id | variant_id), kind, bin, pi, w, raw_argmax, raw_max_fit, cluster,
                         repeat_of; key/salt.hex. The key is NEVER copied into a coder package.
  coder_A/, coder_B/     study/ (470 items), practice1/ (30 Reddit + 10 DEV), practice2/ (10 DEV): each has index.html
                         (the coding form with an embedded item list of opaque IDs) and images/<ID>.<ext>.
  draw_summary.json      counts per bin, weights, Kish n_eff (no labels exist at this point)
"""
import argparse
import hashlib
import json
import secrets
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import hv_common as H


def draw(frame):
    """Stratified SRS: one generator; per bin (B1..B6) sort by post_id, choose 70 positions without replacement."""
    rng = np.random.default_rng(H.SEED_DRAW)
    parts = []
    for b in H.BINS:
        pop = frame[frame.bin == b].sort_values("post_id", kind="mergesort").reset_index(drop=True)
        if len(pop) < H.N_PER_BIN:
            raise SystemExit(f"{b} has only {len(pop)} posts")
        idx = rng.choice(len(pop), H.N_PER_BIN, replace=False)
        pick = pop.iloc[idx].copy()
        pick["N_h"] = len(pop)
        pick["pi"] = H.N_PER_BIN / len(pop)
        pick["w"] = len(pop) / H.N_PER_BIN
        parts.append(pick)
    return pd.concat(parts, ignore_index=True)


def repeats(sample):
    srt = sample.sort_values("post_id", kind="mergesort").reset_index(drop=True)
    idx = np.random.default_rng(H.SEED_REPEAT).choice(len(srt), H.N_REPEATS, replace=False)
    return sorted(srt.post_id.iloc[idx].tolist())


def practice_reddit(frame, sample):
    rest = frame[~frame.post_id.isin(sample.post_id)].sort_values("post_id", kind="mergesort").reset_index(drop=True)
    idx = np.random.default_rng(H.SEED_PRACTICE_REDDIT).choice(len(rest), H.N_PRACTICE_REDDIT, replace=False)
    return rest.iloc[idx].reset_index(drop=True)


def opaque(salt, internal):
    return hashlib.sha256(salt + internal.encode()).hexdigest()[:10]


def coder_order(unique_ids, repeat_pairs, coder_index):
    """Permute the 450 unique study items, then insert each repeat copy >= REPEAT_MIN_GAP after its original."""
    rng = np.random.default_rng([H.SEED_ORDER, coder_index])
    order = [unique_ids[i] for i in rng.permutation(len(unique_ids))]
    n, gap = len(order), H.REPEAT_MIN_GAP
    originals = [o for o, _ in repeat_pairs]  # post_id order
    # phase 1: no repeat original may sit in the last `gap` positions (relocate uniformly into [0, n - gap))
    while True:
        bad = [o for o in originals if order.index(o) > n - gap - 1]
        if not bad:
            break
        order.remove(bad[0])
        order.insert(int(rng.integers(0, n - gap)), bad[0])
    # phase 2: insert each copy uniformly in [pos(original) + gap, current length]; later insertions only widen gaps
    for orig, copy in repeat_pairs:
        lo = order.index(orig) + gap
        order.insert(int(rng.integers(lo, len(order) + 1)), copy)
    return order


def instructions_html():
    import html as _h
    return "".join(f"<li>{_h.escape(t)}</li>" for t in H.CODER_INSTRUCTIONS)


def write_package(out, coder, package_id, ids, files, families, ref_dir):
    d = out / f"coder_{coder}" / package_id
    (d / "images").mkdir(parents=True, exist_ok=True)
    # the frozen offline reference sheet ships inside every package (15 §4-§5); verified before copying
    ok, why = H.verify_reference_assets(ref_dir)
    if not ok:
        raise SystemExit(f"REFUSING: {why}")
    (d / "reference" / "img").mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_dir / "reference_sheet.html", d / "reference" / "reference_sheet.html")
    for p in sorted((ref_dir / "img").iterdir()):
        shutil.copy(p, d / "reference" / "img" / p.name)
    items = []
    for iid in ids:
        src = Path(files[iid])
        ext = src.suffix.lower() if src.suffix else ".img"
        shutil.copy(src, d / "images" / f"{iid}{ext}")
        items.append({"id": iid, "file": f"{iid}{ext}"})
    pkg = {"coder": coder, "package_id": package_id, "items": items, "families": families}
    tpl = (H.HV / "coding_form_template.html").read_text()
    page = tpl.replace("/*__PACKAGE_JSON__*/null", json.dumps(pkg)).replace("<!--__INSTRUCTIONS__-->", instructions_html())
    (d / "index.html").write_text(page)


def build(frame, ref_dir, out, salt=None):
    out.mkdir(parents=True, exist_ok=True)
    sample = draw(frame)
    reps = repeats(sample)
    prac_r = practice_reddit(frame, sample)
    gold = pd.read_csv(ref_dir / "gold_items.csv")
    prac_d = pd.read_csv(ref_dir / "practice_dev.csv")
    ref = pd.read_csv(ref_dir / "reference_manifest.csv")
    families = [H.display_name(f) for f in sorted(ref.family_id.unique())]
    salt = salt if salt is not None else secrets.token_bytes(32)
    key, files = [], {}

    def add(internal, kind, src_id, path, **kw):
        iid = opaque(salt, internal)
        if iid in files:
            raise SystemExit("opaque ID collision")
        files[iid] = path
        key.append(dict(item_id=iid, kind=kind, source_id=src_id, **kw))
        return iid

    pid = {}
    for _, r in sample.iterrows():
        pid[r.post_id] = add(f"study:{r.post_id}", "study", r.post_id, r.local_path, bin=r.bin, pi=r.pi, w=r.w,
                             raw_argmax=r.raw_argmax, raw_max_fit=r.raw_max_fit, cluster=r[H.CLUSTER_COL],
                             subreddit=r.subreddit)
    gids = [add(f"gold:{r.variant_id}", "gold", r.variant_id, r.image_path, gold_family=r.family_id)
            for _, r in gold.iterrows()]
    pairs = []
    for p in reps:
        r = sample[sample.post_id == p].iloc[0]
        c = add(f"repeat:{p}", "repeat", p, r.local_path, repeat_of=pid[p], bin=r.bin, raw_argmax=r.raw_argmax,
                subreddit=r.subreddit)
        pairs.append((pid[p], c))
    p1 = [add(f"practice_reddit:{r.post_id}", "practice_reddit", r.post_id, r.local_path) for _, r in prac_r.iterrows()]
    p1 += [add(f"practice_dev:{r.variant_id}", "practice_dev", r.variant_id, r.image_path, gold_family=r.family_id,
               practice_round=1) for _, r in prac_d[prac_d["round"] == 1].iterrows()]
    p2 = [add(f"practice_dev:{r.variant_id}", "practice_dev", r.variant_id, r.image_path, gold_family=r.family_id,
              practice_round=2) for _, r in prac_d[prac_d["round"] == 2].iterrows()]
    unique_ids = [pid[p] for p in sorted(pid)] + gids
    for ci, coder in enumerate(["A", "B"]):
        write_package(out, coder, "study", coder_order(unique_ids, pairs, ci), files, families, ref_dir)
        o1 = [p1[i] for i in np.random.default_rng([H.SEED_ORDER, ci, 1]).permutation(len(p1))]
        write_package(out, coder, "practice1", o1, files, families, ref_dir)
        write_package(out, coder, "practice2", p2, files, families, ref_dir)
    (out / "key").mkdir(exist_ok=True)
    pd.DataFrame(key).to_csv(out / "key" / "key.csv", index=False)
    (out / "key" / "salt.hex").write_text(salt.hex())
    kish = {}
    for name, bins in [("B5_B6", ["B5", "B6"]), ("B1_B2", ["B1", "B2"]), ("all", H.BINS)]:
        w = sample[sample.bin.isin(bins)].w.to_numpy()
        kish[name] = float(w.sum() ** 2 / (w ** 2).sum())
    summ = dict(n=len(sample), per_bin=sample.bin.value_counts().sort_index().to_dict(),
                N_h=sample.groupby("bin").N_h.first().to_dict(), kish_neff=kish, n_repeats=len(reps),
                n_gold=len(gids), n_practice1=len(p1), n_practice2=len(p2))
    (out / "draw_summary.json").write_text(json.dumps(summ, indent=2))
    return sample, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", help="fixture directory (name must contain 'fixture'): uses its frame/clusters/manifest")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.fixture:
        fx = Path(a.fixture)
        if "fixture" not in fx.name:
            raise SystemExit("REFUSING: --fixture directory name must contain 'fixture'")
        frame = H.load_frame(fx / "scores_all.csv", fx / "clusters.csv", fx / "manifest.csv")
        ref_dir, out = fx / "reference", Path(a.out or fx / "draw")
    else:
        ok, why = H.freeze_ok()
        if not ok:
            raise SystemExit(f"REFUSING real draw: {why}")
        frame = H.load_frame(manifest=H.REDDIT_MANIFEST)
        ref_dir, out = H.HV / "reference", Path(a.out or H.HV / "draw")
    _, summ = build(frame, ref_dir, out)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
