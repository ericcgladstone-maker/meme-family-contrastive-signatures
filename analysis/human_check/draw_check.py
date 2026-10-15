#!/usr/bin/env python3
"""Draw the 60-post sample and build Eric's blinded coding package (15b §3, §5).

REAL MODE refuses to run unless the 15b freeze record verifies (hc_common.freeze_ok) and no draw exists yet.
FIXTURE MODE (--fixture DIR, DIR name must contain 'fixture') runs the identical code on a synthetic frame.

Outputs:
  KEY (--key-dir; real default analysis/human_check/draw/, in Drive): key.csv (item_id -> post_id, kind, bin, N_h,
      raw_argmax, raw_max_fit, subreddit, repeat_of), salt.hex, draw_summary.json (counts only). Never opened before
      the labels are frozen (15b §6).
  PACKAGE (--package-dir; real default <non-Drive>/human_check/package_E/): index.html (the coding form with an
      embedded list of opaque IDs in presentation order), images/<ID>.<ext>, reference/ (the frozen 31-family sheet).
      The package contains no key, score, bin, nomination, community, title or engagement information.
"""
import argparse
import hashlib
import html
import json
import secrets
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import hc_common as C
H = C.H


def draw(frame):
    """Stratified SRS: one generator; per bin (B1..B6) sort by post_id, choose N_PER_BIN positions without replacement."""
    rng = np.random.default_rng(C.SEED_DRAW)
    parts = []
    for b in C.BINS:
        pop = frame[frame.bin == b].sort_values("post_id", kind="mergesort").reset_index(drop=True)
        if len(pop) < C.N_PER_BIN:
            raise SystemExit(f"{b} has only {len(pop)} posts")
        pick = pop.iloc[rng.choice(len(pop), C.N_PER_BIN, replace=False)].copy()
        pick["N_h"] = len(pop)
        parts.append(pick)
    return pd.concat(parts, ignore_index=True)


def repeats(sample):
    srt = sample.sort_values("post_id", kind="mergesort").reset_index(drop=True)
    idx = np.random.default_rng(C.SEED_REPEAT).choice(len(srt), C.N_REPEATS, replace=False)
    return sorted(srt.post_id.iloc[idx].tolist())


def opaque(salt, internal):
    return hashlib.sha256(salt + internal.encode()).hexdigest()[:10]


def presentation_order(unique_ids, repeat_pairs):
    """Permute the 60 unique items; then insert each repeat copy at least REPEAT_MIN_GAP positions after its original
    (originals first moved out of the last REPEAT_MIN_GAP positions; later insertions only widen gaps)."""
    rng = np.random.default_rng(C.SEED_ORDER)
    order = [unique_ids[i] for i in rng.permutation(len(unique_ids))]
    n, gap = len(order), C.REPEAT_MIN_GAP
    originals = [o for o, _ in repeat_pairs]
    while True:
        bad = [o for o in originals if order.index(o) > n - gap - 1]
        if not bad:
            break
        order.remove(bad[0])
        order.insert(int(rng.integers(0, n - gap)), bad[0])
    for orig, copy in repeat_pairs:
        lo = order.index(orig) + gap
        order.insert(int(rng.integers(lo, len(order) + 1)), copy)
    return order


def write_package(pkg_dir, ids, files, ref_dir):
    ok, why = H.verify_reference_assets(ref_dir)
    if not ok:
        raise SystemExit(f"REFUSING: {why}")
    (pkg_dir / "images").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "reference" / "img").mkdir(parents=True, exist_ok=True)
    shutil.copy(ref_dir / "reference_sheet.html", pkg_dir / "reference" / "reference_sheet.html")
    for p in sorted((ref_dir / "img").iterdir()):
        shutil.copy(p, pkg_dir / "reference" / "img" / p.name)
    items = []
    for iid in ids:
        src = Path(files[iid])
        ext = src.suffix.lower() if src.suffix else ".img"
        shutil.copy(src, pkg_dir / "images" / f"{iid}{ext}")
        items.append({"id": iid, "file": f"{iid}{ext}"})
    pkg = {"coder": C.CODER, "package_id": "check", "items": items, "families": C.families()}
    tpl = (C.HC / "coding_form_check.html").read_text()
    instr = "".join(f"<li>{html.escape(t)}</li>" for t in C.CODER_INSTRUCTIONS)
    page = tpl.replace("/*__PACKAGE_JSON__*/null", json.dumps(pkg)).replace("<!--__INSTRUCTIONS__-->", instr)
    (pkg_dir / "index.html").write_text(page)


def build(frame, ref_dir, key_dir, pkg_dir, salt=None):
    for d in (key_dir, pkg_dir):
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"REFUSING: {d} already exists and is not empty (one draw only)")
    key_dir.mkdir(parents=True, exist_ok=True)
    sample = draw(frame)
    reps = repeats(sample)
    salt = salt if salt is not None else secrets.token_bytes(32)
    key, files, pid = [], {}, {}
    for _, r in sample.iterrows():
        iid = opaque(salt, f"study:{r.post_id}")
        pid[r.post_id] = iid
        files[iid] = r.local_path
        key.append(dict(item_id=iid, kind="study", post_id=r.post_id, bin=r.bin, N_h=int(r.N_h),
                        raw_argmax=r.raw_argmax, raw_max_fit=r.raw_max_fit, subreddit=r.subreddit, repeat_of=""))
    pairs = []
    for p in reps:
        r = sample[sample.post_id == p].iloc[0]
        c = opaque(salt, f"repeat:{p}")
        files[c] = r.local_path
        key.append(dict(item_id=c, kind="repeat", post_id=p, bin=r.bin, N_h=int(r.N_h), raw_argmax=r.raw_argmax,
                        raw_max_fit=r.raw_max_fit, subreddit=r.subreddit, repeat_of=pid[p]))
        pairs.append((pid[p], c))
    if len(set(files)) != len(key):
        raise SystemExit("opaque ID collision")
    order = presentation_order([pid[p] for p in sorted(pid)], pairs)
    write_package(pkg_dir, order, files, ref_dir)
    kdf = pd.DataFrame(key)
    kdf["position"] = kdf.item_id.map({iid: k + 1 for k, iid in enumerate(order)})
    kdf.to_csv(key_dir / "key.csv", index=False)
    (key_dir / "salt.hex").write_text(salt.hex())
    summ = dict(n_unique=len(sample), n_repeats=len(reps), n_presentations=len(order),
                per_bin=sample.bin.value_counts().sort_index().to_dict(),
                N_h=sample.groupby("bin").N_h.first().astype(int).to_dict())
    (key_dir / "draw_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", help="fixture directory (name must contain 'fixture')")
    ap.add_argument("--key-dir")
    ap.add_argument("--package-dir")
    a = ap.parse_args()
    if a.fixture:
        fx = Path(a.fixture)
        if "fixture" not in fx.name:
            raise SystemExit("REFUSING: --fixture directory name must contain 'fixture'")
        frame = H.load_frame(fx / "scores_all.csv", fx / "clusters.csv", fx / "manifest.csv")
        ref_dir = fx / "reference" if (fx / "reference").exists() else C.REF_DIR
        key_dir, pkg_dir = Path(a.key_dir or fx / "draw"), Path(a.package_dir or fx / "package_E")
    else:
        ok, why = C.freeze_ok()
        if not ok:
            raise SystemExit(f"REFUSING real draw: {why}")
        frame = H.load_frame(manifest=H.REDDIT_MANIFEST)
        ref_dir = C.REF_DIR
        key_dir = Path(a.key_dir or C.HC / "draw")
        pkg_dir = Path(a.package_dir or C.scratch_dir("human_check") / "package_E")
    print(json.dumps(build(frame, ref_dir, key_dir, pkg_dir)))


if __name__ == "__main__":
    main()
