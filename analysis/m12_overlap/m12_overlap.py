#!/usr/bin/env python3
"""M12 cross-source leakage audit (SPEC.md; 09 F-59). Counts only; refuses unless FREEZE_RECORD.json verifies."""
import hashlib
import json
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
RR = HERE.parent.parent
sys.path.insert(0, str(RR / "analysis" / "human_validation"))
import hv_common as H  # noqa: E402  (frame loader and paths; frozen, read-only)

INPUTS = {"dev_index": H.DEV_INDEX, "reddit_manifest": H.REDDIT_MANIFEST, "frame": H.FRAME}
OUT = HERE / "out" / "m12_overlap.json"
HAMMING_MAX = 6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    rec = json.loads((HERE / "FREEZE_RECORD.json").read_text())
    want = {"SPEC.md": HERE / "SPEC.md", "m12_overlap.py": HERE / "m12_overlap.py", **INPUTS}
    for k, p in want.items():
        if rec["hashes"].get(k) != sha(p):
            raise SystemExit(f"REFUSING: hash mismatch for {k}")


def phash(p):
    return imagehash.phash(Image.open(p).convert("RGB"), hash_size=8)


def main():
    verify()
    if OUT.exists():
        raise SystemExit("REFUSING: output exists (run once)")
    dev = pd.read_csv(H.DEV_INDEX)
    red = H.load_frame(manifest=H.REDDIT_MANIFEST)
    assert len(dev) == 680 and len(red) == 4263, (len(dev), len(red))
    d_sha = [sha(p) for p in dev.image_path]
    r_sha = [sha(p) for p in red.local_path]
    d_ph = np.array([phash(p).hash.flatten() for p in dev.image_path], dtype=bool)
    r_ph = np.array([phash(p).hash.flatten() for p in red.local_path], dtype=bool)
    exact = [(i, j) for j, s in enumerate(r_sha) for i in [k for k, t in enumerate(d_sha) if t == s]]
    dist = (r_ph[:, None, :] != d_ph[None, :, :]).sum(axis=2)  # 4263 x 680 Hamming distances
    near = [(int(i), int(j)) for i, j in zip(*np.nonzero(dist.T <= HAMMING_MAX))]  # dist.T is dev x reddit -> (dev i, reddit j)

    def summ(pairs):
        rj = sorted({j for _, j in pairs}); di = sorted({i for i, _ in pairs})
        return {"reddit_posts_matched": len(rj), "reddit_pct": round(100 * len(rj) / len(red), 3),
                "dev_images_implicated": len(di), "dev_families_implicated": int(dev.iloc[di].family_id.nunique()) if di else 0}
    union = sorted(set(exact) | set(near))
    res = {"n_dev": len(dev), "n_reddit": len(red), "rule_exact": "SHA-256 of file bytes",
           "rule_near": f"imagehash.phash(RGB, hash_size=8), Hamming <= {HAMMING_MAX}",
           "exact": summ(exact), "near": summ(near), "union": summ(union),
           "min_hamming_any_pair": int(dist.min()),
           "pairs_for_traceability_not_inspected": [{"dev_variant": str(dev.iloc[i].variant_id), "dev_family": str(dev.iloc[i].family_id),
                                                     "reddit_post_id": str(red.iloc[j].post_id), "hamming": int(dist[j, i])} for i, j in union]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: res[k] for k in ["exact", "near", "union", "min_hamming_any_pair"]}))


if __name__ == "__main__":
    main()
