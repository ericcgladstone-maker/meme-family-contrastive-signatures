#!/usr/bin/env python3
"""G1 step 2: embed ALL 4,263 denominator images (cached + fetched) with OpenAI CLIP ViT-L/14.

Corrects AI-9: the submitted RUN_downstream_embed_score.py kept only manifest rows with
status == "cached", silently dropping every "fetched" image (r/historymemes, r/shitposting,
r/whenthe). Here both statuses are used.

Identity check A: for cached posts, compare with the submitted 2,543-post checkpoint.
(Identity check B, vs. the stored fr_36 fit, runs in 02_decomposition.py.)
Output: out/reddit_denom_embeddings_all.npz, out/embed_log.json
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from src.analysis.clip_embed import embed_image_paths  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    outc = [r for r in csv.DictReader(C.OUTCOMES.open(newline="")) if r["listing_group"] == "denominator"]
    mani = {r["post_id"]: r for r in csv.DictReader(C.MANIFEST.open(newline=""))}
    recs, missing = [], []
    for r in outc:
        m = mani.get(r["post_id"])
        # CORRECTED FILTER: any status with an existing local file (cached OR fetched)
        if m and m["local_path"] and Path(m["local_path"]).exists():
            recs.append({"variant_id": r["post_id"], "image_path": m["local_path"]})
        else:
            missing.append(r["post_id"])
    status = {}
    for r in recs:
        s = mani[r["variant_id"]]["status"]
        status[s] = status.get(s, 0) + 1
    print(f"[embed] denominator={len(outc)} to_embed={len(recs)} missing={len(missing)} by_status={status}", flush=True)
    t = time.time()
    res = embed_image_paths(recs, pretrained_key="openai", device="auto", l2_normalize=False)
    rid, rvec = list(res["ids"]), res["vectors"]
    skipped = res.get("skipped", [])
    np.savez(OUT / "reddit_denom_embeddings_all.npz", ids=np.array(rid, object), vectors=rvec)
    mins = (time.time() - t) / 60
    print(f"[embed] embedded {len(rid)} in {mins:.1f} min; skipped {len(skipped)}", flush=True)

    # Identity check A: cached posts vs submitted checkpoint
    z = np.load(C.SUBSET_CKPT, allow_pickle=True)
    sid, svec = list(z["ids"]), z["vectors"]
    pos = {v: i for i, v in enumerate(rid)}
    common_ids = [v for v in sid if v in pos]
    a = svec[[sid.index(v) for v in common_ids]]
    b = rvec[[pos[v] for v in common_ids]]
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    cos = (an * bn).sum(1)
    check_a = dict(n_common=len(common_ids), min_cosine=float(cos.min()), mean_cosine=float(cos.mean()),
                   max_abs_diff=float(np.abs(a - b).max()))
    print(f"[identity A] {check_a}", flush=True)
    C.write_json(OUT / "embed_log.json", dict(
        n_denominator=len(outc), n_embedded=len(rid), n_missing=len(missing), missing=missing,
        by_status=status, skipped=skipped, minutes=mins, identity_check_A=check_a))


if __name__ == "__main__":
    main()
