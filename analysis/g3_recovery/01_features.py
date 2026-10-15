#!/usr/bin/env python3
"""G3 step 0: build all corpus features once (680 variants) for the recovery engine.

Saves out/features.npz with: ids, fam, dense (MSAE, 680x6144), clip_openai (unnormalized),
clip_laion, phash bits (64), and out/ocr_texts.json. Everything is in the canonical corpus order
used by fr_25 / RUN_m2m4_full (embeddings_openai id order restricted to the cohort).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from src.analysis.clip_embed import load_npz  # noqa: E402
from src.analysis import surface_baselines as SB  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    cohort, canon, vec, fam = C.load_cohort()
    dense, _, sae = C.sae_dense(canon, vec)
    lai = load_npz(C.EMB_LAION)
    lpos = {v: i for i, v in enumerate(lai["ids"])}
    lvec = np.vstack([lai["vectors"][lpos[v]] for v in canon])
    pid, bits = SB.phash_vectors([{"variant_id": v, "image_path": cohort[v]["image_path"]} for v in canon])
    assert pid == canon, "pHash order/coverage mismatch"
    ocrm = {r["variant_id"]: r for r in csv.DictReader(C.OCR_TEXT.open(newline=""))}
    texts = [ocrm.get(v, {}).get("ocr_text", "") for v in canon]
    np.savez_compressed(OUT / "features.npz", ids=np.array(canon, object), fam=np.array(fam, object), dense=dense,
                        clip_openai=vec, clip_laion=lvec, phash_bits=bits.astype(np.uint8))
    (OUT / "ocr_texts.json").write_text(json.dumps(texts))
    print(f"[features] n={len(canon)} families={len(set(fam))} dense={dense.shape} latent={sae.latent_dim}")


if __name__ == "__main__":
    main()
