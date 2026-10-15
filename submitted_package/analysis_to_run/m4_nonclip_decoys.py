"""
m4_nonclip_decoys.py  --  Repairs issue M4 (adversarial-decoy encoder independence).

The reported adversarial decoys are selected by LAION OpenCLIP and scored by an SAE over
OpenAI CLIP (fr_25:77-78). Two CLIP-family models are NOT fully independent. This script
selects decoys with a genuinely different, NON-CLIP encoder and re-runs the adversarial
discrimination, so the stress test does not share representation structure with the scorer.

Choose one non-CLIP selector:
  * perceptual hash (pHash) nearest neighbours   (already computed; cheapest)
  * a self-supervised vision encoder (e.g. DINOv2) embedding nearest neighbours
For each family: take the 12 nearest non-family neighbours by the non-CLIP selector,
exclude members, ensure cross-family near-dups already removed (fr_12), then score with
the primary SAE representation exactly as fr_25 does, and compute member-vs-decoy AUROC.

Report contrastive PH4 and whole-image CLIP discrimination under the non-CLIP decoys next
to the existing different-data-CLIP result. Expect a somewhat harder test.

Output: nonclip_decoy_results.csv + printed means.

Skeleton (fill selector + scoring to mirror fr_25):
"""
import numpy as np, pandas as pd

def main():
    raise SystemExit(
        "Implement: (1) load family labels (cross_family_corrected_index.csv), "
        "the SAE features (sparse_activations_openai.npy), and a NON-CLIP similarity "
        "matrix (pHash Hamming similarity, or DINOv2 cosine); (2) per family, pick 12 "
        "nearest non-family decoys by the non-CLIP metric; (3) score members and decoys "
        "with the contrastive signature and with CLIP centroid similarity, exactly as "
        "fr_25; (4) AUROC member-vs-decoy per family; (5) print across-family means for "
        "contrastive and CLIP under non-CLIP decoys. Paste into R3 as the stronger stress "
        "test and soften the 'independent encoder' language accordingly."
    )

if __name__ == "__main__":
    main()
