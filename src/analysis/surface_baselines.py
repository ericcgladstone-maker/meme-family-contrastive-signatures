"""Surface-similarity baselines for the stress diagnostics.

These are deliberately simple "does surface resemblance alone recover families?"
controls, to contrast against learned (CLIP / SAE-pharmacophore) representations.

- phash_similarity: perceptual-hash (visual near-duplicate) similarity. The
  strongest purely-surface visual baseline available without captions/OCR.

Template-identity and lexical/title baselines are stubs here: KYM Pool variants
carry no clean per-variant template id or caption text on disk yet, so those
baselines are reported as "unavailable" rather than faked. They become available
if/when per-variant OCR or titles are curated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def phash_vectors(records: list[dict[str, Any]], hash_size: int = 8) -> tuple[list[str], np.ndarray]:
    """Compute a binary pHash bit-vector per image. Returns (ids, bits NxB)."""
    import imagehash
    from PIL import Image
    ids, bits = [], []
    for r in records:
        try:
            img = Image.open(Path(r["image_path"])).convert("RGB")
            h = imagehash.phash(img, hash_size=hash_size)
            flat = h.hash.flatten().astype(np.float32)  # 0/1 array, length hash_size^2
            ids.append(r["variant_id"])
            bits.append(flat)
        except Exception:
            continue
    return ids, (np.vstack(bits) if bits else np.zeros((0, hash_size * hash_size), np.float32))


def phash_similarity(records: list[dict[str, Any]], hash_size: int = 8):
    """Similarity = 1 - normalized Hamming distance between pHash bit-vectors."""
    ids, bits = phash_vectors(records, hash_size=hash_size)
    if not ids:
        return ids, np.zeros((0, 0), np.float32)
    n, b = bits.shape
    # Hamming distance via matrix ops: agreements = bits@bits.T + (1-bits)@(1-bits).T
    agree = bits @ bits.T + (1.0 - bits) @ (1.0 - bits).T
    sim = agree / float(b)
    np.fill_diagonal(sim, 1.0)
    return ids, sim.astype(np.float64)
