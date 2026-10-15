"""Real OpenCLIP image embeddings.

Loads a frozen OpenCLIP ViT-L/14 encoder and embeds image files into L2-
normalized 768-d vectors. Supports two pretrained tags:
  - 'laion2b_s32b_b82k' (locked_design default; cached locally)
  - 'openai'            (required to match the cached MSAE's training space)

Embeddings are cached to a .npz so re-runs are free. This module is the real
replacement for the smoke-test `embed.py` hash-based stand-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PRETRAINED_TAGS = {
    "laion": "laion2b_s32b_b82k",
    "openai": "openai",
}
# OpenAI CLIP weights require the QuickGELU activation variant; the LAION tag
# uses standard GELU. Using the wrong activation corrupts the embedding space.
MODEL_ARCH_BY_KEY = {
    "laion": "ViT-L-14",
    "openai": "ViT-L-14-quickgelu",
}
MODEL_ARCH = "ViT-L-14"  # default; per-key arch resolved in load_encoder
EMBED_DIM = 768


def pick_device(prefer: str = "auto") -> str:
    import torch
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_encoder(pretrained_key: str = "laion", device: str = "auto"):
    """Return (model, preprocess, device_str). Uses the local HF cache only."""
    import os
    import open_clip
    import torch
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    tag = PRETRAINED_TAGS[pretrained_key]
    arch = MODEL_ARCH_BY_KEY[pretrained_key]
    device = pick_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
    model.eval().to(device)
    return model, preprocess, device


def embed_image_paths(
    records: list[dict[str, Any]],
    pretrained_key: str = "laion",
    device: str = "auto",
    batch_size: int = 16,
    l2_normalize: bool = True,
) -> dict[str, Any]:
    """records: [{variant_id, image_path, ...}]. Returns dict with parallel
    arrays `ids` and `vectors` (float32, N x 768) plus skipped list."""
    from PIL import Image
    import torch

    model, preprocess, device = load_encoder(pretrained_key, device)
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    skipped: list[dict[str, str]] = []

    batch_imgs: list[torch.Tensor] = []
    batch_ids: list[str] = []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feat = model.encode_image(x)
            if l2_normalize:
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        feat = feat.float().cpu().numpy()
        for vid, row in zip(batch_ids, feat):
            ids.append(vid)
            vecs.append(row.astype(np.float32))
        batch_imgs.clear()
        batch_ids.clear()

    for rec in records:
        path = Path(rec["image_path"])
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_ids.append(rec["variant_id"])
        except Exception as e:  # corrupt/unsupported image
            skipped.append({"variant_id": rec["variant_id"], "reason": f"{type(e).__name__}: {e}"})
            continue
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    matrix = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, EMBED_DIM), np.float32)
    return {
        "ids": ids,
        "vectors": matrix,
        "skipped": skipped,
        "embedder_id": f"openclip_{MODEL_ARCH_BY_KEY[pretrained_key]}_{PRETRAINED_TAGS[pretrained_key]}",
        "device": device,
        "l2_normalized": l2_normalize,
    }


def save_npz(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.array(result["ids"], dtype=object),
        vectors=result["vectors"],
        embedder_id=result["embedder_id"],
        l2_normalized=result["l2_normalized"],
    )


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "ids": list(data["ids"]),
        "vectors": data["vectors"],
        "embedder_id": str(data["embedder_id"]),
        "l2_normalized": bool(data["l2_normalized"]),
    }
