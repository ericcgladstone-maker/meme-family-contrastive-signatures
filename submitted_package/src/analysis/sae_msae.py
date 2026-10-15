"""Real MSAE sparse-feature extraction (WolodjaZ/MSAE, Zaigrajew et al. 2025).

Loads the cached TopKReLU SAE checkpoint and produces sparse Top-K activation
records in the format `analysis/pharmacophore.py` consumes
(active_indices / active_values / latent_dim / top_k).

Forward pass (verbatim from the repo's sae.py / SAE wrapper):
    preprocess: x = (x - mean_center) * scaling_factor      # model normalize=False
    encode:     pre = (x - pre_bias) @ encoder + latent_bias
                z   = TopKReLU_k( pre )                      # keep top-k, ReLU
    decode:     x_hat = z @ decoder + pre_bias
                x_rec = x_hat / scaling_factor + mean_center

IMPORTANT: the cached checkpoint was trained on OpenAI CLIP ViT-L/14 (cc3m), so
its inputs must be raw (un-normalized) OpenAI-CLIP image features, NOT the LAION
open_clip features. `validate_reconstruction` checks this empirically.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def find_checkpoint() -> Path:
    """Resolve the cached MSAE .pth path from the HF hub cache."""
    home = os.path.expanduser("~")
    pat = home + "/.cache/huggingface/hub/models--WolodjaZ--MSAE/snapshots/*/ViT-L_14/centered/*.pth"
    matches = glob.glob(pat)
    if not matches:
        raise FileNotFoundError(f"MSAE checkpoint not found under {pat}")
    return Path(matches[0])


class MSAE:
    """Inference-only MSAE TopKReLU encoder."""

    def __init__(self, checkpoint_path: Path | None = None, device: str = "cpu"):
        import torch
        self.torch = torch
        self.device = device
        path = Path(checkpoint_path) if checkpoint_path else find_checkpoint()
        self.checkpoint_path = path
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = ckpt["model"]
        self.encoder = m["encoder"].to(device).float()          # (768, 6144)
        self.pre_bias = m["pre_bias"].to(device).float()        # (768,)
        self.latent_bias = m["latent_bias"].to(device).float()  # (6144,)
        self.decoder = m["decoder"].to(device).float()          # (6144, 768)
        self.mean_center = ckpt["mean_center"].to(device).float()  # (768,)
        self.scaling_factor = float(ckpt["scaling_factor"])
        self.latent_dim = int(self.encoder.shape[1])
        self.input_dim = int(self.encoder.shape[0])
        # k parsed from filename (e.g. ..._TopKReLU_64_...)
        self.k = self._parse_k(path.name)
        self.sae_id = path.stem

    @staticmethod
    def _parse_k(name: str, default: int = 64) -> int:
        parts = name.split("_")
        for i, p in enumerate(parts):
            if "TopK" in p and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
        return default

    def _preprocess(self, x):
        return (x - self.mean_center) * self.scaling_factor

    def encode(self, vectors: np.ndarray):
        """vectors: (N, 768) raw OpenAI-CLIP image features.
        Returns (indices (N,k) int, values (N,k) float) of the Top-K ReLU latents."""
        torch = self.torch
        x = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            x = self._preprocess(x)
            pre = (x - self.pre_bias) @ self.encoder + self.latent_bias  # (N, 6144)
            vals, idx = torch.topk(pre, k=self.k, dim=-1)
            vals = torch.relu(vals)
        return idx.cpu().numpy(), vals.cpu().numpy()

    def reconstruct(self, vectors: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.asarray(vectors, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            xp = self._preprocess(x)
            pre = (xp - self.pre_bias) @ self.encoder + self.latent_bias
            vals, idx = torch.topk(pre, k=self.k, dim=-1)
            vals = torch.relu(vals)
            z = torch.zeros_like(pre).scatter_(-1, idx, vals)
            x_hat = z @ self.decoder + self.pre_bias
            x_rec = x_hat / self.scaling_factor + self.mean_center
        return x_rec.cpu().numpy()


def activation_records(ids: list[str], vectors: np.ndarray,
                       sae: MSAE, computed_at: str | None = None) -> list[dict[str, Any]]:
    """Produce pharmacophore-compatible sparse activation records."""
    computed_at = computed_at or datetime.now(tz=timezone.utc).isoformat()
    idx, vals = sae.encode(vectors)
    out: list[dict[str, Any]] = []
    for vid, ii, vv in zip(ids, idx, vals):
        order = np.argsort(ii)  # sort indices ascending for stable storage
        out.append({
            "variant_id": vid,
            "sae_id": sae.sae_id,
            "latent_dim": sae.latent_dim,
            "top_k": sae.k,
            "active_indices": [int(x) for x in ii[order]],
            "active_values": [float(x) for x in vv[order]],
            "is_background_corpus_sample": False,
            "is_null_set_variant": False,
            "computed_at": computed_at,
        })
    return out


def validate_reconstruction(vectors: np.ndarray, sae: MSAE) -> dict[str, float]:
    """Fraction of variance unexplained (FVU) on the input embeddings. Low FVU
    indicates the inputs are in the SAE's expected space (OpenAI CLIP)."""
    rec = sae.reconstruct(vectors)
    x = np.asarray(vectors, dtype=np.float32)
    resid = x - rec
    total_var = ((x - x.mean(axis=0)) ** 2).sum()
    fvu = float((resid ** 2).sum() / total_var) if total_var > 0 else float("nan")
    _, vals = sae.encode(x[: min(len(x), 50)])
    mean_active = float((vals > 0).sum(axis=1).mean())
    return {"fvu": round(fvu, 4), "mean_active_latents_first50": round(mean_active, 2)}
