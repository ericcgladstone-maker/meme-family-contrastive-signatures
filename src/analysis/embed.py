"""Embedding step. Real run uses OpenCLIP ViT-L/14.

Smoke-test variant: produces deterministic pseudo-embeddings from image SHA-256
so the smoke test runs without downloading CLIP weights. The real embed.py is a
thin wrapper around `open_clip.create_model_and_transforms` plus a
parquet writer that satisfies `schemas/embeddings.schema.json`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..curation.audit import AuditWriter, REPO_ROOT


EMBED_DIM = 768


def _deterministic_embedding(image_path: Path, dim: int = EMBED_DIM, seed_offset: int = 0) -> np.ndarray:
    """Hash-of-image → seeded RNG → unit vector. Smoke-test stand-in for CLIP."""
    h = hashlib.sha256(image_path.read_bytes()).digest()
    seed = int.from_bytes(h[:4], "big") ^ seed_offset
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)


def embed_variants(
    variant_records: list[dict[str, Any]],
    audit: AuditWriter,
    embedder_id: str = "toy_openclip_vit_l_14",
    checkpoint_sha256: str = "PLACEHOLDER_TOY",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in variant_records:
        path = REPO_ROOT / v["image_path"]
        vec = _deterministic_embedding(path)
        out.append({
            "variant_id": v["variant_id"],
            "embedder_id": embedder_id,
            "embedder_checkpoint_sha256": checkpoint_sha256,
            "embedding_dim": EMBED_DIM,
            "embedding": vec.tolist(),
            "is_background_corpus_sample": False,
            "is_null_set_variant": False,
            "input_hash": v["image_sha256"],
            "computed_at": v["first_appearance_timestamp"],
        })
    return out


def embed_null_set_variants(
    null_set_variant_records: list[dict[str, Any]],
    toy_variants_root: Path,
    audit: AuditWriter,
    embedder_id: str = "toy_openclip_vit_l_14",
    checkpoint_sha256: str = "PLACEHOLDER_TOY",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for nv in null_set_variant_records:
        prefix = nv["null_set_id"] + "_"
        stem = nv["variant_id"][len(prefix):] if nv["variant_id"].startswith(prefix) else nv["variant_id"]
        vp = toy_variants_root / nv["null_set_id"] / (stem + ".png")
        if not vp.exists():
            vp = toy_variants_root / nv["null_set_id"] / "n01.png"
        vec = _deterministic_embedding(vp)
        import hashlib as _h
        out.append({
            "variant_id": nv["variant_id"],
            "embedder_id": embedder_id,
            "embedder_checkpoint_sha256": checkpoint_sha256,
            "embedding_dim": EMBED_DIM,
            "embedding": vec.tolist(),
            "is_background_corpus_sample": False,
            "is_null_set_variant": True,
            "input_hash": _h.sha256(vp.read_bytes()).hexdigest(),
            "computed_at": "2025-01-01T00:00:00+00:00",
        })
    return out


def synth_background_corpus(
    n: int,
    audit: AuditWriter,
    embedder_id: str = "toy_openclip_vit_l_14",
    checkpoint_sha256: str = "PLACEHOLDER_TOY",
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Deterministic synthetic background corpus for threshold estimation."""
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for i in range(n):
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        vec /= (np.linalg.norm(vec) + 1e-12)
        out.append({
            "variant_id": f"background_{i:05d}",
            "embedder_id": embedder_id,
            "embedder_checkpoint_sha256": checkpoint_sha256,
            "embedding_dim": EMBED_DIM,
            "embedding": vec.tolist(),
            "is_background_corpus_sample": True,
            "is_null_set_variant": False,
            "input_hash": hashlib.sha256(f"bg{i}".encode()).hexdigest(),
            "computed_at": "2025-01-01T00:00:00+00:00",
        })
    return out
