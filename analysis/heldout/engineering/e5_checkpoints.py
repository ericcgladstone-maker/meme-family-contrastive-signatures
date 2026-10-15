#!/usr/bin/env python3
"""E5: download the pinned modern-baseline checkpoints (exact revisions) and verify artifact SHA-256.
Also records the processor configs (hashed) that fix preprocessing. Writes out/checkpoint_manifest.json.
Also hashes the MSAE checkpoint and records the open_clip model identifiers used for CLIP.
"""
import glob
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

OUT = Path(__file__).resolve().parent / "out"
PINS = {
    "siglip2": dict(repo="google/siglip2-so400m-patch14-384", rev="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
                    weights="model.safetensors",
                    sha256="9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2"),
    "dinov2": dict(repo="facebook/dinov2-large", rev="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                   weights="model.safetensors",
                   sha256="399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    man = {}
    for k, p in PINS.items():
        d = snapshot_download(p["repo"], revision=p["rev"], allow_patterns=["*.json", "*.txt", "*.model", p["weights"]])
        w = os.path.join(d, p["weights"])
        got = sha(w)
        cfgs = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(d, "*.json")))}
        man[k] = dict(repo=p["repo"], revision=p["rev"], weights=p["weights"], expected_sha256=p["sha256"],
                      downloaded_sha256=got, verified=(got == p["sha256"]), local_dir=d, config_sha256=cfgs)
        print(k, "verified" if got == p["sha256"] else "MISMATCH", flush=True)
    msae = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--WolodjaZ--MSAE/snapshots/*/ViT-L_14/centered/*.pth"))
    man["msae"] = [dict(path=m, sha256=sha(m)) for m in msae]
    man["clip"] = dict(openai="open_clip ViT-L-14 pretrained='openai' (via src.analysis.clip_embed)",
                       laion="open_clip ViT-L-14 laion2B-s32B-b82K (HF cache models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K)")
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: v.get("verified") if isinstance(v, dict) else len(v) for k, v in man.items()}))


if __name__ == "__main__":
    main()
