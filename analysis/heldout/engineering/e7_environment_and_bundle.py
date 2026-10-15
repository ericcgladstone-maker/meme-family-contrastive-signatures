#!/usr/bin/env python3
"""E7 + freeze bundle.

  --env     : write out/environment_manifest.json (Python, packages, macOS, Vision/ocrmac, hardware)
  --bundle  : write heldout/FREEZE_BUNDLE.json = SHA-256 of 14, every frozen script, frame snapshot files,
              dev_series_parents.json, checkpoint + environment manifests, and one combined bundle hash.
              (Run ONLY after the last protocol-driven change; it does not create FREEZE_RECORD.json - that
              is written at the freeze step after GPT's final audit and Eric's sign-off.)
"""
import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HO = HERE.parent
AN = HO.parent
RR = AN.parent
OUT = HERE / "out"
FROZEN_SCRIPTS = [AN / "common.py", HO / "engine.py", HO / "screen_and_retrieve.py", HO / "build_features.py",
                  HO / "run_heldout.py", HO / "summarize_heldout.py", AN / "render_memo.py",
                  HERE / "e7_environment_and_bundle.py", HERE / "e3_d4_closure_test.py",
                  HERE / "out" / "e3_d4_closure_test.json"]
FRAME_FILES = sorted((HO / "frame").glob("*"))
BUILD = RR.parent / "build"
# imported implementation modules (engine / common / feature builder depend on these)
IMPORTED_MODULES = [BUILD / "src/analysis/clip_embed.py", BUILD / "src/analysis/sae_msae.py",
                    BUILD / "src/analysis/pharmacophore.py", BUILD / "src/analysis/score.py",
                    BUILD / "src/analysis/surface_baselines.py", BUILD / "src/analysis/diagnostics.py",
                    BUILD / "src/curation/audit.py"]
# frozen DEV inputs consumed by the engine / feature builder
DEV_INPUTS = [BUILD / "first_results/cross_family_dedup/cross_family_corrected_index.csv",
              BUILD / "first_results/embeddings_openai.npz", BUILD / "first_results/embeddings_laion.npz",
              BUILD / "first_results/ocr_text.csv", AN / "g3_recovery/out/features.npz",
              AN / "g3_recovery/out/ocr_texts.json", HO / "out/dev_modern_embeddings.npz"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def env():
    pk = {}
    for m in ["numpy", "pandas", "sklearn", "scipy", "torch", "open_clip", "transformers", "imagehash", "PIL",
              "joblib", "huggingface_hub", "safetensors", "ocrmac", "objc"]:
        try:
            x = importlib.import_module(m)
            pk[m] = getattr(x, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            pk[m] = f"MISSING ({type(e).__name__})"
    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"], capture_output=True,
                        text=True).stdout.split("\n")
    res = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               macos=platform.mac_ver()[0], machine=platform.machine(), cpu=hw[:3], packages=pk,
               note="OCR = macOS Vision via ocrmac 1.0.1 (pyobjc 11.1); E4 verified 50/50 exact vs May OCR")
    OUT.mkdir(exist_ok=True)
    (OUT / "environment_manifest.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(pk))


def bundle():
    items = {"14_Heldout_Family_Protocol.md": sha(RR / "14_Heldout_Family_Protocol.md")}
    for p in FROZEN_SCRIPTS:
        if not p.exists():
            raise SystemExit(f"missing frozen script {p.name}")
        items[str(p.relative_to(AN))] = sha(p)
    for p in FRAME_FILES:
        items[str(p.relative_to(AN))] = sha(p)
    for p in [OUT / "checkpoint_manifest.json", OUT / "environment_manifest.json"]:
        items[str(p.relative_to(AN))] = sha(p)
    for p in IMPORTED_MODULES + DEV_INPUTS:
        if not p.exists():
            raise SystemExit(f"missing frozen input {p}")
        items[str(p.relative_to(RR.parent))] = sha(p)
    combined = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
    (HO / "FREEZE_BUNDLE.json").write_text(json.dumps(dict(items=items, bundle_sha256=combined), indent=2))
    print("bundle", combined, len(items), "items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--bundle", action="store_true")
    a = ap.parse_args()
    if a.env:
        env()
    if a.bundle:
        bundle()
