#!/usr/bin/env python3
"""Pre-freeze network dry-run of the frozen Screener.screen() on DEV slugs only (no held-out candidate).

Test harness only: output redirected to engineering/out/devtest/; the DEV slug list is emptied IN THIS PROCESS
so S0/S2 do not stop the DEV entries; test parameters MIN_VARIANTS=10, gallery only, drop NSFW images.
Expected: an ungated DEV slug downloads its first-page images, and D3 removes those that duplicate the DEV
corpus (validating D3); a gated slug (doge) -> nsfw_gallery.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sr", HERE.parent / "screen_and_retrieve.py")
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def main():
    sr.OUT = HERE / "out" / "devtest"
    sr.FREEZE_PARAMS.update(MIN_VARIANTS=10, IMAGE_SOURCES="gallery", EXCLUDE_GATED_IMAGES=True)
    sr.DEV_SLUGS = []
    s = sr.Screener()
    rows = [s.screen(1, "roll-safe", "H", None), s.screen(2, "doge", "H", None)]
    s.save(final=True)
    out = [{k: v for k, v in r.items() if k in ("slug", "disposition", "raw", "d1", "d2", "d3", "n_postdedup",
                                                   "n_gated_images", "parent", "evidence")} for r in rows]
    (HERE / "out" / "e3_screener_devtest.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
