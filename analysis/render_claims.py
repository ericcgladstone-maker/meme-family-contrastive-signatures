#!/usr/bin/env python3
"""render_memo.py (frozen, bundled; unchanged) + held-out sources, for rendering 13 after the held-out results.

Extra sources: ho  = heldout/out/heldout_results.json (pre-specified, mechanical summarizer output)
               hox = heldout/out/exploratory_x1_decoys_vs_fused.json (EXPLORATORY; must be labelled as such)
               hs  = heldout/out/screening_summary.json (closure counts)
Usage: python3 render_claims.py TEMPLATE.md OUTPUT.md
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("render_memo", HERE / "render_memo.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
rm.SRC.update({"ho": HERE / "heldout/out/heldout_results.json",
               "hox": HERE / "heldout/out/exploratory_x1_decoys_vs_fused.json",
               "hs": HERE / "heldout/out/screening_summary.json"})
missing = [k for k, p in rm.SRC.items() if not p.exists()]
if missing:
    raise SystemExit(f"missing sources: {missing}")
rm.main()
