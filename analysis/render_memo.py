#!/usr/bin/env python3
"""Render a Markdown template whose statistics are placeholders bound to the gate result files.

Placeholders: {{g1:full_4263.increments.raw_fit.dR2|+.3f}}  ->  value from g1_results.json, formatted.
Sources: g1 = g1_full_sample/out/g1_results.json, g2 = g2_template_cluster/out/g2_results.json,
g3 = g3_recovery/out/g3_results.json, g1id = g1_full_sample/out/identity_check.json,
g2cl = g2_template_cluster/out/cluster_diagnostics.json.
Unresolvable placeholders abort the render (no silent gaps). A binding log is written next to the output.

Usage: python3 render_memo.py TEMPLATE.md OUTPUT.md
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = {
    "g1": HERE / "g1_full_sample/out/g1_results.json",
    "g1id": HERE / "g1_full_sample/out/identity_check.json",
    "g2": HERE / "g2_template_cluster/out/g2_results.json",
    "g2cl": HERE / "g2_template_cluster/out/cluster_diagnostics.json",
    "g3": HERE / "g3_recovery/out/g3_results.json",
    "rpb": HERE / "repair/out/bootstrap_samelabel.json",
    "rpc": HERE / "repair/out/repair_cheap.json",
    "rpd": HERE / "repair/out/decoys_foldlocal.json",
}
PAT = re.compile(r"\{\{(\w+):([^|}]+)(?:\|([^}]+))?\}\}")


def get(obj, path):
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def main():
    tpl, out = Path(sys.argv[1]), Path(sys.argv[2])
    data = {k: json.loads(p.read_text()) for k, p in SRC.items() if p.exists()}
    log, errors = [], []

    def sub(m):
        src, path, fmt = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
        try:
            v = get(data[src], path)
            s = format(v, fmt) if fmt else str(v)
            if fmt.endswith("f") and s.startswith("-") and float(s) == 0:
                s = s[1:]  # render -0.000 as 0.000
            log.append(f"{src}:{path} -> {s}")
            return s
        except Exception as e:  # noqa: BLE001
            errors.append(f"{src}:{path} ({e})")
            return "??"
    text = PAT.sub(sub, tpl.read_text())
    if errors:
        print("UNRESOLVED PLACEHOLDERS:\n  " + "\n  ".join(errors))
        sys.exit(1)
    out.write_text(text)
    (out.parent / (out.stem + ".bindings.log")).write_text("\n".join(log) + "\n")
    print(f"rendered {out.name}: {len(log)} bound statistics, 0 unresolved")


if __name__ == "__main__":
    main()
