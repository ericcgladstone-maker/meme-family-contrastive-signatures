#!/usr/bin/env python3
"""Best-fit signature concentration in the Reddit denominator sample (final cross-audit, `28` A1). Counts only: how many
posts each of the 31 family signatures is the best fit for, overall and within the top-decile group used for the scale
anchor. No model is fitted. Output: out/bestfit_concentration.json (with source hashes)."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AN = HERE.parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    scores = AN / "g1_full_sample/out/scores_all.csv"
    decile = AN / "r1_7_decile/out/decile_lift.json"
    d = pd.read_csv(scores)
    cut = json.loads(decile.read_text())["decile_cutoff_raw_max_fit"]
    allc = Counter(d.raw_argmax)
    top = d[d.raw_max_fit >= cut]
    topc = Counter(top.raw_argmax)
    (f1, n1), (f2, n2) = allc.most_common(2)
    ft, nt = topc.most_common(1)[0]
    out = {"source_sha256": {"scores_all": sha(scores), "decile_lift": sha(decile)},
           "n_posts": int(len(d)), "n_bestfit_groups": int(len(allc)),
           "top_family": f1, "top_family_label": f1.replace("_", " "), "top_family_n": int(n1), "top_family_share": n1 / len(d),
           "second_family": f2, "second_family_n": int(n2), "top2_share": (n1 + n2) / len(d),
           "top_decile_n": int(len(top)), "top_decile_top_family": ft, "top_decile_top_family_n": int(nt),
           "top_decile_top_family_share": nt / len(top),
           "counts_all": dict(allc.most_common()), "counts_top_decile": dict(topc.most_common())}
    assert ft == f1
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out/bestfit_concentration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith(("counts", "source"))}, indent=1))


if __name__ == "__main__":
    main()
