# Legacy-values recomputation: input manifest (for the GPT code-and-rerun audit, 2026-09-30)

Script: `analysis/legacy_values/recompute_legacy.py` (SHA-256 `77f3726861c04c535b097056d5e559b6c38df756bfa8fb933922393732c2c4a6`, 4691 bytes)
Output: `analysis/legacy_values/out/legacy_values.json` (SHA-256 `c7e217fa992bb4644142371f0432aadecaa523b9e6345641ae25f1d6a489fa0d`, 2626 bytes)

**Reproduction check (Claude, 2026-09-30):** rerunning the script with a Python audit hook that recorded every file opened reproduced `legacy_values.json` byte-identically. The inputs below are the complete set of project files read or imported during that run; the Python standard library and site-packages are excluded.

| Role | Path (relative to `Paper PH4 and Memes Computational/`) | SHA-256 | Bytes | Drive file id (size confirmed = local) |
|---|---|---|---|---|
| data | `SSCR Revise and Resubmit/analysis/g1_full_sample/out/scores_all.csv` | `f474767a8f9f7df58ff905d00d4c4d97edc399b4af469838f38300349229fb4b` | 750539 | `1gH0rUACSJ6m9EiX_XYbAjo9EF1a5N1Dg` |
| data | `build/first_results/ph4_q2_horsepower/q2_denom_outcomes.csv` | `ea67d0bdcd7b75c07dc8d5e58ed067e0f8cc487899cfdbe10da05f4809c777de` | 742502 | `1KQ_7QkSEOGVRgMVKsut_1fe547QI2MXk` (build copy; an identical-size submission-package copy is `1GgtnMYjn4kbTzRGXD3zrnkVkJGK-Cjcq`) |
| data (check only) | `SSCR Revise and Resubmit/analysis/g1_full_sample/out/g1_results.json` | `3c84d2e07b29cba0409cf35ef5e424e7b23ab3fc129d35dc34b8c0a9a31781d0` | 28748 | `1vLE1kLdCiKWkLBKqZ_NwKoe43u_9Fk-0` |
| code (imported) | `SSCR Revise and Resubmit/analysis/common.py` | `4077d8ea9a9e285188485aa7125d3f13f73f674d4cf51b8d119a77f9d277b047` | 3421 | `1c5OsQARNU0jSi_uXouTFr5znHoXJ2ovn` |
| code (imported) | `SSCR Revise and Resubmit/analysis/g1_full_sample/03_decomposition.py` (frame, variants, control set D) | `b135b92da52e531e189c3fbb8280f624232ddac816035c65f65facebdd02725c` | 13849 | `13SeFS2_y-gJMOnHrzV7k1ppsqECP__hM` |
| code (imported) | `SSCR Revise and Resubmit/analysis/repair/01_bootstrap_samelabel.py` (repaired same-label bootstrap) | `d05ac83b7061b8a4944ddd2d1cc6c947dad62f1ad9e12e74d9dc4e0b37c13176` | 6015 | `16GGtJZZEtLW2patG1zU4C7o-7IXySr__` |
| code (imported; submitted engine, frozen) | `SSCR Submission/_Working/09_Reproducibility/analysis_to_run/RUN_e1_m2_reconcile.py` | `51b48d0305f510f2ba676ebdb1769f8148d35982b13fe2f80365bfe55dfd5f0a` | 5652 | `1mTaE5RauvZx3la1WvBS9T09g1Dpxr4x2` |

**Points for the audit (Claude, not judgments):**
- The exclusion values (`excl_*`) use the `raw_fit` variant. The Table S13 outcome swaps use `raw_fit_plus_spec_stored`. Which variant the submitted +0.054, +0.044 and +0.040 used should be confirmed.
- All point estimates use best-fit-family `GroupKFold(5)` with the in-fold binary outcome (`inc()`).
- The selected-listing interval comes from `RB.boot` over `ph4_fit_family` with `RB.picks_for(ug, 0)` on 348 posts, and the number of groups is small. This is the likely source of the wide, asymmetric interval.
