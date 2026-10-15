# M12 cross-source-overlap exclusion sensitivity: specification (frozen before execution; `09` F-62)

**Purpose.** Determine whether the reported Reddit engagement result depends materially on the 14 denominator posts identified prospectively as near-duplicates of development-corpus images, under the study's existing DCT pHash rule (Hamming ≤ 6).

**Exclusion set.** Exactly the 14 Reddit post IDs in `out/m12_overlap.json` (`pairs_for_traceability_not_inspected`). There is no image inspection and no manual change to the set. Result: n = 4,263 − 14 = 4,249.

**Estimands rerun.** Existing estimands only, with identical code paths.
1. Continuous-outcome ΔR² of the full model (control D + community; full adds the raw signature score and the specificity margin) over control. Four schemes, with fold construction identical to `g2_template_cluster/02_grouped_cv.py`:
   - random 5-fold (`KFold(5, shuffle=True, random_state=0)`);
   - pHash-cluster grouped (`cc10_as_planned`, `GroupKFold(5)`);
   - community grouped (`GroupKFold(7)` = leave-one-community-out);
   - best-fit-family grouped (`GroupKFold(5)` on `ph4_fit_family`).
   
   The binary-outcome ΔAUROC that the same engine emits is kept in the output but is not a new reporting endpoint.
2. The representation-magnitude-adjusted raw-signature residual. The point estimate is `g1_full_sample/03_decomposition._inc(df, ph4_fit_family groups, D + [sparse_norm, clip_norm, ocr_len], [raw_max_fit])`. The interval is the same-label cluster bootstrap of `repair/01_bootstrap_samelabel.py`: 2,000 draws, `picks_for(unique families, seed 0)`, percentile 95%.

**Reproduction gate.** Before excluding anything, the script reruns each estimand on the full 4,263 posts. It requires the stored values to be reproduced: `g2_results.json` `pooled.*.dR2` and `g1_results.json` `full_4263.raw_over_D_plus_magnitude.dR2`, to within 1e-9. Otherwise it refuses. The bootstrap is not rerun on the full sample; the stored interval is the reference.

**Not rerun.** The human check, the selected-listing contrast, recovery, and every other sensitivity.

**Output.** `out/m12_exclusion_sensitivity.json`, with original and exclusion values side by side. There is no characterization (robust or unchanged). Stop for interpretation before any prose edit.
