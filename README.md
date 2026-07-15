# Measuring Meme-Family Structure With Contrastive Sparse Features

Reproducibility materials for the article *"Measuring Meme-Family Structure With
Contrastive Sparse Features: Canonical Resemblance and Reddit Engagement."*

This repository contains the analysis code, derived data, result tables, and the
figures used in the manuscript and supplement. It is sufficient to regenerate the
reported numbers, tables, and figures from the shared derived features without
redistributing raw platform content.

> **Peer-review note.** This work is under double-anonymized review. During review,
> code and data are provided through an anonymized OSF view-only link. The public
> repository URL and a citable DOI will be released on acceptance.

## What the study does

A meme "family" is a set of variants sharing a documented template or reference. For
each family we estimate a **contrastive sparse-feature signature**: the difference in
mean sparse-autoencoder (MSAE) activations between family members and nonmembers,
restricted to the top-15 features by cosine fit. We evaluate how well this signature
**recovers** documented family membership (leave-one-variant-out, family as the unit),
compare it against whole-image CLIP, surface, sparse-centroid, nearest-neighbor, and a
supervised classifier, stress it with adversarial look-alike decoys, and then **apply**
it to newly listed Reddit image posts to measure its association with engagement.

The representation is a Matryoshka multimodal sparse autoencoder (6,144 latent
dimensions, k = 64) over CLIP ViT-L/14.

## Repository layout

```
src/analysis/        Signature math and comparison measures (imported by the scripts)
  sae_msae.py          MSAE activation records
  clip_embed.py        CLIP embedding + .npz loaders
  score.py             signature construction and scoring
  surface_baselines.py, text_template_baselines.py, pharmacophore*.py, ...
src/curation/audit.py  Hashing/provenance helper imported by the above
analysis/            Analysis entry points that produced the SSCR results
analysis_to_run/     Grouped CV, cluster bootstrap, and same-representation baselines
figures/             Figure and DOCX generation
data/                Derived inputs (features, public post IDs, intermediate results)
tables/              Corpus composition (Table S1) and family recovery matrix (Table S7)
results/             Result outputs + final figures (PNG + SVG)
```

## Environment

Tested with **Python 3.9**. Install pinned dependencies:

```
pip install -r requirements.txt
```

Key versions: NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, PyTorch with
open_clip, imagehash 4.3.2, and ocrmac (macOS Vision OCR, used only to regenerate
OCR surface features). Regenerating the MSAE features requires the sparse-autoencoder
weights loaded by `src/analysis/sae_msae.py`.

## Random seeds

Seeds are fixed for exact reproduction:

- Recovery negative sampling: **20260512**
- Cross-validation and bootstrap procedures: **0**

## Reproducing the results

Run scripts from this repository root with `PYTHONPATH=.` so the `src` package
resolves, e.g. `PYTHONPATH=. python analysis/RUN_save_nulls.py`. Each script writes to
the paths named at the top of the file; the derived inputs it expects are provided
under `data/`. (The scripts were authored against the original project tree, so a few
top-of-file path constants point at that layout — repoint them at `data/` as needed.)

| Manuscript element | Script(s) | Output |
|---|---|---|
| Family recovery (Table 2, Fig 2, Table S7) | `analysis_to_run/RUN_m2m4_full.py`, `analysis/RUN_si_recovery.py` | `data/m2m4_results.csv`, `results/si_recovery_results.md`, `tables/TableS7_family_recovery_matrix.csv` |
| Adversarial discrimination (Fig 3, Fig S6) | `analysis_to_run/m4_nonclip_decoys.py` (via `RUN_m2m4_full.py`) | `data/m2m4_results.csv` (adv_* columns) |
| Grouped CV & cluster bootstrap (Tables 3–4, Fig S10) | `analysis_to_run/RUN_e1_m2_reconcile.py`, `analysis_to_run/RUN_e2_bootstrap.py` | fold- and bootstrap-level estimates |
| Engagement association (Table 4, Fig 4a/d) | `analysis/RUN_downstream_embed_score.py`, `analysis/RUN_downstream_engagement.py` | `results/downstream_scores.csv`, `results/downstream_engagement_results.csv` |
| Randomization nulls (Fig 4b/c, Fig S11) | `analysis/RUN_save_nulls.py` | `data/null_random_feature.csv`, `data/null_label_perm.csv` |
| Signature interpretability (worked examples) | `analysis_to_run/RUN_interpretability_export.py` | `results/interpretability_signatures.csv` |
| Typicality / magnitude controls | `analysis/RUN_typicality_controls.py` | console / log |
| Consolidated reviewer numerics | `analysis/RUN_reviewer_numerics.py` | `results/reviewer_numerics_results.md` |
| All figures | `figures/make_figures.py` (Fig 1–3), `figures/make_all_figures.py` (Fig 4 + supplement) | `results/figures/*.png`, `*.svg` |

Randomization results: random-feature null (500 banks) p < .002; label-permutation
null (200 banks) p = .025; observed engagement increment = 0.080.

## Data provenance and terms

`data/` contains **derived** representations, publicly listed post identifiers, and
intermediate analysis outputs — not raw user content. Reddit posts are referenced by
public identifier; meme-reference materials are described rather than redistributed.
Users are not identified: usernames are removed from analysis data. Reuse is subject to
the source platforms' terms.

## License

Code is released under the **MIT License** (see `LICENSE`). The derived **data** deposited
on OSF are released under **CC0 1.0**, subject to the platform-terms note above. (This
code/data split is intentional: permissive-with-attribution for software, public-domain
dedication for the derived data.)

## Citation

See `CITATION.cff`. Please cite the article. Archive DOI: https://doi.org/10.17605/OSF.IO/BFWZE
(OSF). The public GitHub URL is released on acceptance.
