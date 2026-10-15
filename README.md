# Measuring Meme-Family Structure With Contrastive Sparse Features — revision package

Reproducibility materials for the revised article *"Measuring Meme-Family Structure With Contrastive Sparse
Features: Canonical Resemblance and Reddit Engagement"* (revise-and-resubmit version).

Eric Gladstone and David C. Thompson. Author's manuscript, revise and resubmit at *Social Science Computer Review*:
https://graystoneindustries.co/research/#measuring-meme-family-structure-with-contrastive-sparse-features-canonical-resemblance-and-reddit-engagement

This release (`v2.0.0-r1`) is the revision package for the R1 version submitted on 5 October 2026. See `RELEASE_NOTES.md`
for what it contains and what is not redistributed.

This README replaces the README of the submitted package. The submitted package is kept unchanged for provenance.
Corrections to the submitted README:

- **Meme family.** A family is not "variants sharing a documented template or reference". A *meme family* is a set of
  variants that a meme-reference source (Know Your Meme) documents as one lineage. Its members share a recurring
  configuration of image, text or layout while differing, often substantially, in surface form. A *template* (one
  reusable base image) is narrower: a family may span several templates, or keep its identity through a phrasal or
  relational pattern when the image changes (manuscript Figure 1).
- **Signature.** The 15 retained features are selected **by contrast**, not "by cosine fit". For family *f* the
  contrast d_f is the mean member activation minus the mean activation of the training background (nonmembers used
  for estimation, disjoint from the evaluation negatives and decoys). The signature keeps the K = 15 features with the
  largest positive contrast, with weights w_f = max(d_f[T_f], 0). This gives 15 (feature index, weight) pairs:
  equivalently, a 6,144-dimensional vector with 15 non-zero positive entries. A candidate is scored by the cosine
  between its activations on those 15 features and the weights (raw resemblance). Cosine is the *scoring* rule, not
  the *selection* rule (SI §S4).
- **Reported numbers.** The submitted README quoted randomization results typed by hand. Those values are superseded.
  This README reports no statistics: every reported value is read from the audited outputs listed in
  `REPRODUCIBILITY_MANIFEST.csv`, and the manuscript and SI are rendered from those outputs.
- **Script provenance.** Superseded submitted scripts are marked in the manifest (status `superseded`),
  including `e1_grouped_cv.py`, which shipped unreferenced with a "primary" docstring and produced no reported result.

## What the study does

For each documented family we estimate a contrastive sparse-feature signature in the sparse code of a multimodal
sparse autoencoder over OpenAI CLIP ViT-L/14 image embeddings (6,144 latent units, 64 active). We evaluate
leave-one-variant-out family recovery against a frequency-based recurrence baseline and against surface,
whole-image (CLIP), sparse-prototype, nearest-neighbor, fused modern-encoder (SigLIP2 + DINOv2) and supervised
alternatives. This is done first in the development corpus (30 families) and then prospectively in held-out families
(random stratum H; purposive look-alike challenge stratum C). We stress-test discrimination with encoder-selected and
perceptual-hash-selected decoys, and then apply the frozen pipeline to newly listed Reddit image posts to study its
association with engagement.

## Layout (paths relative to the package root)

```
analysis/common.py                 shared paths and helpers
analysis/00_freeze/                verification of the frozen submitted inputs
analysis/g3_recovery/              development recovery, corrected protocol; K curve; negative redraws
analysis/heldout/                  held-out-family validation (frozen protocol, screener, engine, summarizer)
analysis/heldout/exploratory/      exploratory decoy comparison vs the fused centroid (labelled exploratory)
analysis/g2_template_cluster/      engagement validation schemes (perceptual-hash clusters, best-fit family, community)
analysis/g1_full_sample/           full-sample (4,263 posts) decomposition and randomization nulls
analysis/repair/                   corrected same-label cluster bootstrap; Monte Carlo p-values; fold-local decoys
analysis/revision_selections/      rule-based case selection (worked example, illustrative cases)
analysis/legacy_values/            recomputed sensitivity values (no hand-typed values remain)
analysis/r1_7_decile/              top-decile hit-rate lift (scale interpretation)
analysis/figure_inputs/            audited export of the submitted signature file for the figures
analysis/human_check/              blinded author-coded human check (draw, coding form, label freeze, analysis)
analysis/human_validation/         superseded two-coder design (frozen before any draw; never run)
analysis/m12_overlap/              cross-source overlap audit and exclusion sensitivity
analysis/prospective/              export of the short-window prospective analysis
analysis/corpus/                   development-corpus counts
analysis/repro_gate/               independent reconstruction of Table 2 (audit)
revision/build/figures/            figure script (plotting only; reads audited outputs; no estimation)
revision/build/build_docx_r1.py    DOCX builder (inline-math fix; change highlighting)
submitted_package/                 the submitted reproducibility package, unchanged (superseded scripts are marked in the manifest)
```

Every script states its inputs, outputs and usage in its docstring. The manifest gives, for every script and output
file, its SHA-256 checksum, the producing script, and the command, grouped by the rows of SI Table S22.

## Environment

Python 3.9 on macOS. Core versions: NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, PyTorch with `open_clip`,
`imagehash` 4.3.2, `ocrmac` (macOS Vision OCR; used only to regenerate OCR surface features), matplotlib and
python-docx for figures and documents (SI Table S3). Run each script from the package root with
`PYTHONDONTWRITEBYTECODE=1 python3 <script>`. Input locations are set in `analysis/common.py`, which points at the
derived inputs of the original project tree. Repoint them there if the tree is relocated.

## Seeds

Recovery negative sampling: 20260512. Cross-validation and bootstrap: 0 for reported point estimates. Seed sweeps and
repeated cross-validation use the seeds named in the SI (Table S5a).

## Data and images

The package contains derived representations, public post identifiers and analysis outputs. Raw meme images and raw
Reddit content are not redistributed where source or platform terms restrict it. Retrieved held-out-family images are
not redistributed and appear in no figure; the manifest lists them by directory, and per-file hashes are in
`analysis/heldout/out/heldout_manifest.csv`. Figure 1 is a schematic redraw of development-corpus examples (no image
pixels are reproduced), and Figure S7 shows family names, feature weights, scores, and the source family and signature fit of each family's highest-scoring nonmember.

## License and citation

Code: MIT License. Derived data deposited on OSF: CC0 1.0, subject to the platform-terms note above. Please cite
the article; the archive identifier is withheld here for anonymized review.
