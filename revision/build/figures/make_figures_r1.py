#!/usr/bin/env python3
"""Revision figures (R1). Presentation layer only: every value plotted is read from an audited output; this script
selects, orders and formats and computes no new estimand (means/intervals are read, not recomputed).

Sources (paths relative to `SSCR Revise and Resubmit/`):
  g1   analysis/g1_full_sample/out/g1_results.json, g1_nulls_*_full_4263_seed0.csv
  g2   analysis/g2_template_cluster/out/g2_results.json, g2_per_fold.csv
  g3   analysis/g3_recovery/out/g3_results.json, g3_family_table.csv
  rpb  analysis/repair/out/bootstrap_samelabel.json
  rpc  analysis/repair/out/repair_cheap.json
  ho   analysis/heldout/out/heldout_results.json
  lv   analysis/legacy_values/out/legacy_values.json
  rev  analysis/revision_selections/out/revision_selections.json
  sig  analysis/figure_inputs/out/signatures_submitted_export.json (audited copy of the submitted signature export)
Figure 1 is a schematic (no image pixels reproduced); its source variants are recorded in
19_Revision/build/provenance/figure1_sources.csv. No held-out-family image is used anywhere.
Unchanged submitted figures (S1 family sizes, S11 prospective) are copied; S2 (dedup schematic) is redrawn with the same
content for layout (F-45). The copies come from the read-only submission tree.
Usage: python3 19_Revision/build/figures/make_figures_r1.py   (writes 19_Revision/build/out/figures/)
"""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Ellipse, Polygon  # noqa: E402

sys.dont_write_bytecode = True
RR = Path(__file__).resolve().parents[3]
AN = RR / "analysis"
OUT = RR / "19_Revision/build/out/figures"
PROV = RR / "19_Revision/build/provenance"
SUB_FIG = RR.parent / "SSCR Submission/_Working/03_Figures"
DEV_INDEX = RR.parent / "build/first_results/cross_family_dedup/cross_family_corrected_index.csv"

J = lambda p: json.loads((AN / p).read_text())  # noqa: E731
g1 = J("g1_full_sample/out/g1_results.json")["full_4263"]
g2 = J("g2_template_cluster/out/g2_results.json")
g3 = J("g3_recovery/out/g3_results.json")
rpb = J("repair/out/bootstrap_samelabel.json")
rpc = J("repair/out/repair_cheap.json")
ho = J("heldout/out/heldout_results.json")
lv = J("legacy_values/out/legacy_values.json")
rev = J("revision_selections/out/revision_selections.json")
sig = J("figure_inputs/out/signatures_submitted_export.json")["families"]
fam = pd.read_csv(AN / "g3_recovery/out/g3_family_table.csv")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
C_DEV, C_H, C_C, C_GREY = "#4C72B0", "#1b6e5a", "#b2542a", "#7f7f7f"
METHODS = [("recurrence", "Recurrence (frequency-based)"), ("sparse_centroid", "Sparse centroid"),
           ("sparse_knn", "Sparse kNN"), ("surf_composite", "Surface composite"),
           ("clip_centroid", "CLIP centroid"), ("contrastive", "Contrastive signature"),
           ("logistic", "Supervised classifier")]
NICE = lambda s: s.replace("_", " ")  # noqa: E731


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1 (schematic)
def _frame(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.0, ec="black"))


def _bar(ax, x, y, w, h=0.035, c="#bdbdbd"):
    ax.add_patch(Rectangle((x, y), w, h, color=c, lw=0))


def _person(ax, x, y, s, color, gesture):
    ax.add_patch(Circle((x, y + 0.55 * s), 0.16 * s, color=color))
    ax.add_patch(FancyBboxPatch((x - 0.22 * s, y), 0.44 * s, 0.38 * s, boxstyle="round,pad=0.01", color=color))
    if gesture == "reject":  # raised palm
        ax.add_patch(Rectangle((x + 0.22 * s, y + 0.35 * s), 0.10 * s, 0.22 * s, color=color))
        ax.text(x + 0.45 * s, y + 0.55 * s, "✗", fontsize=9, ha="center", va="center")
    else:  # pointing
        ax.add_patch(Rectangle((x + 0.22 * s, y + 0.25 * s), 0.25 * s, 0.07 * s, color=color))
        ax.text(x + 0.60 * s, y + 0.30 * s, "✓", fontsize=9, ha="center", va="center")


def figure1():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1.15, 1.0]})
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    # (a) phrasal formula persists over different base images (documented family: nobody / literally no one)
    ax = axes[0]
    ax.set_title("(a) One family:\ndifferent images and captions", fontsize=8, loc="left")
    shapes = [lambda a, x, y: a.add_patch(Polygon([[x, y], [x + .13, y + .16], [x + .26, y]], color="#9ecae1")),
              lambda a, x, y: a.add_patch(Circle((x + .13, y + .09), .085, color="#c6dbef")),
              lambda a, x, y: a.add_patch(Rectangle((x + .02, y + .02), .22, .14, color="#deebf7"))]
    for k, (x, shp) in enumerate(zip([0.02, 0.35, 0.68], shapes)):
        _frame(ax, x, 0.18, 0.30, 0.62)
        ax.text(x + 0.02, 0.73, "nobody:", fontsize=6.5, va="center")
        ax.text(x + 0.02, 0.64, "[line]:", fontsize=6, va="center", color=C_GREY)
        shp(ax, x + 0.02, 0.24)
    ax.text(0.5, 0.06, "shared: caption formula\nvaries: image and wording", ha="center", fontsize=6.5)
    # (b) relational layout persists across different base characters within one documented family (drakeposting)
    ax = axes[1]
    ax.set_title("(b) One family: shared layout,\ndifferent base characters", fontsize=8, loc="left")
    for x, col, lab in [(0.03, "#e6a141", "character 1"), (0.53, "#8c6d31", "character 2")]:
        _frame(ax, x, 0.18, 0.44, 0.62)
        ax.plot([x, x + 0.44], [0.49, 0.49], color="black", lw=0.8)
        ax.plot([x + 0.22, x + 0.22], [0.18, 0.80], color="black", lw=0.8)
        _person(ax, x + 0.08, 0.55, 0.30, col, "reject")
        _person(ax, x + 0.08, 0.24, 0.30, col, "approve")
        _bar(ax, x + 0.25, 0.66, 0.16)
        _bar(ax, x + 0.25, 0.34, 0.16)
        ax.text(x + 0.22, 0.83, lab, ha="center", fontsize=6.5, color=C_GREY)
    ax.text(0.5, 0.06, "shared: reject / approve layout\nvaries: character and content", ha="center", fontsize=6.5)
    # (c) visually similar drawn faces from two different documented families (soyjak/wojak vs Pepe the Frog)
    ax = axes[2]
    ax.set_title("(c) Similar-looking:\ndifferent families", fontsize=8, loc="left")
    _frame(ax, 0.03, 0.18, 0.44, 0.62)
    ax.add_patch(Ellipse((0.25, 0.47), 0.30, 0.44, color="#5b8fd6"))
    ax.add_patch(Ellipse((0.26, 0.45), 0.22, 0.34, color="#9bd48f"))
    ax.add_patch(Circle((0.22, 0.50), 0.018, color="black"))
    ax.add_patch(Circle((0.30, 0.50), 0.018, color="black"))
    ax.plot([0.20, 0.31], [0.38, 0.40], color="black", lw=0.8)
    ax.text(0.25, 0.83, "family A", ha="center", fontsize=6.5, color=C_GREY)
    _frame(ax, 0.53, 0.18, 0.44, 0.62)
    ax.add_patch(Ellipse((0.75, 0.47), 0.36, 0.34, color="#7fbf6a"))
    ax.add_patch(Circle((0.69, 0.55), 0.05, color="white", ec="black", lw=0.6))
    ax.add_patch(Circle((0.81, 0.55), 0.05, color="white", ec="black", lw=0.6))
    ax.add_patch(Circle((0.69, 0.55), 0.02, color="black"))
    ax.add_patch(Circle((0.81, 0.55), 0.02, color="black"))
    ax.add_patch(Ellipse((0.75, 0.39), 0.20, 0.05, color="#c0583f"))
    ax.text(0.75, 0.83, "family B", ha="center", fontsize=6.5, color=C_GREY)
    ax.text(0.5, 0.06, "shared: drawn green face\ndifferent documented families", ha="center", fontsize=6.5)
    save(fig, "Figure1_meme_family_concept")


def figure1_provenance():
    dev = pd.read_csv(DEV_INDEX).set_index("variant_id")
    rows = []
    plan = [("a", "nobody_literally_no_one", sig["nobody_literally_no_one"]["top_member_exemplars"][:3],
             "phrasal formula ('nobody:' / varying line) over different base images"),
            ("b", "drake_hotline_bling_drakeposting", sig["drake_hotline_bling_drakeposting"]["top_member_exemplars"][:2],
             "same reject/approve two-row layout with two different base characters within one documented family"),
            ("c", "wojak_soyjak", sig["wojak_soyjak"]["top_member_exemplars"][:1],
             "family A member (drawn green face in hood)"),
            ("c", "pepe_the_frog", sig["wojak_soyjak"]["top_nonmember_exemplars"][:1],
             "family B: the highest-scoring nonmember under the soyjak/wojak signature (fit %.3f)" %
             sig["wojak_soyjak"]["top_nonmember_fit"][0])]
    for panel, family, ids, role in plan:
        for v in ids:
            p = Path(dev.loc[v, "image_path"])
            rows.append(dict(panel=panel, family=family, variant_id=v, image_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                             role=role))
    PROV.mkdir(parents=True, exist_ok=True)
    with open(PROV / "figure1_sources.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (PROV / "figure1_selection_procedure.md").write_text(
        "# Figure 1: source cases and selection procedure\n\n"
        "The panels are schematic redraws. No image pixels are reproduced; each panel draws the shared layout of\n"
        "the development-corpus variants listed in `figure1_sources.csv` (DEV variant IDs + SHA-256 of the local images).\n\n"
        "**Actual selection procedure (author-chosen pedagogical cases; no fixed rule is claimed):**\n"
        "- Candidate variants were taken from each family's top member exemplars in the submitted signature export\n"
        "  (`analysis/figure_inputs/out/signatures_submitted_export.json`) and viewed by Claude on 2026-09-30.\n"
        "- (a) *nobody / literally no one* was chosen because its members visibly share a caption formula over unrelated\n"
        "  base images (phrasal persistence).\n"
        "- (b) *drakeposting* was chosen because its documented members include the reject/approve two-row layout with\n"
        "  two different base characters (relational-layout persistence across base templates).\n"
        "- (c) the look-alike pair uses soyjak/wojak and its highest-scoring nonmember under that family's signature,\n"
        "  a Pepe the Frog variant (drawn green faces from different documented families). Among the five highest\n"
        "  top-nonmember fits in the export, this pair was chosen as the most legible schematic.\n"
        "- No held-out-family image was consulted or used. **Pending Eric's visual approval.**\n")


# ---------------------------------------------------------------- Figure 2
def figure2():
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_axes([0.0, 0.52, 1.0, 0.46])
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    boxes = [("Image", "variant or\ncandidate"), ("CLIP ViT-L/14", "768-d,\nunnormalized"),
             ("Sparse\nautoencoder", "6,144 latents,\n64 active"), ("Member −\nbackground", "contrast d$_f$"),
             ("Signature", "top K = 15\nfeatures, weights w$_f$"), ("Score", "cos on the 15\nfeatures: r(c, f)")]
    xs = np.linspace(0.1, 8.5, len(boxes))
    for (t, s), x in zip(boxes, xs):
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.3, 1.0, boxstyle="round,pad=0.05", fc="#eef3fa", ec=C_DEV))
        ax.text(x + 0.65, 1.25, t, ha="center", va="center", fontsize=6.5, weight="bold")
        ax.text(x + 0.65, 0.85, s, ha="center", va="center", fontsize=6.5)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + 1.45 + 0.05, 1.05), xytext=(x + 1.35, 1.05),
                    arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(5.0, 0.18, "raw signature score = max$_f$ r(c, f);  specificity margin = top minus second family",
            ha="center", fontsize=6.8)
    ax.text(0.0, 1.95, "(a) Estimation pipeline", fontsize=8, weight="bold", va="top")
    ax2 = fig.add_axes([0.08, 0.06, 0.88, 0.38])
    s = sig["trade_offer"]
    y = np.arange(15)
    ax2.bar(y, s["weights"], color=C_DEV, width=0.7)
    ax2.set_xticks(y)
    ax2.set_xticklabels([str(i) for i in s["feature_indices"]], rotation=60, fontsize=6)
    ax2.set_xlabel("retained sparse feature (index)")
    ax2.set_ylabel("weight w$_f$")
    ax2.set_title("(b) Worked example: the 15-feature signature of trade offer (rule-selected: median recovery)",
                  fontsize=8, loc="left", weight="bold")
    save(fig, "Figure2_measurement_framework")


# ---------------------------------------------------------------- Figure 3
def _heldout_forest(ax):
    # F-59 (M13): consequential comparisons first; the supervised classifier is shown in the same
    # contrastive-minus-comparator orientation (sign-flipped from the bound classifier-minus-contrastive values)
    base = [("logistic", "Supervised classifier (logistic)"), ("clip_centroid", "CLIP centroid"),
            ("fused_centroid", "Fused SigLIP2 + DINOv2 centroid"), ("sparse_centroid", "Sparse centroid"),
            ("sparse_knn", "Sparse kNN"), ("recurrence", "Recurrence (frequency-based)"),
            ("surf_composite", "Surface composite"), ("oracle_surface", "Surface oracle")]
    for j, (k, lab) in enumerate(base):
        for off, st, col in [(-0.16, "H", C_H), (0.16, "C", C_C)]:
            if k == "logistic":
                c = ho[st]["classifier_minus_contrastive"]
                r = {"mean": -c["mean"], "ci95": [-c["ci95"][1], -c["ci95"][0]]}
            else:
                r = ho[st]["rq1"][k]
            ax.errorbar(r["mean"], j + off, xerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]],
                        fmt="s" if k == "logistic" else "o", ms=4, color=col, lw=1.2, capsize=2.5)
    ax.axhline(0.5, color="#999999", lw=0.6, ls="--")  # separates the supervised comparator
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(range(len(base)))
    ax.set_yticklabels([lab for _, lab in base], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("contrastive − comparator recovery AUROC (95% family-bootstrap interval)")
    ax.set_title("(a) Held-out-family validation", fontsize=9, loc="left")
    ax.plot([], [], "o", color=C_H, label="H: randomly sampled held-out families (n = %d)" % ho["H"]["n_families"])
    ax.plot([], [], "o", color=C_C, label="C: purposive look-alike challenge families (n = %d)" % ho["C"]["n_families"])
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1)


def figure3():
    """Main Figure 3 (GPT/Eric, pass 5): (a) held-out validation first and larger; (b) development benchmark."""
    m = pd.DataFrame(rev["matrix"])
    fig = plt.figure(figsize=(7.2, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.55)
    _heldout_forest(fig.add_subplot(gs[0]))
    ax = fig.add_subplot(gs[1])
    for i, (k, lab) in enumerate(METHODS):
        v = m[k].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=6, color=C_GREY,
                   alpha=0.6)
        ax.plot([i - 0.3, i + 0.3], [g3["RQ1_means"][k]["mean_local"]] * 2, color="black", lw=1.5)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([lab for _, lab in METHODS], rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("recovery AUROC")
    ax.set_title("(b) Development benchmark (not validation), n = 30 families", fontsize=8, loc="left")
    save(fig, "Figure3_family_recovery")


def figureS13():
    """SI Figure S13 (moved from submitted/pass-4 Figure 3): development-corpus recovery diagnostics."""
    m = pd.DataFrame(rev["matrix"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    for _, r in m.iterrows():
        ax.plot([0, 1], [r.recurrence, r.contrastive], color="#c7c7c7", lw=0.6)
    mu = g3["RQ1_means"]
    ax.plot([0, 1], [mu["recurrence"]["mean_local"], mu["contrastive"]["mean_local"]], color="black", lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Recurrence", "Contrastive"])
    ax.set_ylabel("recovery AUROC")
    ax.set_title("(a) Development corpus, paired by family (n = 30)", fontsize=8, loc="left")
    ax = axs[1]
    d = np.sort((m.contrastive - m.sparse_centroid).to_numpy())
    ax.bar(range(len(d)), d, color=[C_DEV if x > 0 else C_GREY for x in d])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("development family (sorted)")
    ax.set_ylabel("contrastive − sparse centroid")
    ax.set_title("(b) Development corpus: per-family difference", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "FigureS13_development_diagnostics")


# ---------------------------------------------------------------- Figure 4
def figure4():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})  # F-59 (M14): held-out panel enlarged
    ax = axs[0]
    ks = [k for k, _ in METHODS]
    for off, key, col, lab in [(-0.18, "laion_encoder_selected", C_DEV, "encoder-selected (LAION CLIP)"),
                               (0.18, "phash_selected", C_GREY, "perceptual-hash-selected")]:
        mm = g3["RQ2"][key]["means"]
        ax.bar(np.arange(len(ks)) + off, [mm[k] for k in ks], width=0.36, color=col, label=lab)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([lab for _, lab in METHODS], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("mean discrimination AUROC")
    ax.set_title("(a) Development corpus", fontsize=8, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax = axs[1]
    rows = [("H", "rq2_laion_familylevel", C_H, "o", "H: random,\nencoder-selected"),
            ("C", "rq2_laion_familylevel", C_C, "o", "C: purposive challenge,\nencoder-selected"),
            ("H", "phash_decoys_descriptive", C_H, "D", "H: random,\nperceptual-hash-selected")]
    for j, (st, key, col, mk, lab) in enumerate(rows):
        r = ho[st][key]
        ax.errorbar(r["mean"], j, xerr=[[r["mean"] - r["ci95"][0]], [r["ci95"][1] - r["mean"]]], fmt=mk,
                    color=col, capsize=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for *_, lab in rows], fontsize=6.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("contrastive − CLIP centroid discrimination AUROC\n(95% family-bootstrap interval)")
    ax.set_title("(b) Held-out families", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "Figure4_adversarial")


# ---------------------------------------------------------------- Figure 5 and S10
def _decomp(axs):
    inc = g1["increments"]
    order = [("raw_fit", "Raw signature"), ("sae_centroid", "Sparse centroid"), ("clip_norm", "Whole-image magnitude"),
             ("sparse_norm", "Sparse-activation magnitude"), ("calibrated_fit", "Family-calibrated fit"),
             ("classifier_prob", "Classifier score"), ("random_sig_max", "Random-feature signature")]
    ax = axs[0]
    ax.barh(range(len(order)), [inc[k]["dR2"] for k, _ in order], color=C_DEV)
    # F-59 (M15): the control-adjusted residual, shown apart from the score constructions, with its interval
    yr = len(order) + 0.4
    adj = g1["raw_over_D_plus_magnitude"]["dR2"]
    ci = rpb["g1_full_raw_over_magnitude_seed0"]["dR2"]["ci95"]
    ax.axhline(len(order) - 0.3, color="#999999", lw=0.6, ls="--")
    ax.errorbar(adj, yr, xerr=[[adj - ci[0]], [ci[1] - adj]], fmt="D", color=C_H, capsize=3, ms=4)
    ax.set_yticks(list(range(len(order))) + [yr])
    ax.set_yticklabels([lab for _, lab in order] + ["Raw signature after\nrepresentation-magnitude\ncontrols (95% interval)"],
                       fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("incremental R² over control model")
    ax.set_title("(a) Score construction", fontsize=8, loc="left")
    obs = inc["raw_fit"]["dR2"]
    for ax, kind, lab in [(axs[1], "random_feature", "(b) Random-feature banks"),
                          (axs[2], "label_perm", "(c) Label-permutation banks")]:
        v = pd.read_csv(AN / f"g1_full_sample/out/g1_nulls_{kind}_full_4263_seed0.csv").iloc[:, 0].to_numpy()
        ax.hist(v, bins=30, color="#c7c7c7")
        ax.axvline(obs, color="black", lw=1.2)
        ax.set_xlabel("incremental R²")
        ax.set_title(f"{lab} (B = {len(v)})", fontsize=8, loc="left")
    return obs


def figure5():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    _decomp([axs[0, 0], axs[0, 1], axs[1, 0]])
    ax = axs[1, 1]
    pc = g1["per_community_corr_raw"]
    ks = sorted(pc, key=lambda k: -pc[k])
    ax.bar(range(len(ks)), [pc[k] for k in ks], color=C_DEV)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["r/" + k for k in ks], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylabel("correlation (raw score, log score)")
    ax.set_title("(d) By community (all seven)", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "Figure5_decomposition")


def figureS10():
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.8))  # wider: panel titles collided at 7.2 in (F-58)
    _decomp(list(axs))
    fig.tight_layout()
    save(fig, "FigureS10_decomposition_nulls")


# ---------------------------------------------------------------- SI figures
def figureS3():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    kc = g3["K_curve_clean_local"]
    ks = sorted(kc, key=lambda s: int(s[1:]))
    axs[0].plot([int(k[1:]) for k in ks], [kc[k] for k in ks], "o-", color=C_DEV)
    axs[0].axvline(15, color=C_GREY, ls=":", lw=0.8)
    axs[0].set_xlabel("retained features K")
    axs[0].set_ylabel("mean recovery AUROC (contrastive)")
    axs[0].set_title("(a) K curve (development, corrected protocol)", fontsize=8, loc="left")
    mm = g3["seed_sweep_1_10"]["method_means"]
    labs = [(k + "__E_local", lab) for k, lab in METHODS]
    for i, (k, lab) in enumerate(labs):
        lo, hi = mm[k]
        axs[1].plot([lo, hi], [i, i], color=C_DEV, lw=3)
    axs[1].set_yticks(range(len(labs)))
    axs[1].set_yticklabels([lab for _, lab in labs], fontsize=6.5)
    axs[1].invert_yaxis()
    axs[1].set_xlabel("mean recovery AUROC\n(range over 10 negative redraws)")
    axs[1].set_title("(b) Redraws of evaluation negatives", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "FigureS3_sensitivity")


def figureS4():
    m = pd.DataFrame(rev["matrix"]).sort_values("contrastive")
    ks = [k for k, _ in METHODS]
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    im = ax.imshow(m[ks].to_numpy(), aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.family, fontsize=6)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([lab for _, lab in METHODS], rotation=40, ha="right", fontsize=6.5)
    fig.colorbar(im, ax=ax, label="recovery AUROC")
    save(fig, "FigureS4_recovery_heatmap")


def figureS5():
    m = pd.DataFrame(rev["matrix"])
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for off, (k, lab) in enumerate(METHODS[:5]):
        d = m.contrastive - m[k]
        axs[0].scatter(np.full(len(d), off) + np.random.default_rng(off).uniform(-.15, .15, len(d)), d, s=6, color=C_DEV)
    axs[0].axhline(0, color="black", lw=0.6)
    axs[0].set_xticks(range(5))
    axs[0].set_xticklabels([lab for _, lab in METHODS[:5]], rotation=35, ha="right", fontsize=6.5)
    axs[0].set_ylabel("contrastive − baseline")
    axs[0].set_title("(a) Paired differences by family", fontsize=8, loc="left")
    axs[1].scatter(m.n, m.contrastive, s=8, color=C_DEV)
    axs[1].set_xlabel("family size (members)")
    axs[1].set_ylabel("contrastive recovery AUROC")
    axs[1].set_title("(b) Recovery and family size", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "FigureS5_paired_size")


def figureS6():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, col, lab, key in [(axs[0], "DL", "encoder-selected (LAION CLIP) decoys", "laion_encoder_selected"),
                              (axs[1], "DP", "perceptual-hash-selected decoys", "phash_selected")]:
        for i, (k, name) in enumerate(METHODS):
            v = fam[f"{k}__{col}_local"].to_numpy()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-.15, .15, len(v)), v, s=6, color=C_DEV, alpha=.6)
            ax.plot([i - .3, i + .3], [g3["RQ2"][key]["means"][k]] * 2, color="black")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([n for _, n in METHODS], rotation=35, ha="right", fontsize=6.5)
        ax.set_title(lab, fontsize=8, loc="left")
    axs[0].set_ylabel("family discrimination AUROC")
    fig.tight_layout()
    save(fig, "FigureS6_adversarial")


def figureS7():
    cases = [("trade offer", "trade_offer", "worked example (median recovery)")]
    for key, lab in [("largest_laion_advantage", "largest encoder-decoy advantage"),
                     ("smallest_laion_advantage", "smallest encoder-decoy advantage"),
                     ("largest_classifier_gap", "largest classifier gap")]:
        f = rev["illustrative"][key]["family"]
        cases.append((f, f.replace(" ", "_"), lab))
    m = pd.DataFrame(rev["matrix"]).set_index("family")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    for ax, (name, k, lab) in zip(axs.ravel(), cases):
        s = sig[k]
        ax.bar(range(15), s["weights"], color=C_DEV)
        ax.set_xticks(range(15))
        ax.set_xticklabels([str(i) for i in s["feature_indices"]], rotation=60, fontsize=5.5)
        ax.set_title(f"{name}: {lab}", fontsize=7.5, loc="left")
        ax.text(0.98, 0.95, "contrastive AUROC %.3f\ntop-scoring nonmember:\n%s (fit %.3f)" %
                (m.loc[name, "contrastive"], s["top_nonmember_exemplars"][0].replace("pool_", "").split("__")[0].replace("_", " "),
                 s["top_nonmember_fit"][0]), transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_ylabel("weight", fontsize=7)
    fig.tight_layout()
    save(fig, "FigureS7_worked_examples")


def figureS9():
    pooled = g2["pooled"]
    rows = [("Random folds", pooled["random5"]["dR2"], None),
            ("pHash cluster (planned)", pooled["template_cc10_as_planned"]["dR2"], rpb["g2_template_cc10_as_planned"]),
            ("Best-fit family", pooled["bestfit_family"]["dR2"], rpb["g2_bestfit_family"]),
            ("Community (= LOCO)", pooled["community_gkf7_eq_LOCO"]["dR2"], None),
            ("Exclude r/whenthe", lv["excl_whenthe"]["dR2"], None),
            ("Exclude top 1%", lv["excl_top1"]["dR2"], None),
            ("Exclude top 5%", lv["excl_top5"]["dR2"], None),
            ("Selected listing (n = %d)" % lv["selected_listing"]["n"], lv["selected_listing"]["dR2"],
             {"dR2": lv["selected_listing"]["boot_dR2"]})]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (lab, v, b) in enumerate(rows):
        if b:
            lo, hi = b["dR2"]["ci95"]
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=C_DEV, capsize=2)
        else:
            ax.plot(v, i, "o", color=C_DEV)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("continuous-outcome incremental R²\n(95% cluster-bootstrap interval where computed)")
    save(fig, "FigureS9_sensitivity")


def figureS12():
    pf = pd.read_csv(AN / "g2_template_cluster/out/g2_per_fold.csv")
    schemes = list(dict.fromkeys(pf.scheme))
    rep = g2["repeated_cv_seeds_1_20"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, col in [(axs[0], "dR2"), (axs[1], "dAUC")]:
        for i, s in enumerate(schemes):
            v = pf[pf.scheme == s][col].to_numpy()
            ax.scatter(v, np.full(len(v), i), s=12, color=C_DEV)
            if s == "community_gkf7_eq_LOCO" and col == "dR2":  # `28` A2: label the held-out r/whenthe fold (n = 585)
                w = pf[(pf.scheme == s) & (pf.n_test == 585)][col].to_numpy()
                ax.annotate("r/whenthe held out", (w[0], i), xytext=(4, 5), textcoords="offset points",
                            ha="left", fontsize=6)
            if s in rep:
                ax.plot(rep[s][col]["median"], i, "|", ms=14, color="black")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("per-fold Δ" + ("R²" if col == "dR2" else "AUROC"))
    axs[0].set_yticks(range(len(schemes)))
    lab = {"random5": "Random folds", "bestfit_family": "Best-fit family", "community_gkf7_eq_LOCO": "Community (= LOCO)",
           "template_cc10_as_planned": "pHash cluster (planned, Hamming 10)", "template_cc6": "pHash cluster (Hamming 6)",
           "template_cc8": "pHash cluster (Hamming 8)", "template_greedy12": "pHash cluster (greedy 12)"}
    axs[0].set_yticklabels([lab.get(s, NICE(s)) for s in schemes], fontsize=6.5)
    axs[0].invert_yaxis()
    axs[0].set_title("Points: folds; bar: repeated-CV median (seeds 1–20)", fontsize=7, loc="left")
    fig.tight_layout()
    save(fig, "FigureS12_per_fold")


# ---------------------------------------------------------------- Figure S2 (redrawn for layout only; content as submitted)
def figureS2():
    """Same boxes, labels and note as the submitted FigureS2_dedup_schematic.png; redrawn because the submitted raster
    had labels overlapping the arrows and a clipped final box (PDF proof, 09 F-45). Schematic; no statistics."""
    from matplotlib.patches import FancyBboxPatch
    labels = ["707 within-family\ndeduplicated\nvariants", "SHA-256\nexact-duplicate\nremoval",
              "pHash\nnear-duplicate\nremoval (hash 8,\nHamming \u2264 6)", "Cross-family\noverlap\nresolution",
              "680 analyzed\nvariants\n(31 families)"]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(-0.15, 10.15)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    w, h, gap = 1.7, 1.2, (10 - 5 * 1.7) / 4
    xs = [i * (w + gap) for i in range(5)]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        ax.add_patch(FancyBboxPatch((x, 0.75), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eef4fb", ec="#2c6fb0", lw=1.0))
        ax.text(x + w / 2, 0.75 + h / 2, lab, ha="center", va="center", fontsize=6.8, linespacing=1.15)
        if i < 4:
            ax.annotate("", xy=(xs[i + 1] - 0.04, 0.75 + h / 2), xytext=(x + w + 0.04, 0.75 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color="#2c6fb0", lw=1.0))
    ax.text(5, 0.3, "Cross-family step: 23 exact Wojak/Soyjak duplicates merged into one family; "
                    "1 image filed under four families excluded as contamination.",
            ha="center", va="center", fontsize=6.5, style="italic", color="#444")
    save(fig, "FigureS2_dedup_schematic")


# ---------------------------------------------------------------- Figure S11 (redrawn from the audited export; 09 F-65)
def figureS11():
    """Short-window prospective analysis, full cohort, from analysis/prospective/out/prospective.json (the submitted
    raster used hand-typed values; same estimates, now bound). Symmetric axis so a null is not drawn as an effect."""
    pr = J("prospective/out/prospective.json")
    waves = [("t4h", "+4 h"), ("t8h", "+8 h"), ("t24h", "+24 h"), ("t72h", "+72 h"), ("t7d", "+7 d")]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, (k, lab) in enumerate(waves):
        r = pr["waves"][k]["full"]
        if r["status"] == "ok":
            ax.bar(i, r["dR2"], width=0.55, color=C_DEV)
            ax.text(i, 0.0105, "n = %s" % format(r["n"], ","), ha="center", fontsize=6.5)
        else:
            ax.text(i, 0.006, "not\ncollected", ha="center", va="center", fontsize=6.5, color="#666666")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-0.015, 0.015)
    ax.set_xticks(range(len(waves)))
    ax.set_xticklabels([lab for _, lab in waves])
    ax.set_ylabel("incremental R²\n(after initial engagement)")
    ax.set_title("Raw signature increment for engagement growth, by follow-up wave", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "FigureS11_prospective")


# ---------------------------------------------------------------- Figure S1 (redrawn from the audited corpus counts; 09 F-66)
def figureS1():
    """The submitted raster labelled the median as 24 (the 31-family median); the analysed-family median is 24.5."""
    cc = J("corpus/out/corpus_counts.json")
    sizes = np.array(list(cc["family_sizes"].values()))
    rec = sizes[sizes >= 5]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    bins = np.arange(0.5, rec.max() + 1.5, 1)
    ax.hist(rec, bins=bins, color=C_DEV, edgecolor="white")
    ax.hist(sizes[sizes < 5], bins=bins, color="#bbbbbb", edgecolor="white")
    ax.axvline(cc["recovery_size_median"], color="black", lw=0.8, ls="--")
    ax.text(cc["recovery_size_median"] - 0.3, ax.get_ylim()[1] * 0.92, "median %g (analysed)" % cc["recovery_size_median"],
            ha="right", fontsize=6.5)
    ax.text(2, 1.3, "excluded\n(n = %d)" % cc["excluded_family_size"], ha="center", fontsize=6.5, color="#555555")
    ax.set_xlabel("variants per family")
    ax.set_ylabel("families")
    fig.tight_layout()
    save(fig, "FigureS1_family_sizes")


def copy_unchanged():
    for n in []:  # S1, S2 and S11 are redrawn (figureS1, figureS2, figureS11)
        shutil.copy(SUB_FIG / n, OUT / n)


def main():
    figure1()
    figure1_provenance()
    figure2()
    figure3()
    figure4()
    figure5()
    figureS3()
    figureS4()
    figureS5()
    figureS6()
    figureS7()
    figureS9()
    figureS10()
    figureS12()
    figureS13()
    copy_unchanged()
    figureS1()
    figureS2()
    figureS11()
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
