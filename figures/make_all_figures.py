"""Rebuild main Figure 4 (4-panel with real null histograms) and generate the data-backed
supplementary figures (S1,S3,S4,S5,S6,S9,S10,S11,S12) as 300-dpi grayscale PNGs.
Also export main figures as SVG. Run from build/."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9",
                     "grid.linewidth": 0.6, "axes.edgecolor": "0.3", "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
F = "../SSCR Submission/03_Figures/"; AA = "../SSCR Submission/07_Analysis_Audit/"
# Okabe-Ito colorblind-safe palette
C_CON = "#0072B2"   # contrastive / primary (blue)
C_BASE = "#E69F00"  # baseline / whole-image (orange)
C_NEG = "#D55E00"   # observed reference / negative (vermillion)
C_ACC = "#009E73"   # accent (green)
C_NULL = "#9AA7B0"  # muted gray for null / near-zero
s7 = pd.read_csv("../SSCR Submission/04_Tables/TableS7_family_recovery_matrix.csv")
m2 = pd.read_csv("first_results/ph4_manuscript_natcomms/natcomms_revision_v2/analysis_to_run/m2m4_results.csv")
s1 = pd.read_csv("../SSCR Submission/04_Tables/TableS1_corpus_composition.csv")
rf = pd.read_csv(AA + "null_random_feature.csv")["increment"].values
lp = pd.read_csv(AA + "null_label_perm.csv")["increment"].values
OBS = 0.080

def save(fig, name, svg=False):
    fig.savefig(F + name + ".png")
    if svg: fig.savefig(F + name + ".svg")
    plt.close(fig)

# ---- MAIN FIGURE 4 (rebuilt): decomposition + 2 null histograms + community ----
fig, ax = plt.subplots(2, 2, figsize=(7.6, 5.4))
labs = ["Raw sig.", "Sparse cent.", "CLIP norm", "Sparse norm", "Calib. fit", "Classif.", "Random"]
vals = [0.080, 0.056, 0.036, 0.030, -0.000, -0.002, -0.000]
ax[0,0].barh(range(len(vals))[::-1], vals, color=np.where(np.array(vals) > 0.01, C_CON, C_NULL))
ax[0,0].set_yticks(range(len(vals))[::-1]); ax[0,0].set_yticklabels(labs, fontsize=7.5)
ax[0,0].axvline(0, color="black", lw=0.8); ax[0,0].set_xlabel("Incremental R²"); ax[0,0].set_title("(a) Score decomposition", fontsize=9)
ax[0,1].hist(rf, bins=30, color=C_CON, edgecolor="white", lw=0.3, alpha=0.85); ax[0,1].axvline(OBS, color=C_NEG, lw=2)
ax[0,1].text(OBS, ax[0,1].get_ylim()[1]*0.9, " observed\n 0.080", fontsize=7, va="top", color=C_NEG)
ax[0,1].set_xlabel("Incremental R²"); ax[0,1].set_title("(b) Random-feature null (500)", fontsize=9)
ax[1,0].hist(lp, bins=25, color=C_CON, edgecolor="white", lw=0.3, alpha=0.85); ax[1,0].axvline(OBS, color=C_NEG, lw=2)
ax[1,0].text(OBS, ax[1,0].get_ylim()[1]*0.9, " observed\n 0.080", fontsize=7, va="top", color=C_NEG)
ax[1,0].set_xlabel("Incremental R²"); ax[1,0].set_title("(c) Label-permutation null (200)", fontsize=9)
comm = ["meme", "memes", "shitpost.", "dankmemes", "me_irl", "historymemes", "whenthe"]; cor = [0.339, 0.322, 0.299, 0.167, 0.163, 0.157, 0.020]
ax[1,1].barh(range(len(cor))[::-1], cor, color=C_CON); ax[1,1].set_yticks(range(len(cor))[::-1])
ax[1,1].set_yticklabels([f"r/{x}" for x in comm], fontsize=7); ax[1,1].axvline(0, color="black", lw=0.8)
ax[1,1].set_xlabel("corr. with log engagement"); ax[1,1].set_title("(d) By community", fontsize=9)
fig.tight_layout(); save(fig, "Figure4_engagement", svg=True)

# ---- SI S1: family-size distribution ----
fig, a = plt.subplots(figsize=(4.4, 2.8))
a.hist(s1["final_n"], bins=range(0, 30, 2), color=C_CON, edgecolor="white", alpha=0.85)
a.axvline(24, color=C_NEG, ls="--", lw=1.5); a.text(24, a.get_ylim()[1]*0.9, " median 24", fontsize=8, color=C_NEG)
a.set_xlabel("Final family size (variants)"); a.set_ylabel("Number of families"); a.set_title("Family-size distribution", fontsize=9)
save(fig, "FigureS1_family_sizes")

# ---- SI S3: K-sweep + negative-pool ----
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
axs[0].plot([5,10,15,25,50], [0.877,0.890,0.897,0.903,0.905], "o-", color=C_CON)
axs[0].axvline(15, color=C_NEG, ls=":"); axs[0].set_xlabel("K (retained features)"); axs[0].set_ylabel("Mean recovery AUROC")
axs[0].set_title("(a) K sensitivity", fontsize=9); axs[0].set_ylim(0.85, 0.92)
axs[1].plot(range(1,6), [0.898,0.895,0.894,0.895,0.896], "s", color=C_CON, ms=6)
axs[1].axhline(0.896, color="0.6", ls=":"); axs[1].set_xlabel("Negative sample"); axs[1].set_ylabel("Mean recovery AUROC")
axs[1].set_title("(b) Negative-pool sensitivity", fontsize=9); axs[1].set_ylim(0.88, 0.91); axs[1].set_xticks(range(1,6))
fig.tight_layout(); save(fig, "FigureS3_sensitivity")

# ---- SI S4: recovery heatmap ----
methods = ["recur","clip","surf_oracle","sae_cent","sae_knn","contrastive","logistic"]
mlab = ["Recur.","Whole-img","Surface","Sp.cent.","Sp.kNN","Contrast","Classif."]
d = s7.sort_values("contrastive").reset_index(drop=True)
fig, a = plt.subplots(figsize=(5.2, 6.8))
im = a.imshow(d[methods].values, cmap="cividis", aspect="auto", vmin=0.5, vmax=1.0)
a.set_xticks(range(len(methods))); a.set_xticklabels(mlab, rotation=45, ha="right", fontsize=7)
a.set_yticks(range(len(d))); a.set_yticklabels(d["family"], fontsize=5)
a.set_title("Recovery AUROC by family and method", fontsize=9)
fig.colorbar(im, ax=a, fraction=0.04, label="AUROC")
fig.tight_layout(); save(fig, "FigureS4_recovery_heatmap")

# ---- SI S5: paired diffs + recovery vs size ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
diff_c = (s7["contrastive"]-s7["sae_cent"]).sort_values().values
diff_w = (s7["contrastive"]-s7["clip"]).values[np.argsort((s7["contrastive"]-s7["sae_cent"]).values)]
x = range(len(diff_c))
axs[0].bar(x, diff_c, color=C_CON, width=0.9, label="vs sparse centroid")
axs[0].plot(x, diff_w, ".", color=C_BASE, ms=5, label="vs whole-image")
axs[0].axhline(0, color="black", lw=0.8); axs[0].set_xlabel("Family (sorted)"); axs[0].set_ylabel("Contrastive − baseline AUROC")
axs[0].legend(fontsize=7, frameon=False); axs[0].set_title("(a) Paired differences", fontsize=9)
axs[1].scatter(s7["n"], s7["contrastive"], s=16, color=C_CON)
axs[1].set_xlabel("Family size (variants)"); axs[1].set_ylabel("Contrastive recovery AUROC"); axs[1].set_title("(b) Recovery vs family size", fontsize=9)
fig.tight_layout(); save(fig, "FigureS5_paired_size")

# ---- SI S6: adversarial per-family ----
merged = s7.merge(m2[["family","adv_con_phash","adv_clip_phash"]], on="family", how="inner")
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax_, (cc, cl, ttl) in zip(axs, [("adv_con","adv_clip","(a) Encoder-selected decoys"),
                                     ("adv_con_phash","adv_clip_phash","(b) pHash-selected decoys")]):
    o = merged.sort_values(cc)
    ax_.plot(range(len(o)), o[cc], "o", color=C_CON, ms=4, label="Contrastive")
    ax_.plot(range(len(o)), o[cl], "^", color=C_BASE, ms=4, label="Whole-image")
    ax_.axhline(0.5, color="0.7", ls=":"); ax_.set_xlabel("Family (sorted)"); ax_.set_title(ttl, fontsize=9)
axs[0].set_ylabel("Discrimination AUROC"); axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout(); save(fig, "FigureS6_adversarial")

# ---- SI S9: attrition ----
fig, a = plt.subplots(figsize=(5.0, 2.9))
cm = ["dankmemes","memes","meme","whenthe","historymemes","shitposting","me_irl"]
tot = [749,726,711,585,573,562,357]; ret = [749,726,711,0,0,0,357]
x = np.arange(len(cm))
a.bar(x, tot, color=C_NULL, label="Denominator"); a.bar(x, ret, color=C_CON, label="Re-encodable")
a.set_xticks(x); a.set_xticklabels([f"r/{c}" for c in cm], rotation=45, ha="right", fontsize=7)
a.set_ylabel("Posts"); a.legend(fontsize=7.5, frameon=False); a.set_title("Re-encoding retention by community", fontsize=9)
fig.tight_layout(); save(fig, "FigureS9_attrition")

# ---- SI S10: sensitivity forest ----
fig, a = plt.subplots(figsize=(5.2, 3.0))
items = [("Family-grouped (primary)", 0.072, 0.017, 0.123), ("Community-grouped", 0.052, None, None),
         ("Leave-one-community-out", 0.054, None, None), ("Exclude r/whenthe", 0.054, None, None),
         ("Exclude top 1%", 0.044, None, None), ("Exclude top 5%", 0.040, None, None),
         ("Selected-listing (n=348)", -0.007, -0.012, 0.015)]
for i, (lab, v, lo, hi) in enumerate(items[::-1]):
    y = i
    if lo is not None: a.plot([lo, hi], [y, y], color=C_CON, lw=1.5)
    a.plot(v, y, "o", color=(C_NEG if v < 0 else C_CON), ms=6)
a.axvline(0, color="black", lw=0.8); a.set_yticks(range(len(items))); a.set_yticklabels([it[0] for it in items[::-1]], fontsize=7.5)
a.set_xlabel("Continuous-outcome incremental R²"); a.set_title("Engagement increment across designs and sensitivity", fontsize=9)
fig.tight_layout(); save(fig, "FigureS10_sensitivity")

# ---- SI S11: decomposition + nulls (fuller) ----
fig, axs = plt.subplots(1, 3, figsize=(8.0, 2.9))
axs[0].barh(range(len(vals))[::-1], vals, color=np.where(np.array(vals) > 0.01, C_CON, C_NULL))
axs[0].set_yticks(range(len(vals))[::-1]); axs[0].set_yticklabels(labs, fontsize=7); axs[0].axvline(0, color="black", lw=0.8)
axs[0].set_xlabel("Incremental R²"); axs[0].set_title("(a) Decomposition", fontsize=9)
axs[1].hist(rf, bins=30, color=C_CON, edgecolor="white", lw=0.3, alpha=0.85); axs[1].axvline(OBS, color=C_NEG, lw=2)
axs[1].axvline(np.percentile(rf,95), color=C_BASE, ls="--", lw=1); axs[1].set_xlabel("Incremental R²"); axs[1].set_title("(b) Random-feature null", fontsize=9)
axs[2].hist(lp, bins=25, color=C_CON, edgecolor="white", lw=0.3, alpha=0.85); axs[2].axvline(OBS, color=C_NEG, lw=2)
axs[2].axvline(np.percentile(lp,95), color=C_BASE, ls="--", lw=1); axs[2].set_xlabel("Incremental R²"); axs[2].set_title("(c) Label-permutation null", fontsize=9)
fig.tight_layout(); save(fig, "FigureS11_decomposition_nulls")

# ---- SI S12: prospective ----
fig, a = plt.subplots(figsize=(5.0, 2.8))
waves = ["+4 h", "+8 h", "+24 h", "+72 h", "+7 d"]; inc = [-0.002, -0.002, np.nan, np.nan, np.nan]
xx = np.arange(len(waves))
a.bar(xx, [i if not np.isnan(i) else 0 for i in inc], color=[C_CON,C_CON,"#DDDDDD","#DDDDDD","#DDDDDD"], edgecolor="0.5")
for i in [2,3,4]: a.text(i, 0.001, "not\ncollected", ha="center", fontsize=6.5, color="0.4")
a.axhline(0, color="black", lw=0.8); a.set_xticks(xx); a.set_xticklabels(waves)
a.set_ylabel("Incremental R² after initial engagement"); a.set_title("Short-window prospective analysis", fontsize=9)
fig.tight_layout(); save(fig, "FigureS12_prospective")

import os
print("generated:")
for f in os.listdir(F):
    if f.endswith(".png"): print(" ", f, os.path.getsize(F+f)//1000, "KB")
