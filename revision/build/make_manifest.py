#!/usr/bin/env python3
"""Write 19_Revision/package/REPRODUCIBILITY_MANIFEST.csv (AI-14): every revision analysis script and output, with
SHA-256, size, the producing script and command, and the SI Table S22 row it belongs to. Read-only over analysis/.
Retrieved held-out images (analysis/heldout/out/images/, and the aborted first run's images) are not redistributed; they
appear as one summary row each, and their per-file hashes are in heldout_manifest.csv. The human-validation branch is
listed by script only (its coder-facing materials and results are outside this manifest until that branch closes).
Superseded submitted scripts are listed with the hash of the frozen copy in SSCR Submission/_Working/09_Reproducibility.
Usage (from `SSCR Revise and Resubmit/`): python3 19_Revision/build/make_manifest.py
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
RR = Path(__file__).resolve().parents[2]
AN = RR / "analysis"
REPRO = RR.parent / "SSCR Submission/_Working/09_Reproducibility"
OUT = RR / "19_Revision/package/REPRODUCIBILITY_MANIFEST.csv"

S22 = {
    "g3": "Development recovery, leave-one-variant-out protocol",
    "rpd": "Fold-local decoy sensitivity",
    "ho": "Held-out-family validation",
    "x1": "Exploratory decoy comparison with the fused centroid",
    "g2": "Engagement validation schemes",
    "g1": "Full-sample decomposition and randomization analyses",
    "rp": "Corrected bootstrap intervals and Monte Carlo p-values",
    "rev": "Rule-based case selection",
    "lv": "Recomputed sensitivity values",
    "r17": "Top-decile engagement-rate comparison",
    "sig": "Signature export used by Figures 2b and S7",
    "fig": "Figures 1–5 and S1–S13",
    "sup": "Superseded",
    "infra": "(infrastructure: freeze checks, shared helpers, record rendering)",
    "hv": "(superseded two-coder human-validation design: frozen before any draw; never run)",
    "hc": "Blinded author-coded human check",
    "m12": "Cross-source overlap and exclusion sensitivity",
    "pro": "Prospective short-window analysis",
    "cc": "Development-corpus counts",
    "bfc": "Best-fit signature concentration",
    "audit": "(audit: independent reconstruction of Table 2; reproduction gate)",
    "doc": "(revision DOCX build)",
}
PY = "PYTHONDONTWRITEBYTECODE=1 python3 "
# (regex on the path relative to `SSCR Revise and Resubmit/`, S22 key, producing script, command)
RULES = [
    (r"analysis/00_freeze/(r1_freeze_manifest\.csv|R1_FREEZE_RECORD\.json)", None, None, None),  # freeze records hash the package (circular); `09` F-72
    (r"analysis/heldout/out/(images|_aborted_run1_2026-09-29)/", None, None, None),  # summarized separately
    (r"analysis/heldout/engineering/out/(fixture|devtest|runbook_fixture)/", None, None, None),  # synthetic fixtures
    (r"analysis/human_validation/(?!.*\.py$)", None, None, None),
    (r"analysis/human_check/out/", "hc", "analysis/human_check/analyze_check.py; summarize_check.py", PY + "analysis/human_check/analyze_check.py && " + PY + "analysis/human_check/summarize_check.py"),
    (r"analysis/human_check/draw/", "hc", "analysis/human_check/draw_check.py", PY + "analysis/human_check/draw_check.py"),
    (r"analysis/human_check/(labels/|LABELS_FREEZE)", "hc", "analysis/human_check/freeze_labels.py", PY + "analysis/human_check/freeze_labels.py RESPONSES.csv"),
    (r"analysis/human_check/", "hc", "", ""),
    (r"analysis/m12_overlap/out/m12_overlap", "m12", "analysis/m12_overlap/m12_overlap.py", PY + "analysis/m12_overlap/m12_overlap.py"),
    (r"analysis/m12_overlap/out/m12_exclusion", "m12", "analysis/m12_overlap/m12_exclusion.py", PY + "analysis/m12_overlap/m12_exclusion.py"),
    (r"analysis/m12_overlap/", "m12", "", ""),
    (r"analysis/prospective/out/", "pro", "analysis/prospective/export_prospective.py", PY + "analysis/prospective/export_prospective.py"),
    (r"analysis/prospective/", "pro", "", ""),
    (r"analysis/concentration/out/", "bfc", "analysis/concentration/export_bestfit_concentration.py", PY + "analysis/concentration/export_bestfit_concentration.py"),
    (r"analysis/concentration/", "bfc", "", ""),
    (r"analysis/corpus/out/", "cc", "analysis/corpus/export_corpus_counts.py", PY + "analysis/corpus/export_corpus_counts.py"),
    (r"analysis/corpus/", "cc", "", ""),
    (r"analysis/repro_gate/out/", "audit", "analysis/repro_gate/reconstruct_table2.py", PY + "analysis/repro_gate/reconstruct_table2.py"),
    (r"analysis/repro_gate/", "audit", "", ""),
    (r"analysis/human_validation/.*\.py$", "hv", "", ""),
    (r"analysis/00_freeze/|analysis/common\.py|analysis/render_(claims|memo)\.py", "infra", "", ""),
    (r"analysis/g3_recovery/out/(features\.npz|ocr_texts\.json)", "g3", "analysis/g3_recovery/01_features.py", PY + "analysis/g3_recovery/01_features.py"),
    (r"analysis/g3_recovery/out/repro_fr25", "g3", "analysis/g3_recovery/02_engine.py", PY + "analysis/g3_recovery/02_engine.py fr25_repro"),
    (r"analysis/g3_recovery/out/repro_m2m4", "g3", "analysis/g3_recovery/02_engine.py", PY + "analysis/g3_recovery/02_engine.py m2m4_repro"),
    (r"analysis/g3_recovery/out/clean_primary", "g3", "analysis/g3_recovery/02_engine.py", PY + "analysis/g3_recovery/02_engine.py clean"),
    (r"analysis/g3_recovery/out/clean_seed_sweep", "g3", "analysis/g3_recovery/02_engine.py", PY + "analysis/g3_recovery/02_engine.py sweep"),
    (r"analysis/g3_recovery/out/clean_kcurve", "g3", "analysis/g3_recovery/02_engine.py", PY + "analysis/g3_recovery/02_engine.py kcurve"),
    (r"analysis/g3_recovery/out/", "g3", "analysis/g3_recovery/03_summarize.py", PY + "analysis/g3_recovery/03_summarize.py"),
    (r"analysis/g3_recovery/", "g3", "", ""),
    (r"analysis/repair/out/decoys_foldlocal", "rpd", "analysis/repair/03_decoys_foldlocal.py", PY + "analysis/repair/03_decoys_foldlocal.py"),
    (r"analysis/repair/out/repair_cheap", "rp", "analysis/repair/02_pvalues_singletons_bank30.py", PY + "analysis/repair/02_pvalues_singletons_bank30.py"),
    (r"analysis/repair/out/", "rp", "analysis/repair/01_bootstrap_samelabel.py", PY + "analysis/repair/01_bootstrap_samelabel.py"),
    (r"analysis/repair/03", "rpd", "", ""),
    (r"analysis/repair/", "rp", "", ""),
    (r"analysis/heldout/exploratory/|analysis/heldout/out/exploratory_x1", "x1", "analysis/heldout/exploratory/x1_hard_decoys_vs_fused.py", PY + "analysis/heldout/exploratory/x1_hard_decoys_vs_fused.py"),
    (r"analysis/heldout/out/(ho_primary|run_primary)", "ho", "analysis/heldout/run_heldout.py", PY + "analysis/heldout/run_heldout.py primary"),
    (r"analysis/heldout/out/(ho_sweep|run_sweep)", "ho", "analysis/heldout/run_heldout.py", PY + "analysis/heldout/run_heldout.py sweep"),
    (r"analysis/heldout/out/(ho_foldlocal|run_foldlocal)", "ho", "analysis/heldout/run_heldout.py", PY + "analysis/heldout/run_heldout.py foldlocal"),
    (r"analysis/heldout/out/.*capped10", "ho", "analysis/heldout/run_heldout.py", PY + "analysis/heldout/run_heldout.py capped10"),
    (r"analysis/heldout/out/.*capped15", "ho", "analysis/heldout/run_heldout.py", PY + "analysis/heldout/run_heldout.py capped15"),
    (r"analysis/heldout/out/dev_modern", "ho", "analysis/heldout/build_features.py", PY + "analysis/heldout/build_features.py --dev-modern"),
    (r"analysis/heldout/out/(expanded_|build_features)", "ho", "analysis/heldout/build_features.py", PY + "analysis/heldout/build_features.py --expanded"),
    (r"analysis/heldout/out/(heldout_results|summarize)", "ho", "analysis/heldout/summarize_heldout.py", PY + "analysis/heldout/summarize_heldout.py"),
    (r"analysis/heldout/out/", "ho", "analysis/heldout/screen_and_retrieve.py", PY + "analysis/heldout/screen_and_retrieve.py"),
    (r"analysis/heldout/", "ho", "", ""),
    (r"analysis/g2_template_cluster/out/(phash|clusters|cluster_diag)", "g2", "analysis/g2_template_cluster/01_phash_clusters.py", PY + "analysis/g2_template_cluster/01_phash_clusters.py"),
    (r"analysis/g2_template_cluster/out/", "g2", "analysis/g2_template_cluster/02_grouped_cv.py; 03_bootstrap.py", PY + "analysis/g2_template_cluster/02_grouped_cv.py && " + PY + "analysis/g2_template_cluster/03_bootstrap.py"),
    (r"analysis/g2_template_cluster/", "g2", "", ""),
    (r"analysis/g1_full_sample/out/(reddit_denom|embed_log)", "g1", "analysis/g1_full_sample/01_embed_all.py", PY + "analysis/g1_full_sample/01_embed_all.py"),
    (r"analysis/g1_full_sample/out/(scores_|identity_check)", "g1", "analysis/g1_full_sample/02_scores.py", PY + "analysis/g1_full_sample/02_scores.py"),
    (r"analysis/g1_full_sample/out/", "g1", "analysis/g1_full_sample/03_decomposition.py", PY + "analysis/g1_full_sample/03_decomposition.py"),
    (r"analysis/g1_full_sample/", "g1", "", ""),
    (r"analysis/revision_selections/", "rev", "analysis/revision_selections/select_cases.py", PY + "analysis/revision_selections/select_cases.py"),
    (r"analysis/legacy_values/", "lv", "analysis/legacy_values/recompute_legacy.py", PY + "analysis/legacy_values/recompute_legacy.py"),
    (r"analysis/r1_7_decile/", "r17", "analysis/r1_7_decile/decile_lift.py", PY + "analysis/r1_7_decile/decile_lift.py"),
    (r"analysis/figure_inputs/", "sig", "analysis/figure_inputs/export_signatures.py", PY + "analysis/figure_inputs/export_signatures.py"),
    (r"19_Revision/build/(figures/|out/figures/|provenance/)", "fig", "19_Revision/build/figures/make_figures_r1.py", PY + "19_Revision/build/figures/make_figures_r1.py"),
    (r"19_Revision/build/", "doc", "19_Revision/build/build_docx_r1.py", "cd 19_Revision/build && " + PY + "build_docx_r1.py OUT.docx {manuscript|supplement} [--clean]"),
]
NOTES = {
    "analysis/r1_7_decile/out/decile_lift_STOPPED.json": "stopped first run (provenance guard); superseded by decile_lift.json",
    "analysis/heldout/out/_aborted_run1_MANIFEST.json": "record of the aborted first retrieval run (retained for provenance)",
}
SUPERSEDED = [
    ("analysis_to_run/RUN_m2m4_full.py", "development recovery and decoys; replaced by analysis/g3_recovery/"),
    ("analysis_to_run/RUN_e2_bootstrap.py", "cluster bootstrap with a label-mismatch defect; replaced by analysis/repair/01_bootstrap_samelabel.py"),
    ("analysis/RUN_downstream_embed_score.py", "subset filter (2,543 posts); replaced by analysis/g1_full_sample/ (4,263 posts)"),
    ("analysis/RUN_save_nulls.py", "raw exceedance p-values; replaced by analysis/repair/02_pvalues_singletons_bank30.py"),
    ("analysis_to_run/e1_grouped_cv.py", "unreferenced script with a 'primary' docstring in the submitted package; not used for any reported result"),
]


# Deposit layout (09 F-68): internal folder names are mapped to portable distribution paths in the manifest.
# The package assembly at upload must use exactly this layout (23_Upload_Checklist.md).
DIST = [("SSCR Submission/_Working/09_Reproducibility/", "submitted_package/"), ("19_Revision/", "revision/")]


def dist(s):
    for a, b in DIST:
        s = s.replace(a, b)
    return s


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = sorted([p for p in AN.rglob("*") if p.is_file()] +
                   [p for p in (RR / "19_Revision/build").rglob("*") if p.is_file()],
                   key=lambda p: str(p))
    rows = []
    for p in files:
        rel = str(p.relative_to(RR))
        if "__pycache__" in rel or p.suffix == ".pyc" or p.name == ".DS_Store" or rel.endswith("build_docx_submitted_copy.py"):
            continue
        rule = next((r for r in RULES if re.match(r[0], rel)), None)
        if rule is None or rule[1] is None:
            continue
        _, key, producer, cmd = rule
        role = "script" if p.suffix == ".py" else "output"
        if role == "script":
            producer, cmd = "", ""
        rows.append(dict(s22_row=S22[key], role=role, path=rel, sha256=sha(p), bytes=p.stat().st_size,
                         produced_by=producer, command=cmd, status="current", note=NOTES.get(rel, "") or ("" if role == "script" or producer else
                         "run log, freeze record or engineering check (see the scripts in this directory)")))
    for d in ["analysis/heldout/out/images", "analysis/heldout/out/_aborted_run1_2026-09-29"]:
        n = sum(1 for q in (RR / d).rglob("*") if q.is_file())
        rows.append(dict(s22_row=S22["ho"], role="output (not redistributed)", path=d + "/", sha256="", bytes="",
                         produced_by="analysis/heldout/screen_and_retrieve.py", command=PY + "analysis/heldout/screen_and_retrieve.py",
                         status="current" if d.endswith("images") else "aborted run (provenance)",
                         note=f"{n} files; retrieved images are not redistributed; per-file hashes in heldout_manifest.csv"))
    for rel, why in SUPERSEDED:
        q = REPRO / rel
        rows.append(dict(s22_row=S22["sup"], role="script (submitted package)", path="SSCR Submission/_Working/09_Reproducibility/" + rel,
                         sha256=sha(q) if q.exists() else "", bytes=q.stat().st_size if q.exists() else "",
                         produced_by="", command="", status="superseded", note=why))
    for r in rows:
        for k in ("path", "produced_by", "command"):
            r[k] = dist(str(r[k]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(RR)}: {len(rows)} rows "
          f"({sum(r['role'] == 'script' for r in rows)} scripts, {sum(r['role'] == 'output' for r in rows)} outputs)")


if __name__ == "__main__":
    main()
