"""Shared constants, loaders and the freeze guard for the human-validation study (15 v0.5).

Everything here implements 15 literally; any change after the joint freeze requires a 15 §13 amendment.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HV = Path(__file__).resolve().parent
AN = HV.parent
RR = AN.parent
PROJ = RR.parent
PROTOCOL = RR / "15_Human_Validation_Protocol.md"
FRAME = AN / "g1_full_sample/out/scores_all.csv"
CLUSTERS = AN / "g2_template_cluster/out/clusters.csv"
DEV_INDEX = PROJ / "build/first_results/cross_family_dedup/cross_family_corrected_index.csv"
REDDIT_MANIFEST = PROJ / "build/first_results/ph4_q2_horsepower/q2_denom_images_manifest.csv"  # post_id -> local_path
OUTCOMES = PROJ / "build/first_results/ph4_q2_horsepower/q2_denom_outcomes.csv"  # post_id -> subreddit (community)
FREEZE_RECORD = HV / "FREEZE_RECORD.json"

# Test / fixture / scratch output lives OUTSIDE Google Drive (Drive sync stalls on many small files).
# Override with PH4_NONDRIVE_DIR; the path must not resolve inside a Google Drive folder.
NONDRIVE = Path(os.environ.get("PH4_NONDRIVE_DIR",
                               "~/Desktop/PH4_Computational_Revise_Resubmit_Non_Drive_Materials")).expanduser()


def scratch_dir(sub="human_validation"):
    d = (NONDRIVE / sub).resolve()
    if "My Drive" in str(d) or "Google Drive" in str(d) or "GoogleDrive" in str(d):
        raise SystemExit(f"REFUSING: scratch dir {d} is inside Google Drive; set PH4_NONDRIVE_DIR elsewhere")
    d.mkdir(parents=True, exist_ok=True)
    return d

# 15 §3 (frozen)
CUTPOINTS = [0.7422, 0.7962, 0.8323, 0.8682, 0.9110]
N_PER_BIN = 70
BINS = ["B1", "B2", "B3", "B4", "B5", "B6"]
SEED_DRAW, SEED_PRACTICE_REDDIT, SEED_ORDER = 20260930, 20260931, 20260932
SEED_GOLD, SEED_EXEMPLAR, SEED_BOOT, SEED_REPEAT, SEED_PRACTICE_DEV = 20260933, 20260934, 20260935, 20260936, 20260938
N_EXEMPLARS, N_REPEATS, REPEAT_MIN_GAP, N_PRACTICE_REDDIT, N_PRACTICE_DEV = 4, 20, 50, 30, 20
N_BOOT = 2000
CLUSTER_COL = "cc10_as_planned"
LABEL_NONE, LABEL_UNCERTAIN = "none", "uncertain"
# 15 §5: frozen coder instructions (verbatim decision rules + the external-source rule), shown in every package
CODER_INSTRUCTIONS = [
    "Choose a family if the post uses that meme's recognizable format: its template image, structure or caption "
    "convention, as shown on the reference sheet. It still counts if the text, characters or other elements have been "
    "changed.",
    "Choose \"none\" if the post is a meme or image that does not use any listed family's format.",
    "Choose \"uncertain\" only if you cannot decide between \"a family\" and \"none\", or between two families.",
    "Use only the frozen reference sheet; do not consult Know Your Meme, search engines or any other external source.",
]
SEED_BOOT_STREAMS = {"reliability": 0, "adj": 1, "B": 2}  # default_rng([SEED_BOOT, k]) per bootstrap (15 §8)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assign_bin(x):
    """B1..B6 from the frozen 4-dp cutpoints (15 §3 table: B2 = [0.7422, 0.7962), ..., B6 = >= 0.9110)."""
    return np.array(BINS)[np.searchsorted(CUTPOINTS, np.asarray(x, float), side="right")]


def display_name(family_id):
    """15 §4: slug-derived display name."""
    return family_id.removeprefix("pool_").replace("_", " ").title()


def load_frame(frame=FRAME, clusters=CLUSTERS, manifest=None, outcomes=OUTCOMES):
    """Frame = scores (post_id, raw_max_fit, raw_argmax) + community (subreddit) + pHash cluster + local image path.
    The community comes from the frame file if it has a `subreddit` column (fixtures), else from the outcomes file."""
    raw = pd.read_csv(frame)
    s = raw[["post_id", "raw_max_fit", "raw_argmax"] + (["subreddit"] if "subreddit" in raw.columns else [])]
    if "subreddit" not in s.columns:
        o = pd.read_csv(outcomes)[["post_id", "subreddit"]]
        s = s.merge(o, on="post_id", how="left", validate="one_to_one")
    if s["subreddit"].isna().any():
        raise SystemExit("frame posts without a community (subreddit)")
    c = pd.read_csv(clusters)[["post_id", CLUSTER_COL]]
    d = s.merge(c, on="post_id", how="left", validate="one_to_one")
    if d[CLUSTER_COL].isna().any():
        raise SystemExit("frame posts without a pHash cluster")
    if manifest is not None:
        m = pd.read_csv(manifest)[["post_id", "local_path"]]
        d = d.merge(m, on="post_id", how="left", validate="one_to_one")
    d["bin"] = assign_bin(d.raw_max_fit)
    return d


def load_dev(dev_index=DEV_INDEX):
    d = pd.read_csv(dev_index)
    sizes = d.family_id.value_counts()
    fams = sorted(sizes.index)
    eligible = [f for f in fams if sizes[f] >= 5]
    return d, fams, eligible


# ---------------------------------------------------------------- freeze guard
FROZEN_ITEMS = ["15_Human_Validation_Protocol.md", "analysis/human_validation/hv_common.py",
                "analysis/human_validation/make_reference_sheet.py", "analysis/human_validation/draw_sample.py",
                "analysis/human_validation/coding_form_template.html", "analysis/human_validation/analyze_labels.py",
                "analysis/g1_full_sample/out/scores_all.csv", "analysis/g2_template_cluster/out/clusters.csv",
                "analysis/human_validation/reference/reference_manifest.csv",
                "analysis/human_validation/reference/gold_items.csv", "analysis/human_validation/reference/practice_dev.csv",
                "../build/first_results/ph4_q2_horsepower/q2_denom_images_manifest.csv",
                "../build/first_results/cross_family_dedup/cross_family_corrected_index.csv",
                "../build/first_results/ph4_q2_horsepower/q2_denom_outcomes.csv",
                "analysis/human_validation/qualify.py",
                "analysis/human_validation/reference/reference_sheet.html",
                "analysis/human_validation/reference/reference_assets.csv"]
REF_DIR_REL = "analysis/human_validation/reference"


def verify_reference_assets(ref_dir):
    """Every coder-visible reference image must match reference_assets.csv (SHA-256), with no missing or extra files."""
    ref_dir = Path(ref_dir)
    man = pd.read_csv(ref_dir / "reference_assets.csv")
    on_disk = sorted(p.name for p in (ref_dir / "img").iterdir() if p.is_file())
    if sorted(man.file) != on_disk:
        return False, "reference img/ files differ from reference_assets.csv"
    for r in man.itertuples():
        if sha(ref_dir / "img" / r.file) != r.sha256:
            return False, f"reference asset hash mismatch: {r.file}"
    return True, "ok"


# Audit evidence (15 v0.8 §21): hashed into the joint freeze record, but NOT checked by freeze_ok (rerunning the tests
# after the freeze rewrites the report and must not lock the draw).
AUDIT_ARTIFACTS = ["analysis/human_validation/test_hv.py", "analysis/human_validation/fixture_test_report.json"]

def freeze_ok(record=FREEZE_RECORD, root=RR):
    """Real-data draw is allowed only if the joint freeze record exists, every frozen item re-hashes to its recorded
    value, every coder-visible reference image matches reference_assets.csv, and H-1 (Coder B) and H-7 (institutional
    determination) are recorded as satisfied (15 §12 steps 4-5)."""
    if not Path(record).exists():
        return False, "no FREEZE_RECORD.json (joint freeze not done)"
    rec = json.loads(Path(record).read_text())
    for rel in FROZEN_ITEMS:
        h = rec.get("items", {}).get(rel)
        if h is None:
            return False, f"item not in record: {rel}"
        p = Path(root) / rel
        if not p.exists() or sha(p) != h:
            return False, f"hash mismatch: {rel}"
    ok, why = verify_reference_assets(Path(root) / REF_DIR_REL)
    if not ok:
        return False, why
    if not rec.get("h1_coder_b_recorded") or not rec.get("h7_determination_recorded"):
        return False, "H-1 and/or H-7 not recorded as satisfied"
    return True, "UNLOCKED"
