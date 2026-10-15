"""Shared constants, loaders and freeze guards for the blinded author-coded human check (15b; R2-4 redesign, 09 F-51).

Supersedes the two-coder design of 15 (whose files and FREEZE_RECORD stay unchanged as an audit record). The score
frame, the six score bins (frozen cutpoints), the display-name rule and the 31-family reference sheet are reused
read-only from analysis/human_validation/ so that the strata are exactly the frozen ones; those files are hashed into
this design's own freeze record.
"""
import json
import sys
from pathlib import Path

HC = Path(__file__).resolve().parent
AN = HC.parent
RR = AN.parent
sys.path.insert(0, str(AN / "human_validation"))
import hv_common as H  # noqa: E402  (frozen, read-only reuse: load_frame, assign_bin, BINS, display_name, sha)

PROTOCOL = RR / "15b_Human_Check_Protocol.md"
FREEZE_RECORD = HC / "FREEZE_RECORD.json"
LABELS_RECORD = HC / "LABELS_FREEZE.json"
REF_DIR = AN / "human_validation" / "reference"

# 15b §3 (frozen at the 15b freeze)
BINS = H.BINS
N_PER_BIN = 10
N_REPEATS = 10
REPEAT_MIN_GAP = 20
SEED_DRAW, SEED_REPEAT, SEED_ORDER = 20261001, 20261002, 20261003
CODER = "E"  # Eric (sole coder)
LABEL_NONE, LABEL_UNCERTAIN = H.LABEL_NONE, H.LABEL_UNCERTAIN
LOW, HIGH = ["B1", "B2"], ["B5", "B6"]  # main descriptive comparison (15b §8)

# 15b §5: coder instructions, shown on every item (the 15 §5 decision rules; flag/second-choice rules dropped)
CODER_INSTRUCTIONS = [
    "Choose a family if the post uses that meme's recognizable format: its template image, structure or caption "
    "convention, as shown on the reference sheet. It still counts if the text, characters or other elements have been "
    "changed.",
    "Choose \"none\" if the post is a meme or image that does not use any listed family's format.",
    "Choose \"uncertain\" only if you cannot decide between \"a family\" and \"none\", or between two families.",
    "Use only the reference sheet; do not consult Know Your Meme, search engines or any other external source.",
]

sha = H.sha
display_name = H.display_name
scratch_dir = H.scratch_dir

FROZEN_ITEMS = [
    "15b_Human_Check_Protocol.md",
    "analysis/human_check/hc_common.py",
    "analysis/human_check/draw_check.py",
    "analysis/human_check/coding_form_check.html",
    "analysis/human_check/freeze_labels.py",
    "analysis/human_check/analyze_check.py",
    # reused read-only from the superseded design (strata, frame, display names, reference sheet)
    "analysis/human_validation/hv_common.py",
    "analysis/g1_full_sample/out/scores_all.csv",
    "analysis/g2_template_cluster/out/clusters.csv",
    "../build/first_results/ph4_q2_horsepower/q2_denom_images_manifest.csv",
    "../build/first_results/ph4_q2_horsepower/q2_denom_outcomes.csv",
    "../build/first_results/cross_family_dedup/cross_family_corrected_index.csv",
    "analysis/human_validation/reference/reference_manifest.csv",
    "analysis/human_validation/reference/reference_sheet.html",
    "analysis/human_validation/reference/reference_assets.csv",
]
AUDIT_ARTIFACTS = ["analysis/human_check/test_check.py", "analysis/human_check/fixture_test_report.json"]


def families():
    """The 31 display names on the reference sheet (alphabetical), the only family labels allowed."""
    import pandas as pd
    return [display_name(f) for f in sorted(pd.read_csv(REF_DIR / "reference_manifest.csv").family_id.unique())]


def valid_labels():
    return set(families()) | {LABEL_NONE, LABEL_UNCERTAIN}


def freeze_ok(record=FREEZE_RECORD, root=RR):
    """The real draw is allowed only if this design's freeze record exists, every frozen item re-hashes to its
    recorded value and every reference image matches reference_assets.csv. (No coder gates: the sole coder is Eric.)"""
    if not Path(record).exists():
        return False, "no 15b FREEZE_RECORD.json (freeze not done)"
    rec = json.loads(Path(record).read_text())
    for rel in FROZEN_ITEMS:
        h = rec.get("items", {}).get(rel)
        if h is None:
            return False, f"item not in record: {rel}"
        p = Path(root) / rel
        if not p.exists() or sha(p) != h:
            return False, f"hash mismatch: {rel}"
    ok, why = H.verify_reference_assets(Path(root) / "analysis/human_validation/reference")
    if not ok:
        return False, why
    return True, "UNLOCKED"
