"""
m5_interpretability_and_type_coding.py  --  Repairs issues 11 and 25.

Two deliverables that let the paper make the interpretability and PH4-type claims
non-exploratorily, or else demote them cleanly.

(A) Interpretability evidence per family:
    * top-K contrastive features and their weights;
    * top activating MEMBER exemplars and top activating NON-MEMBER exemplars per top
      feature (image thumbnails or ids);
    * a coherence flag per feature (does the same visual/textual motif recur across its
      top activators?).
    Produce a per-family "signature card" for two worked successes (e.g. Loss) and one
    surface-dominated failure. If motifs are not coherent, report the measure as
    "inspectable" rather than "interpretable" in the manuscript.

(B) Blind PH4-type coding:
    * export a coding sheet: family id + the signature card + the four type options
      (structural / character-style / lexical / template), WITHOUT any recovery or
      engagement numbers visible to coders;
    * collect two independent coders' labels;
    * compute agreement (Cohen's kappa) and resolve disagreements by discussion;
    * only if agreement is acceptable may the per-type engagement gradient (Fig 4c) be
      reported as confirmatory; otherwise it stays exploratory.

Output: signature_cards/ (per-family), ph4_type_coding_sheet.csv, and after coding,
ph4_type_agreement.csv (kappa).

Skeleton:
"""
import numpy as np, pandas as pd

def build_coding_sheet():
    raise SystemExit(
        "Implement: load contrastive signatures + top features from fr_25 outputs; for "
        "each family render top features and top member/non-member activators; write "
        "ph4_type_coding_sheet.csv with columns [family, card_path, type_blank] and NO "
        "performance columns. Hand to two coders. Then load their labels and compute "
        "cohen_kappa_score; write ph4_type_agreement.csv."
    )

if __name__ == "__main__":
    build_coding_sheet()
