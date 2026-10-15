"""Shared paths and helpers for the SSCR R&R gate analyses (G1-G3).

All submitted materials are read-only. Outputs go under analysis/gN_*/out/.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BUILD = PROJECT / "build"
FR = BUILD / "first_results"
SUBMISSION = PROJECT / "SSCR Submission" / "_Working"
REPRO = SUBMISSION / "09_Reproducibility"
AUDIT = SUBMISSION / "07_Analysis_Audit"
RR = PROJECT / "SSCR Revise and Resubmit"
ANALYSIS = RR / "analysis"

CFD_INDEX = FR / "cross_family_dedup" / "cross_family_corrected_index.csv"
EMB_OPENAI = FR / "embeddings_openai.npz"
EMB_LAION = FR / "embeddings_laion.npz"
OCR_TEXT = FR / "ocr_text.csv"
Q2 = FR / "ph4_q2_horsepower"
OUTCOMES = Q2 / "q2_denom_outcomes.csv"
MANIFEST = Q2 / "q2_denom_images_manifest.csv"
RAW_INDEX = Q2 / "q2_denom_raw_index.csv"
REDDIT_IMAGES = Q2 / "reddit_images"
SUBSET_CKPT = AUDIT / "reddit_denom_embeddings.npz"  # submitted 2,543-post checkpoint

K = 15
CTRL_D = ["max_clip_family_sim", "max_phash_family", "max_ocr_family", "log_age"]
PH4_F = ["max_contrastive_ph4_fit", "ph4_spec_margin"]

sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(REPRO / "analysis_to_run"))


def submitted_engine():
    """The submitted CV engine (RUN_e1_m2_reconcile), imported read-only."""
    import RUN_e1_m2_reconcile as E  # noqa: E402
    return E


def load_cohort():
    """Corpus: canonical ids, OpenAI CLIP vectors (unnormalized), family labels, index rows."""
    from src.analysis.clip_embed import load_npz
    cohort = {r["variant_id"]: r for r in csv.DictReader(CFD_INDEX.open(newline=""))}
    emb = load_npz(EMB_OPENAI)
    canon = [v for v in emb["ids"] if v in cohort]
    vec = emb["vectors"][[i for i, v in enumerate(emb["ids"]) if v in cohort]]
    fam = [cohort[v]["family_id"] for v in canon]
    return cohort, canon, vec, fam


def sae_dense(ids, vectors, sae=None):
    from src.analysis import sae_msae
    sae = sae or sae_msae.MSAE(device="cpu")
    recs = sae_msae.activation_records(list(ids), vectors, sae)
    dense = np.zeros((len(ids), sae.latent_dim), np.float32)
    for i, r in enumerate(recs):
        dense[i, r["active_indices"]] = r["active_values"]
    return dense, recs, sae


def sig_scores(X, top, w):
    """Vectorized cosine between X[:, top] and w (0 where either norm is 0), matching cosv()."""
    sub = X[:, top].astype(np.float64)
    num = sub @ w
    den = np.linalg.norm(sub, axis=1) * np.linalg.norm(w)
    out = np.zeros(len(X))
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def contrastive_signature(dense, members, background, k=K):
    diff = dense[members].mean(0) - dense[background].mean(0)
    top = np.argsort(-diff)[:k]
    w = np.clip(diff[top], 0, None).astype(np.float64)
    return top, w


def fam_index(fam):
    d = defaultdict(list)
    for i, f in enumerate(fam):
        d[f].append(i)
    return d


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))
