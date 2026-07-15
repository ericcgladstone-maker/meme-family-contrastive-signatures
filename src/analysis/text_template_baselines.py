"""Lexical (OCR) and template-proxy surface baselines.

Lexical similarity from OCR text (word TF-IDF cosine, char-n-gram cosine, token
Jaccard) and a proxy template-identity similarity from greedy pHash clustering at
a loose threshold. Template clusters are PROXIES, not ground-truth template IDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---- lexical ----

def _tfidf_cosine(texts, **kw):
    from sklearn.feature_extraction.text import TfidfVectorizer
    nonempty = [t for t in texts if t.strip()]
    if len(nonempty) < 2:
        n = len(texts)
        return np.eye(n)
    vec = TfidfVectorizer(**kw)
    X = vec.fit_transform([t if t.strip() else "" for t in texts])
    # rows with empty text -> zero vector -> zero similarity (kept, not dropped)
    Xn = X.toarray().astype(np.float64)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    Xn = Xn / np.clip(norms, 1e-12, None)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, 1.0)
    return sim


def lexical_sims(ids: list[str], texts: list[str]) -> dict[str, np.ndarray]:
    word = _tfidf_cosine(texts, analyzer="word", ngram_range=(1, 2), min_df=1)
    char = _tfidf_cosine(texts, analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    # token Jaccard
    toks = [set(t.lower().split()) for t in texts]
    n = len(ids)
    jac = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if toks[i] or toks[j]:
                inter = len(toks[i] & toks[j])
                union = len(toks[i] | toks[j])
                jac[i, j] = jac[j, i] = inter / union if union else 0.0
    return {"lexical_tfidf_word": word, "lexical_char_ngram": char, "lexical_token_jaccard": jac}


# ---- template proxy via loose pHash clustering ----

def phash_template_proxy(records: list[dict[str, Any]], thresh: int = 12, hash_size: int = 8):
    """Greedy pHash clustering at a loose Hamming threshold -> proxy template ids.
    Returns (ids, labels, sim) where sim[i,j]=1 iff same proxy template.
    PROXY ONLY: captures template-level visual sharing, not curated template IDs."""
    import imagehash
    from PIL import Image
    from pathlib import Path
    ids, hashes = [], []
    for r in records:
        try:
            h = imagehash.phash(Image.open(Path(r["image_path"])).convert("RGB"), hash_size=hash_size)
            ids.append(r["variant_id"])
            hashes.append(h)
        except Exception:
            continue
    labels = [-1] * len(ids)
    reps = []  # (hash, label)
    next_label = 0
    for i, h in enumerate(hashes):
        assigned = None
        for rh, lab in reps:
            if (h - rh) <= thresh:
                assigned = lab
                break
        if assigned is None:
            assigned = next_label
            reps.append((h, assigned))
            next_label += 1
        labels[i] = assigned
    n = len(ids)
    lab = np.array(labels)
    sim = (lab[:, None] == lab[None, :]).astype(np.float64)
    return ids, labels, sim, next_label
