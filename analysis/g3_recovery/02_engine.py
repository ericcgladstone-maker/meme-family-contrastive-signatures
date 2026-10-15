#!/usr/bin/env python3
"""G3 recovery engine: one engine, three modes.

  fr25_repro  : replicates build/scripts/fr_25_contrastive_profiles.py (recurrence, CLIP centroid,
                sparse centroid, surface, contrastive; LAION decoys) incl. its RNG consumption
  m2m4_repro  : replicates RUN_m2m4_full.py (contrastive, sparse centroid, kNN, logistic with
                StratifiedKFold; pHash- and OpenAI-selected decoys)
  clean       : the 08 G3 protocol - per-family RNG; evaluation negatives E_f and all decoys excluded
                from a disjoint training background B_f; every method fit on the same n-1 members
                (and B_f where negatives are used); fold-local ranking; classifier inside LOVO;
                recurrence thresholds from B_f; OCR TF-IDF fit on T u B_f; non-transductive template;
                pre-specified fixed surface composite (z-scored on B_f). Also the 2x2 attribution
                factorial for contrastive (background contaminated/clean x pooled/fold-local).

Usage: python3 02_engine.py [fr25_repro|m2m4_repro|clean|sweep|kcurve]
Outputs under out/.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
from src.analysis.pharmacophore import construct_pharmacophore, estimate_background_thresholds  # noqa: E402
from src.analysis.score import candidate_to_pharmacophore_score  # noqa: E402

OUT = HERE / "out"
K, NSAMP, SEED, P_SUB, MAXF, TAU, TMPL_THR = 15, 200, 20260512, 0.80, 15, 95, 12
_F = {}


# ------------------------------------------------------------------ shared data
def data():
    if _F:
        return _F
    z = np.load(OUT / "features.npz", allow_pickle=True)
    ids, fam = list(z["ids"]), list(z["fam"])
    dense = z["dense"].astype(np.float32)
    vo, vl = z["clip_openai"], z["clip_laion"]
    bits = z["phash_bits"].astype(np.int16)
    texts = json.loads((OUT / "ocr_texts.json").read_text())
    n = len(ids)
    vn = vo / np.clip(np.linalg.norm(vo, axis=1, keepdims=True), 1e-12, None)
    ln = vl / np.clip(np.linalg.norm(vl, axis=1, keepdims=True), 1e-12, None)
    dn = dense / np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
    ham = (bits[:, None, :] != bits[None, :, :]).sum(2)
    phash = 1.0 - ham / bits.shape[1]
    np.fill_diagonal(phash, 1.0)
    srecs = []
    for i, v in enumerate(ids):
        nz = np.nonzero(dense[i])[0]
        srecs.append({"variant_id": v, "active_indices": nz.tolist(), "active_values": dense[i, nz].tolist(),
                      "is_background_corpus_sample": True})
    fi = defaultdict(list)
    for i, f in enumerate(fam):
        fi[f].append(i)
    _F.update(ids=ids, fam=fam, dense=dense, vn=vn, ln=ln, dn=dn, ham=ham, phash=phash, texts=texts, n=n,
              srecs=srecs, fi=fi, clipO=vn @ vn.T, clipL=ln @ ln.T, latent=dense.shape[1])
    return _F


def fams_eval():
    F = data()
    return [(k, f, idxs) for k, (f, idxs) in enumerate(F["fi"].items()) if len(idxs) >= 5]


# ------------------------------------------------------------------ method pieces
def auc_pooled(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    p = np.asarray(pos)[:, None]
    ng = np.asarray(neg)[None, :]
    return float(((p > ng).sum() + 0.5 * (p == ng).sum()) / (p.size * ng.size))


def rank_local(s_q, s_neg):
    s_neg = np.asarray(s_neg)
    return float((s_neg < s_q).mean() + 0.5 * (s_neg == s_q).mean())


def thresholds_from(indices):
    F = data()
    return estimate_background_thresholds([F["srecs"][i] for i in indices], latent_dim=F["latent"], percentile=TAU)


def recurrence_vec(f, train, thr):
    """Returns (idx, loadings) of the recurrence pharmacophore; scoring vectorized below."""
    F = data()
    ph = construct_pharmacophore(f, [F["srecs"][i] for i in train], thr, P_SUB, MAXF, F["latent"],
                                 scope="retrospective_full", cutoff_timestamp=None, config_version="con")
    return np.array(ph["retained_feature_indices"], int), np.array(ph["lineage_typical_loadings"], float), ph


def recurrence_score(idx, load, cand):
    F = data()
    if len(idx) == 0:
        return np.zeros(len(cand))
    act = F["dense"][np.ix_(cand, idx)].astype(np.float64)
    return (np.minimum(act, load[None, :]) * load[None, :]).sum(1) / (load.sum() or 1.0)


def greedy_template_labels():
    F = data()
    reps, lab = [], np.full(F["n"], -1)
    for i in range(F["n"]):
        for rep, l in reps:
            if F["ham"][i, rep] <= TMPL_THR:
                lab[i] = l
                break
        else:
            lab[i] = len(reps)
            reps.append((i, len(reps)))
    return lab


def tfidf_matrix(fit_idx, all_idx):
    """char_wb 2-4 TF-IDF fit on fit_idx texts; returns row-normalized dense rows for all_idx."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    F = data()
    tx = [F["texts"][i] if F["texts"][i].strip() else "" for i in fit_idx]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit(tx)
    X = vec.transform([F["texts"][i] if F["texts"][i].strip() else "" for i in all_idx]).toarray()
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


# ------------------------------------------------------------------ fr_25 replication
def run_fr25():
    F = data()
    rng = np.random.default_rng(SEED)
    thr_all = thresholds_from(range(F["n"]))
    Xall = tfidf_matrix(range(F["n"]), range(F["n"]))
    ocrsim = Xall @ Xall.T
    np.fill_diagonal(ocrsim, 1.0)
    tl = greedy_template_labels()
    tmpl = (tl[:, None] == tl[None, :]).astype(float)
    rows = []
    for k, (f, idxs) in enumerate(F["fi"].items()):
        if len(idxs) < 5:
            continue
        nonfam = [i for i in range(F["n"]) if F["fam"][i] != f]
        near_l = sorted(nonfam, key=lambda i: -max(F["clipL"][i, j] for j in idxs))[:12]
        samp = list(rng.choice(nonfam, size=min(NSAMP, len(nonfam)), replace=False))
        S = {m: {"mem": [], "nm": [], "nl": []} for m in ["con", "rec", "clip", "sae", "phash", "ocr", "tmpl"]}
        for q in idxs:
            tr = [i for i in idxs if i != q]
            cc = F["vn"][tr].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
            sc = F["dn"][tr].mean(0); sc /= (np.linalg.norm(sc) + 1e-12)
            top, w = C.contrastive_signature(F["dense"], tr, nonfam)
            ridx, rload, _ = recurrence_vec(f, tr, thr_all)
            for tag, cs in [("mem", [q]), ("nm", samp), ("nl", near_l)]:
                cs = np.array(cs)
                S["con"][tag] += C.sig_scores(F["dense"][cs], top, w).tolist()
                S["rec"][tag] += recurrence_score(ridx, rload, cs).tolist()
                S["clip"][tag] += (F["vn"][cs] @ cc).tolist()
                S["sae"][tag] += (F["dn"][cs] @ sc).tolist()
                S["phash"][tag] += F["phash"][np.ix_(cs, tr)].mean(1).tolist()
                S["ocr"][tag] += ocrsim[np.ix_(cs, tr)].mean(1).tolist()
                S["tmpl"][tag] += tmpl[np.ix_(cs, tr)].mean(1).tolist()
        for _ in range(15):  # fr_25's bootstrap Jaccard consumes the shared RNG here
            rng.choice(idxs, size=len(idxs), replace=True)
        r = {"family": f.replace("pool_", ""), "n": len(idxs)}
        for m in S:
            r[f"{m}_auc"] = auc_pooled(S[m]["mem"], S[m]["nm"])
        r["best_surface_auc"] = max(r["phash_auc"], r["ocr_auc"], r["tmpl_auc"])
        for m in ["con", "rec", "clip"]:
            r[f"adv_{m}_laion"] = auc_pooled(S[m]["mem"], S[m]["nl"])
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ RUN_m2m4_full replication
def run_m2m4():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    F = data()
    rng = np.random.default_rng(SEED)
    rows = []
    for k, (f, idxs) in enumerate(F["fi"].items()):
        if len(idxs) < 5:
            continue
        nonfam = [i for i in range(F["n"]) if F["fam"][i] != f]
        samp = list(rng.choice(nonfam, size=min(NSAMP, len(nonfam)), replace=False))
        S = {m: {"mem": [], "nm": []} for m in ["con", "cen", "knn"]}
        for q in idxs:
            tr = [i for i in idxs if i != q]
            top, w = C.contrastive_signature(F["dense"], tr, nonfam)
            sc = F["dn"][tr].mean(0); sc /= (np.linalg.norm(sc) + 1e-12)
            for tag, cs in [("mem", [q]), ("nm", samp)]:
                cs = np.array(cs)
                S["con"][tag] += C.sig_scores(F["dense"][cs], top, w).tolist()
                S["cen"][tag] += (F["dn"][cs] @ sc).tolist()
                S["knn"][tag] += (F["dn"][cs] @ F["dn"][tr].T).max(1).tolist()
        Xc = np.vstack([F["dense"][idxs], F["dense"][samp]])
        yc = np.r_[np.ones(len(idxs)), np.zeros(len(samp))]
        oof = np.zeros(len(yc))
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(Xc, yc):
            m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xc[tr], yc[tr])
            oof[te] = m.predict_proba(Xc[te])[:, 1]
        near_p = sorted(nonfam, key=lambda i: -max(F["phash"][i, j] for j in idxs))[:12]
        near_c = sorted(nonfam, key=lambda i: -max(F["clipO"][i, j] for j in idxs))[:12]
        A = {k2: [] for k2 in ["cpm", "cpd", "lpm", "lpd", "ccd", "lcd"]}
        for q in idxs:
            tr = [i for i in idxs if i != q]
            top, w = C.contrastive_signature(F["dense"], tr, nonfam)
            cc = F["vn"][tr].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
            A["cpm"] += C.sig_scores(F["dense"][[q]], top, w).tolist(); A["lpm"].append(float(F["vn"][q] @ cc))
            A["cpd"] += C.sig_scores(F["dense"][near_p], top, w).tolist(); A["lpd"] += (F["vn"][near_p] @ cc).tolist()
            A["ccd"] += C.sig_scores(F["dense"][near_c], top, w).tolist(); A["lcd"] += (F["vn"][near_c] @ cc).tolist()
        rows.append(dict(family=f.replace("pool_", ""), n=len(idxs), contrastive=auc_pooled(S["con"]["mem"], S["con"]["nm"]),
                         sae_centroid=auc_pooled(S["cen"]["mem"], S["cen"]["nm"]), sae_knn=auc_pooled(S["knn"]["mem"], S["knn"]["nm"]),
                         logistic_sparse=float(roc_auc_score(yc, oof)),
                         adv_con_phash=auc_pooled(A["cpm"], A["cpd"]), adv_clip_phash=auc_pooled(A["lpm"], A["lpd"]),
                         adv_con_clip=auc_pooled(A["cpm"], A["ccd"]), adv_clip_clip=auc_pooled(A["lpm"], A["lcd"])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ clean protocol
METHODS = ["contrastive", "recurrence", "clip_centroid", "sparse_centroid", "sparse_knn", "logistic",
           "surf_phash", "surf_ocr", "surf_template", "surf_template_transductive", "surf_composite"]


def clean_family(k, f, idxs, seed, Ks=(K,), with_factorial=True, with_logistic=True):
    from sklearn.linear_model import LogisticRegression
    F = data()
    n = F["n"]
    rng = np.random.default_rng([seed, k])
    nonfam = [i for i in range(n) if F["fam"][i] != f]
    Eneg = np.array(sorted(rng.choice(nonfam, size=min(NSAMP, len(nonfam)), replace=False)))
    DL = sorted(nonfam, key=lambda i: -max(F["clipL"][i, j] for j in idxs))[:12]
    DP = sorted(nonfam, key=lambda i: -max(F["phash"][i, j] for j in idxs))[:12]
    DO = sorted(nonfam, key=lambda i: -max(F["clipO"][i, j] for j in idxs))[:12]
    excl = set(Eneg.tolist()) | set(DL) | set(DP) | set(DO)
    B = np.array([i for i in nonfam if i not in excl])
    thrB = thresholds_from(B.tolist())
    tl = greedy_template_labels() if not hasattr(clean_family, "_tl") else clean_family._tl
    clean_family._tl = tl
    sets = {"E": Eneg, "DL": np.array(DL), "DP": np.array(DP), "DO": np.array(DO)}
    # per-method, per-set accumulators: fold-local ranks and pooled score lists
    loc = {m: {s: [] for s in sets} for m in METHODS}
    pool = {m: {"mem": [], "E": []} for m in METHODS}
    fact = {"contam_pooled": {"mem": [], "E": []}, "contam_local": [], "clean_pooled": {"mem": [], "E": []}, "clean_local": []}
    kloc = {kk: [] for kk in Ks}
    allc = np.unique(np.concatenate([[*idxs], Eneg, DL, DP, DO, B]))
    for q in idxs:
        tr = [i for i in idxs if i != q]
        cands = np.concatenate([[q], Eneg, DL, DP, DO])
        sl = {"q": slice(0, 1), "E": slice(1, 1 + len(Eneg)), "DL": slice(1 + len(Eneg), 13 + len(Eneg)),
              "DP": slice(13 + len(Eneg), 25 + len(Eneg)), "DO": slice(25 + len(Eneg), 37 + len(Eneg))}
        sc = {}
        top, w = C.contrastive_signature(F["dense"], tr, B)
        sc["contrastive"] = C.sig_scores(F["dense"][cands], top, w)
        ridx, rload, _ = recurrence_vec(f, tr, thrB)
        sc["recurrence"] = recurrence_score(ridx, rload, cands)
        cc = F["vn"][tr].mean(0); cc /= (np.linalg.norm(cc) + 1e-12)
        sc["clip_centroid"] = F["vn"][cands] @ cc
        scn = F["dn"][tr].mean(0); scn /= (np.linalg.norm(scn) + 1e-12)
        sc["sparse_centroid"] = F["dn"][cands] @ scn
        sc["sparse_knn"] = (F["dn"][cands] @ F["dn"][tr].T).max(1)
        if with_logistic:
            X = np.vstack([F["dense"][tr], F["dense"][B]])
            y = np.r_[np.ones(len(tr)), np.zeros(len(B))]
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
            sc["logistic"] = clf.decision_function(F["dense"][cands])
        else:
            sc["logistic"] = np.zeros(len(cands))
        # surface (clean): fit only on T u B; score candidates and B (for composite standardization)
        both = np.concatenate([cands, B])
        sc_ph = F["phash"][np.ix_(both, tr)].mean(1)
        Xt = tfidf_matrix(list(tr) + B.tolist(), both.tolist() + list(tr))
        Xc, XT = Xt[:len(both)], Xt[len(both):]
        sc_oc = (Xc @ XT.T).mean(1)
        sc_tp = (F["ham"][np.ix_(both, tr)] <= TMPL_THR).mean(1)
        nb = len(cands)
        sc["surf_phash"], sc["surf_ocr"], sc["surf_template"] = sc_ph[:nb], sc_oc[:nb], sc_tp[:nb]
        sc["surf_template_transductive"] = (tl[cands][:, None] == tl[tr][None, :]).mean(1)
        zs = []
        for arr in (sc_ph, sc_oc, sc_tp):
            mu, sd = arr[nb:].mean(), arr[nb:].std()
            zs.append((arr[:nb] - mu) / sd if sd > 0 else np.zeros(nb))
        sc["surf_composite"] = np.mean(zs, axis=0)
        for m in METHODS:
            s = sc[m]
            for sname in sets:
                loc[m][sname].append(rank_local(s[0], s[sl[sname]]))
            pool[m]["mem"].append(float(s[0]))
            pool[m]["E"] += s[sl["E"]].tolist()
        if with_factorial:
            topc, wc = C.contrastive_signature(F["dense"], tr, nonfam)  # submitted (contaminated) background
            s_c = C.sig_scores(F["dense"][np.concatenate([[q], Eneg])], topc, wc)
            fact["contam_pooled"]["mem"].append(float(s_c[0])); fact["contam_pooled"]["E"] += s_c[1:].tolist()
            fact["contam_local"].append(rank_local(s_c[0], s_c[1:]))
            s_k = sc["contrastive"][:1 + len(Eneg)]
            fact["clean_pooled"]["mem"].append(float(s_k[0])); fact["clean_pooled"]["E"] += s_k[1:].tolist()
            fact["clean_local"].append(rank_local(s_k[0], s_k[1:]))
        for kk in Ks:
            if kk == K:
                continue
            t2, w2 = C.contrastive_signature(F["dense"], tr, B, k=kk)
            s2 = C.sig_scores(F["dense"][np.concatenate([[q], Eneg])], t2, w2)
            kloc[kk].append(rank_local(s2[0], s2[1:]))
    out = {"family": f.replace("pool_", ""), "k": k, "n": len(idxs), "seed": seed, "n_background": len(B),
           "n_E": len(Eneg), "overlap_E_decoys": len(set(Eneg.tolist()) & (set(DL) | set(DP) | set(DO)))}
    for m in METHODS:
        out[f"{m}__E_local"] = float(np.mean(loc[m]["E"]))
        out[f"{m}__E_pooled"] = auc_pooled(pool[m]["mem"], pool[m]["E"])
        for sname in ["DL", "DP", "DO"]:
            out[f"{m}__{sname}_local"] = float(np.mean(loc[m][sname]))
    if with_factorial:
        out["fact_contam_pooled"] = auc_pooled(fact["contam_pooled"]["mem"], fact["contam_pooled"]["E"])
        out["fact_contam_local"] = float(np.mean(fact["contam_local"]))
        out["fact_clean_pooled"] = auc_pooled(fact["clean_pooled"]["mem"], fact["clean_pooled"]["E"])
        out["fact_clean_local"] = float(np.mean(fact["clean_local"]))
    for kk in Ks:
        if kk != K:
            out[f"K{kk}__E_local"] = float(np.mean(kloc[kk]))
    return out


def run_clean(seeds, **kw):
    fe = fams_eval()
    tasks = [(k, f, idxs, s) for s in seeds for (k, f, idxs) in fe]
    res = Parallel(n_jobs=11, verbose=0)(delayed(clean_family)(k, f, idxs, s, **kw) for (k, f, idxs, s) in tasks)
    return pd.DataFrame(res)


def main():
    OUT.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"
    if mode == "fr25_repro":
        df = run_fr25(); df.to_csv(OUT / "repro_fr25.csv", index=False)
    elif mode == "m2m4_repro":
        df = run_m2m4(); df.to_csv(OUT / "repro_m2m4.csv", index=False)
    elif mode == "clean":
        df = run_clean([SEED]); df.to_csv(OUT / "clean_primary.csv", index=False)
    elif mode == "sweep":
        df = run_clean(list(range(1, 11)), with_factorial=False); df.to_csv(OUT / "clean_seed_sweep.csv", index=False)
    elif mode == "kcurve":
        df = run_clean([SEED], Ks=(5, 10, 15, 25, 50), with_factorial=False, with_logistic=False)
        df.to_csv(OUT / "clean_kcurve.csv", index=False)
    print(f"[{mode}] wrote {len(df)} rows", flush=True)
    cols = [c for c in df.columns if df[c].dtype != object and c not in ("k", "n", "seed")]
    print(df[cols].mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
