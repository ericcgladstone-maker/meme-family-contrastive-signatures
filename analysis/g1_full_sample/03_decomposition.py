#!/usr/bin/env python3
"""G1 steps 4-10: decomposition of the engagement association, subset (validation) and full sample.

Engine: the submitted RUN_e1_m2_reconcile.run (control D + community; in-fold labels;
GroupKFold(5) on stored ph4_fit_family) unless stated.

A. VALIDATION on the submitted 2,543 subset:
   - score-construction increments must reproduce downstream_engagement_results.csv
   - raw-over-magnitude point estimate must reproduce 0.026
   - the RUN_reviewer_numerics RNG chain (seed 0: 2,000 bootstrap draws, then 500 random-feature
     banks, 200 label-permutation banks, 2,000 adjusted-bootstrap draws) is replicated draw-for-draw,
     so the submitted null means / p-values and the adjusted CI can be reproduced exactly.
B. FULL SAMPLE (4,263): the same increments, correlations, nulls (same banks), and family-cluster
   bootstraps (2,000; seed 0) for raw, calibrated, classifier (prob + decision), sparse centroid, and
   raw-over-magnitude. Observed values are computed, never typed.
C. Seed sweep: nulls and raw / adjusted bootstraps with seeds 1-3.
D. Figure 4d per-community correlations; Table S20 recomputed as cached-vs-fetched composition.
E. Cross-scheme sensitivity: random5 (seed 0), community GroupKFold(7) [= LOCO], template cc10.
Output: out/g1_results.json, out/g1_nulls_*.csv, out/g1_boot_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

E = C.submitted_engine()
OUT = Path(__file__).resolve().parent / "out"
G2OUT = Path(__file__).resolve().parents[1] / "g2_template_cluster" / "out"
D = C.CTRL_D
GEN = ["sparse_norm", "clip_norm", "ocr_len"]
NJOBS = 11
VARIANTS = {  # name -> columns added to D
    "raw_fit": ["raw_max_fit"],
    "raw_fit_plus_spec_stored": ["max_contrastive_ph4_fit", "ph4_spec_margin"],
    "calibrated_fit": ["calib_max_fit"],
    "classifier_prob": ["clf_max_prob"],
    "classifier_decision": ["clf_max_decision"],
    "sae_centroid": ["sae_centroid_max"],
    "clip_similarity": ["max_clip_family_sim"],
    "sparse_norm": ["sparse_norm"],
    "clip_norm": ["clip_norm"],
    "random_sig_max": ["rand_sig_max"],
    "magnitude_block": GEN,
}


# ---------------------------------------------------------------- workers
def _inc(df, groups, ctrl, add, splitter=None):
    E.CTRL, E.PH4 = ctrl, add
    E.build_dummies(df)
    sp = splitter if splitter is not None else GroupKFold(min(5, len(np.unique(groups))))
    return E.run(df, sp, groups=groups if not isinstance(sp, KFold) else None, infold=True)


def _null_chunk(df, groups, score_mat):
    out = []
    for s in score_mat:
        d = df.copy()
        d["_s"] = s
        out.append(_inc(d, groups, D, ["_s"])["dR2"])
    return out


def _boot_chunk(df, gcol, picks, ctrl, add):
    by = {g: df[df[gcol] == g] for g in df[gcol].unique()}
    out = []
    for pick in picks:
        bt = pd.concat([by[g].assign(__rep=f"{g}_{i}") for i, g in enumerate(pick)], ignore_index=True)
        g = bt.__rep.astype("category").cat.codes.to_numpy()
        try:
            r = _inc(bt, g, ctrl, add)
            out.append((r["dR2"], r["dAUC"]))
        except Exception:
            out.append((np.nan, np.nan))
    return out


def chunks(seq, n):
    k = max(1, len(seq) // n)
    return [seq[i:i + k] for i in range(0, len(seq), k)]


def par_nulls(df, groups, mats):
    res = Parallel(n_jobs=NJOBS)(delayed(_null_chunk)(df, groups, c) for c in chunks(list(mats), NJOBS * 3))
    return np.array([x for r in res for x in r])


def par_boot(df, gcol, picks, ctrl, add):
    res = Parallel(n_jobs=NJOBS)(delayed(_boot_chunk)(df, gcol, c, ctrl, add) for c in chunks(list(picks), NJOBS * 3))
    v = np.array([x for r in res for x in r])
    return v[~np.isnan(v[:, 0])]


def summ(v):
    v = np.asarray(v)
    return dict(n=int(len(v)), median=float(np.median(v)), mean=float(v.mean()),
                ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], share_gt0=float((v > 0).mean()))


# ---------------------------------------------------------------- data
def frame(scores_csv):
    o = pd.read_csv(C.OUTCOMES)
    o = o[o.listing_group == "denominator"]
    s = pd.read_csv(scores_csv)
    df = o.merge(s, on="post_id", how="inner").reset_index(drop=True)
    df["ocr_len"] = df["ocr_len"].astype(float)
    return df


def rdense_for(ids, npz):
    z = np.load(npz, allow_pickle=True)
    zid = list(z["ids"])
    pos = {v: i for i, v in enumerate(zid)}
    vec = z["vectors"][[pos[v] for v in ids]]
    dense, _, _ = C.sae_dense(ids, vec)
    return dense


def reviewer_numerics_chain(seed, sae_dim, cdense, cfam, fams_full, fams_sub):
    """Replicate RUN_reviewer_numerics RNG consumption draw-for-draw."""
    rng = np.random.default_rng(seed)
    boot1 = [rng.choice(fams_full, len(fams_full), replace=True) for _ in range(2000)]
    rf = [[(rng.choice(sae_dim, C.K, replace=False), np.abs(rng.normal(size=C.K))) for _ in range(30)] for _ in range(500)]
    lp = []
    for _ in range(200):
        perm = cfam.copy()
        rng.shuffle(perm)
        sigs = []
        for f in sorted(set(perm)):  # order irrelevant to the max; sorted for determinism
            idx = np.where(perm == f)[0]
            if len(idx) < 5:
                continue
            nm = np.where(perm != f)[0]
            sigs.append(C.contrastive_signature(cdense, idx, nm))
        lp.append(sigs)
    adj = [rng.choice(fams_sub, len(fams_sub), replace=True) for _ in range(2000)]
    return boot1, rf, lp, adj


def bank_max(rdense, bank):
    return np.column_stack([C.sig_scores(rdense, t, w) for t, w in bank]).max(1)


# ---------------------------------------------------------------- main
def main():
    res = {}
    cohort, canon, cvec, cfam = C.load_cohort()
    cdense, _, sae = C.sae_dense(canon, cvec)
    cfam = np.array(cfam)
    full_o = pd.read_csv(C.OUTCOMES)
    full_o = full_o[full_o.listing_group == "denominator"].reset_index(drop=True)
    fams_full = full_o.ph4_fit_family.unique()

    for label, csvp, npz in [("subset_2543", OUT / "scores_subset.csv", C.SUBSET_CKPT),
                             ("full_4263", OUT / "scores_all.csv", OUT / "reddit_denom_embeddings_all.npz")]:
        df = frame(csvp)
        g = df.ph4_fit_family.astype("category").cat.codes.to_numpy()
        r = dict(n=len(df))
        # increments + correlations
        r["increments"] = {k: _inc(df, g, D, v) for k, v in VARIANTS.items()}
        r["raw_over_D_plus_magnitude"] = _inc(df, g, D + GEN, ["raw_max_fit"])
        r["raw_stored_over_D_plus_magnitude"] = _inc(df, g, D + GEN, ["max_contrastive_ph4_fit"])
        r["corr_with_log_score"] = {c: float(df[c].corr(df["log_score"])) for c in
                                    ["raw_max_fit", "max_contrastive_ph4_fit", "calib_max_fit", "clf_max_prob",
                                     "clf_max_decision", "sae_centroid_max", "max_clip_family_sim",
                                     "sparse_norm", "clip_norm", "rand_sig_max", "ocr_len"]}
        print(f"[{label}] n={len(df)} raw dR2={r['increments']['raw_fit']['dR2']:+.4f} "
              f"adj={r['raw_over_D_plus_magnitude']['dR2']:+.4f}", flush=True)
        # nulls (reviewer_numerics chain, seed 0) + seeds 1-3
        rdense = rdense_for(list(df.post_id), npz)
        fams_sub = df.ph4_fit_family.unique()
        obs = r["increments"]["raw_fit"]["dR2"]
        r["nulls"] = {}
        for seed in [0, 1, 2, 3]:
            _, rf, lp, adj_picks = reviewer_numerics_chain(seed, sae.latent_dim, cdense, cfam, fams_full, fams_sub)
            nrf = par_nulls(df, g, [bank_max(rdense, b) for b in rf])
            nlp = par_nulls(df, g, [bank_max(rdense, b) for b in lp])
            pd.DataFrame({"random_feature": nrf}).to_csv(OUT / f"g1_nulls_random_feature_{label}_seed{seed}.csv", index=False)
            pd.DataFrame({"label_perm": nlp}).to_csv(OUT / f"g1_nulls_label_perm_{label}_seed{seed}.csv", index=False)
            r["nulls"][f"seed{seed}"] = dict(
                observed_raw_dR2=obs,
                random_feature=dict(mean=float(nrf.mean()), p95=float(np.percentile(nrf, 95)), max=float(nrf.max()),
                                    p_ge_obs=float((nrf >= obs).mean()), n=len(nrf)),
                label_perm=dict(mean=float(nlp.mean()), p95=float(np.percentile(nlp, 95)), max=float(nlp.max()),
                                p_ge_obs=float((nlp >= obs).mean()), n=len(nlp),
                                mean_share_of_observed=float(nlp.mean() / obs)))
            # submitted-analysis replication of p used the typed OBS=0.080 on the subset
            if label == "subset_2543" and seed == 0:
                r["nulls"]["seed0"]["label_perm_p_ge_typed_0.080"] = float((nlp >= 0.080).mean())
                r["nulls"]["seed0"]["random_feature_p_ge_typed_0.080"] = float((nrf >= 0.080).mean())
            print(f"[{label} nulls seed{seed}] rf mean {nrf.mean():+.4f} p {(nrf>=obs).mean():.4f} | "
                  f"lp mean {nlp.mean():+.4f} p {(nlp>=obs).mean():.4f}", flush=True)
            # adjusted bootstrap: seed-0 chain replicates the submitted adjusted CI on the subset
            if seed == 0:
                va = par_boot(df, "ph4_fit_family", adj_picks, D + GEN, ["max_contrastive_ph4_fit"])
                r["adjusted_bootstrap_submitted_spec_seed0"] = dict(dR2=summ(va[:, 0]))
                pd.DataFrame(va, columns=["dR2", "dAUC"]).to_csv(OUT / f"g1_boot_adjusted_stored_{label}_seed0.csv", index=False)
                print(f"[{label} adjusted boot seed0] {r['adjusted_bootstrap_submitted_spec_seed0']}", flush=True)
        # family-cluster bootstraps for C1/C2 (full sample only; seed 0, plus seeds 1-3 for raw/adjusted)
        if label == "full_4263":
            r["bootstraps"] = {}
            specs = {"raw_fit": (D, ["raw_max_fit"]), "calibrated_fit": (D, ["calib_max_fit"]),
                     "classifier_prob": (D, ["clf_max_prob"]), "classifier_decision": (D, ["clf_max_decision"]),
                     "sae_centroid": (D, ["sae_centroid_max"]), "raw_over_magnitude": (D + GEN, ["raw_max_fit"])}
            for name, (ctrl, add) in specs.items():
                for seed in ([0, 1, 2, 3] if name in ("raw_fit", "raw_over_magnitude") else [0]):
                    rng = np.random.default_rng(seed)
                    picks = [rng.choice(fams_sub, len(fams_sub), replace=True) for _ in range(2000)]
                    v = par_boot(df, "ph4_fit_family", picks, ctrl, add)
                    pd.DataFrame(v, columns=["dR2", "dAUC"]).to_csv(OUT / f"g1_boot_{name}_{label}_seed{seed}.csv", index=False)
                    r["bootstraps"][f"{name}_seed{seed}"] = dict(dR2=summ(v[:, 0]), dAUC=summ(v[:, 1]))
                    print(f"[{label} boot {name} seed{seed}] dR2 {summ(v[:,0])['ci95']}", flush=True)
            # D. per-community correlations (Figure 4d) and Table S20 recompute
            r["per_community_corr_raw"] = {s: float(gg["raw_max_fit"].corr(gg["log_score"])) for s, gg in df.groupby("subreddit")}
            r["per_community_corr_stored"] = {s: float(gg["max_contrastive_ph4_fit"].corr(gg["log_score"])) for s, gg in df.groupby("subreddit")}
            mani = pd.read_csv(C.MANIFEST)[["post_id", "status"]]
            dm = df.merge(mani, on="post_id")
            dm["upvote_ratio"] = pd.to_numeric(dm["upvote_ratio"], errors="coerce")
            r["tableS20_recomputed_cached_vs_fetched"] = {
                st: dict(n=len(gg), communities=sorted(gg.subreddit.unique()), mean_age_days=float(gg.age_days.mean()),
                         mean_comments=float(gg.num_comments.mean()), mean_upvote_ratio=float(gg.upvote_ratio.mean()),
                         mean_stored_raw_fit=float(gg.max_contrastive_ph4_fit.mean()),
                         mean_clip_sim=float(gg.max_clip_family_sim.mean()))
                for st, gg in dm.groupby("status")}
            # E. cross-scheme sensitivity
            cl = pd.read_csv(G2OUT / "clusters.csv", dtype={"post_id": str})
            dfx = df.merge(cl, on="post_id", how="left")
            assert dfx.cc10_as_planned.notna().all()
            gcom = dfx.subreddit.astype("category").cat.codes.to_numpy()
            gtpl = dfx.cc10_as_planned.to_numpy()
            xs = {}
            for sname, groups, splitter in [("random5_seed0", None, KFold(5, shuffle=True, random_state=0)),
                                            ("community_gkf7_eq_LOCO", gcom, GroupKFold(7)),
                                            ("template_cc10_as_planned", gtpl, GroupKFold(5))]:
                xs[sname] = {}
                for vname in ["raw_fit", "calibrated_fit", "classifier_prob", "classifier_decision", "sae_centroid"]:
                    xs[sname][vname] = _inc(dfx, groups, D, VARIANTS[vname], splitter)
                xs[sname]["raw_over_magnitude"] = _inc(dfx, groups, D + GEN, ["raw_max_fit"], splitter)
            r["cross_scheme"] = xs
        res[label] = r
        C.write_json(OUT / "g1_results.json", res)
    # submitted reference values (from submitted outputs, for side-by-side)
    ref = pd.read_csv(C.AUDIT / "downstream_engagement_results.csv", index_col=0)
    res["submitted_reference"] = dict(
        downstream_engagement_results=ref.to_dict(orient="index"),
        reviewer_numerics_md=(C.AUDIT / "reviewer_numerics_results.md").read_text(),
        typicality_audit={"raw_over_D_plus_magnitude": 0.026, "sparse_norm": 0.030, "clip_norm": 0.036,
                          "random_sig_max": -0.000, "magnitude_block": 0.054, "source": "SSCR_Analysis_Audit.md"})
    C.write_json(OUT / "g1_results.json", res)
    print("[done] g1_results.json", flush=True)


if __name__ == "__main__":
    main()
