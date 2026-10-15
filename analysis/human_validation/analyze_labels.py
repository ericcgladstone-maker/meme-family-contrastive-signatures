#!/usr/bin/env python3
"""Pre-specified analysis of the human-validation labels (15 v0.6 §7–§10). Frozen with the other scripts BEFORE any
label exists; tested only on synthetic labels.

Inputs (--draw DIR, the draw output):
  key/key.csv; labels/coder_A.csv, labels/coder_B.csv (exported by the form: item_id, primary, flag, ...);
  labels/adjudicated.csv (item_id, primary) for every study item on which A and B disagreed before adjudication
  (items flagged by either coder need no adjudication: they are missing in the adjudicated set).
Output: DIR/results/hv_results.json (all §8 endpoints with the specified intervals, and the mechanical §10 rule for
each label set).

Flag rules (15 §5, §7, §8): a blank primary is allowed only with flag = 1 (else the loader refuses). A flagged item is
missing: for the coder who flagged it (coder-specific sets and repeat agreement), and in the adjudicated set if either
coder flagged it. Exception: a flagged or "uncertain" gold item counts as not correct.
Bootstraps (15 §8): global pHash-cluster resampling, B = 2,000, generators default_rng([20260935, k]) with
k = 0 (reliability), 1 (adjudicated labels), 2 (Coder-B labels); percentile intervals; weights fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hv_common as H

CATS = ["confirmed", "other_family", "none", "uncertain"]
MEMBER = ("confirmed", "other_family")


# ------------------------------------------------------------------ basic estimators
def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def hajek(y, w):
    y, w = np.asarray(y, float), np.asarray(w, float)
    return float((w * y).sum() / w.sum()) if len(w) and w.sum() > 0 else float("nan")


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else float("nan")


def cohen_kappa(a, b):
    a, b = np.asarray(a, object), np.asarray(b, object)
    if len(a) == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def kripp_alpha_nominal(a, b):
    """Two coders, no missing values: alpha = 1 - D_o / D_e on the pooled value distribution."""
    a, b = np.asarray(a, object), np.asarray(b, object)
    if len(a) == 0:
        return float("nan")
    n = 2 * len(a)
    _, counts = np.unique(np.concatenate([a, b]).astype(str), return_counts=True)
    do = float((a != b).mean())
    de = float((n * n - (counts ** 2).sum()) / (n * (n - 1)))
    return float(1 - do / de) if de > 0 else float("nan")


def specific_agreement(a_member, b_member):
    a, b = np.asarray(a_member, bool), np.asarray(b_member, bool)
    both, neither, disc = (a & b).sum(), (~a & ~b).sum(), (a != b).sum()
    pos = 2 * both / (2 * both + disc) if (2 * both + disc) else float("nan")
    neg = 2 * neither / (2 * neither + disc) if (2 * neither + disc) else float("nan")
    return float(pos), float(neg)


def weighted_logit(x, y, w, iters=100):
    """IRLS for a weighted logistic regression y ~ b0 + b1 x; weights rescaled to mean 1 (15 §8)."""
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    w = w / w.mean()
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = w * p * (1 - p) + 1e-12
        z = eta + (y - p) / (p * (1 - p) + 1e-12)
        new = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-9 * np.eye(2), X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    return beta


def resample_clusters(clusters, rng):
    """GLOBAL cluster bootstrap (15 §8): draw cluster IDs with replacement across all bins; return row indices.
    A cluster is never split: all its rows enter together, once per time the cluster is drawn."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rows = {c: np.where(clusters == c)[0] for c in uniq}
    pick = rng.choice(uniq, len(uniq), replace=True)
    return np.concatenate([rows[c] for c in pick])


def pct(vals):
    v = np.asarray(vals, float)
    if np.all(np.isnan(v)):
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


# ------------------------------------------------------------------ label handling
def classify(primary, flag, argmax_disp):
    if flag:
        return None
    if primary == H.LABEL_UNCERTAIN:
        return "uncertain"
    if primary == H.LABEL_NONE:
        return "none"
    return "confirmed" if primary == argmax_disp else "other_family"


STUDY_PACKAGE_KINDS = ("study", "gold", "repeat")


def read_coder_csv(path, valid_labels, expected_ids=None, coder=None, package_id="study"):
    """Explicit NaN handling: primary is read as text ('' when blank); flag is 0/1. A blank primary is legal only with
    flag = 1; any non-blank primary must be a valid label.
    Guard 2 (15 v0.7): completeness before analysis -- when expected_ids is given, the file must contain every
    study-package item (420 study + 30 gold + 20 repeat copies) exactly once, no unknown IDs, and every row must carry
    the expected package_id and the expected coder (the coder column is required); flag must be 0 or 1."""
    r = pd.read_csv(path, dtype={"item_id": str, "primary": str, "package_id": str, "coder": str},
                    keep_default_na=False)
    r["primary"] = r["primary"].fillna("").astype(str).str.strip()
    # binary flag (GPT pre-freeze schema guard): the column is required and every value must be exactly 0 or 1
    if "flag" not in r.columns:
        raise SystemExit(f"{path.name}: missing flag column")
    fl = r["flag"].astype(str).str.strip()
    if not fl.isin(["0", "1"]).all():
        raise SystemExit(f"{path.name}: flag must be 0 or 1 on every row, e.g. {fl[~fl.isin(['0', '1'])].iloc[0]!r}")
    r["flag"] = fl.astype(int)
    bad_blank = r[(r.primary == "") & (r.flag == 0)]
    if len(bad_blank):
        raise SystemExit(f"{path.name}: {len(bad_blank)} rows with a blank primary label and no flag")
    bad_label = r[(r.primary != "") & ~r.primary.isin(valid_labels)]
    if len(bad_label):
        raise SystemExit(f"{path.name}: {len(bad_label)} rows with an unknown label, e.g. {bad_label.primary.iloc[0]!r}")
    if r.item_id.duplicated().any():
        raise SystemExit(f"{path.name}: duplicate item_id rows")
    if expected_ids is not None:
        exp = set(expected_ids)
        got = set(r.item_id)
        if got - exp:
            raise SystemExit(f"{path.name}: {len(got - exp)} unknown item_id(s), e.g. {sorted(got - exp)[0]!r}")
        if exp - got:
            raise SystemExit(f"{path.name}: incomplete: {len(exp - got)} of {len(exp)} study-package items have no response")
        if "package_id" not in r.columns or (r.package_id != package_id).any():
            raise SystemExit(f"{path.name}: package_id must be {package_id!r} on every row")
        if coder is not None:  # GPT pre-freeze schema guard: the coder column is required and must match on every row
            if "coder" not in r.columns:
                raise SystemExit(f"{path.name}: missing coder column")
            if (r.coder.astype(str).str.strip() != coder).any():
                raise SystemExit(f"{path.name}: coder column does not match coder {coder!r}")
    return r.set_index("item_id")


def valid_label_set():
    fams = [H.display_name(f) for f in sorted(pd.read_csv(H.DEV_INDEX).family_id.unique())]
    return set(fams) | {H.LABEL_NONE, H.LABEL_UNCERTAIN}


def load_labels(d):
    key = pd.read_csv(d / "key" / "key.csv", dtype={"item_id": str, "repeat_of": str})
    valid = valid_label_set()
    expected = key[key.kind.isin(STUDY_PACKAGE_KINDS)].item_id.tolist()
    lab = {c: read_coder_csv(d / "labels" / f"coder_{c}.csv", valid, expected_ids=expected, coder=c) for c in ["A", "B"]}
    return key, lab


def validate_adjudication(adjudicated, need_ids, valid_labels):
    """Guard 1 (15 v0.7): adjudicated.csv must contain EXACTLY the unflagged study items on which the two coders'
    pre-adjudication primary labels disagree -- no missing, extra or duplicate IDs -- each with a valid label
    (a family display name, 'none' or 'uncertain'). Items flagged by either coder skip adjudication and must not appear."""
    need = set(need_ids)
    if adjudicated is None:
        if need:
            raise SystemExit(f"adjudicated.csv missing: {len(need)} disagreements require an adjudicated label")
        return pd.Series(dtype=object)
    a = adjudicated.copy()
    a["primary"] = a["primary"].fillna("").astype(str).str.strip()
    if a.item_id.duplicated().any():
        raise SystemExit(f"adjudicated.csv: duplicate item_id(s), e.g. {a.item_id[a.item_id.duplicated()].iloc[0]!r}")
    got = set(a.item_id)
    if need - got:
        raise SystemExit(f"adjudicated.csv: {len(need - got)} disagreement(s) lack an adjudicated label")
    if got - need:
        raise SystemExit(f"adjudicated.csv: {len(got - need)} extra item_id(s) that are not unflagged disagreements "
                         f"(agreements, flagged items or unknown IDs), e.g. {sorted(got - need)[0]!r}")
    bad = a[~a.primary.isin(valid_labels)]
    if len(bad):
        raise SystemExit(f"adjudicated.csv: {len(bad)} invalid label(s), e.g. {bad.primary.iloc[0]!r}")
    return a.set_index("item_id").primary


def study_frame(key, lab, adjudicated):
    st = key[key.kind == "study"].copy().reset_index(drop=True)
    st["argmax_disp"] = st.raw_argmax.map(H.display_name)
    for c in ["A", "B"]:
        L = lab[c]
        missing = [i for i in st.item_id if i not in L.index]
        if missing:
            raise SystemExit(f"coder {c}: {len(missing)} study items without a response row")
        st[f"prim_{c}"] = st.item_id.map(L.primary)
        st[f"flag_{c}"] = st.item_id.map(L.flag).astype(int)
    either_flag = (st.flag_A == 1) | (st.flag_B == 1)
    agree = (~either_flag) & (st.prim_A == st.prim_B)
    need = st.loc[(~either_flag) & (st.prim_A != st.prim_B), "item_id"]
    adj = validate_adjudication(adjudicated, need, valid_label_set())
    st["prim_adj"] = np.where(agree, st.prim_A, st.item_id.map(adj).fillna(""))
    st["flag_adj"] = either_flag.astype(int)  # flagged by either coder -> missing in the adjudicated set
    for v in ["A", "B", "adj"]:
        st[f"cat_{v}"] = [classify(p, f, a) for p, f, a in zip(st[f"prim_{v}"], st[f"flag_{v}"], st.argmax_disp)]
    return st, agree


# ------------------------------------------------------------------ endpoints for one label set
def _stats(cat, w, b, x, fam_human, argmax, sub, with_curve=None):
    """All point estimates for one (possibly resampled) label set; arrays aligned."""
    cat = np.asarray(cat, object)
    w, b, x = np.asarray(w, float), np.asarray(b, object), np.asarray(x, float)
    out = {}
    for h in H.BINS:
        m = b == h
        g = cat[m]
        for c in CATS:
            out[f"bin_{h}_{c}"] = float((g == c).mean()) if m.any() else float("nan")
        gu = g[g != "uncertain"]
        out[f"bin_{h}_confirmed_excl_uncertain"] = float((gu == "confirmed").mean()) if len(gu) else float("nan")
    mem = np.isin(cat, MEMBER)
    out["pop_member_w"] = hajek(mem, w)
    for c in CATS:
        out[f"pop_{c}_w"] = hajek(cat == c, w)
    hi, lo = np.isin(b, ["B5", "B6"]), np.isin(b, ["B1", "B2"])
    out["B5B6_none_w"] = hajek(cat[hi] == "none", w[hi])
    out["B5B6_none_unw"] = float((cat[hi] == "none").mean()) if hi.any() else float("nan")
    mh = hi & mem
    out["B5B6_other_among_members_w"] = hajek(cat[mh] == "other_family", w[mh]) if mh.any() else float("nan")
    out["B5B6_other_among_members_unw"] = float((cat[mh] == "other_family").mean()) if mh.any() else float("nan")
    out["B1B2_member_w"] = hajek(mem[lo], w[lo])
    out["B1B2_member_unw"] = float(mem[lo].mean()) if lo.any() else float("nan")
    beta_w = weighted_logit(x, (cat == "confirmed").astype(float), w)
    out["slope_w"], out["intercept_w"] = float(beta_w[1]), float(beta_w[0])
    if with_curve is not None:
        out["_curve"] = 1 / (1 + np.exp(-(beta_w[0] + beta_w[1] * with_curve)))
    return out


def endpoints(st, v):
    s = st[st[f"cat_{v}"].notna()].reset_index(drop=True)
    cat = s[f"cat_{v}"].to_numpy(object)
    w, b, x = s.w.to_numpy(float), s.bin.to_numpy(object), s.raw_max_fit.to_numpy(float)
    human = s[f"prim_{v}"].to_numpy(object)
    pt = _stats(cat, w, b, x, human, s.raw_argmax.to_numpy(object), s.subreddit.to_numpy(object))
    ep = {"n_nonmissing": int(len(s)), "n_missing_flagged": int(st[f"cat_{v}"].isna().sum()), "per_bin": {}}
    for h in H.BINS:
        m = b == h
        g = cat[m]
        ep["per_bin"][h] = {c: pt[f"bin_{h}_{c}"] for c in CATS} | {
            "confirmed_excl_uncertain": pt[f"bin_{h}_confirmed_excl_uncertain"], "n": int(m.sum()),
            "wilson": {c: wilson(int((g == c).sum()), int(m.sum())) for c in CATS} | {
                "confirmed_excl_uncertain": wilson(int((g[g != "uncertain"] == "confirmed").sum()),
                                                   int((g != "uncertain").sum()))}}
    ep["population_w"] = {c: pt[f"pop_{c}_w"] for c in CATS} | {"member": pt["pop_member_w"]}
    ep["member_share_w"] = pt["pop_member_w"]
    ep["none_share_w"], ep["uncertain_share_w"] = pt["pop_none_w"], pt["pop_uncertain_w"]
    ep["B5B6_none_share_w"], ep["B5B6_none_share_unw"] = pt["B5B6_none_w"], pt["B5B6_none_unw"]
    ep["B5B6_other_family_among_members_w"] = pt["B5B6_other_among_members_w"]
    ep["B5B6_other_family_among_members_unw"] = pt["B5B6_other_among_members_unw"]
    ep["B1B2_member_share_w"], ep["B1B2_member_share_unw"] = pt["B1B2_member_w"], pt["B1B2_member_unw"]
    lo = np.isin(b, ["B1", "B2"]) & np.isin(cat, MEMBER)
    ep["B1B2_member_families"] = pd.Series(human[lo]).value_counts().to_dict()
    none_x = x[cat == "none"]
    ep["none_score_distribution"] = dict(
        n=int(len(none_x)), by_bin={h: int(((cat == "none") & (b == h)).sum()) for h in H.BINS},
        quantiles=({q: float(np.quantile(none_x, q / 100)) for q in (0, 25, 50, 75, 100)} if len(none_x) else {}))
    comp = {}
    for h in H.BINS:
        m = b == h
        fams = pd.Series(s.raw_argmax.to_numpy(object)[m])
        rows = {}
        for f, n in fams.value_counts().items():
            mm = m & (s.raw_argmax.to_numpy(object) == f)
            rows[H.display_name(f)] = dict(n=int(n), confirmed=float((cat[mm] == "confirmed").mean()),
                                           sparse_n_lt_10=bool(n < 10))
        comp[h] = rows
    ep["nomination_family_composition"] = comp
    com = {}
    for c_, g in s.groupby("subreddit"):
        cc = g[f"cat_{v}"].to_numpy(object)
        com[str(c_)] = dict(n=int(len(g)), **{k: hajek(cc == k, g.w) for k in CATS},
                            member=hajek(np.isin(cc, MEMBER), g.w))
    ep["community_prevalence_w_descriptive"] = com
    ep["kish"] = {"B5_B6": kish(w[np.isin(b, ["B5", "B6"])]), "B1_B2": kish(w[np.isin(b, ["B1", "B2"])]), "all": kish(w)}
    ep["trend_beta"] = [pt["intercept_w"], pt["slope_w"]]
    ep["trend_beta_unweighted"] = weighted_logit(x, (cat == "confirmed").astype(float), np.ones_like(w)).tolist()
    return ep, s


def bootstrap(st, v, B=H.N_BOOT):
    s = st[st[f"cat_{v}"].notna()].reset_index(drop=True)
    rng = np.random.default_rng([H.SEED_BOOT, H.SEED_BOOT_STREAMS[v]])
    cat = s[f"cat_{v}"].to_numpy(object)
    w, b, x = s.w.to_numpy(float), s.bin.to_numpy(object), s.raw_max_fit.to_numpy(float)
    grid = np.linspace(x.min(), x.max(), 50)
    reps, curves = [], []
    for _ in range(B):
        ix = resample_clusters(s.cluster.to_numpy(), rng)
        r = _stats(cat[ix], w[ix], b[ix], x[ix], None, None, None, with_curve=grid)
        curves.append(r.pop("_curve"))
        reps.append(r)
    df = pd.DataFrame(reps)
    ci = {k: pct(df[k]) for k in df.columns}
    ci["slope"] = ci["slope_w"]
    ci["B6_conf"] = ci["bin_B6_confirmed"]
    ci["member_w"] = ci["pop_member_w"]
    ci["B5B6_none_w"] = ci["B5B6_none_w"]
    band = np.nanpercentile(np.array(curves), [2.5, 97.5], axis=0)
    return ci, {"grid": grid.tolist(), "lo": band[0].tolist(), "hi": band[1].tolist()}


# ------------------------------------------------------------------ §10 rules
def r0_reasons(kappa_binary, gold_acc_A, gold_acc_B):
    """Guard 3 (15 v0.7): an undefined binary kappa (NaN) FAILS R0 (conservative)."""
    why = []
    if kappa_binary is None or not np.isfinite(kappa_binary):
        why.append("binary member/none kappa undefined")
    elif kappa_binary < 0.40:
        why.append("binary kappa < 0.40")
    for c, g in (("A", gold_acc_A), ("B", gold_acc_B)):
        if g is None or not np.isfinite(g) or g < 0.70:
            why.append(f"coder {c} gold accuracy < 0.70 or undefined")
    return why


def r0_gate(kappa_binary, gold_acc_A, gold_acc_B):
    return bool(r0_reasons(kappa_binary, gold_acc_A, gold_acc_B))


def rule_for(ep, ci):
    """Order of application R4 -> R2 -> R1 -> else R3 (R0 is global and handled before)."""
    b6 = ep["per_bin"]["B6"]["confirmed"]
    slope_lo, slope_hi = ci["slope"]
    if ep["B5B6_none_share_w"] >= 0.50 and slope_lo <= 0:
        return "R4"
    if b6 >= 0.70 and ci["B6_conf"][0] >= 0.50 and slope_lo > 0:
        return "R2"
    if ep["member_share_w"] < 0.30 and b6 < 0.50:
        return "R1"
    return "R3"


CONSERVATIVE = {"R4": 4, "R3": 3, "R1": 2, "R2": 1}


def combine(rule_adj, rule_b):
    return rule_adj if CONSERVATIVE[rule_adj] >= CONSERVATIVE[rule_b] else rule_b


# ------------------------------------------------------------------ reliability and controls
def _rel_stats(pa, pb, ca, cb):
    ma, mb = np.isin(ca, MEMBER), np.isin(cb, MEMBER)
    bok = (ca != "uncertain") & (cb != "uncertain")
    pos, neg = specific_agreement(ma[bok], mb[bok])
    return dict(kappa_full=cohen_kappa(pa, pb), kappa_binary=cohen_kappa(ma[bok], mb[bok]),
                alpha_nominal=kripp_alpha_nominal(pa, pb), raw_agreement=float((pa == pb).mean()) if len(pa) else float("nan"),
                specific_positive=pos, specific_negative=neg)


def reliability(st, agree, B=H.N_BOOT):
    """Pre-adjudication labels; items flagged by either coder are excluded (no label to compare)."""
    ok = st.cat_A.notna() & st.cat_B.notna()
    s = st[ok].reset_index(drop=True)
    pa, pb = s.prim_A.to_numpy(object), s.prim_B.to_numpy(object)
    ca, cb = s.cat_A.to_numpy(object), s.cat_B.to_numpy(object)
    pt = _rel_stats(pa, pb, ca, cb)
    rng = np.random.default_rng([H.SEED_BOOT, H.SEED_BOOT_STREAMS["reliability"]])
    reps = []
    for _ in range(B):
        ix = resample_clusters(s.cluster.to_numpy(), rng)
        reps.append(_rel_stats(pa[ix], pb[ix], ca[ix], cb[ix]))
    df = pd.DataFrame(reps)
    bok = (ca != "uncertain") & (cb != "uncertain")
    return dict(point=pt, ci={k: pct(df[k]) for k in df.columns}, n_items=int(len(s)),
                n_flagged_excluded=int((~ok).sum()), n_binary=int(bok.sum()), n_uncertain_excluded=int((~bok).sum()),
                pre_adjudication_agreement_nonflagged=float(agree[ok].mean()) if ok.any() else float("nan"))


def controls(key, lab):
    out = {}
    gold = key[key.kind == "gold"]
    rep = key[key.kind == "repeat"]
    for c in ["A", "B"]:
        L = lab[c]
        # gold: correct only if the primary equals the known family and the item is not flagged (flag/uncertain = wrong)
        corr = [(i in L.index) and (L.flag.get(i, 0) == 0) and (L.primary.get(i) == H.display_name(f))
                for i, f in zip(gold.item_id, gold.gold_family)]
        k = int(sum(corr))
        # repeats: flagged observations are missing
        pairs = [(i, o) for i, o in zip(rep.item_id, rep.repeat_of)
                 if i in L.index and o in L.index and L.flag.get(i, 0) == 0 and L.flag.get(o, 0) == 0]
        same = [L.primary[i] == L.primary[o] for i, o in pairs]
        out[c] = dict(gold_correct=k, gold_n=len(gold), gold_acc=k / len(gold) if len(gold) else float("nan"),
                      gold_wilson=wilson(k, len(gold)), repeat_same=int(sum(same)), repeat_n=len(pairs),
                      repeat_n_definition="repeated posts whose original and repeat responses are both unflagged",
                      repeat_excluded_flagged=int(len(rep) - len(pairs)), repeat_wilson=wilson(int(sum(same)), len(pairs)))
    return out


def run(d, B=H.N_BOOT):
    key, lab = load_labels(d)
    adjp = d / "labels" / "adjudicated.csv"
    adjudicated = pd.read_csv(adjp, dtype={"item_id": str, "primary": str}, keep_default_na=False) if adjp.exists() else None
    st, agree = study_frame(key, lab, adjudicated)
    rel = reliability(st, agree, B=B)
    ctl = controls(key, lab)
    r0_why = r0_reasons(rel["point"]["kappa_binary"], ctl["A"]["gold_acc"], ctl["B"]["gold_acc"])
    r0 = bool(r0_why)
    res = dict(reliability=rel, controls=ctl, R0_fails=r0, R0_reasons=r0_why, label_sets={})
    for v in ["adj", "B"]:
        ep, _ = endpoints(st, v)
        ci, band = bootstrap(st, v, B=B)
        res["label_sets"][v] = dict(endpoints=ep, ci=ci, calibration_band=band,
                                    rule=None if r0 else rule_for(ep, ci))
    res["rule_final"] = "R0" if r0 else combine(res["label_sets"]["adj"]["rule"], res["label_sets"]["B"]["rule"])
    (d / "results").mkdir(exist_ok=True)
    (d / "results" / "hv_results.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True)
    a = ap.parse_args()
    d = Path(a.draw)
    if "fixture" not in str(d):
        ok, why = H.freeze_ok()
        if not ok:
            raise SystemExit(f"REFUSING: {why}")
    r = run(d)
    print(json.dumps({"R0_fails": r["R0_fails"], "rule_final": r["rule_final"]}))


if __name__ == "__main__":
    main()
