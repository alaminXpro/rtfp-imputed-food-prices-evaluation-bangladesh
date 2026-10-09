"""Table 5: within-market comparison on switching markets (Task 2.4, methodology D3; H3 in the draft analysis plan).

Sample (latest release, data/interim/panel_latest.parquet): market x month x item cells from 2020-01 in markets
whose name does not end in ' Division'. Cell types by the observation status of t and t-1:
  OO  observed t and t-1 -> enters with e = e_obs = survey log change - observed-cell month mean (mu_obs)
  II  imputed t and t-1  -> enters with e = e_pub = published log change - the same mu_obs
  OI / IO (one month observed, one imputed) -> excluded (OI added in sensitivity 'incl_mixed')
  cells with e undefined (no observed change anywhere that month, so mu_obs is undefined) -> excluded
A market x item enters if it keeps at least one OO and one II cell ('switching'). Exact counts by reason are
written to T5_switchers_sample.csv.
Model, per item: PPML of 100|e| on imputed + market FE + month FE (= market x item and month x item FE);
SEs clustered by market and two-way by market and month (Cameron-Gelbach-Miller; rtfp.ppml, tested).
Benchmark (D3): a noise-removing, signal-preserving imputer gives coefficient log sqrt(rel) under Gaussian e;
rel from Table 1 (T1_noise_reliability.csv): A_main rel_naive (the lower of the two, so the more demanding
benchmark for 'more than noise removed'; the naive estimator overstates noise only under white survey noise, A4) and
A_trim1 rel_naive. H3: coefficient below the benchmark;
one-sided test with the larger of the two SEs; Holm across the four items. A two-way pigeonhole bootstrap
re-estimates rel and the coefficient on the same draw, so the comparison carries the noise-estimate uncertainty.
Event time: spells of the same status; months -6..+6 around each observed<->imputed boundary, bounded by the
adjacent spells; mean |e| by event month (CR1 market-clustered CIs) and an event-study PPML with
market x item and month x item FE; pre-trend = joint Wald test of k = -6..-2 (reference k = -1) and the
linear slope over k = -6..-1.
Writes outputs/tables/T5_switchers*.csv and outputs/results_notes/T5.md (generated; do not edit by hand).
"""
import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from rtfp.metrics import pigeonhole_weights
from rtfp.noise import autocovariances, noise_from_autocov
from rtfp.ppml import (CELL_TYPES, classify_cells, cluster_mean, event_windows, lincom, noise_benchmark, ppml,
                       wald)

OUT = Path("outputs/tables")
NOTE = Path("outputs/results_notes/T5.md")
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
START = "2020-01-01"
NOISE_START = "2020-08-01"  # Table 1 noise sample A
CLUSTERS = {"market": ["mkt_uid"], "twoway": ["mkt_uid", "month"]}
BENCH = {"A_main": ("A_main", 0.0), "A_trim1": ("A_trim1", 0.01)}  # T1 variant, trim share
# matched benchmarks for the c_-based observed side: same Table 1 sample A cells, published (c_) changes
BENCH_C = {"A_main_c": 0.0, "A_trim1_c": 0.01}
WINDOW = 6
MIN_OBS_CHANGES = 10  # sensitivity mu_min10; chosen after seeing item-months whose mu rests on 1 change (disclosed)
SEED = 20261009
Z = norm.ppf(0.975)


# --- inputs ------------------------------------------------------------------------------------------------------

def noise_sample(panel):
    """Table 1 noise sample A: observed local changes, non-Division markets, price months >= 2020-08."""
    p = panel[panel.e_obs.notna()]
    return p[~p.mkt_name.str.endswith(" Division") & (p.month >= NOISE_START)]


def reliabilities(panel, t1):
    """rel_naive per item for A_main and A_trim1, recomputed and checked against Table 1."""
    a = noise_sample(panel)
    rows = []
    for name, (variant, trim) in BENCH.items():
        g = autocovariances(a, trim=trim)
        for it in ITEMS:
            est = noise_from_autocov(g.loc[it])
            ref = t1[(t1.variant == variant) & (t1["item"] == it)].iloc[0]
            if not (abs(est["rel_naive"] - ref.rel_naive) < 1e-10 and int(g.loc[it, "n0"]) == int(ref.n_changes)):
                raise AssertionError(f"recomputed rel differs from Table 1 for {variant}/{it}")
            cut = a[a["item"] == it].e_obs.abs().quantile(1 - trim) if trim else np.nan
            rows.append({"benchmark": name, "t1_variant": variant, "item": it, "n_changes": int(ref.n_changes),
                         "rel_naive": ref.rel_naive, "bench_abs": noise_benchmark(ref.rel_naive, 1),
                         "bench_sq": noise_benchmark(ref.rel_naive, 2), "ratio_abs": np.sqrt(ref.rel_naive),
                         "trim_cut_abs_e_obs": cut, "source": "Table 1 (recomputed, identical)"})
    for name, trim in BENCH_C.items():
        g = autocovariances(a, value="e_pub", trim=trim)
        for it in ITEMS:
            r = noise_from_autocov(g.loc[it])["rel_naive"]
            rows.append({"benchmark": name, "t1_variant": "A_main sample, value e_pub", "item": it,
                         "n_changes": int(g.loc[it, "n0"]), "rel_naive": r, "bench_abs": noise_benchmark(r, 1),
                         "bench_sq": noise_benchmark(r, 2), "ratio_abs": np.sqrt(r) if r > 0 else np.nan,
                         "trim_cut_abs_e_obs": np.nan, "source": "computed here (not in Table 1)"})
    return pd.DataFrame(rows)


def ever_observed_flag(panel, stack):
    """True for observed cells, and for imputed cells whose t and t-1 were never observed in any release."""
    ever = stack.groupby(["mkt_uid", "month", "item"], observed=True).obs.any().rename("ever").reset_index()
    ever["item"] = ever["item"].astype(str)
    p = panel[["mkt_uid", "month", "item", "obs"]].merge(ever, on=["mkt_uid", "month", "item"], how="left")
    p["ever"] = p.ever.fillna(False).astype(bool)
    prev = p[["mkt_uid", "item", "month", "ever"]].assign(month=lambda d: d.month + pd.offsets.MonthBegin(1))
    p = p.merge(prev.rename(columns={"ever": "ever_prev"}), on=["mkt_uid", "item", "month"], how="left")
    return (p.obs | ~(p.ever | p.ever_prev.fillna(False).astype(bool))).to_numpy()


def trim_flag(panel, rel):
    """Matched 1% trimming: OO cells above Table 1's A_trim1 cut for |e_obs|; II cells above the 99th percentile
    of |e_pub| among II cells of the period (same share removed on each side)."""
    c = classify_cells(panel, start=START)
    c = c[~c.mkt_name.str.endswith(" Division") & c.e.notna()]
    cut_obs = rel[rel.benchmark == "A_trim1"].set_index("item").trim_cut_abs_e_obs
    cut_imp = c[c.cell_type == "II"].groupby("item").abs_e.quantile(0.99)
    bad = ((c.cell_type == "OO") & (c.abs_e > c["item"].map(cut_obs))) | \
          ((c.cell_type == "II") & (c.abs_e > c["item"].map(cut_imp)))
    drop = c[bad][["mkt_uid", "month", "item"]].assign(drop=True)
    p = panel[["mkt_uid", "month", "item"]].merge(drop, on=["mkt_uid", "month", "item"], how="left")
    return (~p["drop"].fillna(False).astype(bool)).to_numpy(), cut_imp


# --- PPML table --------------------------------------------------------------------------------------------------

def fit_item(cells, item, power=1):
    d = cells[cells["item"] == item].copy()
    d["y"] = (100 * d.abs_e) ** power
    return ppml(d, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=CLUSTERS), d


def shape_ratio(e):
    """mean |e| / sqrt(mean e^2); sqrt(2/pi) = 0.798 for a zero-mean Gaussian."""
    e = np.asarray(e, float)
    return float(np.mean(np.abs(e)) / np.sqrt(np.mean(e ** 2)))


def spec_rows(spec, cells, rel, power, bench_names):
    rows, fits = [], {}
    for it in ITEMS:
        r, d = fit_item(cells, it, power)
        fits[it] = r
        se_m, se_2 = r.se["market"]["imputed"], r.se["twoway"]["imputed"]
        se_used = max(se_m, se_2) if np.isfinite(se_2) else se_m
        b = r.coef["imputed"]
        o, i = d[d.imputed == 0], d[d.imputed == 1]
        row = {"spec": spec, "item": it, "outcome": "abs_e" if power == 1 else "e_squared", "n_cells": r.n,
               "n_obs_cells": len(o), "n_imp_cells": len(i), "n_dropped_separation": r.n_dropped,
               "n_markets": d.mkt_uid.nunique(), "n_months": d.month.nunique(),
               "coef": b, "se_market": se_m, "ci_lo_market": b - Z * se_m, "ci_hi_market": b + Z * se_m,
               "se_twoway": se_2, "ci_lo_twoway": b - Z * se_2, "ci_hi_twoway": b + Z * se_2,
               "twoway_psd": r.psd["twoway"], "se_used": se_used, "ratio_imp_to_obs": np.exp(b),
               "mean_abs_e_obs_pct": 100 * o.abs_e.mean(), "mean_abs_e_imp_pct": 100 * i.abs_e.mean(),
               "shape_obs": shape_ratio(o.e), "shape_imp": shape_ratio(i.e),
               "shape_term": np.log(shape_ratio(i.e) / shape_ratio(o.e))}
        for bn in bench_names:
            rr = rel[(rel.benchmark == bn) & (rel["item"] == it)].iloc[0]
            bench = noise_benchmark(rr.rel_naive, power)
            row.update({f"rel_{bn}": rr.rel_naive, f"bench_{bn}": bench, f"diff_{bn}": b - bench,
                        f"below_point_{bn}": bool(b < bench), f"p_below_{bn}": norm.cdf((b - bench) / se_used)})
        rows.append(row)
    t = pd.DataFrame(rows)
    for bn in bench_names:
        t[f"p_holm_{bn}"] = multipletests(t[f"p_below_{bn}"], method="holm")[1]
        t[f"below_sig_{bn}"] = t[f"p_holm_{bn}"] < 0.05
    # mean over items of (coef - benchmark), market-clustered across items (a market's rows form one cluster)
    avg = {"spec": spec, "item": "mean_of_items", "outcome": t.outcome.iloc[0], "n_cells": int(t.n_cells.sum())}
    res = [fits[it] for it in ITEMS]
    for label, cols in CLUSTERS.items():
        est, se = lincom(res, [0.25] * 4, "imputed", cols)
        avg.update({"coef": est, f"se_{label}": se})
    avg["se_used"] = max(avg["se_market"], avg["se_twoway"]) if np.isfinite(avg["se_twoway"]) else avg["se_market"]
    for label in CLUSTERS:
        avg[f"ci_lo_{label}"], avg[f"ci_hi_{label}"] = avg["coef"] - Z * avg[f"se_{label}"], avg["coef"] + Z * avg[f"se_{label}"]
    for bn in bench_names:
        bench = t[f"bench_{bn}"].mean()
        avg.update({f"bench_{bn}": bench, f"diff_{bn}": avg["coef"] - bench, f"below_point_{bn}": bool(avg["coef"] < bench),
                    f"p_below_{bn}": norm.cdf((avg["coef"] - bench) / avg["se_used"])})
    return pd.concat([t, pd.DataFrame([avg])], ignore_index=True)


# --- bootstrap with rel resampled jointly -------------------------------------------------------------------------

def bootstrap(panel, cells, B):
    a = noise_sample(panel)
    universe = panel[~panel.mkt_name.str.endswith(" Division") & (panel.month >= START)]
    markets, months = sorted(universe.mkt_uid.unique()), sorted(universe.month.unique())
    rows = []
    for b in range(B):
        w_a = pigeonhole_weights(a, np.random.default_rng(SEED + b), markets, months)
        w_c = pigeonhole_weights(cells, np.random.default_rng(SEED + b), markets, months)
        rels = {bn: autocovariances(a.assign(w=w_a), trim=trim) for bn, (_, trim) in BENCH.items()}
        cw = cells.assign(w=w_c, y=100 * cells.abs_e)
        for it in ITEMS:
            d = cw[cw["item"] == it]
            try:
                coef = ppml(d, "y", ["imputed"], fe=["mkt_uid", "month"], weights="w").coef["imputed"]
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                coef = np.nan
            row = {"b": b, "item": it, "coef": coef}
            for bn, g in rels.items():
                rel = noise_from_autocov(g.loc[it])["rel_naive"] if it in g.index else np.nan
                row[f"rel_{bn}"] = rel
                # rel <= 0: estimated noise exceeds the total variance, the benchmark is -inf and the
                # coefficient cannot be below it; such draws count as 'not below', they are not dropped
                row[f"diff_{bn}"] = np.inf if (rel == rel and rel <= 0 and coef == coef) else coef - noise_benchmark(rel, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_summary(bt, main):
    rows = []
    m = bt.pivot(index="b", columns="item")
    for it in ITEMS + ["mean_of_items"]:
        point = main[(main["item"] == it)].iloc[0]
        coef_b = m["coef"][ITEMS].mean(axis=1) if it == "mean_of_items" else m["coef"][it]
        for bn in BENCH:
            diff = m[f"diff_{bn}"][ITEMS].mean(axis=1, skipna=False) if it == "mean_of_items" else m[f"diff_{bn}"][it]
            ok = diff.dropna()
            rel_b = m[f"rel_{bn}"][ITEMS] if it == "mean_of_items" else m[f"rel_{bn}"][[it]]
            rows.append({"item": it, "benchmark": bn, "B": len(bt.b.unique()), "n_valid": len(ok),
                         "n_draws_rel_nonpositive": int((rel_b <= 0).any(axis=1).sum()),
                         "coef": point.coef, "boot_sd_coef": coef_b.std(ddof=1), "se_used_analytic": point.se_used,
                         "diff_point": point[f"diff_{bn}"], "boot_sd_diff_finite": ok[np.isfinite(ok)].std(ddof=1),
                         # order statistics, no interpolation (some draws are +inf)
                         "diff_ci_lo": ok.quantile(0.025, interpolation="lower"),
                         "diff_ci_hi": ok.quantile(0.975, interpolation="higher"),
                         "share_diff_below_0": float((ok < 0).mean()), "p_boot_one_sided": float((ok >= 0).mean())})
    out = pd.DataFrame(rows)
    out["p_boot_holm"] = np.nan
    for bn in BENCH:
        sel = (out.benchmark == bn) & out["item"].isin(ITEMS)
        out.loc[sel, "p_boot_holm"] = multipletests(out.loc[sel, "p_boot_one_sided"], method="holm")[1]
    return out


# --- event time --------------------------------------------------------------------------------------------------

def event_tables(classified, tag="all_cells"):
    c = classified[~classified.mkt_name.str.endswith(" Division")]
    ev = event_windows(c[["mkt_uid", "item", "month", "obs"]], window=WINDOW)
    ev = ev.merge(c[["mkt_uid", "item", "month", "cell_type", "e", "abs_e", "enters", "imputed"]],
                  on=["mkt_uid", "item", "month"], how="left")
    main = ev[ev.enters].copy()
    mixed = ev[(ev.cell_type == "OI") & ev.abs_e.notna()].copy()  # imputed t after observed t-1 (k = 0, obs->imp)
    means = []
    for label, d in [("main", main), ("balanced", main[main.balanced]), ("mixed_OI", mixed)]:
        for (direction, k), g in d.groupby(["direction", "k"]):
            for it in ["all"] + ITEMS:
                gg = g if it == "all" else g[g["item"] == it]
                if len(gg) < 2:
                    continue
                m, se = cluster_mean(100 * gg.abs_e.to_numpy(), gg.mkt_uid.to_numpy())
                means.append({"cells": tag, "sample": label, "direction": direction, "k": int(k), "item": it, "n_cells": len(gg),
                              "n_events": gg.event_id.nunique(), "n_markets": gg.mkt_uid.nunique(),
                              "share_imputed": gg.imputed.mean(), "mean_abs_e_pct": m, "se": se,
                              "ci_lo": m - Z * se, "ci_hi": m + Z * se})
    coefs, tests = [], []
    for direction, d in main.groupby("direction"):
        d = d.copy()
        ks = sorted(k for k in d.k.unique() if k != -1)
        xs = [f"k{k:+d}" for k in ks]
        for k, x in zip(ks, xs):
            d[x] = (d.k == k).astype(float)
        d["y"] = 100 * d.abs_e
        d["fe_mi"] = d.mkt_uid + "|" + d["item"]
        d["fe_ti"] = d.month.dt.strftime("%Y-%m") + "|" + d["item"]
        r = ppml(d, "y", xs, fe=["fe_mi", "fe_ti"], clusters=CLUSTERS)
        for k, x in zip(ks, xs):
            row = {"cells": tag, "direction": direction, "k": k, "coef": r.coef[x], "n_cells_k": int((d.k == k).sum())}
            for label in CLUSTERS:
                se = r.se[label][x]
                row.update({f"se_{label}": se, f"ci_lo_{label}": r.coef[x] - Z * se, f"ci_hi_{label}": r.coef[x] + Z * se})
            coefs.append(row)
        pre = [x for k, x in zip(ks, xs) if -WINDOW <= k <= -2]
        post = [x for k, x in zip(ks, xs) if k >= 1]
        kk = np.array([int(x[1:]) for x in pre] + [-1], float)
        a = (kk - kk.mean()) / ((kk - kk.mean()) ** 2).sum()  # least-squares slope weights; beta(-1) = 0
        for label in CLUSTERS:
            V = r.vcov[label]
            base = {"cells": tag, "direction": direction, "cluster": label, "n_cells": r.n, "n_events": d.event_id.nunique(),
                    "psd": r.psd[label]}
            stat, df, p = wald(r.coef[pre].to_numpy(), V.loc[pre, pre].to_numpy())
            tests.append({**base, "test": "joint_pre_k-6..-2_eq_0", "estimate": np.nan, "se": np.nan,
                          "stat": stat, "df": df, "p": p})
            slope = float(a[:-1] @ r.coef[pre].to_numpy())
            vs = float(a[:-1] @ V.loc[pre, pre].to_numpy() @ a[:-1])
            se = np.sqrt(vs) if vs >= 0 else np.nan
            tests.append({**base, "test": "linear_pre_slope_per_month", "estimate": slope, "se": se,
                          "stat": (slope / se) ** 2, "df": 1, "p": 2 * norm.sf(abs(slope / se))})
            w = np.full(len(post), 1 / len(post))
            est = float(w @ r.coef[post].to_numpy())
            vp = float(w @ V.loc[post, post].to_numpy() @ w)
            se = np.sqrt(vp) if vp >= 0 else np.nan
            tests.append({**base, "test": "mean_post_k1..6_minus_k-1", "estimate": est, "se": se,
                          "stat": (est / se) ** 2, "df": 1, "p": 2 * norm.sf(abs(est / se))})
    return pd.DataFrame(means), pd.DataFrame(coefs), pd.DataFrame(tests), ev


# --- results note ------------------------------------------------------------------------------------------------

def f(x, nd=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def pval(p):
    return "n/a" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def registration_status():
    txt = Path("docs/analysis_plan.md").read_text() if Path("docs/analysis_plan.md").exists() else ""
    m = re.search(r"REGISTERED:\s*(\S+)", txt)
    if not m or "_" in m.group(1):
        return ("**Status: exploratory.** `docs/analysis_plan.md` (which lists this test as H3) is still a draft "
                "with no registration date, so this result was computed before registration.")
    return f"Registered analysis H3 (analysis plan registered {m.group(1)}); existing data, not prospective."


def reversal_sentence(per, tab):
    """Explain items whose raw means run against the FE-adjusted coefficient (computed, not hand-typed)."""
    rev = [it for it in ITEMS if (per.loc[it, "mean_abs_e_imp_pct"] > per.loc[it, "mean_abs_e_obs_pct"]) != (per.loc[it, "coef"] > 0)]
    if not rev:
        return ""
    m10 = tab[tab.spec == "mu_min10"].set_index("item")
    parts = []
    for it in rev:
        txt = (f"For {it} the raw means run the other way from the coefficient ({f(per.loc[it, 'coef'])}): they compare "
               f"different months, while the coefficient compares cells within market × item and month × item.")
        if it in m10.index:
            txt += (f" Dropping item-months whose μ rests on fewer than {MIN_OBS_CHANGES} observed changes (`mu_min10`) "
                    f"gives raw means {f(m10.loc[it, 'mean_abs_e_obs_pct'], 2)} / {f(m10.loc[it, 'mean_abs_e_imp_pct'], 2)} "
                    f"and coefficient {f(m10.loc[it, 'coef'])}, "
                    + ("so the reversal is concentrated in the excluded months; this is consistent with noisy μ "
                       "inflating imputed cells' |e| (inference; the spec also drops market-items that stop switching, "
                       "see 'Noisy μ' below)." if m10.loc[it, "mean_abs_e_imp_pct"] < m10.loc[it, "mean_abs_e_obs_pct"]
                       else "so the reversal is not explained by thinly observed months."))
        parts.append(txt)
    return " " + " ".join(parts)


def write_note(vintage, sample, rel, tab, bsum, means, coefs, tests, cut_imp, B, facts):
    share_few = facts["share_few"]
    single = ", ".join(f"{r.item} {r.month:%Y-%m}" for r in facts["single"].itertuples())
    main = tab[tab.spec == "main"].set_index("item")
    m10 = tab[tab.spec == "mu_min10"].set_index("item")
    per = main.loc[ITEMS]
    L = ["# T5 — Within-market comparison on switching markets (Task 2.4, methodology D3)", "",
         f"*Generated by `analysis/t5_switchers.py` on {date.today()} from release {vintage}; regenerate rather "
         "than edit. Every number below is read from the T5 tables written in the same run.*", "",
         registration_status(), ""]
    n_sig = {bn: int(per[f"below_sig_{bn}"].sum()) for bn in BENCH}
    n_pt = {bn: int(per[f"below_point_{bn}"].sum()) for bn in BENCH}
    lo, hi = per.coef.min(), per.coef.max()
    mt = tab[(tab.spec == "matched_trim1") & tab["item"].isin(ITEMS)]
    cb = tab[(tab.spec == "c_based_obs") & tab["item"].isin(ITEMS)]
    bmain = bsum[(bsum["item"] == "mean_of_items")].set_index("benchmark") if len(bsum) else None
    L += ["## Claim", "",
          f"Within the same market × item and month × item, imputed cells' absolute local monthly change is "
          + (f"smaller than observed cells' in all four items (every 95% CI, market and two-way, below 0)"
             if bool(((per.ci_hi_market < 0) & (per.ci_hi_twoway < 0)).all()) else
             f"smaller than observed cells' (95% CI below 0 with both SEs) in "
             f"{int(((per.ci_hi_market < 0) & (per.ci_hi_twoway < 0)).sum())} of 4 items")
          + f": PPML coefficients on `imputed` range from {f(lo)} to {f(hi)} across the "
          f"four items (imputed/observed ratio of mean |e| {f(np.exp(lo))} to {f(np.exp(hi))}).", "",
          f"Whether this is more than noise removal depends on the survey-noise estimate. Against the A_main "
          f"benchmark (naive estimator on untrimmed changes; "
          + ("the lower reliability in every item, so the more demanding test of 'more than noise removed'; "
             if bool((per.rel_A_main < per.rel_A_trim1).all()) else "")
          + f"the naive estimator overstates noise under white survey noise but not under stale-quote noise, Table A4) "
          f"the coefficient "
          f"is below the benchmark in {n_pt['A_main']} of 4 items at the point estimate and significantly below "
          f"(one-sided, Holm, 5%) in {n_sig['A_main']} of 4. Against A_trim1 (same naive estimator, 1% largest changes "
          f"trimmed) it is below in {n_pt['A_trim1']} of 4 at the point estimate and significantly below in "
          f"{n_sig['A_trim1']} of 4; on the population A_trim1 was estimated on (`matched_trim1`), significantly "
          f"below in {int(mt.below_sig_A_trim1.sum())} of 4. With the c_-based observed side and its own matched "
          f"rel, significantly below in {int(cb.below_sig_A_main_c.sum())} of 4 (A_main_c) and "
          f"{int(cb.below_sig_A_trim1_c.sum())} of 4 (A_trim1_c).", ""]
    sq = tab[(tab.spec == "squared") & tab["item"].isin(ITEMS)]
    if len(sq):
        L += [f"**The squared-change version needs no Gaussian assumption and gets equal weight.** With outcome e² "
              f"and benchmark log rel, the coefficient is below the A_main benchmark in "
              f"{int(sq.below_point_A_main.astype(bool).sum())} of 4 items at the point estimate (significantly in "
              f"{int(sq.below_sig_A_main.astype(bool).sum())}) and below A_trim1 in "
              f"{int(sq.below_point_A_trim1.astype(bool).sum())} of 4 (significantly in "
              f"{int(sq.below_sig_A_trim1.astype(bool).sum())}). The |e| verdict counts above depend on a Gaussian shape "
              f"that the data reject (see the 'Gaussian benchmark' caveat), so the two outcomes are reported side by side.", ""]
    sens = tab[tab["item"].isin(ITEMS) & (tab.spec != "main")]
    rng_txt = []
    for bn in BENCH:
        k = sens.groupby("spec")[f"below_sig_{bn}"].apply(lambda x: np.nan if x.isna().any() else int(x.astype(bool).sum())).dropna()
        span = f"{int(k.min())}" if k.min() == k.max() else f"{int(k.min())}–{int(k.max())}"
        rng_txt.append(f"{bn}: {span} of 4 over {len(k)} specs")
    L += ["Across the sensitivity specs (below), the count of items significantly below the benchmark ranges "
          + "; ".join(rng_txt) + ".", ""]
    if bmain is not None:
        bi = bsum[bsum["item"].isin(ITEMS)]
        nb = {bn: int((bi[bi.benchmark == bn].p_boot_holm < 0.05).sum()) for bn in BENCH}
        L += [f"**These verdicts treat rel as known.** With rel re-estimated on each bootstrap draw (B = {B}), the "
              f"number of items significantly below the benchmark (bootstrap one-sided p, Holm, 5%) is "
              f"{nb['A_main']} of 4 for A_main and {nb['A_trim1']} of 4 for A_trim1; for the item average the share of "
              f"draws below the benchmark is {f(bmain.loc['A_main', 'share_diff_below_0'])} (A_main) and "
              f"{f(bmain.loc['A_trim1', 'share_diff_below_0'])} (A_trim1). "
              + _bench_sd_sentence(), ""]
    L += ["*Reading (inference):* a coefficient below the benchmark means imputed local changes are less dispersed "
          "than the true local changes would be once survey noise is removed, i.e. more than noise was removed. "
          "The verdict is therefore stated for each noise estimate, not as one answer.", ""]
    # main table
    t = per.reset_index()
    show = pd.DataFrame({
        "item": t["item"],
        "n (obs/imp)": [f"{int(a)} ({int(b)}/{int(c)})" for a, b, c in zip(t.n_cells, t.n_obs_cells, t.n_imp_cells)],
        "markets": t.n_markets.astype(int), "coef": t.coef.map(f),
        "95% CI two-way": [f"[{f(a)}, {f(b)}]" for a, b in zip(t.ci_lo_twoway, t.ci_hi_twoway)],
        "SE market / two-way": [f"{f(a)} / {f(b)}" for a, b in zip(t.se_market, t.se_twoway)],
        "bench A_main (rel)": [f"{f(a)} ({f(b, 2)})" for a, b in zip(t.bench_A_main, t.rel_A_main)],
        "p Holm A_main": t.p_holm_A_main.map(pval),
        "bench A_trim1 (rel)": [f"{f(a)} ({f(b, 2)})" for a, b in zip(t.bench_A_trim1, t.rel_A_trim1)],
        "p Holm A_trim1": t.p_holm_A_trim1.map(pval)})
    m = main.loc["mean_of_items"]
    L += [f"## Main specification ({START[:7]} on, OO vs II cells, outcome 100·|e|)", "", md_table(show), "",
          f"Mean over items: coefficient {f(m.coef)} (SE market {f(m.se_market)}, two-way {f(m.se_twoway)}); "
          f"mean benchmark {f(m.bench_A_main)} (A_main), {f(m.bench_A_trim1)} (A_trim1); one-sided p "
          f"{pval(m.p_below_A_main)} and {pval(m.p_below_A_trim1)}.", "",
          "p: one-sided test of coefficient < benchmark using the larger of the market and two-way SEs, Holm across "
          "the four items; the benchmark is treated as known (the bootstrap below relaxes this). Raw mean |e| (%), "
          "observed / imputed cells: " + "; ".join(
              f"{it} {f(per.loc[it, 'mean_abs_e_obs_pct'], 2)} / {f(per.loc[it, 'mean_abs_e_imp_pct'], 2)}" for it in ITEMS) + "."
          + reversal_sentence(per, tab), ""]
    # bootstrap
    if len(bsum):
        bs = bsum.copy()
        bshow = pd.DataFrame({"item": bs["item"], "benchmark": bs.benchmark, "coef − bench": bs.diff_point.map(f),
                              "bootstrap 95% CI": [f"[{f(a)}, {f(b)}]" for a, b in zip(bs.diff_ci_lo, bs.diff_ci_hi)],
                              "share of draws below 0": bs.share_diff_below_0.map(f),
                              "p boot (Holm)": [f"{pval(a)} ({pval(b)})" for a, b in zip(bs.p_boot_one_sided, bs.p_boot_holm)],
                              "boot SD coef − bench": bs.boot_sd_diff_finite.map(f),
                              "draws with rel ≤ 0": bs.n_draws_rel_nonpositive.astype(int),
                              "boot SD coef / analytic SE": [f"{f(a)} / {f(b)}" for a, b in
                                                             zip(bs.boot_sd_coef, bs.se_used_analytic)]})
        L += [f"## Benchmark uncertainty: two-way pigeonhole bootstrap (B = {B}), rel re-estimated on the same draw",
              "", md_table(bshow), "",
              f"Each draw resamples markets and months ({START[:7]} on) jointly for the noise sample (Table 1 sample A) and "
              "the PPML sample, as Table 1 does. A draw with rel ≤ 0 (estimated noise exceeds total variance) has "
              "benchmark −∞ and counts as not below (coef − bench = +∞, so a CI bound can be inf; the SD of "
              "coef − bench uses finite draws). p boot = share of draws with coef − bench ≥ 0. The two-way pigeonhole "
              "bootstrap is known to be conservative for crossed designs (Owen 2007); bootstrap SD of the coefficient "
              + ("exceeds the analytic SE used in every item"
                 if bool((bs[(bs.benchmark == 'A_main') & bs['item'].isin(ITEMS)].eval('boot_sd_coef > se_used_analytic')).all())
                 else "vs the analytic SE used")
              + " (" + ", ".join(
                  f"{r.item} {f(r.boot_sd_coef)} vs {f(r.se_used_analytic)}" for r in bs[(bs.benchmark == 'A_main') & bs['item'].isin(ITEMS)].itertuples())
              + ").", ""]
    # sensitivity
    rows = []
    for spec, g in tab[tab["item"].isin(ITEMS)].groupby("spec", sort=False):
        g = g.set_index("item").loc[ITEMS]
        bns = [b for b in list(BENCH) + list(BENCH_C) if f"below_sig_{b}" in g.columns and g[f"below_sig_{b}"].notna().all()]
        rows.append({"spec": spec, "outcome": g.outcome.iloc[0],
                     **{it: f"{f(g.loc[it, 'coef'])} ({f(g.loc[it, 'se_used'])})" for it in ITEMS},
                     "items sig. below (Holm 5%)": "; ".join(f"{b}: {int(g[f'below_sig_{b}'].astype(bool).sum())}/4" for b in bns)})
    L += ["## Sensitivity (coefficient (SE used) by item)", "", md_table(pd.DataFrame(rows)), "",
          "Specs: `from_2020_08` = noise-sample period only; `incl_mixed` = adds imputed-t/observed-(t−1) cells; "
          "`c_based_obs` = observed side uses the published change (c_) instead of the survey change, against rel "
          "estimated on c_ changes of the same Table 1 sample A cells (A_main_c, A_trim1_c; computed here, not in "
          "Table 1); `never_observed_imputed` = drops imputed cells whose t or t−1 was a survey value in any "
          "release; `matched_trim1` = drops OO cells above Table 1's A_trim1 cut and II cells above their own 99th "
          "percentile (the population the A_trim1 rel describes; only that benchmark applies); `mu_min10` = only "
          f"item-months whose μ rests on ≥ {MIN_OBS_CHANGES} observed changes; `squared` = outcome e², benchmark "
          "log rel (no Gaussian assumption needed).", ""]
    rb = rel.pivot(index="item", columns="benchmark", values="rel_naive").loc[ITEMS]
    L += ["Reliabilities used (rel_naive): " + "; ".join(
        f"{b}: " + ", ".join(f"{it} {f(rb.loc[it, b], 3)}" for it in ITEMS) for b in rb.columns) + ".", ""]
    # event time
    L += [f"## Event time (months −{WINDOW}..+{WINDOW} around observed↔imputed spell boundaries; all items pooled)", ""]
    for cells_tag in ["all_cells"]:
        em = means[(means.cells == cells_tag) & (means["sample"] == "main") & (means["item"] == "all")]
        ec = coefs[coefs.cells == cells_tag].set_index(["direction", "k"])
        ev_rows = []
        for (direction, k), r in em.set_index(["direction", "k"]).sort_index().iterrows():
            c = ec.loc[(direction, k)] if (direction, k) in ec.index else None
            ev_rows.append({"direction": direction, "k": int(k), "n cells": int(r.n_cells),
                            "share imputed": f(r.share_imputed, 2),
                            "mean |e| % [95% CI]": f"{f(r.mean_abs_e_pct, 2)} [{f(r.ci_lo, 2)}, {f(r.ci_hi, 2)}]",
                            "PPML coef vs k=−1 [95% CI, market]": "ref." if k == -1 else (
                                "—" if c is None else f"{f(c.coef)} [{f(c.ci_lo_market)}, {f(c.ci_hi_market)}]")})
        L += [f"**Cells: {cells_tag}.**", "", md_table(pd.DataFrame(ev_rows)), ""]
    mx = means[(means.cells == "all_cells") & (means["sample"] == "mixed_OI") & (means["item"] == "all") & (means.k == 0)]
    if len(mx):
        r0 = mx.iloc[0]
        L += [f"k = 0 is empty by construction in both directions (the first month of a new spell is a mixed change). "
              f"Shown separately: imputed cells right after an observed month (OI, obs→imp, k = 0): mean |e| "
              f"{f(r0.mean_abs_e_pct, 2)}% [{f(r0.ci_lo, 2)}, {f(r0.ci_hi, 2)}], n = {int(r0.n_cells)}.", ""]
    tt = tests.copy()
    tshow = pd.DataFrame({"cells": tt.cells, "direction": tt.direction, "cluster": tt.cluster, "test": tt.test,
                          "estimate (SE)": [("—" if not np.isfinite(e) else f"{f(e)} ({f(s)})") for e, s in zip(tt.estimate, tt.se)],
                          "stat (df)": [f"{f(s, 2)} ({int(d)})" for s, d in zip(tt.stat, tt["df"])], "p": tt.p.map(pval),
                          "V PSD": tt.psd})
    L += ["The same tables for the `matched_trim1` and `mu_min10` cell samples are in T5_switchers_event_means.csv "
          "and T5_switchers_event_ppml.csv (column `cells`).", "",
          "Pre-trend and post tests (event-study PPML, market × item and month × item FE, reference k = −1):", "",
          md_table(tshow), "",
          "Where the two-way V is not PSD, read the market-clustered row.", ""]
    # which cells enter
    s_all = sample.groupby(["cell_type", "reason"]).n.sum().unstack(fill_value=0)
    L += [f"## Which cells enter ({START[:7]} on, four items, latest release)", "",
          md_table(s_all.reset_index().rename(columns={"cell_type": "cell type"})), "",
          "Cell types: " + "; ".join(f"{k} = {v}" for k, v in CELL_TYPES.items()) + ". Reasons, first that "
          "applies: division_market > mixed_change > excluded_by_rule > no_outcome (e undefined: no observed change "
          "in that item-month, so the observed-cell month mean is undefined) > not_switching (market × item without "
          "both an entering OO and an entering II cell) > enters. Per-item counts: T5_switchers_sample.csv.", "",
          "Outcome: OO cells use e_obs = log(y_t / y_{t−1}) − μ_t (survey values, both months observed); II cells "
          "use e_pub = log(c_t / c_{t−1}) − μ_t (published values, both months imputed); μ_t = mean survey change of "
          "all observed cells in that item-month of the latest release (panel column mu_obs; it includes the "
          f"'… Division' series while they are observed, last observed {facts['div_last']:%Y-%m}).", ""]
    # caveats
    so, si = per.shape_obs.agg(["min", "max"]), per.shape_imp.agg(["min", "max"])
    tw = tab[tab["item"].isin(ITEMS)]
    psd_bad = int((~tw.twoway_psd.astype(bool)).sum())
    L += ["## Caveats", "",
          f"- **Gaussian benchmark.** log sqrt(rel) assumes Gaussian e and noise. The shape ratio mean|e| / rms(e) "
          f"is {f(so['min'])}–{f(so['max'])} for observed and {f(si['min'])}–{f(si['max'])} for imputed cells "
          f"(Gaussian {f(np.sqrt(2 / np.pi))}). The PPML coefficient on |e| ≈ log(rms ratio) + log(shape_imp / "
          "shape_obs), so different shapes move it away from the benchmark even for a pure noise remover; the shape "
          "term log(shape_imp / shape_obs) by item is " + ", ".join(f"{it} {f(per.loc[it, 'shape_term'])}" for it in ITEMS)
          + " (column shape_term). The `squared` spec avoids the assumption but weights large changes heavily.",
          f"- **Population of rel.** rel comes from observed changes in non-Division markets from {NOISE_START[:7]} "
          f"(Table 1, sample A); the main sample starts {START[:7]} (`from_2020_08` matches the period). The A_trim1 "
          "rel describes changes below the 1% trim cut, so applying it to untrimmed |e| mixes populations; "
          "`matched_trim1` is the matched comparison (imputed-side 99th-percentile cuts, % units: "
          + ", ".join(f"{it} {f(100 * cut_imp[it], 2)}" for it in ITEMS) + ").",
          f"- **Noisy μ in thinly observed months.** μ is the mean of the observed changes in the item-month; where "
          f"few markets are observed it is noisy, and that noise enters every imputed cell's e_pub of that month but "
          f"not the observed cells' e_obs. Item-months with entering cells whose μ rests on a single observed change: "
          f"{single or 'none'}. Share of "
          f"entering cells in item-months with fewer than {MIN_OBS_CHANGES} observed changes: "
          + ", ".join(f"{it} {f(share_few[it], 3)}" for it in ITEMS) + ". `mu_min10` drops them and then re-classifies "
          "switching, so it also drops market-items that no longer switch; cells and markets, main → mu_min10: "
          + ", ".join(f"{it} {int(main.loc[it, 'n_cells'])}→{int(m10.loc[it, 'n_cells'])} cells, "
                      f"{int(main.loc[it, 'n_markets'])}→{int(m10.loc[it, 'n_markets'])} markets" for it in ITEMS)
          + ". The threshold was chosen after seeing these months.",
          "- **Imputations are the latest release's.** Inference from the release design (history is re-estimated "
          "in every release): imputed values in the latest release can draw on survey values from later months, so "
          "this compares spells as published now, not real-time nowcasts. The hold-out tables (T3) are the "
          "real-time test.",
          f"- **Two-way SEs.** CGM variances need not be PSD; {psd_bad} of {len(tw)} item-level two-way variances "
          "were negative (the market-clustered SE is then used).",
          "- **Event windows** are bounded by the adjacent spells; spells are short, so composition changes across "
          "event months. `balanced` rows in T5_switchers_event_means.csv keep only events with full windows on both "
          "sides. The event-study PPML pools items with item-specific FE.",
          "- **Mixed changes excluded.** The first month of each new spell is a mixed (survey→model or model→survey) "
          "change and is excluded from the main sample; `incl_mixed` adds the imputed-after-observed cells.", ""]
    L += ["## Files", "",
          "`outputs/tables/T5_switchers.csv` (all specs), `T5_switchers_benchmarks.csv` (rel and benchmarks; Table 1 "
          "values recomputed and checked), `T5_switchers_bootstrap.csv`, `T5_switchers_sample.csv` (cells by item × "
          "type × reason), `T5_switchers_event_means.csv`, `T5_switchers_event_ppml.csv`, `T5_switchers_pretrend.csv`, "
          "`T5_switchers_bootstrap_draws.csv` (one row per draw × item).",
          ""]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(L))


# --- main --------------------------------------------------------------------------------------------------------

def _bench_sd_sentence() -> str:
    """Which items have more benchmark (noise-estimate) uncertainty than coefficient uncertainty, from the draws."""
    d = pd.read_csv(OUT / "T5_switchers_bootstrap_draws.csv")
    out = []
    for bn in BENCH:
        bench = (d.coef - d[f"diff_{bn}"]).replace([np.inf, -np.inf], np.nan)
        g = d.assign(bench=bench).groupby("item").agg(sd_bench=("bench", "std"), sd_coef=("coef", "std"))
        g = g.loc[[i for i in ITEMS if i in g.index]]
        dom = [i for i, r in g.iterrows() if r.sd_bench > r.sd_coef]
        out.append(f"{bn}: benchmark SD exceeds coefficient SD for {', '.join(dom) if dom else 'no item'} ("
                   + ", ".join(f"{i} {r.sd_bench:.3f} vs {r.sd_coef:.3f}" for i, r in g.iterrows()) + ")")
    return "Bootstrap SD of the benchmark itself vs of the coefficient — " + "; ".join(out) + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws (methodology D1: 1,000; 0 = skip)")
    a = ap.parse_args()

    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    panel["item"] = panel["item"].astype(str)
    vintage = str(panel.vintage.iloc[0])
    t1 = pd.read_csv(OUT / "T1_noise_reliability.csv")
    stack = pd.read_parquet("data/interim/vintages.parquet", columns=["vintage", "mkt_uid", "month", "item", "obs"])
    rel = reliabilities(panel, t1)

    classified = classify_cells(panel, start=START)
    sample = classified.groupby(["item", "cell_type", "reason"]).size().rename("n").reset_index()
    cells = classified[classified.enters].copy()

    panel["ok_never"] = ever_observed_flag(panel, stack)
    n_chg = panel[panel.e_obs.notna()].groupby(["item", "month"]).size().rename("n_obs_changes").reset_index()
    panel["n_obs_changes"] = panel[["item", "month"]].merge(n_chg, on=["item", "month"], how="left").n_obs_changes.fillna(0).to_numpy()
    panel["ok_mu10"] = panel.n_obs_changes >= MIN_OBS_CHANGES
    panel["ok_trim"], cut_imp = trim_flag(panel, rel)
    both = list(BENCH)
    specs = [("main", dict(), 1, both), ("from_2020_08", dict(start=NOISE_START), 1, both),
             ("incl_mixed", dict(mixed=True), 1, both), ("c_based_obs", dict(obs_side="c"), 1, list(BENCH_C)),
             ("never_observed_imputed", dict(cell_ok="ok_never"), 1, both),
             ("matched_trim1", dict(cell_ok="ok_trim"), 1, ["A_trim1"]), ("mu_min10", dict(cell_ok="ok_mu10"), 1, both),
             ("squared", dict(), 2, both)]
    tabs = []
    for name, kw, power, bn in specs:
        kw = {"start": START, **kw}
        cs = cells if name in ("main", "squared") else classify_cells(panel, **kw).query("enters")
        tabs.append(spec_rows(name, cs, rel, power, bn))
    tab = pd.concat(tabs, ignore_index=True)
    main_tab = tab[tab.spec == "main"]

    if a.boot > 0:
        bt = bootstrap(panel, cells, a.boot)
        bt.to_csv(OUT / "T5_switchers_bootstrap_draws.csv", index=False)
        bsum = bootstrap_summary(bt, main_tab)
    else:
        bsum = pd.DataFrame(columns=["item", "benchmark", "B", "n_valid", "n_draws_rel_nonpositive", "coef", "boot_sd_coef", "se_used_analytic",
                                     "diff_point", "boot_sd_diff_finite", "diff_ci_lo", "diff_ci_hi", "share_diff_below_0", "p_boot_holm",
                                     "p_boot_one_sided"])
    ev = [event_tables(classified, "all_cells"),
          event_tables(classify_cells(panel, start=START, cell_ok="ok_trim"), "matched_trim1"),
          event_tables(classify_cells(panel, start=START, cell_ok="ok_mu10"), "mu_min10")]
    means, coefs, tests = (pd.concat([e[i] for e in ev], ignore_index=True) for i in range(3))

    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "T5_switchers.csv", index=False)
    rel.to_csv(OUT / "T5_switchers_benchmarks.csv", index=False)
    bsum.to_csv(OUT / "T5_switchers_bootstrap.csv", index=False)
    sample.to_csv(OUT / "T5_switchers_sample.csv", index=False)
    means.to_csv(OUT / "T5_switchers_event_means.csv", index=False)
    coefs.to_csv(OUT / "T5_switchers_event_ppml.csv", index=False)
    tests.to_csv(OUT / "T5_switchers_pretrend.csv", index=False)
    ent = classified[classified.enters].merge(panel[["mkt_uid", "item", "month", "ok_mu10", "n_obs_changes"]],
                                              on=["mkt_uid", "item", "month"])
    facts = {"share_few": 1 - ent.groupby("item").ok_mu10.mean(),
             "single": ent[ent.n_obs_changes == 1][["item", "month"]].drop_duplicates().sort_values(["item", "month"]),
             "div_last": panel[panel.mkt_name.str.endswith(" Division") & panel.obs].month.max()}
    write_note(vintage, sample, rel, tab, bsum, means, coefs, tests, cut_imp, a.boot, facts)

    pd.set_option("display.width", 250)
    cols = ["spec", "item", "n_cells", "coef", "se_market", "se_twoway", "bench_A_main", "p_holm_A_main",
            "bench_A_trim1", "p_holm_A_trim1"]
    print(tab[[c for c in cols if c in tab.columns]].round(4).to_string(index=False))
    print(bsum.round(4).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    print(f"wrote {NOTE}")


if __name__ == "__main__":
    main()
