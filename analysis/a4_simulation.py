"""Table A4: framework simulation calibrated to the Bangladesh panel (Task 3.1; methodology M2 'Simulation').

Calibration, per item (src/rtfp/simulate.py: calibrate_common, calibrate_local):
  markets       the 104 non-Division markets of the latest release, with their observed share over 2020-08..2026-10
                as observation propensity, their division as region, and their coordinates (for IDW)
  months        2020-08..2026-10 (75; the Table 1 A_main noise-sample window)
  noise         Var(nu) = Table 1 A_main naive estimate; g0 = Table 1 A_main Var(e_obs)
  local         Var(d local), persistence and regional share from observed cells (demeaning-corrected)
  common        trend + month dummies + AR(1) fitted to the national mean log published price, 2007-01..2026-10
  release       share of latest-observed cells already in the imputing release, set so that the expected number
                of cross-sectional both-imputed pairs equals the empirical n (Table 3, own_mu)
  stale         carry-forward probability = empirical share of exact-zero observed changes (A_main sample)
Variants: noise in {white, stale, average} x imputer in {cond_mean, idw_seasonal, chain_mean, outlier_smooth};
R panels each (white 200, others 100). Metrics on cross-sectional both-imputed pairs, own_mu and within_month
(rtfp.metrics), noise estimators on latest-observed changes (rtfp.noise).
Comparison with Table 3: the empirical point estimate is 'in range' when it lies within the 2.5-97.5% range of
the simulated panel statistics (same expected n, so the range reflects sampling at the empirical sample size).

Outputs: outputs/tables/T_A4_simulation.csv, T_A4_simulation_match.csv, T_A4_simulation_noise_check.csv,
         T_A4_simulation_calibration.csv; outputs/results_notes/A4.md (generated).
Run unsandboxed if the host restricts Accelerate threads (large Cholesky factorizations).
"""
import os
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.simulate import (IMPUTERS, NOISE_VARIANTS, SimSpec, calibrate_common, calibrate_local, impute,
                           panel_metrics, simulate_panel)

ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
OUT = Path("outputs/tables")
NOTES = Path("outputs/results_notes")
SEED = 20261012
REPS = {"white": 200, "stale": 100, "average": 100}
WINDOW_START = "2020-08-01"
METRICS = ["corr", "beta_mz", "beta_rev", "lam"]
MATCH_METRICS = ["corr", "beta_mz", "lam"]


def _km(lat, lon):
    lat0, lon0 = np.nanmean(lat), np.nanmean(lon)
    return np.column_stack([(lon - lon0) * 111.32 * np.cos(np.radians(lat0)), (lat - lat0) * 110.57])


def calibrate(panel: pd.DataFrame, t1: pd.DataFrame, t3: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """SimSpec per item and a calibration table (one row per item)."""
    nd = panel[~panel.mkt_name.str.endswith(" Division")]
    win = nd[nd.month >= WINDOW_START]
    specs, rows = {}, []
    for it in ITEMS:
        a = win[win["item"] == it]
        mk = a.groupby("mkt_uid").agg(pi=("obs", "mean"), region=("adm1_name", "first"), lat=("lat", "first"),
                                      lon=("lon", "first")).sort_index()
        reg_code = pd.factorize(mk.region, sort=True)[0]
        coords = _km(mk.lat.to_numpy(float), mk.lon.to_numpy(float))
        obs = a[a.obs].assign(log_y=lambda d: np.log(d.y), region=lambda d: d.mkt_uid.map(
            dict(zip(mk.index, reg_code))))[["mkt_uid", "region", "month", "log_y", "e_obs"]]
        r1 = t1[(t1.variant == "A_main") & (t1["item"] == it)].iloc[0]
        loc = calibrate_local(obs, var_nu=float(r1.var_nu_naive), g0=float(r1.g0))
        nat = nd[nd["item"] == it].assign(lc=lambda d: np.log(d.c)).groupby("month").lc.mean()
        com = calibrate_common(nat)
        T = a.month.nunique()
        pi = mk.pi.to_numpy(float)
        n_emp = int(t3[(t3.stratum == "cross_sectional") & (t3.pair_type == "both_imputed") & (t3["mode"] == "own_mu")
                       & (t3["item"] == it)].n.iloc[0])
        p_release = float(np.clip(1 - np.sqrt(n_emp / ((T - 1) * (pi ** 2).sum())), 0.0, 1.0))
        e = a[a.e_obs.notna()]
        p_stale = float((e.dlog_obs == 0).mean())
        drift_window = com["drift"]
        start_idx = int(((pd.Timestamp(WINDOW_START).year - pd.Timestamp(com["start"]).year) * 12
                         + pd.Timestamp(WINDOW_START).month - pd.Timestamp(com["start"]).month))
        spec = SimSpec(n_months=T, start=WINDOW_START, propensity=tuple(pi), region=tuple(int(g) for g in reg_code),
                       coords_km=tuple(map(tuple, coords)), log_base=com["log_base"] + drift_window * start_idx,
                       drift=drift_window, season=com["season"], phi_common=com["phi_common"],
                       sd_common=com["sd_common"], phi_local=loc["phi_local"], var_dlocal=loc["var_dlocal"],
                       regional_share=loc["regional_share"], sd_market=loc["sd_market"],
                       var_noise=float(r1.var_nu_naive), noise="white", p_stale=p_stale, p_release=p_release)
        specs[it] = spec
        rows.append({"item": it, "n_markets": len(pi), "n_months": T, "mean_propensity": pi.mean(),
                     "expected_release_obs_per_month": (pi * p_release).sum(), "n_pairs_empirical": n_emp,
                     "p_release": p_release, "var_noise": spec.var_noise, "g0_empirical": float(r1.g0),
                     **{k: loc[k] for k in ("var_dlocal", "v_local", "phi_local", "regional_share",
                                            "regional_share_raw", "kappa", "sd_market", "n_level_obs")},
                     "p_stale": p_stale, "phi_common": com["phi_common"], "sd_common": com["sd_common"],
                     "drift_per_month": com["drift"], "season_amplitude": float(np.ptp(com["season"]))})
    return specs, pd.DataFrame(rows)


def run_rep(args):
    it, noise, rep, spec = args
    ss = np.random.SeedSequence([SEED, ITEMS.index(it), NOISE_VARIANTS.index(noise), rep])
    g_panel, g_chain = [np.random.default_rng(s) for s in ss.spawn(2)]
    sp = SimSpec(**{**spec.__dict__, "noise": noise})
    p = simulate_panel(sp, g_panel)
    rows, noise_est = [], None
    zero_share = float((np.diff(p.y, axis=1)[p.obs_latest[:, 1:] & p.obs_latest[:, :-1]] == 0).mean())
    rel_obs = p.obs_release.sum(axis=0)
    for imp in IMPUTERS:
        res = panel_metrics(p, impute(p, imp, rng=g_chain), noise=noise_est is None)
        if noise_est is None:
            noise_est = {k: res["own_mu"].get(k, np.nan) for k in
                         ("g0", "var_nu_naive", "rel_naive", "phi", "var_nu_ar1", "rel_ar1", "ar1_identified")}
        for mode, s in res.items():
            rows.append({"item": it, "noise": noise, "imputer": imp, "rep": rep, "mode": mode,
                         **{k: s.get(k, np.nan) for k in ["n"] + METRICS + ["corr_true", "rel_true"]},
                         **noise_est, "zero_change_share": zero_share,
                         "release_obs_per_month_median": float(np.median(rel_obs))})
    return rows


def _q(s, q):
    return s.quantile(q)


def summarise(sims: pd.DataFrame, t3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sims = sims.assign(rho_star_naive=sims["corr"] / np.sqrt(sims.rel_naive))
    agg = {"n_reps": ("rep", "size"), "n_pairs_mean": ("n", "mean")}
    for m in METRICS + ["corr_true", "rel_true", "rho_star_naive"]:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_median"] = (m, "median")
        agg[f"{m}_sd"] = (m, "std")
        agg[f"{m}_lo"] = (m, lambda s: _q(s, 0.025))
        agg[f"{m}_hi"] = (m, lambda s: _q(s, 0.975))
    t = sims.groupby(["item", "noise", "imputer", "mode"]).agg(**agg).reset_index()
    emp = t3[(t3.stratum == "cross_sectional") & (t3.pair_type == "both_imputed")][
        ["item", "mode", "n"] + METRICS].rename(columns={"n": "n_emp", **{m: f"{m}_emp" for m in METRICS}})
    t = t.merge(emp, on=["item", "mode"], how="left")
    for m in MATCH_METRICS:
        t[f"{m}_in_range"] = (t[f"{m}_emp"] >= t[f"{m}_lo"]) & (t[f"{m}_emp"] <= t[f"{m}_hi"])
    t["n_in_range"] = t[[f"{m}_in_range" for m in MATCH_METRICS]].sum(axis=1)
    t["all_in_range"] = t.n_in_range == len(MATCH_METRICS)
    t["distance"] = sum(((t[f"{m}_emp"] - t[f"{m}_median"]) / t[f"{m}_sd"]) ** 2 for m in MATCH_METRICS)
    best = (t.sort_values(["item", "mode", "noise", "distance"]).groupby(["item", "mode", "noise"]).head(1)
            [["item", "mode", "noise", "imputer", "n_in_range", "distance"]])
    return t, best


def noise_check(sims: pd.DataFrame, specs: dict, calib: pd.DataFrame) -> pd.DataFrame:
    one = sims[(sims.imputer == "cond_mean") & (sims["mode"] == "within_month")]
    rows = []
    for (it, nz), g in one.groupby(["item", "noise"]):
        ident = g.ar1_identified.astype(bool)
        rows.append({"item": it, "noise": nz, "n_reps": len(g), "var_nu_true": specs[it].var_noise,
                     "var_nu_naive_mean": g.var_nu_naive.mean(),
                     "var_nu_ar1_mean_identified": g.var_nu_ar1[ident].mean(), "ar1_identified_share": ident.mean(),
                     "g0_sim_mean": g.g0.mean(), "g0_empirical": calib.set_index("item").g0_empirical[it],
                     "zero_change_share_sim": g.zero_change_share.mean(),
                     "zero_change_share_empirical": specs[it].p_stale,
                     "release_obs_per_month_median_sim": g.release_obs_per_month_median.mean(),
                     "rel_true_pairs_mean": g.rel_true.mean(), "rel_naive_mean": g.rel_naive.mean(),
                     "corr_true_mean": g.corr_true.mean(),
                     "corr_over_sqrt_rel_true_mean": (g["corr"] / np.sqrt(g.rel_true)).mean(),
                     "corr_over_sqrt_rel_naive_mean": (g["corr"] / np.sqrt(g.rel_naive)).mean()})
    return pd.DataFrame(rows)


def _f(x, d=2):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def _md(df: pd.DataFrame, d: int = 2) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(_f(v, d) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def write_note(t: pd.DataFrame, best: pd.DataFrame, nc: pd.DataFrame, calib: pd.DataFrame, t3: pd.DataFrame,
               reps: dict) -> None:
    w = t[t.noise == "white"]
    # pattern summary per imputer (white noise): mean over items of median metrics, and matches out of 8 cells
    pat = (w.groupby(["imputer", "mode"]).agg(beta_mz=("beta_mz_median", "mean"), lam=("lam_median", "mean"),
                                              corr=("corr_median", "mean"), corr_true=("corr_true_median", "mean"),
                                              cells_all_in_range=("all_in_range", "sum"),
                                              cells_beta_in_range=("beta_mz_in_range", "sum"))
           .reset_index())
    emp = (t3[(t3.stratum == "cross_sectional") & (t3.pair_type == "both_imputed")]
           .groupby("mode")[["beta_mz", "lam", "corr"]].agg(["min", "max"]))
    by_imp = w.groupby("imputer").all_in_range.sum()
    by_imp_any = t.groupby(["noise", "imputer"]).all_in_range.sum()
    top = by_imp_any[by_imp_any == by_imp_any.max()]
    cm = w[(w.imputer == "cond_mean")]
    cm_wm, cm_own = cm[cm["mode"] == "within_month"], cm[cm["mode"] == "own_mu"]
    n_cells = len(w[w.imputer == "cond_mean"])
    nc_w = nc[nc.noise == "white"]
    tbl_pat = _md(pat.rename(columns={"beta_mz": "beta_mz (mean of item medians)", "lam": "lambda", "corr": "corr",
                                      "corr_true": "corr vs noise-free truth",
                                      "cells_all_in_range": "item cells with corr, beta_mz, lambda all in range",
                                      "cells_beta_in_range": "item cells with beta_mz in range"}))
    tbl_cal = _md(calib[["item", "n_pairs_empirical", "p_release", "expected_release_obs_per_month", "var_noise",
                         "var_dlocal", "phi_local", "regional_share", "p_stale", "phi_common", "sd_common"]], 4)
    tbl_nc = _md(nc_w[["item", "var_nu_true", "var_nu_naive_mean", "var_nu_ar1_mean_identified",
                       "ar1_identified_share", "g0_sim_mean", "g0_empirical", "rel_true_pairs_mean",
                       "rel_naive_mean", "corr_true_mean", "corr_over_sqrt_rel_true_mean",
                       "corr_over_sqrt_rel_naive_mean"]], 4)
    tbl_best = _md(best[best.noise == "white"], 2)
    n_beta_in, n_lam_in = int(cm.beta_mz_in_range.sum()), int(cm.lam_in_range.sum())
    lam_out = cm[~cm.lam_in_range]
    ok_all = cm[cm.all_in_range]
    ok_txt = ", ".join(f"{r.item} {r.mode}" for r in ok_all.itertuples()) or "none"
    nr = nc.assign(naive=nc.var_nu_naive_mean / nc.var_nu_true, ar1=nc.var_nu_ar1_mean_identified / nc.var_nu_true)
    nrw, nro = nr[nr.noise == "white"], nr[nr.noise != "white"]
    emp_n = t3[(t3.stratum == "cross_sectional") & (t3.pair_type == "both_imputed")].n
    sd_wm = w[w["mode"] == "within_month"].groupby("imputer").beta_mz_sd.mean()
    top_txt = "; ".join(f"{imp} with {nz} survey noise ({int(v)} of {n_cells} item-mode cells)" for (nz, imp), v in top.items())
    note = f"""# A4 — Framework simulation calibrated to Bangladesh (Task 3.1)

*Generated by `analysis/a4_simulation.py`; do not edit numbers by hand. Simulation results describe synthetic panels
calibrated to the data; they are not estimates from the data.*

**Claim.** In panels calibrated to the Bangladesh panel (104 markets, Table 1 noise, the empirical observation
pattern and hold-out size), a calibrated conditional-mean imputer (an oracle that knows the true covariance) does
not reproduce the empirical pattern, and the statistic that rules it out is λ, not β_MZ. Its simulated 95% range
contains the empirical β_MZ in {n_beta_in} of {n_cells} item × mode cells (simulated within-month β_MZ is
{_f(cm_wm.beta_mz_mean.mean())} on average but varies widely across panels), while it contains the empirical λ in only
{n_lam_in} of {n_cells}: the empirical λ lies above the simulated range in {', '.join(f"{r.item} {r.mode} ({_f(r.lam_emp, 3)} vs {_f(r.lam_lo, 3)}–{_f(r.lam_hi, 3)})" for r in lam_out.itertuples())}.
The cells where all three statistics (corr, β_MZ, λ) fit the oracle: {ok_txt}. Imputer type(s) whose simulated 95%
ranges contain all three empirical statistics in the most item × mode cells: {top_txt}.

**Simulated statistics by imputer** (white survey noise; mean over the 4 items of the per-item median across panels;
'in range' counts are out of 4 items):

{tbl_pat}

Empirical (Table 3, cross-sectional both-imputed): own_mu β_MZ {_f(emp.loc['own_mu', ('beta_mz', 'min')])}–{_f(emp.loc['own_mu', ('beta_mz', 'max')])},
λ {_f(emp.loc['own_mu', ('lam', 'min')])}–{_f(emp.loc['own_mu', ('lam', 'max')])}, corr {_f(emp.loc['own_mu', ('corr', 'min')])}–{_f(emp.loc['own_mu', ('corr', 'max')])};
within_month β_MZ {_f(emp.loc['within_month', ('beta_mz', 'min')])}–{_f(emp.loc['within_month', ('beta_mz', 'max')])},
λ {_f(emp.loc['within_month', ('lam', 'min')])}–{_f(emp.loc['within_month', ('lam', 'max')])}, corr {_f(emp.loc['within_month', ('corr', 'min')])}–{_f(emp.loc['within_month', ('corr', 'max')])}.

**Closest imputer per item and mode** (white noise; smallest standardized distance over corr, β_MZ, λ):

{tbl_best}

**Own-release demeaning lowers β_MZ even for a calibrated imputer.** With own_mu, the published side is demeaned by
the release's observed-market mean, whose sampling error the truth side does not share. Calibrated conditional mean,
own_mu: β_MZ {_f(cm_own.beta_mz_mean.mean())} (mean over items) vs {_f(cm_wm.beta_mz_mean.mean())} within-month.
So part of the empirical own_mu shortfall below 1 is a property of the metric at this number of observed markets,
not of the imputer (inference from the simulation).

**Noise estimators and disattenuation** (white noise, calibrated conditional mean, within-month):

{tbl_nc}

- `var_nu_naive_mean` overstates the true Var(ν) by the mean-reversion term, as derived in methodology D2; the AR(1)
  estimator is identified in the share of panels shown.
- `corr_over_sqrt_rel_true_mean` vs `corr_true_mean` checks the disattenuation ρ* = corr/√rel; using the naive
  reliability (`corr_over_sqrt_rel_naive_mean`) overstates ρ* when rel_naive understates the reliability.

**Calibration** (per item; sources in the script docstring):

{tbl_cal}

**Method.** `src/rtfp/simulate.py` (generator, four imputers, hold-out pairs, calibration); metrics from
`src/rtfp/metrics.py`, noise estimators from `src/rtfp/noise.py`. White noise: {int(reps['white'])} panels per item;
stale and average variants: {int(reps['stale'])} each. Tests: `tests/test_simulate.py` (β_MZ = 1 within ±0.05 over 200
panels under within-month demeaning; Var(ν) recovered under AR(1) + white noise; dense-matrix check of the conditional
mean).

**Caveats.**
- The calibrated conditional mean is an oracle: it knows the true covariance structure. A real imputer can do no
  better in expectation; 'calibrated' here means β_MZ = 1 by construction (within-month).
- Regional share, persistence and noise are estimated with error; the regional share fixes how much local information
  neighbours carry and therefore the attainable correlation. Wide empirical CIs (n = {int(emp_n.min())}–{int(emp_n.max())} pairs) mean several
  imputer types can be 'in range'; for the conditional mean, λ does nearly all the discriminating (Claim).
- Imputer types are stylized: RTFP's model is not known to us. 'Reproduces' means 'consistent with', not 'is'.
- The two fit criteria disagree: by joint in-range counts chain_mean ties for best under all three noise variants
  (so the noise model is not discriminated), while by standardized distance idw_seasonal is closest in most cells.
  Some item-mode cells fit no imputer (e.g. lentils own_mu, wheat flour within-month), and several are near-ties.
  Read the comparison as "mean-type imputers with residual chain noise or distance/seasonal fill are consistent
  with the data; a calibrated conditional mean is not", not as identification of one mechanism.
- Within-month β_MZ varies a lot across single panels (per-panel SD, mean over items: {', '.join(f'{k} {_f(v)}' for k, v in sd_wm.items())}),
  so small simulation runs are noisy; the {int(reps['white'])}-panel summaries are the ones to quote.
- Noise estimators depend on the noise model. Under white survey noise the naive estimator overstates Var(ν)
  (ratio to truth {_f(nrw.naive.min())}–{_f(nrw.naive.max())}) and the AR(1) estimator, where identified
  ({100 * nrw.ar1_identified_share.min():.0f}–{100 * nrw.ar1_identified_share.max():.0f}% of panels), understates it
  ({_f(nrw.ar1.min())}–{_f(nrw.ar1.max())}), so the true value lies between the two. Under the stale-quote and
  averaging variants **both** understate it (naive {_f(nro.naive.min())}–{_f(nro.naive.max())}, AR(1)
  {_f(nro.ar1.min())}–{_f(nro.ar1.max())}), so the naive estimator is not a bound there and is not guaranteed to be
  conservative for ρ*.
- `rel_true_pairs_mean` is computed on hold-out pairs and `rel_naive_mean` on the noise sample; their gap mixes
  estimator bias with sample differences, so compare them only qualitatively.

**Files.** `outputs/tables/T_A4_simulation.csv` (all summaries), `T_A4_simulation_match.csv`,
`T_A4_simulation_noise_check.csv`, `T_A4_simulation_calibration.csv`.
"""
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "A4.md").write_text(note)


def note_only():
    """Rewrite A4.md from the saved tables (no re-simulation)."""
    t = pd.read_csv(OUT / "T_A4_simulation.csv")
    nc = pd.read_csv(OUT / "T_A4_simulation_noise_check.csv")
    reps = t.groupby("noise").n_reps.max().to_dict()
    write_note(t, pd.read_csv(OUT / "T_A4_simulation_match.csv"), nc,
               pd.read_csv(OUT / "T_A4_simulation_calibration.csv"), pd.read_csv(OUT / "T3_local_changes.csv"), reps)


def main():
    if sys.argv[1:] == ["note"]:
        return note_only()
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    panel["item"] = panel["item"].astype(str)
    t1 = pd.read_csv(OUT / "T1_noise_reliability.csv")
    t3 = pd.read_csv(OUT / "T3_local_changes.csv")
    specs, calib = calibrate(panel, t1, t3)
    scale = float(os.environ.get("A4_REPS_SCALE", "1"))  # < 1 only for smoke tests
    reps = {k: max(2, int(round(v * scale))) for k, v in REPS.items()}
    tasks = [(it, nz, r, specs[it]) for it in ITEMS for nz in NOISE_VARIANTS for r in range(reps[nz])]
    # default 2 workers, single-threaded BLAS each: the user's laptop overheated with one worker per core
    nproc = int(os.environ.get("A4_PROCS", 2))
    with Pool(nproc) as pool:
        rows = [row for res in pool.imap_unordered(run_rep, tasks, chunksize=4) for row in res]
    sims = pd.DataFrame(rows).sort_values(["item", "noise", "imputer", "rep", "mode"])
    t, best = summarise(sims, t3)
    nc = noise_check(sims, specs, calib)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "T_A4_simulation.csv", index=False)
    best.to_csv(OUT / "T_A4_simulation_match.csv", index=False)
    nc.to_csv(OUT / "T_A4_simulation_noise_check.csv", index=False)
    calib.to_csv(OUT / "T_A4_simulation_calibration.csv", index=False)
    write_note(t, best, nc, calib, t3, reps)
    pd.set_option("display.width", 250)
    print(calib.round(4).to_string(index=False))
    print(t[t.noise == "white"][["item", "imputer", "mode", "n_pairs_mean", "corr_median", "beta_mz_median",
                                 "lam_median", "corr_emp", "beta_mz_emp", "lam_emp", "n_in_range"]].round(3)
          .to_string(index=False))
    print(nc.round(5).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
