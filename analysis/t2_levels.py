"""Table 2: level accuracy of RTFP imputations on the hold-out (Task 2.2, levels part).

For every hold-out cell, the published estimate from the imputing release (c_final, and c_first) is
compared with the survey value in the latest release (y_eval, which equals the WFP feed).
Reported by item x stratum; clean strata (forecast, cross_sectional) are the evaluation sample, the
others are shown for comparison only. CIs: two-way (market x month) pigeonhole bootstrap, B draws.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.metrics import level_mz, level_stats, pigeonhole_weights

B = 500
SEED = 20261010
OUT = Path("outputs/tables")


def summarise(d, cvar):
    s = level_stats(d[cvar], d.y_eval)
    mz_m = level_mz(d.rename(columns={cvar: "c", "y_eval": "y"}), demean="month")["slope"]
    mz_n = level_mz(d.rename(columns={cvar: "c", "y_eval": "y"}), demean="none")["slope"]
    lc, ly = np.log(d[cvar]), np.log(d.y_eval)
    band = d.l_final.notna() & d.h_final.notna()
    inside = ((d.y_eval >= d.l_final) & (d.y_eval <= d.h_final))[band]
    return {**s, "mz_slope_month_demeaned": mz_m, "mz_slope_raw": mz_n, "corr_log_levels": np.corrcoef(lc, ly)[0, 1],
            "band_coverage": inside.mean() if cvar == "c_final" else np.nan,
            "band_halfwidth_pct": (50 * (d.h_final - d.l_final) / d.c_final)[band].median() if cvar == "c_final" else np.nan}


def main():
    h = pd.read_parquet("data/interim/holdout.parquet")
    h["item"] = h["item"].astype(str)
    h = h[h.y_eval.notna() & h.c_final.notna()]
    rows = []
    for (it, st), d in h.groupby(["item", "stratum"]):
        for cvar in ("c_final", "c_first"):
            dd = d.dropna(subset=[cvar])
            rows.append({"item": it, "stratum": st, "estimate": cvar, **summarise(dd, cvar)})
    t = pd.DataFrame(rows)

    clean = h[h.stratum.isin(["forecast", "cross_sectional"])]
    markets, months = sorted(set(h.mkt_uid)), sorted(set(h.month))
    boot = []
    for b in range(B):
        w = pigeonhole_weights(clean, np.random.default_rng(SEED + b), markets, months)
        d = clean.assign(w=w)
        d = d[d.w > 0]
        ape = np.abs(d.c_final / d.y_eval - 1)
        lr = np.log(d.c_final / d.y_eval)
        for (it, st), g in d.assign(ape=ape, lr=lr).groupby(["item", "stratum"]):
            boot.append({"b": b, "item": it, "stratum": st, "mape": 100 * np.average(g.ape, weights=g.w),
                         "bias_logpct": 100 * np.average(g.lr, weights=g.w)})
    bt = pd.DataFrame(boot)
    ci = bt.groupby(["item", "stratum"]).agg(mape_lo=("mape", lambda s: s.quantile(0.025)),
                                             mape_hi=("mape", lambda s: s.quantile(0.975)),
                                             bias_lo=("bias_logpct", lambda s: s.quantile(0.025)),
                                             bias_hi=("bias_logpct", lambda s: s.quantile(0.975))).reset_index()
    t = t.merge(ci.assign(estimate="c_final"), on=["item", "stratum", "estimate"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "T2_levels.csv", index=False)
    pd.set_option("display.width", 260)
    show = t[t.estimate == "c_final"][["item", "stratum", "n", "mape", "mape_lo", "mape_hi", "median_ape", "bias_logpct",
                                        "bias_lo", "bias_hi", "share_err_gt10", "mz_slope_month_demeaned",
                                        "mz_slope_raw", "corr_log_levels", "band_coverage", "band_halfwidth_pct"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
