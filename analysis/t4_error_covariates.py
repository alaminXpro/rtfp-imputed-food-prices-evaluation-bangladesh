"""Table 4: what predicts the external error of an imputed cell? (Task 2.3)

Sample: clean hold-out cells (forecast + cross_sectional). Outcome: absolute log level error in percent,
100 * |log(c_final / y_eval)|. Regressors, all measured in the imputing release v_final:
gap (months since the market's last observation of the item), distance to the nearest observed market (km),
RTFP trust score (1-10) and vintage age (months from price month to v_final). RTFP's data_coverage
field is one value per release for every market, so it is not a cell covariate and is excluded.
Distance is undefined for forecast cells and enters only the cross-sectional regression.
Item x month fixed effects; SEs clustered two-way by market and month.
Also: binned mean error by trust score, the producer's own reliability indicator.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.metrics import fe_ols

OUT = Path("outputs/tables")


def main():
    h = pd.read_parquet("data/interim/holdout.parquet")
    h["item"] = h["item"].astype(str)
    d = h[h.stratum.isin(["forecast", "cross_sectional"]) & h.y_eval.notna()].copy()
    d["abs_err_pct"] = 100 * np.abs(np.log(d.c_final / d.y_eval))
    d["vintage_age_m"] = (d.v_final.dt.year - d.month.dt.year) * 12 + (d.v_final.dt.month - d.month.dt.month)
    d["dist_100km"] = d.dist_obs_final / 100
    d["fe"] = d["item"] + "|" + d.month.dt.strftime("%Y-%m")
    # data_coverage is excluded: it is one value per release for all markets (dataset-level), not a cell covariate
    base = ["gap_final", "trust_final", "vintage_age_m"]
    xs = base + ["dist_100km"]  # for describe()
    rows = []
    # distance is undefined for forecast cells (no market observed that month), so it enters only the
    # cross-sectional regression
    for name, sub, regs in [("pooled", d, base), ("cross_sectional", d[d.stratum == "cross_sectional"], base + ["dist_100km"]),
                            ("forecast", d[d.stratum == "forecast"], base)]:
        r = fe_ols(sub, "abs_err_pct", regs, fe="fe", clusters=["mkt_uid", "month"]).reset_index(names="regressor")
        rows.append(r.assign(sample=name))
        for x in regs:  # one regressor at a time, same FE and clustering
            r1 = fe_ols(sub, "abs_err_pct", [x], fe="fe", clusters=["mkt_uid", "month"]).reset_index(names="regressor")
            rows.append(r1.assign(sample=f"{name}:univariate"))
    t = pd.concat(rows, ignore_index=True)
    trust_bins = d.assign(trust_bin=pd.cut(d.trust_final, [0, 5.999, 7.999, 9.999, 10])).groupby(
        ["trust_bin"], observed=True).agg(n=("abs_err_pct", "size"), mean_abs_err_pct=("abs_err_pct", "mean"),
                                          median_abs_err_pct=("abs_err_pct", "median")).reset_index()
    desc = d[xs + ["abs_err_pct"]].describe().T
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "T4_error_covariates.csv", index=False)
    trust_bins.to_csv(OUT / "T4b_error_by_trust.csv", index=False)
    desc.to_csv(OUT / "T4c_covariate_summary.csv")
    pd.set_option("display.width", 220)
    print(t[~t["sample"].str.contains("univariate")].round(4).to_string(index=False))
    print(t[t["sample"] == "pooled:univariate"].round(4).to_string(index=False))
    print(trust_bins.round(3).to_string(index=False))
    print(desc.round(3).to_string())


if __name__ == "__main__":
    main()
