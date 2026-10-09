"""Figure 7 / Table 7n: does RTFP's national signal track independent official food inflation? (Task 4.3)

Compares monthly and 3-month national log changes of (a) RTFP's published index (all cells, equal-weight
items) and (b) an observed-only index (survey values only) with BBS national food CPI changes
(within one CPI base; no splice across the April 2023 rebasing). Descriptive (methodology M4).
Caveat: four RTFP items vs BBS's full food basket.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from rtfp.bbs import within_base_log_change
from rtfp.metrics import national_changes

OUT = Path("outputs/tables")


def hac_fit(y, x, lags, period=None):
    d = pd.concat([y, x], axis=1).dropna()
    f = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"n_months": len(d), "corr": d.corr().iloc[0, 1], "slope": f.params.iloc[1], "slope_se_hac": f.bse.iloc[1]}
    if period is not None:
        out.update(first_month=period.loc[d.index].min(), last_month=period.loc[d.index].max())
    return out


def main():
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    nat = national_changes(panel)
    nat["period"] = nat.month.dt.to_period("M").astype(str)
    bbs = pd.read_csv("data/external/bbs/bbs_food_cpi_monthly.csv", dtype={"month": str, "base": str})
    from rtfp.download import as_of_date
    if as_of_date():  # data cutoff (RTFP_AS_OF): BBS months up to the cutoff month
        bbs = bbs[bbs.month <= as_of_date()[:7]].reset_index(drop=True)
    bbs["d1"] = within_base_log_change(bbs, 1)
    bbs["d3"] = within_base_log_change(bbs, 3)
    m = nat.merge(bbs[["month", "base", "d1", "d3"]].rename(columns={"month": "period"}), on="period")
    for k in (1, 3):
        for col in ("dlog_pub_index", "dlog_obs_index"):
            m[f"{col}_{k}m"] = m[col].rolling(k).sum() if k > 1 else m[col]
    rows = []
    for k, lags in ((1, 3), (3, 6)):
        for src in ("dlog_pub_index", "dlog_obs_index"):
            for period, mask in {"all": m.period >= "2014-12", "2020-08+": m.period >= "2020-08",
                                 "2005-06 base (to 2023-03)": m.base == "2005-06", "2021-22 base": m.base == "2021-22"}.items():
                r = hac_fit(m.loc[mask, f"d{k}"], m.loc[mask, f"{src}_{k}m"], lags, m.loc[mask, "period"])
                rows.append({"horizon_months": k, "rtfp_index": src, "period": period, **r})
    t = pd.DataFrame(rows)
    # published vs survey-only national index (partly mechanical: observed cells enter both)
    pv = []
    for per, mask in {"2007-2026": nat.month >= "2007-02-01", "2020-08+": nat.month >= "2020-08-01",
                      "2007-2019": nat.month < "2020-01-01"}.items():
        d = nat.loc[mask, ["dlog_pub_index", "dlog_obs_index"]].dropna()
        f = sm.OLS(d.dlog_obs_index, sm.add_constant(d.dlog_pub_index)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        pv.append({"period": per, "n": len(d), "corr_pub_vs_obs": round(float(np.corrcoef(d.dlog_pub_index, d.dlog_obs_index)[0, 1]), 3),
                   "slope_obs_on_pub": round(float(f.params.iloc[1]), 3), "se_hac": round(float(f.bse.iloc[1]), 3),
                   "sd_pub": round(float(d.dlog_pub_index.std()), 4), "sd_obs": round(float(d.dlog_obs_index.std()), 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pv).to_csv(OUT / "T7n_published_vs_survey_national.csv", index=False)
    t.to_csv(OUT / "T7n_bbs_national.csv", index=False)
    # BBS-derived files are published under BBS's own terms (acknowledgement, no commercial reproduction),
    # NOT under this repository's CC BY 4.0 / MIT: see data/third_party/bbs/README.md
    tp = Path("data/third_party/bbs")
    tp.mkdir(parents=True, exist_ok=True)
    bbs[["month", "base", "food_index_national", "food_inflation_yoy_national", "source_file"]].to_csv(
        tp / "bbs_food_cpi_monthly.csv", index=False)
    m.to_csv(tp / "T7n_bbs_national_series.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
