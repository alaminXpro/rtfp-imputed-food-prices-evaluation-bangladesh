"""Table A5: two-source check of measurement noise in Dhaka (Task 1.4 / methodology D2, supporting).

TCB (Trading Corporation of Bangladesh) daily Dhaka retail prices, averaged to months (mid of min/max), vs
the WFP survey values in RTFP for 'Dhaka Sadar' (the only Dhaka market with substantial observed coverage).
TCB is not among WFP's listed sources (verified in the HDX metadata), so their measurement errors are taken as
independent. Then Cov(dTCB, dWFP) estimates the variance of the true monthly change, and
rel_two_source = Cov / Var(dWFP) is the reliability of WFP's monthly changes for this market.
Caveats: one market; ~30-60 months per item; TCB's range spans several Dhaka markets and grades.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/tables")
MAP = {"rice": [("rice_coarse", "none")], "wheat_flour": [("wheat_flour_atta_loose", "none")],
       "oil": [("palm_oil_loose", "none")],
       "lentils": [("lentil_masur", "large_grain"), ("lentil_masur", "medium_grain"), ("lentil_masur", "small_grain")]}
B = 2000
SEED = 5


def main():
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    panel["item"] = panel["item"].astype(str)
    w = panel[(panel.mkt_name == "Dhaka Sadar") & panel.obs][["item", "month", "y"]].copy()
    w["period"] = w.month.dt.to_period("M").astype(str)
    tcb = pd.read_csv("data/external/tcb/tcb_monthly.csv", dtype={"variant": str}, keep_default_na=False)
    from rtfp.download import as_of_date
    if as_of_date():  # data cutoff (RTFP_AS_OF): TCB months up to the cutoff month
        tcb = tcb[tcb.month <= as_of_date()[:7]]
    rows = []
    for item, variants in MAP.items():
        for (ten, var) in variants:
            t = tcb[(tcb.item_en == ten) & (tcb.variant == var)][["month", "mean_mid"]].rename(columns={"month": "period"})
            j = w[w["item"] == item].merge(t, on="period").sort_values("period")
            j["mi"] = pd.PeriodIndex(j.period, freq="M").astype("int64")
            j["d_wfp"] = np.log(j.y).diff().where(j.mi.diff() == 1)
            j["d_tcb"] = np.log(j.mean_mid).diff().where(j.mi.diff() == 1)
            k = j.dropna(subset=["d_wfp", "d_tcb"])
            if len(k) < 12:
                rows.append({"item": item, "tcb_series": f"{ten}:{var}", "n_months": len(k)})
                continue
            cov = np.cov(k.d_wfp, k.d_tcb)[0, 1]
            rel = cov / k.d_wfp.var()
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(B):  # moving-block bootstrap over months, block length 3
                starts = rng.integers(0, len(k) - 2, int(np.ceil(len(k) / 3)))
                idx = np.concatenate([np.arange(s, s + 3) for s in starts])[: len(k)]
                kk = k.iloc[idx]
                bs.append(np.cov(kk.d_wfp, kk.d_tcb)[0, 1] / kk.d_wfp.var())
            rows.append({"item": item, "tcb_series": f"{ten}:{var}", "n_months": len(k),
                         "first": k.period.min(), "last": k.period.max(),
                         "level_ratio_wfp_tcb_median": float((j.y / j.mean_mid).median()),
                         "corr_changes": float(np.corrcoef(k.d_wfp, k.d_tcb)[0, 1]),
                         "rel_two_source": float(rel), "rel_lo": float(np.quantile(bs, 0.025)),
                         "rel_hi": float(np.quantile(bs, 0.975)),
                         "share_zero_wfp_changes": float((k.d_wfp == 0).mean())})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "T_A5_tcb_two_source.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
