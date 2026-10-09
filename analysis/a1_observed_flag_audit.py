"""Table A1: are RTFP's plain item columns the WFP/HDX retail survey values? (Task 1.2, stop-gate)

Inputs:  data/interim/panel_latest.parquet, newest HDX file in data/raw/hdx_wfp/
Outputs: outputs/tables/T_A1_observed_flag_audit.csv, outputs/tables/T_A1b_c_vs_y.csv,
         data/interim/audit_cells.parquet
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.audit import audit_cells, match_markets

GATE = 0.90


def main():
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    from rtfp.download import hdx_file

    hdx_path = str(hdx_file("data/raw/hdx_wfp"))  # respects RTFP_AS_OF
    hdx = pd.read_csv(hdx_path)
    mm = match_markets(panel[["mkt_uid", "mkt_name", "lat", "lon"]],
                       hdx[["market", "market_id", "latitude", "longitude"]])
    a = audit_cells(panel, hdx, mm)
    Path("data/interim").mkdir(exist_ok=True)
    a.to_parquet("data/interim/audit_cells.parquet", index=False)

    counts = a.pivot_table(index="item", columns="status", values="mkt_uid", aggfunc="size", fill_value=0)
    for s in ["match", "value_mismatch", "no_hdx_row", "hdx_only"]:
        counts[s] = counts.get(s, 0)
    counts["rtfp_observed"] = counts[["match", "value_mismatch", "no_hdx_row"]].sum(axis=1)
    counts["share_matched"] = (counts.match / counts.rtfp_observed).round(4)
    counts["share_hdx_in_rtfp"] = ((counts.match + counts.value_mismatch) /
                                   (counts.match + counts.value_mismatch + counts.hdx_only)).round(4)
    mis = a[a.status == "value_mismatch"].groupby("item").ratio_y_hdx.median().rename("median_ratio_mismatch")
    flag = a[a.status == "match"].groupby("item").priceflag.apply(
        lambda s: (s == "aggregate").mean()).rename("share_aggregate_among_matched")
    t = counts.join(mis).join(flag)
    total = t[["match", "rtfp_observed"]].sum()
    t.loc["ALL", ["match", "rtfp_observed"]] = total
    t.loc["ALL", "share_matched"] = round(total.match / total.rtfp_observed, 4)

    # c_ vs y on observed cells: how often, how large, which direction, outlier-like?
    o = a[a.obs].sort_values(["mkt_uid", "item", "month"]).copy()
    o["dev"] = np.log(o.c / o.y)
    o["differs"] = np.abs(o.c / o.y - 1) > 0.01  # |c/y - 1| > 1%, as stated in the results note
    roll = o.groupby(["mkt_uid", "item"]).y.transform(lambda s: np.log(s).rolling(5, center=True, min_periods=3).median())
    o["y_off_local_median"] = np.abs(np.log(o.y) - roll)
    diff = o[o.differs]
    b = pd.DataFrame({
        "n_observed": o.groupby("item").size(),
        "n_c_ne_y": o.differs.groupby(o.item).sum(),
        "share_c_ne_y": o.differs.groupby(o.item).mean().round(4),
        "median_abs_dev_pct": (100 * diff.dev.abs()).groupby(diff.item).median().round(2),
        "share_c_above_y": (diff.dev > 0).groupby(diff.item).mean().round(3),
        "median_y_off_local_median_pct_when_c_ne_y": (100 * diff.y_off_local_median).groupby(diff.item).median().round(2),
        "median_y_off_local_median_pct_when_c_eq_y": (100 * o[~o.differs].y_off_local_median)
        .groupby(o[~o.differs].item).median().round(2),
    })
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    t.to_csv("outputs/tables/T_A1_observed_flag_audit.csv")
    b.to_csv("outputs/tables/T_A1b_c_vs_y.csv")
    pd.set_option("display.width", 250)
    print(f"HDX file: {hdx_path}; market match methods: {mm.how.value_counts().to_dict()}")
    print(t.to_string())
    print(b.to_string())
    passed = t.loc["ALL", "share_matched"] >= GATE
    print(f"GATE (share matched >= {GATE}): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
