"""Stacked RTFP vintages and the natural hold-out H (methodology M1).

A cell (mkt_uid, month, item) enters H at its first imputed -> observed transition across vintages:
- v_truth  first vintage in which the cell is observed after having been imputed
- v_final  last vintage before v_truth in which it was imputed (the estimate being judged)
- c_first  the cell's first imputed value (earliest imputing vintage)
- y_eval   the cell's value in the LATEST vintage (equal to the WFP/HDX feed, Table A1); the truth used
           for evaluation. y_truth (value at v_truth) is kept for revision statistics.
- stratum  precedence: toggle (observed in a vintage before v_final) > unconfirmed (not observed in
           the latest vintage) > backfill (arrived more than max_lag_months after the price month)
           > forecast (no market observed for the item-month in v_final) > cross_sectional
Vintages that lack the item simply have no row for it, so they never count as imputed.
"""
import glob
import os

import numpy as np
import pandas as pd

from rtfp.crosswalk import MARKET_COLS, build_crosswalk, vintage_of
from rtfp.panel import build_panel

KEY = ["mkt_uid", "month", "item"]
STACK_COLS = ["vintage", "mkt_uid", "month", "item", "y", "c", "h", "l", "trust", "gap", "dist_obs_km",
              "n_obs_month", "obs", "dlog_obs", "dlog_pub", "mu_obs", "data_coverage", "data_coverage_recent"]


def build_stack(vintage_dir: str, items=None, prefix: str = "BGD", on_drift: str = "raise") -> pd.DataFrame:
    from rtfp.panel import ITEMS

    items = ITEMS if items is None else items
    frames, markets = [], []
    for f in sorted(glob.glob(os.path.join(vintage_dir, f"{prefix}_RTFP_mkt_*", "*.csv")), key=vintage_of):
        raw = pd.read_csv(f)
        markets.append(raw[MARKET_COLS].drop_duplicates().assign(vintage=vintage_of(f)))
        p = build_panel(raw, items)
        p["vintage"] = pd.Timestamp(vintage_of(f))
        frames.append(p[[c for c in STACK_COLS if c in p.columns]])
    # raises if a market name jumps location (on_drift='raise'), or drops that market (on_drift='drop')
    xw = build_crosswalk(pd.concat(markets, ignore_index=True), on_drift=on_drift)
    s = pd.concat(frames, ignore_index=True)
    dropped = xw.attrs.get("dropped_markets", [])
    if dropped:
        from rtfp.crosswalk import _uid
        s = s[~s.mkt_uid.isin({_uid(n) for n in dropped})].reset_index(drop=True)
    s["mu_pub_all"] = s.groupby(["vintage", "item", "month"]).dlog_pub.transform("mean")
    for c in ["y", "c", "h", "l", "dlog_obs", "dlog_pub", "mu_obs", "mu_pub_all", "dist_obs_km", "gap"]:
        s[c] = s[c].astype("float64")
    s["item"] = s["item"].astype("category")
    s.attrs["dropped_markets"] = list(dropped)
    return s


def latest_rows(stack: pd.DataFrame) -> pd.DataFrame:
    """Rows of the latest vintage that contains each item (rice is absent from many vintages)."""
    last_v = stack.groupby("item", observed=True).vintage.transform("max")
    return stack[stack.vintage == last_v]


def build_holdout(stack: pd.DataFrame, max_lag_months: int = 6) -> pd.DataFrame:
    s = stack.sort_values(KEY + ["vintage"]).reset_index(drop=True)
    g = s.groupby(KEY, observed=True, sort=False)
    imp = (~s.obs) & s.c.notna()
    seen_imp_before = imp.astype(int).groupby([s[k] for k in KEY], observed=True).cumsum().sub(imp.astype(int)) > 0
    truth = s[s.obs & seen_imp_before].groupby(KEY, observed=True).head(1)
    truth = truth[KEY + ["vintage", "y"]].rename(columns={"vintage": "v_truth", "y": "y_truth"})

    pre = s.merge(truth[KEY + ["v_truth"]], on=KEY)
    pre = pre[pre.vintage < pre.v_truth]
    imp_rows = pre[(~pre.obs) & pre.c.notna()]
    final = imp_rows.groupby(KEY, observed=True).tail(1)
    first = imp_rows.groupby(KEY, observed=True).head(1)[KEY + ["c", "vintage"]].rename(
        columns={"c": "c_first", "vintage": "v_first"})
    final = final.rename(columns={"vintage": "v_final", "c": "c_final", "h": "h_final", "l": "l_final",
                                  "trust": "trust_final", "gap": "gap_final", "dist_obs_km": "dist_obs_final",
                                  "n_obs_month": "n_obs_month_final"})
    keep = KEY + ["v_final", "c_final", "h_final", "l_final", "trust_final", "gap_final", "dist_obs_final",
                  "n_obs_month_final"] + [c for c in ("data_coverage", "data_coverage_recent") if c in final.columns]
    h = final[keep].merge(first, on=KEY).merge(truth, on=KEY)

    obs_before_final = pre.merge(h[KEY + ["v_final"]], on=KEY)
    obs_before_final = obs_before_final[obs_before_final.obs & (obs_before_final.vintage < obs_before_final.v_final)]
    toggled = obs_before_final[KEY].drop_duplicates().assign(toggle=True)
    h = h.merge(toggled, on=KEY, how="left")
    latest = latest_rows(s)[KEY + ["y", "obs"]].rename(columns={"y": "y_eval", "obs": "obs_latest"})
    h = h.merge(latest, on=KEY, how="left")
    confirmed = h.obs_latest.fillna(False).astype(bool)
    h.loc[~confirmed, "y_eval"] = np.nan
    h["truth_revised"] = confirmed & (np.abs(h.y_eval / h.y_truth - 1) > 0.001)
    h["lag_truth_m"] = (h.v_truth.dt.year - h.month.dt.year) * 12 + (h.v_truth.dt.month - h.month.dt.month)
    h["stratum"] = np.select(
        [h.toggle.fillna(False).astype(bool), ~confirmed, h.lag_truth_m > max_lag_months, h.n_obs_month_final == 0],
        ["toggle", "unconfirmed", "backfill", "forecast"], "cross_sectional")
    return h.drop(columns=["toggle", "obs_latest"]).reset_index(drop=True)


def build_change_pairs(holdout: pd.DataFrame, stack: pd.DataFrame) -> pd.DataFrame:
    """Monthly change pairs (t-1, t) for hold-out cells t, both published values from v_final(t).

    Truth side: y at t and t-1 from the latest vintage containing the item (y_eval); requires t-1 observed there.
    pair_type: both_imputed if t-1 was imputed in v_final(t), else imputed_t_observed_prev.
    """
    s = stack.copy()
    s["month_next"] = s.month + pd.offsets.MonthBegin(1)
    prev = s[["vintage", "mkt_uid", "item", "month_next", "c", "y", "obs"]].rename(
        columns={"month_next": "month", "c": "c_prev", "y": "y_prev", "obs": "obs_prev"})
    h = holdout.merge(prev.rename(columns={"vintage": "v_final", "y_prev": "y_prev_vfinal"}),
                      on=["v_final", "mkt_uid", "item", "month"], how="inner")
    last_v = stack.groupby("item", observed=True).vintage.max().rename("last_v").reset_index()
    prev = prev.merge(last_v, on="item")
    tv = prev[prev.vintage == prev.last_v][["mkt_uid", "item", "month", "y_prev", "obs_prev"]].rename(
        columns={"obs_prev": "obs_prev_truth", "y_prev": "y_prev_truth"})
    h = h[h.y_eval.notna()].merge(tv, on=["mkt_uid", "item", "month"], how="inner")
    h = h[h.obs_prev_truth.astype(bool) & h.c_prev.notna()]
    h["pair_type"] = np.where(h.obs_prev.astype(bool), "imputed_t_observed_prev", "both_imputed")
    h["dlog_pub"] = np.log(h.c_final) - np.log(h.c_prev)
    h["dlog_truth"] = np.log(h.y_eval) - np.log(h.y_prev_truth)
    if {"mu_obs", "mu_pub_all"} <= set(stack.columns):
        mu_v = stack[["vintage", "item", "month", "mu_obs", "mu_pub_all"]].drop_duplicates(["vintage", "item", "month"])
        h = h.merge(mu_v.rename(columns={"vintage": "v_final", "mu_obs": "mu_obs_vfinal",
                                         "mu_pub_all": "mu_pub_vfinal"}), on=["v_final", "item", "month"], how="left")
        mu_v = mu_v.merge(last_v, on="item")
        mu_l = mu_v[mu_v.vintage == mu_v.last_v][["item", "month", "mu_obs"]].rename(
            columns={"mu_obs": "mu_obs_latest"})
        h = h.merge(mu_l, on=["item", "month"], how="left")
    return h.drop(columns=["obs_prev", "obs_prev_truth"]).reset_index(drop=True)


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", default="data/raw/rtfp_vintages")
    ap.add_argument("--out", default="data/interim")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stack = build_stack(a.vintages)
    stack.to_parquet(out / "vintages.parquet", index=False)
    h = build_holdout(stack)
    h.to_parquet(out / "holdout.parquet", index=False)
    pairs = build_change_pairs(h, stack)
    pairs.to_parquet(out / "change_pairs.parquet", index=False)
    print(f"stack {len(stack):,} rows; holdout {len(h):,} cells; change pairs {len(pairs):,}")
    print(h.groupby(["item", "stratum"], observed=True).size().unstack(fill_value=0))
    print(pairs.groupby(["item", "stratum", "pair_type"], observed=True).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
