"""Long market × month × item panel of one RTFP vintage, with observation flags and covariates.

Conventions (methodology M1):
- `y` is the survey value (plain item column); `c` is the published estimate (`c_<item>`).
- The observed side uses log y and needs both months observed; the published side uses log c.
- `mu_obs` is the month × item mean of observed changes only; both sides are demeaned by it.
"""
import numpy as np
import pandas as pd

from rtfp.crosswalk import AGGREGATE_NAMES, _haversine_km, _uid

ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
ID_COLS = ["mkt_name", "adm1_name", "adm2_name", "lat", "lon", "geo_id", "price_date",
           "data_coverage", "data_coverage_recent"]
FIELDS = {"": "y", "c_": "c", "o_": "o", "h_": "h", "l_": "l", "trust_": "trust"}


def _long(wide: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for it in ITEMS:
        cols = {f"{p}{it}": name for p, name in FIELDS.items() if f"{p}{it}" in wide.columns}
        if f"c_{it}" not in wide.columns:
            continue  # item absent from this vintage (e.g. rice in 42 of 83 vintages)
        f = wide[[c for c in ID_COLS if c in wide.columns] + list(cols)].rename(columns=cols)
        for name in FIELDS.values():
            if name not in f.columns:
                f[name] = np.nan
        frames.append(f.assign(item=it))
    p = pd.concat(frames, ignore_index=True)
    p = p[~p.mkt_name.isin(AGGREGATE_NAMES)].copy()
    p["mkt_uid"] = p.mkt_name.map(_uid)
    p["month"] = pd.to_datetime(p.price_date)
    p["obs"] = p.y.notna()
    return p.drop(columns="price_date")


def _changes(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["mkt_uid", "item", "month"])
    p["mi"] = p.month.dt.year * 12 + p.month.dt.month
    prev = p[["mkt_uid", "item", "mi", "y", "c", "obs"]].assign(mi=lambda d: d.mi + 1)
    p = p.merge(prev, on=["mkt_uid", "item", "mi"], how="left", suffixes=("", "_prev"))
    both = p.obs & (p.obs_prev == True)  # noqa: E712  (NaN-safe)
    p["dlog_obs"] = np.where(both, np.log(p.y) - np.log(p.y_prev), np.nan)
    p["dlog_pub"] = np.log(p.c) - np.log(p.c_prev)
    p["mu_obs"] = p.groupby(["item", "month"]).dlog_obs.transform("mean")
    p["e_obs"] = p.dlog_obs - p.mu_obs
    p["e_pub"] = p.dlog_pub - p.mu_obs
    return p.drop(columns=["y_prev", "c_prev", "obs_prev"])


def _gap(p: pd.DataFrame) -> pd.Series:
    last = p.mi.where(p.obs)
    last = last.groupby([p.mkt_uid, p.item]).ffill()
    return p.mi - last


def _distance(p: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (_, _), g in p.groupby(["item", "month"], sort=False):
        o = g[g.obs & g.lat.notna()]
        if o.empty:
            d = np.full(len(g), np.nan)
        else:
            d = _haversine_km(g.lat.to_numpy()[:, None], g.lon.to_numpy()[:, None],
                              o.lat.to_numpy()[None, :], o.lon.to_numpy()[None, :]).min(axis=1)
            d[g.obs.to_numpy()] = 0.0
        out.append(pd.DataFrame({"dist_obs_km": d, "n_obs_month": len(o)}, index=g.index))
    return pd.concat(out)


def build_panel(wide: pd.DataFrame) -> pd.DataFrame:
    """Panel from one vintage's wide CSV contents (one row per market × month)."""
    p = _changes(_long(wide))
    p["gap"] = _gap(p)
    p = p.join(_distance(p))
    return p.drop(columns="mi").reset_index(drop=True)


def load_panel(csv_path: str) -> pd.DataFrame:
    return build_panel(pd.read_csv(csv_path))


def main() -> None:
    import argparse
    from pathlib import Path

    from rtfp.download import latest_rtfp_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--vintages", default="data/raw/rtfp_vintages")
    ap.add_argument("--out", default="data/interim/panel_latest.parquet")
    a = ap.parse_args()
    src = latest_rtfp_csv(Path(a.vintages))
    p = load_panel(str(src)).assign(vintage=src.stem[-10:])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(a.out, index=False)
    print(f"{src.name}: {len(p)} rows, {p.mkt_uid.nunique()} markets, {p.obs.sum()} observed cells -> {a.out}")


if __name__ == "__main__":
    main()
