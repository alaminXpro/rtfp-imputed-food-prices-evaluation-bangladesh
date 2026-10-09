"""Task 1.2: audit the assumption that RTFP's plain item columns are the WFP/HDX survey values.

Every RTFP cell is classified against the HDX retail feed for the same market, month and item:
- match           observed in RTFP and an HDX row exists with |y/price − 1| ≤ tol
- value_mismatch  observed in RTFP, HDX row exists, values differ by more than tol
- no_hdx_row      observed in RTFP, no HDX row
- hdx_only        not observed in RTFP, but an HDX row exists
- neither         not observed in RTFP and no HDX row
"""
import numpy as np
import pandas as pd

from rtfp.crosswalk import _haversine_km

HDX_COMMODITY = {"Rice (coarse)": "rice", "Wheat flour": "wheat_flour", "Oil (palm)": "oil",
                 "Lentils (masur)": "lentils"}


def match_markets(panel_markets: pd.DataFrame, hdx_markets: pd.DataFrame, max_km: float = 5.0) -> pd.DataFrame:
    """Map each RTFP mkt_uid to an HDX market_id: exact name first, else nearest within max_km."""
    hm = hdx_markets.drop_duplicates("market_id")
    by_name = dict(zip(hm.market, hm.market_id))
    out = []
    for r in panel_markets.drop_duplicates("mkt_uid").itertuples(index=False):
        if r.mkt_name in by_name:
            out.append((r.mkt_uid, by_name[r.mkt_name], "name", 0.0))
            continue
        d = _haversine_km(r.lat, r.lon, hm.latitude.to_numpy(), hm.longitude.to_numpy())
        if len(d) and np.nanmin(d) <= max_km:
            i = int(np.nanargmin(d))
            out.append((r.mkt_uid, hm.market_id.iloc[i], "coords", float(d[i])))
        else:
            out.append((r.mkt_uid, np.nan, "none", np.nan))
    return pd.DataFrame(out, columns=["mkt_uid", "market_id", "how", "km"])


def audit_cells(panel: pd.DataFrame, hdx: pd.DataFrame, market_map: pd.DataFrame, tol: float = 0.01) -> pd.DataFrame:
    h = hdx[(hdx.pricetype == "Retail") & hdx.commodity.isin(HDX_COMMODITY)].copy()
    h["item"] = h.commodity.map(HDX_COMMODITY)
    h["month"] = pd.to_datetime(h.date).dt.to_period("M").dt.to_timestamp()
    h = h[["market_id", "month", "item", "price", "priceflag"]]
    p = panel[["mkt_uid", "mkt_name", "month", "item", "y", "c", "obs"]].merge(
        market_map[["mkt_uid", "market_id"]], on="mkt_uid", how="left")
    a = p.merge(h, on=["market_id", "month", "item"], how="left")
    has = a.price.notna()
    close = has & (np.abs(a.y / a.price - 1) <= tol)
    a["status"] = np.select(
        [a.obs & close, a.obs & has, a.obs & ~has, ~a.obs & has],
        ["match", "value_mismatch", "no_hdx_row", "hdx_only"], "neither")
    a["ratio_y_hdx"] = a.y / a.price
    return a
