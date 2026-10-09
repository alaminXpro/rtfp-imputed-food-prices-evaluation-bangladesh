"""Measurement noise in observed local price changes (methodology D2).

Model: observed local log level y = l + nu, with l AR(1) (persistence phi, variance V) and nu white
noise. For observed local changes:
    g(0) = Var(dl) + 2 Var(nu),       Var(dl) = 2 V (1 - phi)
    g(1) = -Var(nu) - V (1 - phi)^2
    g(k) = -V (1 - phi)^2 phi^(k-1),  k >= 2
Estimators:
    naive  Var(nu) = -g(1)                 (ignores mean reversion; overstates noise)
    AR(1)  phi = g(3)/g(2);  Var(nu) = -g(1) + g(2)/phi
Reliability of observed local changes: rel = 1 - 2 Var(nu) / g(0).
"""
import numpy as np
import pandas as pd


def autocovariances(df: pd.DataFrame, value: str = "e_obs", lags=(0, 1, 2, 3), demean: bool = True,
                    trim: float = 0.0) -> pd.DataFrame:
    """Pooled autocovariances by item, using pairs exactly k calendar months apart within a market.

    Optional column `w` weights observations (bootstrap); a pair takes the weight of its later month.
    trim > 0 drops, per item, the changes whose absolute value is above the (1 - trim) quantile before
    computing any lag (robust to gross entry errors, which appear as a spike and its reversal).
    """
    cols = ["mkt_uid", "item", "month", value] + (["w"] if "w" in df.columns else [])
    d = df[cols].dropna(subset=[value]).copy()
    if "w" not in d.columns:
        d["w"] = 1.0
    d = d[d.w > 0]
    d["item"] = d["item"].astype(str)
    if trim > 0:
        cut = d.groupby("item")[value].transform(lambda x: x.abs().quantile(1 - trim))
        d = d[d[value].abs() <= cut]
    if demean:
        wm = (d[value] * d.w).groupby(d["item"]).sum() / d.w.groupby(d["item"]).sum()
        d[value] = d[value] - d["item"].map(wm)
    d["mi"] = d.month.dt.year * 12 + d.month.dt.month
    out = {}
    for k in lags:
        lagged = d[["mkt_uid", "item", "mi", value]].assign(mi=lambda x: x.mi + k)
        j = d.merge(lagged, on=["mkt_uid", "item", "mi"], suffixes=("", "_lag"))
        wp = j[value] * j[f"{value}_lag"] * j.w
        out[f"g{k}"] = wp.groupby(j["item"]).sum() / j.w.groupby(j["item"]).sum()
        out[f"n{k}"] = j.w.groupby(j["item"]).size()
    return pd.DataFrame(out)


def noise_from_autocov(g: pd.Series) -> dict:
    g0, g1, g2, g3 = g["g0"], g["g1"], g["g2"], g["g3"]
    var_naive = max(0.0, -g1)
    est = {"g0": g0, "var_nu_naive": var_naive, "rel_naive": 1 - 2 * var_naive / g0}
    phi = g3 / g2 if g2 != 0 else np.nan
    ok = (g2 < 0) and (g3 < 0) and (0 < phi < 1)
    var_ar1 = (-g1 + g2 / phi) if ok else np.nan
    ok = ok and var_ar1 >= 0
    est.update({"phi": phi if ok else np.nan, "var_nu_ar1": var_ar1 if ok else np.nan,
                "rel_ar1": (1 - 2 * var_ar1 / g0) if ok else np.nan, "ar1_identified": bool(ok)})
    return est
