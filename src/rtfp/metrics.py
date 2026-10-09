"""Local-change agreement statistics and two-way (market x month) cluster bootstrap (methodology D1).

Notation: e_pub = published (imputed) local change, e_obs = observed (truth) local change.
    corr      corr(e_pub, e_obs)
    beta_mz   slope of e_obs on e_pub  (calibration; regressor attenuation factor)
    beta_rev  slope of e_pub on e_obs  (identity: beta_rev = beta_mz * lam)
    lam       Var(e_pub) / Var(e_obs)
"""
import numpy as np
import pandas as pd


def local_components(pairs: pd.DataFrame, mode: str = "own_mu") -> pd.DataFrame:
    """Return pairs with e_pub and e_obs.

    own_mu:       e_pub = dlog_pub - mu_obs_vfinal (fallback mu_pub_vfinal when no market was observed);
                  e_obs = dlog_truth - mu_obs_latest.
    within_month: both sides demeaned within item x month over the evaluation sample itself
                  (item-months with a single pair are dropped).
    """
    p = pairs.copy()
    if mode == "own_mu":
        mu_pub = p.mu_obs_vfinal.where(p.mu_obs_vfinal.notna(), p.mu_pub_vfinal)
        p["e_pub"] = p.dlog_pub - mu_pub
        p["e_obs"] = p.dlog_truth - p.mu_obs_latest
    elif mode == "within_month":
        g = p.groupby(["item", "month"], observed=True)
        p = p[g.dlog_pub.transform("size") >= 2].copy()
        g = p.groupby(["item", "month"], observed=True)
        p["e_pub"] = p.dlog_pub - g.dlog_pub.transform("mean")
        p["e_obs"] = p.dlog_truth - g.dlog_truth.transform("mean")
    else:
        raise ValueError(mode)
    return p.dropna(subset=["e_pub", "e_obs"])


def weighted_local_stats(e_pub, e_obs, w) -> dict:
    e_pub, e_obs, w = (np.asarray(x, dtype=float) for x in (e_pub, e_obs, w))
    W = w.sum()
    mp, mo = (w * e_pub).sum() / W, (w * e_obs).sum() / W
    vp, vo = (w * (e_pub - mp) ** 2).sum() / W, (w * (e_obs - mo) ** 2).sum() / W
    c = (w * (e_pub - mp) * (e_obs - mo)).sum() / W
    return {"n": int(round(W)), "corr": c / np.sqrt(vp * vo), "beta_mz": c / vp, "beta_rev": c / vo,
            "lam": vp / vo, "var_pub": vp, "var_obs": vo}


def local_stats(e_pub, e_obs) -> dict:
    return weighted_local_stats(e_pub, e_obs, np.ones(len(np.asarray(e_pub))))


def pigeonhole_weights(df: pd.DataFrame, rng, markets, months) -> np.ndarray:
    """Two-way cluster bootstrap weights: (times market drawn) x (times month drawn)."""
    mk = pd.Series(np.bincount(rng.integers(0, len(markets), len(markets)), minlength=len(markets)), index=markets)
    mo = pd.Series(np.bincount(rng.integers(0, len(months), len(months)), minlength=len(months)), index=months)
    return (df.mkt_uid.map(mk).to_numpy() * df.month.map(mo).to_numpy()).astype(int)
