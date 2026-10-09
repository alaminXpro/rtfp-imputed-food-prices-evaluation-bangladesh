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


def level_stats(c, y) -> dict:
    """Level accuracy of published estimate c against truth y (percent units)."""
    c, y = np.asarray(c, float), np.asarray(y, float)
    ape = np.round(np.abs(c / y - 1), 12)  # round so that exactly 10% is not '> 10%' through float error
    return {"n": len(c), "mape": 100 * ape.mean(), "median_ape": 100 * np.median(ape),
            "bias_logpct": 100 * np.mean(np.log(c / y)), "share_err_gt10": float((ape > 0.10).mean())}


def level_mz(df: pd.DataFrame, demean: str = "month") -> dict:
    """Slope of log y on log c; demean='month' removes item x month means from both sides first."""
    ly, lc = np.log(df.y.to_numpy(float)), np.log(df.c.to_numpy(float))
    if demean == "month":
        g = [df["item"].astype(str).to_numpy(), df.month.to_numpy()]
        ly = ly - pd.Series(ly).groupby(g).transform("mean").to_numpy()
        lc = lc - pd.Series(lc).groupby(g).transform("mean").to_numpy()
    elif demean != "none":
        raise ValueError(demean)
    lcd, lyd = lc - lc.mean(), ly - ly.mean()
    sx, sy = np.sqrt((lcd**2).mean()), np.sqrt((lyd**2).mean())
    return {"slope": (lcd * lyd).sum() / (lcd**2).sum(), "n": len(lc),
            "sd_ratio_c_over_y": sx / sy, "corr": (lcd * lyd).mean() / (sx * sy)}


def fe_ols(df: pd.DataFrame, y: str, xs: list, fe: str, clusters: list) -> pd.DataFrame:
    """OLS of y on xs with fixed effects `fe` (absorbed as dummies) and one- or two-way clustered SEs."""
    import statsmodels.api as sm

    d = df[[y] + xs + [fe] + clusters].dropna()
    # a regressor with (almost) no variation within FE groups is absorbed: report it, do not estimate it
    within = {x: (d[x] - d.groupby(fe)[x].transform("mean")).var() for x in xs}
    absorbed = [x for x in xs if not within[x] > 1e-10 * max(d[x].var(), 1e-300)]
    keep = [x for x in xs if x not in absorbed]
    if not keep:
        raise ValueError(f"all regressors absorbed by fixed effects: {absorbed}")
    xs_all, xs = xs, keep
    X = pd.concat([d[xs].astype(float), pd.get_dummies(d[fe].astype(str), prefix="fe", drop_first=True, dtype=float)],
                  axis=1)
    X = sm.add_constant(X)
    groups = np.column_stack([pd.factorize(d[c].astype(str))[0] for c in clusters])
    groups = groups[:, 0] if groups.shape[1] == 1 else groups
    import warnings

    fit = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    with warnings.catch_warnings():
        # two-way cluster covariance (Cameron-Gelbach-Miller) need not be PSD for absorbed dummies;
        # only the reported regressors must have valid SEs, checked below.
        warnings.simplefilter("ignore", RuntimeWarning)
        se, pv = fit.bse[xs], fit.pvalues[xs]
    if not np.isfinite(se).all():
        raise ValueError(f"non-finite clustered SE for {list(se[~np.isfinite(se)].index)}")
    out = pd.DataFrame({"coef": fit.params[xs], "se": se, "p": pv}).reindex(xs_all)
    out["absorbed"] = out.index.isin(absorbed)
    out["within_sd"] = [np.sqrt(within[x]) for x in xs_all]
    out["n"] = int(fit.nobs)
    return out


def persistent_spikes(df: pd.DataFrame, jump: float = 0.10, keep: float = 0.5, moved: float = 0.05) -> pd.DataFrame:
    """Flag survey spikes and whether the published series caught them (methodology D1.3).

    is_spike: |dlog_truth| > jump and the level at t+1 retains >= keep of the jump with the same sign,
              i.e. sign(j) * (j + dlog_truth_next) >= keep * |j|.
    caught:   the published change moved >= `moved` in the same direction.
    """
    j, nxt, pub = df.dlog_truth, df.dlog_truth_next, df.dlog_pub
    retained = np.sign(j) * (j + nxt)
    out = df.copy()
    out["is_spike"] = (j.abs() > jump) & (retained >= keep * j.abs())
    out["caught"] = out.is_spike & (np.sign(pub) == np.sign(j)) & (pub.abs() >= moved)
    return out


def national_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly national log changes, equal weight across items.

    dlog_pub_index: change of the mean log published price across all markets (balanced panel), per item.
    dlog_obs_index: mean observed log change (cells observed in both months), per item.
    Both are then averaged across items with equal weights.
    """
    p = panel.copy()
    p["item"] = p["item"].astype(str)
    lvl = p.assign(lc=np.log(p.c)).groupby(["item", "month"]).lc.mean().unstack("item").sort_index()
    pub = lvl.diff()
    obs = p.groupby(["item", "month"]).dlog_obs.mean().unstack("item").reindex(pub.index)
    return pd.DataFrame({"month": pub.index, "dlog_pub_index": pub.mean(axis=1, skipna=False).to_numpy(),
                         "dlog_obs_index": obs.mean(axis=1, skipna=False).to_numpy()}).dropna(how="all", subset=["dlog_pub_index", "dlog_obs_index"])
