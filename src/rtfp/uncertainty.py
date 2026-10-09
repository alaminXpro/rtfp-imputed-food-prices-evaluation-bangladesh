"""Calibrated uncertainty for imputed RTFP cells (methodology D5).

Target r = log(y / c): how far the survey value will be from the published estimate.
Quantile gradient boosting gives q_lo, q_hi; conformalized quantile regression (Romano et al. 2019) widens
them by a threshold estimated on a later calibration block, separately by Mondrian group when the group
has enough calibration points. Coverage is claimed only inside the hold-out's covariate support.
"""
import math

import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score (inf if > n)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= n else float("inf")


def cqr_intervals(q_lo, q_hi, threshold):
    return np.asarray(q_lo) - np.asarray(threshold), np.asarray(q_hi) + np.asarray(threshold)


def cqr_scores(r, q_lo, q_hi):
    return np.maximum(np.asarray(q_lo) - np.asarray(r), np.asarray(r) - np.asarray(q_hi))


def mondrian_thresholds(scores, groups, alpha: float, min_group: int = 30) -> dict:
    """Threshold per group; groups with fewer than `min_group` calibration points use the pooled threshold."""
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    pooled = conformal_quantile(scores, alpha)
    out = {}
    for g in np.unique(groups):
        s = scores[groups == g]
        out[g] = conformal_quantile(s, alpha) if len(s) >= min_group else pooled
    out["__pooled__"] = pooled
    return out


def time_blocks(vintage: pd.Series, t1, t2) -> pd.Series:
    """train: vintage <= t1; calib: t1 < vintage <= t2; test: vintage > t2 (whole vintages per block)."""
    v = pd.to_datetime(vintage)
    return pd.Series(np.select([v <= pd.Timestamp(t1), v <= pd.Timestamp(t2)], ["train", "calib"], "test"),
                     index=vintage.index)


def obs_share(stack: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Share of the previous `window` months (t-window .. t-1) in which the market observed the item,
    within each release. NaN when no previous month exists in the release."""
    s = stack[["vintage", "mkt_uid", "item", "month", "obs"]].sort_values(["vintage", "mkt_uid", "item", "month"]).copy()
    g = s.groupby(["vintage", "mkt_uid", "item"], observed=True).obs
    prev = g.shift(1)
    s["obs_share"] = prev.astype(float).groupby([s.vintage, s.mkt_uid, s["item"]], observed=True).transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    return s


def group_quantiles(r, groups, levels, min_group: int = 30) -> dict:
    """Empirical quantiles of r per group (pooled fallback for groups smaller than min_group)."""
    r, groups = np.asarray(r, float), np.asarray(groups)
    pooled = {a: float(np.quantile(r, a)) for a in levels}
    out = {"__pooled__": pooled}
    for g in np.unique(groups):
        s = r[groups == g]
        out[g] = {a: float(np.quantile(s, a)) for a in levels} if len(s) >= min_group else pooled
    return out


def aci_next_alpha(alpha_t: float, target: float, err: float, gamma: float = 0.05) -> float:
    """Adaptive conformal inference update (Gibbs & Candes 2021): alpha_{t+1} = alpha_t + gamma (target - err)."""
    return float(np.clip(alpha_t + gamma * (target - err), 1e-6, 1 - 1e-6))


def aci_sequential(calib_scores, batches, target: float, gamma: float = 0.05):
    """Coverage per batch for a static threshold (calibration only) and for online ACI.

    Batches arrive in time order (e.g. one per release). ACI uses the calibration scores plus all scores
    revealed in earlier batches, at level alpha_t, then updates alpha_t from the batch's miscoverage.
    Returns (static_coverage, adaptive_coverage) arrays, one value per batch.
    """
    calib = np.asarray(calib_scores, float)
    static_th = conformal_quantile(calib, target)
    pool, alpha = list(calib), target
    st, ad = [], []
    for b in batches:
        b = np.asarray(b, float)
        st.append(np.mean(b <= static_th))
        th = conformal_quantile(np.asarray(pool), alpha)
        cov = np.mean(b <= th)
        ad.append(cov)
        alpha = aci_next_alpha(alpha, target, 1 - cov, gamma)
        pool.extend(b.tolist())
    return np.array(st), np.array(ad)
