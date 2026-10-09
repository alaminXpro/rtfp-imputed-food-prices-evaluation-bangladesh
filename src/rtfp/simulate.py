"""Framework simulation (methodology M2 'Simulation'; Task 3.1) and the alert benchmark for Task 4.1.

Data-generating process (log prices, market m, month t, region g(m)):
    x_mt = mean0_t + c_t + r_{g(m),t} + l_mt + a_m
      mean0_t = log_base + drift * t + season[calendar month]      deterministic, known to the oracle imputer
      c_t     common AR(1):  c_t = phi_common c_{t-1} + u_t,      sd(u) = sd_common
      r, l    local AR(1)s with the same persistence phi_local: the regional factor r (shared by markets in a
              region; carries a share `regional_share` of the local innovation variance) and the idiosyncratic l.
              Innovation variance s_u^2 = var_dlocal (1 + phi_local) / 2, so Var(d(r + l)) = var_dlocal.
      a_m     market fixed offset, sd_market.
    All AR(1)s start from their stationary distribution.
Reported (survey) value y, noise variants:
    white    y = x + nu, nu ~ N(0, var_noise) iid
    stale    with probability p_stale the market repeats its previous report (carry-forward), else y = x + nu
    average  y_t = (x_t + x_{t-1}) / 2 + nu_t   (monthly averaging of the latent path; first month y_0 = x_0 + nu_0)
Missingness: cell observed in the latest data with market propensity pi_m (iid over months); an observed cell was
already in the imputing release with probability p_release. Hold-out = observed in the latest data but not in the
release (later arrival), as in methodology M1. Pairs (t-1, t) with both months in the hold-out are 'both_imputed';
'cross_sectional' if some market was observed in month t in the release.

Imputers (published value = survey value y at release-observed cells, as RTFP does for ~90% of them):
    cond_mean       Gaussian conditional mean E[x | y_release] under the true white-noise model and parameters
                    (best linear predictor from all markets' observed values, the common and regional factors).
                    With one information set for both months, the imputed change is E[dx | I], so the calibration
                    slope of the true change on it is exactly 1 (beta_mz = 1 in expectation).
    idw_seasonal    data-only fill: two-way additive fit (market + month effects) on release-observed values, plus
                    inverse-squared-distance weighted residuals of markets observed that month (w = 1/(d_km^2 + 1));
                    months without any observation: month effect carried forward with the mean calendar-month change
                    (seasonal fill); markets never observed: IDW of other markets' effects.
    chain_mean      cond_mean + mean of n_chains independent posterior-marginal draws per cell (residual chain noise
                    with variance posterior_var / n_chains), drawn at the cells used for evaluation.
    outlier_smooth  cond_mean posterior path smoothed over time with weights (1/4, 1/2, 1/4) (end points renormalised):
                    temporal smoothing of the kind used to suppress outliers; it shrinks month-to-month changes.

Metrics per panel (`panel_metrics`): rtfp.metrics.local_components (own_mu, within_month) and local_stats on the
cross-sectional both-imputed pairs; corr_true uses the noise-free truth (dx, demeaned the same way); rel_true is
Var(e_true) / Var(e_obs) on those pairs; noise estimators rtfp.noise on the latest-observed local changes.

Calibration helpers (`calibrate_common`, `calibrate_local`) turn the empirical panel and Table 1 into a SimSpec.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, solve_triangular  # not scipy's cholesky: it segfaults on this setup (scipy 1.18.1, Accelerate, macOS 27)

from rtfp.alerts import alps_realtime, ipa_realtime
from rtfp.metrics import local_components, local_stats
from rtfp.noise import autocovariances, noise_from_autocov

IMPUTERS = ("cond_mean", "idw_seasonal", "chain_mean", "outlier_smooth")
NOISE_VARIANTS = ("white", "stale", "average")


@dataclass(frozen=True)
class SimSpec:
    n_months: int = 48
    start: str = "2020-01-01"
    propensity: tuple = (0.5,) * 30
    region: tuple = None
    coords_km: tuple = None
    n_regions: int = 5
    log_base: float = 4.0
    drift: float = 0.0
    season: tuple = (0.0,) * 12
    phi_common: float = 0.9
    sd_common: float = 0.02
    phi_local: float = 0.7
    var_dlocal: float = 0.001
    regional_share: float = 0.3
    sd_market: float = 0.05
    var_noise: float = 0.0005
    noise: str = "white"
    p_stale: float = 0.0
    p_release: float = 0.8

    @property
    def n_markets(self) -> int:
        return len(self.propensity)


@dataclass
class Panel:
    spec: SimSpec
    months: pd.DatetimeIndex
    region: np.ndarray
    coords: np.ndarray
    mean0: np.ndarray
    common: np.ndarray
    market: np.ndarray
    local: np.ndarray
    x: np.ndarray
    y: np.ndarray
    obs_latest: np.ndarray
    obs_release: np.ndarray
    cache: dict = field(default_factory=dict)


def _ar1(rng, n_series: int, T: int, phi: float, s2: float) -> np.ndarray:
    out = np.zeros((n_series, T))
    if s2 <= 0:
        return out
    s = np.sqrt(s2)
    out[:, 0] = rng.normal(0, s / np.sqrt(1 - phi ** 2), n_series)
    for t in range(1, T):
        out[:, t] = phi * out[:, t - 1] + rng.normal(0, s, n_series)
    return out


def _local_innov(spec: SimSpec):
    su2 = spec.var_dlocal * (1 + spec.phi_local) / 2
    return spec.regional_share * su2, (1 - spec.regional_share) * su2


def simulate_panel(spec: SimSpec, rng) -> Panel:
    if not 0 <= spec.phi_local < 1 or not 0 <= spec.phi_common < 1:
        raise ValueError("AR(1) persistence must be in [0, 1)")
    N, T = spec.n_markets, spec.n_months
    months = pd.date_range(spec.start, periods=T, freq="MS")
    region = np.asarray(spec.region if spec.region is not None else np.arange(N) % spec.n_regions)
    G = int(region.max()) + 1
    if spec.coords_km is not None:
        coords = np.asarray(spec.coords_km, float)
    else:
        ang = 2 * np.pi * region / G
        coords = np.column_stack([100 * np.cos(ang), 100 * np.sin(ang)]) + rng.normal(0, 15, (N, 2))
    mean0 = spec.log_base + spec.drift * np.arange(T) + np.asarray(spec.season, float)[months.month - 1]
    common = _ar1(rng, 1, T, spec.phi_common, spec.sd_common ** 2)[0]
    s2r, s2l = _local_innov(spec)
    reg = _ar1(rng, G, T, spec.phi_local, s2r)
    idio = _ar1(rng, N, T, spec.phi_local, s2l)
    local = reg[region] + idio
    market = rng.normal(0, spec.sd_market, N)
    x = mean0[None, :] + common[None, :] + local + market[:, None]
    nu = rng.normal(0, np.sqrt(spec.var_noise), (N, T)) if spec.var_noise > 0 else np.zeros((N, T))
    if spec.noise == "white":
        y = x + nu
    elif spec.noise == "stale":
        y = x + nu
        stale = rng.random((N, T)) < spec.p_stale
        for t in range(1, T):
            y[stale[:, t], t] = y[stale[:, t], t - 1]
    elif spec.noise == "average":
        y = np.column_stack([x[:, :1], (x[:, 1:] + x[:, :-1]) / 2]) + nu
    else:
        raise ValueError(spec.noise)
    pi = np.asarray(spec.propensity, float)
    obs_latest = rng.random((N, T)) < pi[:, None]
    obs_release = obs_latest & (rng.random((N, T)) < spec.p_release)
    return Panel(spec, months, region, coords, mean0, common, market, local, x, y, obs_latest, obs_release)


# ---------------------------------------------------------------- Gaussian conditioning (calibrated conditional mean)

def _lag_covs(spec: SimSpec, T: int):
    k = np.arange(T)
    s2r, s2l = _local_innov(spec)
    kc = spec.sd_common ** 2 / (1 - spec.phi_common ** 2) * spec.phi_common ** k
    kr = s2r / (1 - spec.phi_local ** 2) * spec.phi_local ** k
    kl = s2l / (1 - spec.phi_local ** 2) * spec.phi_local ** k
    return kc, kr, kl, spec.sd_market ** 2


def _cov(p: Panel, ma, ta, mb, tb) -> np.ndarray:
    kc, kr, kl, sa2 = _lag_covs(p.spec, p.x.shape[1])
    lag = np.abs(ta[:, None] - tb[None, :])
    same_m = ma[:, None] == mb[None, :]
    return (kc[lag] + (p.region[ma][:, None] == p.region[mb][None, :]) * kr[lag] + same_m * (kl[lag] + sa2))


def _factor(p: Panel):
    if "chol" not in p.cache:
        mo, to = np.nonzero(p.obs_release)
        K = _cov(p, mo, to, mo, to) + (p.spec.var_noise + 1e-12) * np.eye(len(mo))
        L = np.linalg.cholesky(K)  # lower-triangular, same as scipy's cholesky(K, lower=True)
        alpha = cho_solve((L, True), p.y[mo, to] - p.mean0[to])
        p.cache.update(chol=L, alpha=alpha, mo=mo, to=to)
    return p.cache


def conditional_mean(p: Panel) -> np.ndarray:
    """Posterior mean of the latent log price x at every cell given release-observed y (white-noise model)."""
    if "cmean" in p.cache:
        return p.cache["cmean"].copy()
    f = _factor(p)
    N, T = p.x.shape
    G = int(p.region.max()) + 1
    kc, kr, kl, sa2 = _lag_covs(p.spec, T)
    lag = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    A = np.bincount(f["to"], f["alpha"], minlength=T)
    R = np.bincount(p.region[f["mo"]] * T + f["to"], f["alpha"], minlength=G * T).reshape(G, T)
    M = np.zeros((N, T))
    M[f["mo"], f["to"]] = f["alpha"]
    out = (p.mean0[None, :] + (kc[lag] @ A)[None, :] + (R @ kr[lag])[p.region] + M @ (kl[lag] + sa2))
    p.cache["cmean"] = out
    return out.copy()


def posterior_var(p: Panel, mask: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """Posterior variance of x at the cells in `mask` (NaN elsewhere)."""
    f = _factor(p)
    kc, kr, kl, sa2 = _lag_covs(p.spec, p.x.shape[1])
    out = np.full(p.x.shape, np.nan)
    ms, ts = np.nonzero(mask)
    prior = kc[0] + kr[0] + kl[0] + sa2
    for i in range(0, len(ms), chunk):
        mm, tt = ms[i:i + chunk], ts[i:i + chunk]
        v = solve_triangular(f["chol"], _cov(p, f["mo"], f["to"], mm, tt), lower=True)
        out[mm, tt] = prior - (v ** 2).sum(axis=0)
    return out


# ---------------------------------------------------------------- imputers

def _publish(p: Panel, c: np.ndarray) -> np.ndarray:
    c = c.copy()
    c[p.obs_release] = p.y[p.obs_release]
    return c


def _eval_cells(p: Panel) -> np.ndarray:
    h = p.obs_latest & ~p.obs_release
    prev = np.zeros_like(h)
    prev[:, :-1] = h[:, 1:]
    return (h | prev) & ~p.obs_release


def _idw_weights(coords: np.ndarray) -> np.ndarray:
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    w = 1.0 / (d2 + 1.0)
    np.fill_diagonal(w, 0.0)
    return w


def _idw_seasonal(p: Panel, max_iter: int = 2000, tol: float = 1e-12) -> np.ndarray:
    obs = np.where(p.obs_release, p.y, np.nan)
    N, T = obs.shape
    a = np.zeros(N)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for _ in range(max_iter):
                b = np.nanmean(obs - a[:, None], axis=0)
                a_new = np.nanmean(obs - b[None, :], axis=1)
                done = np.nanmax(np.abs(np.nan_to_num(a_new - a))) < tol
                a = a_new
                if done:
                    break
            b = np.nanmean(obs - a[:, None], axis=0)
    w = _idw_weights(p.coords)
    have_a = np.isfinite(a)
    if not have_a.all():
        ww = w[:, have_a]
        a = np.where(have_a, a, (ww @ a[have_a]) / ww.sum(axis=1))
    have_b = np.isfinite(b)
    if not have_b.all():
        cal = p.months.month.to_numpy() - 1
        db = np.full(12, 0.0)
        steps = np.diff(b)
        ok = np.isfinite(steps)
        for c in range(12):
            sel = ok & (cal[1:] == c)
            if sel.any():
                db[c] = steps[sel].mean()
        for t in range(1, T):
            if not np.isfinite(b[t]) and np.isfinite(b[t - 1]):
                b[t] = b[t - 1] + db[cal[t]]
        for t in range(T - 2, -1, -1):
            if not np.isfinite(b[t]) and np.isfinite(b[t + 1]):
                b[t] = b[t + 1] - db[cal[t + 1]]
    r = np.where(p.obs_release, p.y - a[:, None] - b[None, :], 0.0)
    num = w @ r
    den = w @ p.obs_release.astype(float)
    rhat = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return a[:, None] + b[None, :] + rhat


def impute(p: Panel, method: str, rng=None, n_chains: int = 5, var_mask=None) -> np.ndarray:
    """Published log price for every cell under imputer `method` (one of IMPUTERS)."""
    if method == "cond_mean":
        return _publish(p, conditional_mean(p))
    if method == "idw_seasonal":
        return _publish(p, _idw_seasonal(p))
    if method == "chain_mean":
        rng = np.random.default_rng() if rng is None else rng
        mask = _eval_cells(p) if var_mask is None else var_mask & ~p.obs_release
        c = conditional_mean(p)
        sd = np.sqrt(np.clip(posterior_var(p, mask)[mask], 0, None))
        c[mask] = c[mask] + sd * rng.standard_normal((n_chains, mask.sum())).mean(axis=0)
        return _publish(p, c)
    if method == "outlier_smooth":
        c = conditional_mean(p)
        s = c.copy()
        s[:, 1:-1] = 0.25 * c[:, :-2] + 0.5 * c[:, 1:-1] + 0.25 * c[:, 2:]
        s[:, 0] = (0.5 * c[:, 0] + 0.25 * c[:, 1]) / 0.75
        s[:, -1] = (0.5 * c[:, -1] + 0.25 * c[:, -2]) / 0.75
        return _publish(p, s)
    raise ValueError(method)


# ---------------------------------------------------------------- hold-out pairs and metrics

def _month_means(d: np.ndarray, sel: np.ndarray) -> np.ndarray:
    n = sel.sum(axis=0)
    s = np.where(sel, d, 0.0).sum(axis=0)
    return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)


def holdout_pairs(p: Panel, chat: np.ndarray) -> pd.DataFrame:
    """Both-imputed hold-out change pairs (t-1, t) in the format of data/interim/change_pairs.parquet."""
    N, T = p.x.shape
    h = p.obs_latest & ~p.obs_release
    dy, dx, dc = np.diff(p.y, axis=1), np.diff(p.x, axis=1), np.diff(chat, axis=1)
    rel_pair = p.obs_release[:, 1:] & p.obs_release[:, :-1]
    lat_pair = p.obs_latest[:, 1:] & p.obs_latest[:, :-1]
    mu_v, mu_l, mu_lx = _month_means(dy, rel_pair), _month_means(dy, lat_pair), _month_means(dx, lat_pair)
    mu_pub = dc.mean(axis=0)
    n_rel = p.obs_release.sum(axis=0)[1:]
    pair = h[:, 1:] & h[:, :-1]
    m, j = np.nonzero(pair)
    return pd.DataFrame({
        "item": "sim", "mkt_uid": np.char.add("m", m.astype(str)), "t": j + 1, "month": p.months[j + 1],
        "stratum": np.where(n_rel[j] > 0, "cross_sectional", "forecast"), "pair_type": "both_imputed",
        "dlog_pub": dc[m, j], "dlog_truth": dy[m, j], "dlog_true": dx[m, j],
        "mu_obs_vfinal": mu_v[j], "mu_pub_vfinal": mu_pub[j], "mu_obs_latest": mu_l[j], "mu_true_latest": mu_lx[j],
    })


def noise_sample(p: Panel) -> pd.DataFrame:
    """Latest-observed local changes e_obs = dy - (mean observed dy that month), as in Table 1's noise sample."""
    dy = np.diff(p.y, axis=1)
    sel = p.obs_latest[:, 1:] & p.obs_latest[:, :-1]
    mu = _month_means(dy, sel)
    m, j = np.nonzero(sel)
    return pd.DataFrame({"mkt_uid": np.char.add("m", m.astype(str)), "item": "sim", "month": p.months[j + 1],
                         "e_obs": dy[m, j] - mu[j]})


def panel_metrics(p: Panel, chat: np.ndarray, stratum: str = "cross_sectional", noise: bool = True) -> dict:
    pr = holdout_pairs(p, chat)
    pr = pr[pr.stratum == stratum]
    out = {}
    if noise:
        est = noise_from_autocov(autocovariances(noise_sample(p)).loc["sim"])
    for mode in ("own_mu", "within_month"):
        e = local_components(pr, mode=mode)
        et = local_components(pr.assign(dlog_truth=pr.dlog_true, mu_obs_latest=pr.mu_true_latest), mode=mode)
        idx = e.index.intersection(et.index)
        e, et = e.loc[idx], et.loc[idx]
        if len(e) < 3:
            out[mode] = {"n": len(e)}
            continue
        s = local_stats(e.e_pub, e.e_obs)
        s["corr_true"] = float(np.corrcoef(e.e_pub, et.e_obs)[0, 1])
        s["rel_true"] = float(np.var(et.e_obs) / np.var(e.e_obs))
        if noise:
            s.update({k: est[k] for k in ("g0", "var_nu_naive", "rel_naive", "phi", "var_nu_ar1", "rel_ar1",
                                          "ar1_identified")})
        out[mode] = s
    return out


# ---------------------------------------------------------------- alert benchmark (Task 4.1)

def alert_benchmark(spec: SimSpec, n_eval: int, rng, price_scale: float = 1.0) -> pd.DataFrame:
    """ALPS / IPA z-scores in the last `n_eval` months for three month-t values appended to the same history.

    history (months < t): the published-like series exp(y) where observed in the latest data, exp(x) elsewhere.
    candidates at t: survey exp(y_t) (the noisy reference); oracle exp(x_t) (noise-removing, signal-preserving,
    hence calibrated); common-only exp(mean0_t + c_t + a_m) (calibrated, no local information).
    """
    p = simulate_panel(spec, rng)
    N, T = p.x.shape
    hist = price_scale * np.exp(np.where(p.obs_latest, p.y, p.x))
    cand = price_scale * np.exp(np.stack([p.y, p.x, p.mean0[None, :] + p.common[None, :] + p.market[:, None]
                                          + 0 * p.x]))
    rows = []
    for m in range(N):
        for t in range(T - n_eval, T):
            months = p.months[:t + 1]
            c = cand[:, m, t]
            z = {"alps": alps_realtime(hist[m, :t], months, c),
                 "ipa_baquedano": ipa_realtime(hist[m, :t], months, c, variant="baquedano"),
                 "ipa_sdg": ipa_realtime(hist[m, :t], months, c, variant="sdg")}
            for ind, v in z.items():
                rows.append((f"m{m}", p.months[t], t, ind, v[0], v[1], v[2]))
    return pd.DataFrame(rows, columns=["mkt_uid", "month", "t", "indicator", "z_survey", "z_oracle", "z_common"])


# ---------------------------------------------------------------- calibration from the empirical panel

def calibrate_common(log_index: pd.Series) -> dict:
    """Trend + month dummies (no constant) on a monthly national log-price index; AR(1) on the residuals."""
    m = pd.DatetimeIndex(log_index.index)
    L = log_index.to_numpy(float)
    tt = np.arange(len(L), dtype=float)
    X = np.column_stack([tt, (m.month.to_numpy()[:, None] == np.arange(1, 13)[None, :]).astype(float)])
    beta, *_ = np.linalg.lstsq(X, L, rcond=None)
    u = L - X @ beta
    phi = float((u[1:] * u[:-1]).sum() / (u[:-1] ** 2).sum())
    sd = float(np.std(u[1:] - phi * u[:-1], ddof=1))
    season = beta[1:] - beta[1:].mean()
    return {"log_base": float(beta[1:].mean()), "drift": float(beta[0]), "season": tuple(float(s) for s in season),
            "phi_common": phi, "sd_common": sd, "start": str(m[0].date()), "n_months": len(L)}


def two_way_residuals(df: pd.DataFrame, value: str, max_iter: int = 1000, tol: float = 1e-12) -> pd.Series:
    """Residuals of value on market + month fixed effects (alternating projections)."""
    r = df[value] - df[value].mean()
    for _ in range(max_iter):
        r0 = r
        r = r - r.groupby(df.mkt_uid).transform("mean")
        r = r - r.groupby(df.month).transform("mean")
        if (r - r0).abs().max() < tol:
            break
    return r


def calibrate_local(obs: pd.DataFrame, var_nu: float, g0: float, n_iter: int = 50) -> dict:
    """Local-component moments from observed cells (columns mkt_uid, region, month, log_y, e_obs).

    Empirical local changes are demeaned by the month mean of the n_t observed markets, which removes part of the
    local variance: a share s in a regional factor loses sum_g (n_gt / n_t)^2 of its variance (H_t), the
    idiosyncratic part loses 1 / n_t. With kappa(s) = mean_t over changes of [s (1 - H_t) + (1 - s)(1 - 1/n_t)] and
    omega = mean of (1 - 1/n_t):
        g0 = var_dlocal * kappa(s) + 2 var_nu * omega          (Table 1: g0 = Var(e_obs), naive Var(nu))
        s  = [mean e_i e_j over same-region pairs in a month - same over other-region pairs] / var_dlocal
             (the other-region term removes the demeaning bias of cross products)
    solved jointly by fixed-point iteration. Levels: Var(two-way FE residual of log y), degrees-of-freedom corrected,
    = v_local * kappa(s) + var_nu * omega, and phi_local = 1 - var_dlocal / (2 v_local) (AR(1): Var(dl) = 2V(1-phi)).
    sd_market = SD of the market fixed effects.
    """
    lev = obs.dropna(subset=["log_y"])
    r = two_way_residuals(lev, "log_y")
    n, nm, nt = len(lev), lev.mkt_uid.nunique(), lev.month.nunique()
    v_res = float(r.var() * n / (n - nm - nt))
    e = obs.dropna(subset=["e_obs"])[["mkt_uid", "region", "month", "e_obs"]]
    n_t = e.groupby("month").mkt_uid.transform("size").to_numpy(float)
    h_t = (e.groupby(["month", "region"]).mkt_uid.transform("size") / e.groupby("month").mkt_uid.transform("size"))
    h_t = (h_t ** 2).groupby([e.month, e.region]).first().groupby(level=0).sum()
    h_t = e.month.map(h_t).to_numpy(float)
    omega = float(np.mean(1 - 1 / n_t))
    j = e.merge(e, on="month")
    j = j[j.mkt_uid_x < j.mkt_uid_y]
    same = j.region_x == j.region_y
    prod = j.e_obs_x * j.e_obs_y
    num = float(prod[same].mean() - prod[~same].mean())
    s = 0.0
    for _ in range(n_iter):
        kappa = float(np.mean(s * (1 - h_t) + (1 - s) * (1 - 1 / n_t)))
        var_dlocal = (g0 - 2 * var_nu * omega) / kappa
        s = float(np.clip(num / var_dlocal, 0.0, 1.0))
    kappa = float(np.mean(s * (1 - h_t) + (1 - s) * (1 - 1 / n_t)))
    var_dlocal = (g0 - 2 * var_nu * omega) / kappa
    v_local = (v_res - var_nu * omega) / kappa
    phi = 1 - var_dlocal / (2 * v_local) if v_local > 0 else np.nan
    fe = (lev.log_y - r).groupby(lev.mkt_uid).mean()
    return {"var_dlocal": float(var_dlocal), "v_local": float(v_local), "v_res": v_res, "phi_local": float(phi),
            "regional_share": s, "regional_share_raw": num / (g0 - 2 * var_nu), "kappa": kappa, "omega": omega,
            "sd_market": float((fe - fe.mean()).std()), "n_level_obs": int(n), "n_same_region_pairs": int(same.sum())}


def spec_with(spec: SimSpec, **kw) -> SimSpec:
    return replace(spec, **kw)
