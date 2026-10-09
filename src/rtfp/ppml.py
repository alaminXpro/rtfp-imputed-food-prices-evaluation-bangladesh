"""Poisson pseudo-maximum likelihood (PPML) with dummy fixed effects, and the switching-market design (D3).

Estimator (Santos Silva & Tenreyro 2006): GLM Poisson, log link, E[y | x, FE] = exp(x'b + FE). Consistent for b
whenever the conditional mean is log-linear, whatever the distribution of y (non-integer y and exact zeros are
fine). Fixed effects enter as dummies (statsmodels GLM). Before fitting:
  - rows in any fixed-effect group whose outcome sums to zero are dropped: such a group's effect diverges to
    -inf (separation) and the rows carry no information on b. Dropping zero-outcome rows cannot change another
    group's sum, so one pass over all FE dimensions suffices.
  - linearly dependent dummy columns (e.g. market x item and month x item dummies, which both sum to the item
    indicator) are removed by a column-pivoted QR; a regressor of interest in the span of the FE raises.
Inference: influence functions psi_i = H^-1 x_i (y_i - mu_i), H = sum_i mu_i x_i x_i' (the Poisson sandwich).
  one-way cluster:  V = c_G sum_g (sum_{i in g} psi_i)(...)'                      c_G = G / (G - 1)
  two-way (Cameron, Gelbach & Miller 2011): V = V_1 + V_2 - V_12, each with its own c_G, V_12 clustered on the
  intersection of the two dimensions. V_12 can make V non-PSD; that is reported, never hidden.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import chi2


@dataclass
class PPMLResult:
    coef: pd.Series
    n: int
    n_dropped: int
    converged: bool
    n_iter: int
    vcov: dict = field(default_factory=dict)  # label -> DataFrame
    se: dict = field(default_factory=dict)  # label -> Series (NaN where the variance is negative)
    psd: dict = field(default_factory=dict)  # label -> bool, all eigenvalues of the reported block >= 0
    influence: pd.DataFrame | None = None  # psi for the regressors + the cluster columns, one row per used obs


def drop_separated(df: pd.DataFrame, y: str, fe: list, weights: str | None = None) -> tuple[pd.DataFrame, int]:
    """Drop rows in fixed-effect groups (any dimension) whose (weighted) outcome sums to zero."""
    yy = df[y] * (df[weights] if weights else 1.0)
    bad = np.zeros(len(df), dtype=bool)
    for f in fe:
        bad |= (yy.groupby(df[f].to_numpy()).transform("sum") <= 0).to_numpy()
    return df[~bad], int(bad.sum())


def cluster_vcov(psi: np.ndarray, groups: list, correction: bool = True) -> np.ndarray:
    """Cluster-robust covariance from influence functions psi (n x k). One group array: one-way; two: CGM."""
    psi = np.asarray(psi, dtype=float)
    psi = psi[:, None] if psi.ndim == 1 else psi

    def one(g):
        codes, uniq = pd.factorize(pd.Series(np.asarray(g)))
        G = len(uniq)
        S = np.zeros((G, psi.shape[1]))
        np.add.at(S, codes, psi)
        return (G / (G - 1) if correction else 1.0) * S.T @ S

    if len(groups) == 1:
        return one(groups[0])
    if len(groups) == 2:
        g1, g2 = (pd.Series(np.asarray(g)).astype(str).to_numpy() for g in groups)
        return one(g1) + one(g2) - one(np.char.add(np.char.add(g1.astype(str), "\x1f"), g2.astype(str)))
    raise ValueError("one or two cluster dimensions")


def _design(d: pd.DataFrame, x: list, fe: list) -> tuple[np.ndarray, list]:
    """[regressors, constant, independent FE dummies]; raises if a regressor lies in the span of the FE."""
    blocks, names = [np.ones((len(d), 1))], ["const"]
    for f in fe:
        dm = pd.get_dummies(d[f].astype(str), prefix=f, drop_first=True, dtype=float)
        blocks.append(dm.to_numpy())
        names += list(dm.columns)
    F = np.hstack(blocks)
    norms = np.linalg.norm(F, axis=0)
    _, R, piv = qr(F / norms, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    keep = np.sort(piv[: int((diag > diag[0] * 1e-10).sum())])
    F, names = F[:, keep], [names[i] for i in keep]
    Xx = d[x].to_numpy(float)
    # residual norm of each regressor after projecting on the FE (and preceding regressors)
    Rx = qr(np.hstack([F / np.linalg.norm(F, axis=0), Xx / np.linalg.norm(Xx, axis=0)]), mode="r")[0]
    resid = np.abs(np.diag(Rx))[F.shape[1]:]
    absorbed = [v for v, r in zip(x, resid) if r < 1e-8]
    if absorbed:
        raise ValueError(f"regressor absorbed by fixed effects: {absorbed}")
    return np.hstack([Xx, F]), list(x) + names


def ppml(df: pd.DataFrame, y: str, x: list, fe: list, clusters: dict | None = None, weights: str | None = None,
         correction: bool = True, maxiter: int = 200, tol: float = 1e-10) -> PPMLResult:
    """PPML of y on x with dummy fixed effects `fe`.

    clusters: {label: [col]} (one-way) or {label: [col1, col2]} (two-way CGM). weights: integer frequency
    weights (bootstrap); variances are only computed without weights.
    """
    import statsmodels.api as sm

    clusters = clusters or {}
    if weights and clusters:
        raise ValueError("cluster variances are not computed with frequency weights")
    ccols = sorted({c for cols in clusters.values() for c in cols})
    cols = list(dict.fromkeys([y] + list(x) + list(fe) + ccols + ([weights] if weights else [])))
    d = df[cols].dropna()
    if weights:
        d = d[d[weights] > 0]
    if (d[y] < 0).any():
        raise ValueError("PPML needs a non-negative outcome")
    d, n_dropped = drop_separated(d, y, fe, weights)
    X, names = _design(d, x, fe)
    yv = d[y].to_numpy(float)
    w = d[weights].to_numpy(float) if weights else None
    fit = sm.GLM(yv, X, family=sm.families.Poisson(), freq_weights=w).fit(maxiter=maxiter, tol=tol)
    params = pd.Series(fit.params, index=names)
    out = PPMLResult(coef=params[list(x)], n=int(len(d) if w is None else w.sum()), n_dropped=n_dropped,
                     converged=bool(fit.converged), n_iter=int(fit.fit_history["iteration"]))
    if not out.converged:
        raise RuntimeError("PPML did not converge")
    if clusters:
        mu = fit.mu
        H = (X * mu[:, None]).T @ X
        idx = [names.index(v) for v in x]
        E = np.zeros((len(names), len(idx)))
        E[idx, range(len(idx))] = 1.0
        B = np.linalg.solve(H, E)  # columns of H^-1 for the regressors (H symmetric)
        psi = (X * (yv - mu)[:, None]) @ B
        out.influence = pd.DataFrame(psi, columns=list(x), index=d.index).assign(**{c: d[c].to_numpy() for c in ccols})
        for label, cc in clusters.items():
            V = cluster_vcov(psi, [d[c].to_numpy() for c in cc], correction=correction)
            out.vcov[label] = pd.DataFrame(V, index=list(x), columns=list(x))
            dg = np.diag(V)
            out.se[label] = pd.Series(np.where(dg >= 0, np.sqrt(np.abs(dg)), np.nan), index=list(x))
            out.psd[label] = bool(np.linalg.eigvalsh((V + V.T) / 2).min() >= -1e-14 * max(1.0, np.abs(V).max()))
    return out


def lincom(results: list, weights: list, coef: str, cluster_cols: list, correction: bool = True) -> tuple:
    """Estimate and cluster-robust SE of sum_k w_k b_k(coef) across separately fitted models (e.g. one per item).

    Models share no parameters, so the joint influence of the combination is w_k psi_k on model k's rows;
    clustering then pools rows of all models (a market's rows across items form one cluster)."""
    est = float(sum(wk * r.coef[coef] for r, wk in zip(results, weights)))
    inf = pd.concat([r.influence[[coef] + list(cluster_cols)].assign(**{coef: wk * r.influence[coef]})
                     for r, wk in zip(results, weights)], ignore_index=True)
    V = cluster_vcov(inf[coef].to_numpy(), [inf[c].to_numpy() for c in cluster_cols], correction=correction)
    v = float(V[0, 0])
    return est, (np.sqrt(v) if v >= 0 else np.nan)


def wald(beta: np.ndarray, V: np.ndarray) -> tuple:
    """Wald test of beta = 0: statistic, degrees of freedom, chi-square p-value."""
    beta = np.asarray(beta, float)
    stat = float(beta @ np.linalg.solve(np.asarray(V, float), beta))
    return stat, len(beta), float(chi2.sf(stat, len(beta)))


def noise_benchmark(rel: float, power: int = 1) -> float:
    """PPML coefficient on 'imputed' for |e|^power if the imputer only removed measurement noise.

    Observed local change = e + noise with rel = Var(e) / Var(observed); imputed = e. For power = 2 the ratio
    of means is rel exactly (zero-mean components, independent noise); for power = 1 the ratio is sqrt(rel)
    when e and the noise are Gaussian (E|x| = sqrt(2/pi) sd). Benchmark = (power / 2) * log(rel)."""
    if not (rel == rel) or not (0 < rel <= 1):
        return np.nan
    return 0.5 * power * float(np.log(rel))


# --- switching-market design (methodology D3) --------------------------------------------------------------------

CELL_TYPES = {"OO": "observed t and t-1", "II": "imputed t and t-1", "OI": "imputed t, observed t-1",
              "IO": "observed t, imputed t-1", "start": "no previous month in the panel"}


def _month_index(month: pd.Series) -> pd.Series:
    return month.dt.year * 12 + month.dt.month


def classify_cells(panel: pd.DataFrame, start: str = "2020-01-01", mixed: bool = False, obs_side: str = "y",
                   cell_ok: str | None = None) -> pd.DataFrame:
    """Classify every market x month x item cell from `start` on and say whether it enters the D3 sample.

    Needs columns mkt_name, mkt_uid, item, month, obs, e_obs, e_pub (latest-release panel, rtfp.panel).
    cell_type uses the observation status of t and t-1 (t-1 may precede `start`).
    Outcome e: OO -> e_obs (survey change minus observed-cell month mean; obs_side='c' uses e_pub instead);
               II -> e_pub (published change minus the same month mean); OI -> e_pub only if mixed=True.
    reason (first that applies): division_market > mixed_change > excluded_by_rule (cell_ok column False)
               > no_outcome (e undefined, e.g. no observed change in the month) > not_switching (the market-item
               has no entering observed or no entering imputed cell) > enters.
    """
    p = panel.copy()
    p["mi"] = _month_index(p.month)
    prev = p[["mkt_uid", "item", "mi", "obs"]].assign(mi=lambda d: d.mi + 1).rename(columns={"obs": "obs_prev"})
    p = p.merge(prev, on=["mkt_uid", "item", "mi"], how="left")
    p = p[p.month >= pd.Timestamp(start)].copy()
    o, op = p.obs.astype(bool), p.obs_prev
    p["cell_type"] = np.select([o & (op == True), ~o & (op == False), ~o & (op == True), o & (op == False)],  # noqa: E712
                               ["OO", "II", "OI", "IO"], "start")
    obs_val = p.e_obs if obs_side == "y" else p.e_pub
    if obs_side not in ("y", "c"):
        raise ValueError(obs_side)
    allowed = ["OO", "II"] + (["OI"] if mixed else [])
    p["e"] = np.where(p.cell_type == "OO", obs_val, np.where(p.cell_type.isin(["II", "OI"]), p.e_pub, np.nan))
    p["abs_e"] = p.e.abs()
    p["imputed"] = (~o).astype(float)
    reason = pd.Series("enters", index=p.index, dtype=object)
    division = p.mkt_name.str.endswith(" Division")
    ok = p[cell_ok].astype(bool) if cell_ok else pd.Series(True, index=p.index)
    stages = [(division, "division_market"), (~p.cell_type.isin(allowed), "mixed_change"),
              (~ok, "excluded_by_rule"), (p.e.isna(), "no_outcome")]
    for mask, why in stages:
        reason[(reason == "enters") & mask] = why
    cand = reason == "enters"
    has = p[cand].groupby(["mkt_uid", "item"]).imputed.agg(["min", "max"])
    switching = has[(has["min"] == 0) & (has["max"] == 1)].index
    is_sw = pd.MultiIndex.from_frame(p[["mkt_uid", "item"]]).isin(switching)
    reason[cand & ~is_sw] = "not_switching"
    p["reason"] = reason
    p["enters"] = p.reason == "enters"
    return p.drop(columns=["mi", "obs_prev"]).reset_index(drop=True)


def switcher_cells(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    return classify_cells(panel, **kw).query("enters").reset_index(drop=True)


def event_windows(status: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Cells within `window` months of each observed<->imputed spell boundary of a market x item.

    A spell is a run of consecutive months with the same status (a missing month ends a spell; no event
    bridges it). For a boundary at month s (first month of the new spell): pre cells are the last <= window
    months of the old spell (k = -window..-1), post cells the first <= window + 1 months of the new spell
    (k = 0..window). balanced = the old spell has >= window months and the new spell >= window + 1.
    """
    cols = ["event_id", "direction", "mkt_uid", "item", "month", "k", "switch_month", "old_len", "new_len", "balanced"]
    s = status[["mkt_uid", "item", "month", "obs"]].sort_values(["mkt_uid", "item", "month"]).reset_index(drop=True)
    s["mi"] = _month_index(s.month)
    g = s.groupby(["mkt_uid", "item"], sort=False)
    new_spell = (s.obs != g.obs.shift()) | (s.mi - g.mi.shift() != 1)
    s["spell"] = new_spell.cumsum()
    sp = s.groupby("spell").agg(mkt_uid=("mkt_uid", "first"), item=("item", "first"), obs=("obs", "first"),
                                start=("mi", "min"), end=("mi", "max"), n=("mi", "size")).reset_index()
    nxt = sp.shift(-1)
    ev = sp[(nxt.mkt_uid == sp.mkt_uid) & (nxt.item == sp.item) & (nxt.start == sp.end + 1)]
    out = []
    for i, old in ev.iterrows():
        new = sp.loc[i + 1]
        s0 = int(new.start)
        cells = s[(s.spell == old.spell) & (s.mi >= s0 - window) | (s.spell == new.spell) & (s.mi <= s0 + window)]
        out.append(cells[["mkt_uid", "item", "month"]].assign(
            k=(cells.mi - s0).to_numpy(), direction="obs_to_imp" if old.obs else "imp_to_obs",
            switch_month=cells.month[cells.mi == s0].iloc[0], old_len=int(old.n), new_len=int(new.n),
            balanced=bool(old.n >= window and new.n >= window + 1)))
    if not out:
        return pd.DataFrame(columns=cols)
    e = pd.concat(out, ignore_index=True)
    e["event_id"] = e.mkt_uid.astype(str) + "|" + e["item"].astype(str) + "|" + e.switch_month.dt.strftime("%Y-%m")
    return e[cols]


def cluster_mean(values, groups) -> tuple:
    """Mean and its CR1 cluster-robust standard error."""
    x = np.asarray(values, float)
    m = x.mean()
    V = cluster_vcov((x - m) / len(x), [np.asarray(groups)])
    return float(m), float(np.sqrt(V[0, 0]))
