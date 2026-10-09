"""Price-alert indicators: WFP ALPS and FAO GIEWS IPA (methodology D4; Task 4.1).

Sources (downloaded 2026-10-09; sha256 of the PDFs read):
  [WFP14] WFP (2014). Technical Guidance Note: Calculation and Use of the Alert for Price Spikes (ALPS) Indicator.
          April 2014. https://documents.wfp.org/stellent/groups/public/documents/manual_guide_proced/wfp264186.pdf
          (linked from https://reliefweb.int/report/world/technical-guidance-note-calculation-and-use-alert-price-spikes-alps-indicator-april)
          sha256 3c6a4cb48cf4564c018121e8625924072fbaea8fa6b3c27696dc8d783ebd7478
  [B15]   Baquedano, F.G. (2015). Developing an indicator of price anomalies as an early warning tool: A compound
          growth approach. FAO GIEWS, Rome. https://www.fao.org/fileadmin/user_upload/foodprice/docs/resources/a-i7550e.pdf
          sha256 7aa313905eba78addd6f024abada3bd301816168a7b144661cb5b07458b15590
  [SDG24] UN SDG indicator metadata 2.c.1, Indicator of Food Price Anomalies (IFPA), FAO, last updated 2024-07-29.
          https://unstats.un.org/sdgs/metadata/files/Metadata-02-0C-01.pdf
          sha256 369c8fcd9f967752a3876cc0ee70857ba404bde72369c7c0bbab6027f194e7cb

ALPS [WFP14, section 3.1 and the Stata appendix]
  Step 1 (eq. 1): "Price_t = alpha * T_t + sum_{m=1..12} beta_m * D_mt + eps_t", OLS, "The constant term is excluded
          from this regression"; Stata: `regress prices t d1-d12, noconstant` with t = ym(year, month).
  Step 2 (eq. 3): "eps_hat_t = Price_t - Price_hat_t = Difference between market price and estimated price, residuals"
  Step 3 (eq. 4): "ALPS = (Price_it - Price_hat_it) / sigma_eps_hat"; Stata: `egen p_sd = sd(p_res)`,
          `generate p_indicator = p_res / p_sd` (egen sd uses the n - 1 denominator).
  Step 4 (Table 1): "Normal ALPS < 0.25; Stress 0.25 <= ALPS < 1; Alert 1 <= ALPS < 2; Crisis ALPS >= 2".
  Section 3.2.2: "the indicator is calculated on monthly prices when the series comprises at least 3 years, i.e.,
          ideally it would include 36 observations."
  Section 3.2.3: "The seasonal price trend is calculated using the available entire price series. Price trends can
          be updated every month, as new market prices become available."  `alps_realtime` therefore refits the
          trend on the history plus the month being classified.
  Prices enter in levels (Stata regresses `prices`), not logs, and are not deflated.

IPA [B15, eqs. 1-7 and Annex 1; SDG24 section 4]
  eq. 1: "CGR_t = (P_tn / P_t0)^(1/(tn - t0)) - 1", with t0 = t - 3 (quarterly, CQGR) or t - 12 (annual, CAGR):
          "calculating the CQGR and CAGR as a moving average over the immediately preceding 3-month or 12-month
          period of month t".
  eq. 3: "CXGR_t x (1 - sigma[P_t0 - P_tn]) = vCXGR_t"; "We define volatility as the standard deviation of log
          differences" over the same 3 or 12 months.
  eq. 4: "vCXGRW_t = 1 / (sum_y w_y) * sum_y w_y vCXGR_yt" (same calendar month t across years y);
          "The weights are increasing time weights, so the more recent past has a higher weight".
  eq. 5: "sigma_vCXGR_W_t = sqrt( sum_y w_y (vCXGR_yt - vCXGRW_t)^2 / ( sum_y w_y (Y_hat - 1) / Y_hat ) )",
          "Y_hat is the total number of weights".
  eq. 6: "(vCXGR_t - vCXGRW_t) / sigma_vCXGR_W_t = X_IPA_t"; "0.5 <= X_IPA < 1 Price Watch; X_IPA >= 1 Price Alert;
          o.w. Normal".
  eq. 7: "IPA_t = gamma Q_IPA_t + (1 - gamma) A_IPA_t"; Annex 1, eq. A.2: "alpha = lambda_Q / sigma^2_{Sum Q,A}",
          lambda_Q the first (largest) root of the 2x2 covariance matrix of the two growth rates (eq. A.7; the
          printed "-(sigma_Q^2 + sigma_A^2)" in A.7 is a sign typo: the roots of A.6 are
          ((s_Q + s_A) +/- sqrt((s_Q + s_A)^2 - 4 (s_Q s_A - c^2))) / 2).
  SDG24 eq. 2-3: same z-score without the volatility adjustment, "gamma is a weight with a value of 0.4"; thresholds
          "0.5 <= IFPA < 1 Moderately High; IFPA >= 1 Abnormally High; -0.5 <= IFPA < 0.5 Normal"; section 4.b:
          "an uninterrupted monthly price series ... of at least 5 years, which include the year being analysed and
          the 4 preceding years".
  Variants implemented:
    variant="baquedano"  [B15] vIPA: volatility-adjusted CGRs, weighted mean/SD, gamma by PCA (eq. A.2).
    variant="sdg"        [SDG24] IFPA: unadjusted CGRs, weighted mean/SD, gamma = 0.4.
  Choices the documents leave open (ours, stated so they can be varied):
    - time weights w_y = 1, 2, ..., Y for the available years of that calendar month, oldest = 1 (linear);
    - SDs (volatility, PCA covariance) use the n - 1 denominator;
    - the year being classified is included in the weighted mean and SD (SDG24 4.b: "include the year being
      analysed"); real-time use: only years up to and including the current one;
    - nominal prices (both documents deflate by CPI; SDG24 4.b allows "nominal prices or a non-food CPI"; no
      general CPI is held for Bangladesh in this project).

Agreement (`agreement`): survey-based classes are the reference. recall = TP / (TP + FN); precision = TP / (TP + FP);
Cohen's kappa = (p_o - p_e) / (1 - p_e) on the binary alert indicator. Optional integer weights (bootstrap).
"""
import numpy as np
import pandas as pd

ALPS_CUTS = (0.25, 1.0, 2.0)
ALPS_CLASSES = ("Normal", "Stress", "Alert", "Crisis")
IPA_CUTS = (0.5, 1.0)
IPA_CLASSES = ("Normal", "Watch", "Alert")
ALPS_MIN_OBS = 36
IPA_MIN_YEARS = 5
SDG_GAMMA = 0.4


def _classify(z, cuts, labels) -> np.ndarray:
    z = np.asarray(z, float)
    out = np.full(z.shape, "", dtype=object)
    ok = np.isfinite(z)
    out[ok] = np.asarray(labels, dtype=object)[np.searchsorted(np.asarray(cuts), z[ok], side="right")]
    return out


def alps_class(z) -> np.ndarray:
    """Normal < 0.25 <= Stress < 1 <= Alert < 2 <= Crisis ([WFP14] Table 1); '' where z is missing."""
    return _classify(z, ALPS_CUTS, ALPS_CLASSES)


def ipa_class(z) -> np.ndarray:
    """Normal < 0.5 <= Watch < 1 <= Alert ([B15] eq. 6; SDG24: moderately / abnormally high); '' where missing."""
    return _classify(z, IPA_CUTS, IPA_CLASSES)


def _design(months) -> np.ndarray:
    m = pd.DatetimeIndex(months)
    t = (m.year - 1960) * 12 + (m.month - 1)  # Stata %tm: months since 1960m1
    d = (m.month.to_numpy()[:, None] == np.arange(1, 13)[None, :]).astype(float)
    return np.column_stack([np.asarray(t, float), d])


def alps_insample(prices, months) -> np.ndarray:
    """ALPS for every month of a series, trend fitted on the entire series (the [WFP14] Stata syntax)."""
    p = np.asarray(prices, float)
    X = _design(months)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum() < ALPS_MIN_OBS:
        return out
    beta, *_ = np.linalg.lstsq(X[ok], p[ok], rcond=None)
    res = p[ok] - X[ok] @ beta
    out[ok] = res / np.std(res, ddof=1)
    return out


def alps_realtime(history, months, candidates) -> np.ndarray:
    """ALPS of month t for each candidate value, appended to `history` (months up to t-1); trend refitted.

    months: the len(history) + 1 consecutive months, the last being t.
    """
    h = np.asarray(history, float)
    c = np.atleast_1d(np.asarray(candidates, float))
    X = _design(months)
    if X.shape[0] != len(h) + 1:
        raise ValueError("months must have len(history) + 1 entries")
    ok = np.r_[np.isfinite(h), True]
    if ok.sum() < ALPS_MIN_OBS:
        return np.full(len(c), np.nan)
    Y = np.repeat(np.r_[h, np.nan][:, None], len(c), axis=1)
    Y[-1] = c
    Xo, Yo = X[ok], Y[ok]
    beta, *_ = np.linalg.lstsq(Xo, Yo, rcond=None)
    res = Yo - Xo @ beta
    return res[-1] / np.std(res, axis=0, ddof=1)


def compound_growth(prices, k: int) -> np.ndarray:
    """[B15] eq. 1 with tn - t0 = k months: (P_t / P_{t-k})^(1/k) - 1; NaN for the first k months."""
    p = np.asarray(prices, float)
    out = np.full(len(p), np.nan)
    out[k:] = (p[k:] / p[:-k]) ** (1.0 / k) - 1
    return out


def log_volatility(prices, k: int) -> np.ndarray:
    """[B15] eq. 3 volatility: SD (n - 1) of the k monthly log differences ending at t; NaN for the first k months."""
    lp = np.log(np.asarray(prices, float))
    out = np.full(len(lp), np.nan)
    if len(lp) > k:
        win = np.lib.stride_tricks.sliding_window_view(np.diff(lp), k)
        out[k:] = win.std(axis=1, ddof=1)
    return out


def weighted_mean_sd(x, w):
    """[B15] eqs. 4-5: weighted mean and SD with the (Y_hat - 1) / Y_hat correction, Y_hat = number of weights."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    W, n = w.sum(), len(w)
    m = (w * x).sum() / W
    if n < 2:
        return m, np.nan
    return m, np.sqrt((w * (x - m) ** 2).sum() / (W * (n - 1) / n))


def pca_gamma(q, a) -> float:
    """[B15] Annex 1 eq. A.2: largest eigenvalue of Cov(q, a) divided by Var(q) + Var(a)."""
    q, a = np.asarray(q, float), np.asarray(a, float)
    ok = np.isfinite(q) & np.isfinite(a)
    if ok.sum() < 3:
        return np.nan
    c = np.cov(q[ok], a[ok])
    tr, det = c[0, 0] + c[1, 1], c[0, 0] * c[1, 1] - c[0, 1] ** 2
    if not tr > 0:
        return np.nan
    lam = (tr + np.sqrt(max(tr ** 2 - 4 * det, 0.0))) / 2
    return lam / tr


def _growth_rates(p: np.ndarray, adjust: bool):
    q, a = compound_growth(p, 3), compound_growth(p, 12)
    if adjust:
        q = q * (1 - log_volatility(p, 3))
        a = a * (1 - log_volatility(p, 12))
    return q, a


def _month_z(x: np.ndarray, t: int, min_years: int) -> float:
    """z of x[t] against the same calendar month in earlier years and itself, linear increasing weights."""
    idx = np.arange(t % 12, t + 1, 12)
    v = x[idx]
    v = v[np.isfinite(v)]
    if len(v) < min_years or not np.isfinite(x[t]):
        return np.nan
    m, s = weighted_mean_sd(v, np.arange(1, len(v) + 1, dtype=float))
    return (x[t] - m) / s if s > 0 else np.nan


def ipa_realtime(history, months, candidates, variant: str = "baquedano", min_years: int = IPA_MIN_YEARS) -> np.ndarray:
    """IPA of month t for each candidate value appended to `history` (consecutive months up to t-1).

    months: the len(history) + 1 consecutive months, the last being t (only their count and calendar position
    are used; the series must be uninterrupted).
    """
    if variant not in ("baquedano", "sdg"):
        raise ValueError(variant)
    h = np.asarray(history, float)
    c = np.atleast_1d(np.asarray(candidates, float))
    if len(pd.DatetimeIndex(months)) != len(h) + 1:
        raise ValueError("months must have len(history) + 1 entries")
    t = len(h)
    out = np.full(len(c), np.nan)
    for j, v in enumerate(c):
        p = np.r_[h, v]
        q, a = _growth_rates(p, adjust=(variant == "baquedano"))
        zq, za = _month_z(q, t, min_years), _month_z(a, t, min_years)
        g = pca_gamma(q, a) if variant == "baquedano" else SDG_GAMMA
        out[j] = g * zq + (1 - g) * za
    return out


def agreement(ref, pred, w=None) -> dict:
    """Binary agreement of `pred` with the reference `ref` (both boolean); optional integer/float weights."""
    ref, pred = np.asarray(ref, bool), np.asarray(pred, bool)
    w = np.ones(len(ref)) if w is None else np.asarray(w, float)
    tp, fn = (w * (ref & pred)).sum(), (w * (ref & ~pred)).sum()
    fp, tn = (w * (~ref & pred)).sum(), (w * (~ref & ~pred)).sum()
    n = tp + fn + fp + tn
    out = {"n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "n_ref_pos": tp + fn, "n_pred_pos": tp + fp}
    out["recall"] = tp / (tp + fn) if tp + fn > 0 else np.nan
    out["precision"] = tp / (tp + fp) if tp + fp > 0 else np.nan
    if n > 0:
        po = (tp + tn) / n
        pe = ((tp + fn) / n) * ((tp + fp) / n) + ((fp + tn) / n) * ((fn + tn) / n)
        out["kappa"] = (po - pe) / (1 - pe) if pe < 1 else np.nan
    else:
        out["kappa"] = np.nan
    if w is None or np.allclose(w, np.round(w)):
        for k in ("n", "tp", "fn", "fp", "tn", "n_ref_pos", "n_pred_pos"):
            out[k] = int(round(out[k]))
    return out
