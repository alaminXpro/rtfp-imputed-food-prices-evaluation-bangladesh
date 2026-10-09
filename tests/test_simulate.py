"""Framework simulation (methodology M2 'Simulation', Task 3.1)."""
import numpy as np
import pandas as pd
import pytest

from rtfp.noise import autocovariances, noise_from_autocov
from rtfp.simulate import (IMPUTERS, SimSpec, alert_benchmark, conditional_mean, holdout_pairs, impute,
                           noise_sample, panel_metrics, posterior_var, simulate_panel)


def _spec(**kw):
    base = dict(n_months=48, propensity=(0.7,) * 60, n_regions=6, phi_local=0.7, var_dlocal=0.002,
                regional_share=0.6, var_noise=0.0008, p_release=0.5)
    base.update(kw)
    return SimSpec(**base)


def test_panel_shapes_missingness_and_white_noise():
    sp = _spec(n_months=60, propensity=(0.4,) * 50 + (0.9,) * 50, var_noise=0.001)
    p = simulate_panel(sp, np.random.default_rng(1))
    assert p.x.shape == p.y.shape == (100, 60)
    assert not (p.obs_release & ~p.obs_latest).any()                    # release is a subset of latest
    assert p.obs_latest[:50].mean() == pytest.approx(0.4, abs=0.03)
    assert p.obs_latest[50:].mean() == pytest.approx(0.9, abs=0.03)
    assert (p.obs_release.sum() / p.obs_latest.sum()) == pytest.approx(0.5, abs=0.03)
    assert np.var(p.y - p.x) == pytest.approx(0.001, rel=0.05)


def test_local_component_has_the_requested_change_variance_and_regional_share():
    sp = _spec(n_months=200, propensity=(1.0,) * 120, var_dlocal=0.002, regional_share=0.5, n_regions=4)
    p = simulate_panel(sp, np.random.default_rng(2))
    dl = np.diff(p.local, axis=1)
    assert dl.var() == pytest.approx(0.002, rel=0.08)
    same = [np.mean(dl[i] * dl[j]) for i in range(40) for j in range(i + 1, 40) if p.region[i] == p.region[j]]
    assert np.mean(same) / 0.002 == pytest.approx(0.5, abs=0.08)


def test_stale_variant_carries_the_previous_report_forward():
    sp = _spec(noise="stale", p_stale=0.3, n_months=100)
    p = simulate_panel(sp, np.random.default_rng(3))
    assert (np.diff(p.y, axis=1) == 0).mean() == pytest.approx(0.3, abs=0.02)


def test_average_variant_reports_the_mean_of_month_start_and_end():
    sp = _spec(noise="average", var_noise=0.0)
    p = simulate_panel(sp, np.random.default_rng(4))
    assert p.y[:, 1:] == pytest.approx((p.x[:, 1:] + p.x[:, :-1]) / 2)


def test_conditional_mean_matches_dense_gaussian_conditioning():
    sp = _spec(n_months=12, propensity=(0.5,) * 8, n_regions=2, sd_market=0.1)
    p = simulate_panel(sp, np.random.default_rng(5))
    N, T = p.x.shape
    m, t = np.divmod(np.arange(N * T), T)
    g = p.region[m]
    lag = np.abs(t[:, None] - t[None, :])
    s = sp
    su2 = s.var_dlocal * (1 + s.phi_local) / 2
    K = (s.sd_common ** 2 / (1 - s.phi_common ** 2) * s.phi_common ** lag
         + (g[:, None] == g[None, :]) * s.regional_share * su2 / (1 - s.phi_local ** 2) * s.phi_local ** lag
         + (m[:, None] == m[None, :]) * ((1 - s.regional_share) * su2 / (1 - s.phi_local ** 2) * s.phi_local ** lag
                                         + s.sd_market ** 2))
    o = p.obs_release.ravel()
    mu0 = np.tile(p.mean0, N)
    Koo = K[np.ix_(o, o)] + s.var_noise * np.eye(o.sum())
    dense = mu0 + K[:, o] @ np.linalg.solve(Koo, p.y.ravel()[o] - mu0[o])
    dense_var = np.diag(K) - np.einsum("ij,ji->i", K[:, o], np.linalg.solve(Koo, K[o, :]))
    assert conditional_mean(p).ravel() == pytest.approx(dense, abs=1e-10)
    mask = ~p.obs_release
    assert posterior_var(p, mask)[mask] == pytest.approx(dense_var.reshape(N, T)[mask], abs=1e-12)


def test_calibrated_conditional_mean_gives_beta_mz_of_one_over_200_panels():
    sp = _spec()
    betas, betas_own, ns = [], [], []
    for r in range(200):
        p = simulate_panel(sp, np.random.default_rng(1000 + r))
        res = panel_metrics(p, impute(p, "cond_mean"))
        betas.append(res["within_month"]["beta_mz"])
        betas_own.append(res["own_mu"]["beta_mz"])
        ns.append(res["within_month"]["n"])
    assert min(ns) >= 200
    assert np.mean(betas) == pytest.approx(1.0, abs=0.05)
    # own-release demeaning: the published side carries the sampling error of the release's observed-market mean
    # (few markets observed), which the truth side does not share, so even a calibrated imputer shows beta_mz < 1
    # (diagnosed: ~0.64 with 60 markets, ~0.94 with 300). Within-month demeaning is exactly calibrated.
    assert np.mean(betas_own) < np.mean(betas) - 0.1


def test_noise_estimator_recovers_var_nu_under_ar1_plus_white_noise():
    # 6,000 markets: with 300 the AR(1) estimate has ~28% sampling SD (seed sweep); here <= 5% across seeds
    sp = SimSpec(n_months=120, propensity=(0.95,) * 6000, n_regions=1, regional_share=0.0, phi_local=0.6,
                 var_dlocal=0.0015, var_noise=0.0004, p_release=1.0)
    p = simulate_panel(sp, np.random.default_rng(7))
    est = noise_from_autocov(autocovariances(noise_sample(p)).loc["sim"])
    assert est["ar1_identified"]
    assert est["var_nu_ar1"] == pytest.approx(0.0004, rel=0.15)
    assert est["phi"] == pytest.approx(0.6, abs=0.1)
    # naive estimator overstates noise by V(1 - phi)^2 = var_dlocal (1 - phi) / 2
    assert est["var_nu_naive"] == pytest.approx(0.0004 + 0.0015 * 0.4 / 2, rel=0.1)


def test_disattenuated_correlation_recovers_noise_free_correlation():
    sp = _spec(var_noise=0.002)
    rho, true = [], []
    for r in range(40):
        p = simulate_panel(sp, np.random.default_rng(2000 + r))
        res = panel_metrics(p, impute(p, "cond_mean"))["within_month"]
        rho.append(res["corr"] / np.sqrt(res["rel_true"]))
        true.append(res["corr_true"])
    assert np.mean(rho) == pytest.approx(np.mean(true), abs=0.03)


def test_chain_mean_adds_residual_noise_with_one_fifth_of_the_posterior_variance():
    sp = _spec()
    p = simulate_panel(sp, np.random.default_rng(8))
    cm, ch = impute(p, "cond_mean"), impute(p, "chain_mean", rng=np.random.default_rng(9))
    mask = p.obs_latest & ~p.obs_release
    pv = posterior_var(p, mask)
    assert np.var((ch - cm)[mask]) == pytest.approx(pv[mask].mean() / 5, rel=0.15)
    assert ((ch - cm)[p.obs_release] == 0).all()


def test_outlier_smoothing_shrinks_imputed_local_changes():
    sp = _spec()
    lam_cm, lam_os = [], []
    for r in range(20):
        p = simulate_panel(sp, np.random.default_rng(3000 + r))
        lam_cm.append(panel_metrics(p, impute(p, "cond_mean"))["within_month"]["lam"])
        lam_os.append(panel_metrics(p, impute(p, "outlier_smooth"))["within_month"]["lam"])
    assert np.mean(lam_os) < 0.8 * np.mean(lam_cm)


def test_idw_fill_on_a_toy_month():
    sp = SimSpec(n_months=2, propensity=(1.0, 1.0, 1.0), region=(0, 0, 0),
                 coords_km=((0.0, 0.0), (10.0, 0.0), (0.0, 20.0)), p_release=1.0)
    p = simulate_panel(sp, np.random.default_rng(0))
    p.y[:] = np.array([[1.0, 1.2], [2.0, 2.0], [3.0, 3.4]])
    p.obs_release[:] = True
    p.obs_release[0, 1] = False   # market 0 imputed in month 1
    c = impute(p, "idw_seasonal")
    # expected value recomputed here: two-way (market + month) fit on the observed cells by alternating means,
    # then the inverse-squared-distance average of markets 1 and 2's month-1 residuals.
    obs = p.y.copy()
    obs[0, 1] = np.nan
    a, b = np.zeros(3), np.zeros(2)
    for _ in range(500):
        b = np.nanmean(obs - a[:, None], axis=0)
        a = np.nanmean(obs - b[None, :], axis=1)
    r = obs - a[:, None] - b[None, :]
    w = np.array([1 / (10 ** 2 + 1), 1 / (20 ** 2 + 1)])
    expected = a[0] + b[1] + (w * r[1:, 1]).sum() / w.sum()
    assert c[0, 1] == pytest.approx(expected, abs=1e-6)
    assert c[1, 1] == p.y[1, 1]


def test_holdout_pairs_are_both_imputed_and_later_observed():
    p = simulate_panel(_spec(), np.random.default_rng(11))
    pr = holdout_pairs(p, impute(p, "cond_mean"))
    mi = pr.mkt_uid.str[1:].astype(int).to_numpy()
    ti = pr.t.to_numpy()
    for tt in (ti, ti - 1):
        assert p.obs_latest[mi, tt].all() and not p.obs_release[mi, tt].any()
    assert set(pr.pair_type) == {"both_imputed"}


def test_all_imputer_types_run():
    p = simulate_panel(_spec(n_months=24, propensity=(0.6,) * 20), np.random.default_rng(12))
    for name in IMPUTERS:
        c = impute(p, name, rng=np.random.default_rng(1))
        assert np.isfinite(c).all()
        assert (c[p.obs_release] == p.y[p.obs_release]).all()


def test_alert_benchmark_oracle_matches_survey_without_noise():
    sp = SimSpec(n_months=84, propensity=(0.5,) * 6, var_noise=0.0, sd_common=0.05, phi_common=0.95)
    out = alert_benchmark(sp, n_eval=12, rng=np.random.default_rng(13))
    assert set(out.indicator) == {"alps", "ipa_baquedano", "ipa_sdg"}
    ok = out.z_survey.notna()
    assert ok.sum() > 0
    assert np.allclose(out.z_survey[ok], out.z_oracle[ok])
    assert len(out) == 6 * 12 * 3


def test_calibrate_common_recovers_trend_season_and_ar1():
    from rtfp.simulate import calibrate_common

    rng = np.random.default_rng(21)
    months = pd.date_range("2007-01-01", periods=480, freq="MS")
    season = 0.03 * np.sin(2 * np.pi * np.arange(12) / 12)
    u = np.zeros(480)
    for t in range(1, 480):
        u[t] = 0.9 * u[t - 1] + rng.normal(0, 0.02)
    L = pd.Series(4.0 + 0.005 * np.arange(480) + season[months.month - 1] + u, index=months)
    c = calibrate_common(L)
    assert c["drift"] == pytest.approx(0.005, abs=0.002)
    assert c["phi_common"] == pytest.approx(0.9, abs=0.05)
    assert c["sd_common"] == pytest.approx(0.02, rel=0.1)
    assert np.corrcoef(c["season"], season)[0, 1] > 0.9


def test_calibrate_local_recovers_persistence_and_regional_share():
    from rtfp.simulate import calibrate_local

    sp = SimSpec(n_months=150, propensity=(0.5,) * 104, n_regions=8, phi_local=0.7, var_dlocal=0.002,
                 regional_share=0.4, var_noise=0.0005, p_release=1.0)
    p = simulate_panel(sp, np.random.default_rng(22))
    ns = noise_sample(p)
    m, t = np.nonzero(p.obs_latest)
    lev = pd.DataFrame({"mkt_uid": np.char.add("m", m.astype(str)), "month": p.months[t], "log_y": p.y[m, t]})
    obs = lev.merge(ns[["mkt_uid", "month", "e_obs"]], on=["mkt_uid", "month"], how="left")
    obs["region"] = p.region[obs.mkt_uid.str[1:].astype(int)]
    g0 = float(np.mean((ns.e_obs - ns.e_obs.mean()) ** 2))
    c = calibrate_local(obs, var_nu=0.0005, g0=g0)
    assert c["var_dlocal"] == pytest.approx(0.002, rel=0.15)
    assert c["phi_local"] == pytest.approx(0.7, abs=0.1)
    assert c["regional_share"] == pytest.approx(0.4, abs=0.12)
