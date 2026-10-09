"""Noise estimators (methodology D2) must recover known parameters on simulated AR(1) + white noise."""
import numpy as np
import pandas as pd
import pytest

from rtfp.noise import autocovariances, noise_from_autocov


def _simulate(n_mkts=400, n_months=60, phi=0.6, sd_innov=0.03, sd_noise=0.02, seed=0):
    """Local log level l_t = phi*l_{t-1} + u_t; observed y = l + nu. Returns long frame of observed local changes."""
    rng = np.random.default_rng(seed)
    rows = []
    months = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    for m in range(n_mkts):
        l = np.zeros(n_months)
        l[0] = rng.normal(0, sd_innov / np.sqrt(1 - phi**2))
        for t in range(1, n_months):
            l[t] = phi * l[t - 1] + rng.normal(0, sd_innov)
        y = l + rng.normal(0, sd_noise, n_months)
        dy = np.r_[np.nan, np.diff(y)]
        rows.append(pd.DataFrame({"mkt_uid": f"m{m}", "item": "rice", "month": months, "e_obs": dy}))
    return pd.concat(rows, ignore_index=True)


def test_autocovariances_match_theory_for_ar1_plus_noise():
    phi, sd_u, sd_nu = 0.6, 0.03, 0.02
    # 3,000 markets: at 400 the lag-2 sampling error is ~15% (checked across seeds).
    g = autocovariances(_simulate(phi=phi, sd_innov=sd_u, sd_noise=sd_nu, n_mkts=3000)).loc["rice"]
    V = sd_u**2 / (1 - phi**2)
    assert g["g1"] == pytest.approx(-sd_nu**2 - V * (1 - phi) ** 2, rel=0.08)
    assert g["g2"] == pytest.approx(-V * (1 - phi) ** 2 * phi, rel=0.15)


def test_ar1_estimator_recovers_noise_variance_and_naive_overstates_it():
    phi, sd_u, sd_nu = 0.6, 0.03, 0.02
    g = autocovariances(_simulate(phi=phi, sd_innov=sd_u, sd_noise=sd_nu, n_mkts=1500)).loc["rice"]
    est = noise_from_autocov(g)
    assert est["phi"] == pytest.approx(phi, abs=0.12)
    assert est["var_nu_ar1"] == pytest.approx(sd_nu**2, rel=0.2)
    assert est["var_nu_naive"] > est["var_nu_ar1"]
    true_rel = 2 * (sd_u**2 / (1 - phi**2)) * (1 - phi) / (2 * (sd_u**2 / (1 - phi**2)) * (1 - phi) + 2 * sd_nu**2)
    assert est["rel_ar1"] == pytest.approx(true_rel, abs=0.05)


def test_pure_noise_gives_reliability_near_zero_and_random_walk_gives_one():
    g_noise = autocovariances(_simulate(phi=0.0, sd_innov=1e-6, sd_noise=0.02)).loc["rice"]
    assert noise_from_autocov(g_noise)["rel_naive"] == pytest.approx(0.0, abs=0.05)
    g_rw = autocovariances(_simulate(phi=0.999, sd_innov=0.03, sd_noise=1e-6)).loc["rice"]
    assert noise_from_autocov(g_rw)["rel_naive"] == pytest.approx(1.0, abs=0.05)


def test_ar1_estimator_reports_not_identified_when_signs_are_wrong():
    est = noise_from_autocov(pd.Series({"g0": 1.0, "g1": -0.2, "g2": 0.05, "g3": 0.01, "n1": 100, "n2": 100, "n3": 100}))
    assert np.isnan(est["var_nu_ar1"]) and not est["ar1_identified"]


def test_autocovariances_use_exact_month_lags_not_row_order():
    d = pd.DataFrame({"mkt_uid": "a", "item": "rice",
                      "month": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"]),
                      "e_obs": [1.0, -1.0, 5.0]})
    g = autocovariances(d, demean=False).loc["rice"]
    assert g["n1"] == 1  # only Jan-Feb are one month apart; Feb-Apr is two months
    assert g["g1"] == pytest.approx(-1.0)


def test_weights_zero_on_a_market_equal_dropping_it_and_ones_change_nothing():
    d = _simulate(n_mkts=30, seed=5)
    base = autocovariances(d)
    ones = autocovariances(d.assign(w=1))
    assert np.allclose(base[["g0", "g1", "g2", "g3"]], ones[["g0", "g1", "g2", "g3"]])
    dropped = autocovariances(d[d.mkt_uid != "m0"])
    zero = autocovariances(d.assign(w=np.where(d.mkt_uid == "m0", 0, 1)))
    assert np.allclose(dropped[["g0", "g1", "g2", "g3"]], zero[["g0", "g1", "g2", "g3"]])


def test_trimmed_autocovariance_resists_a_single_gross_error():
    d = _simulate(n_mkts=300, phi=0.6, sd_innov=0.03, sd_noise=0.005, seed=9)
    clean = noise_from_autocov(autocovariances(d).loc["rice"])
    # inject one -85% entry error that recovers next month (spike reversal), like Patuakhali 2025-03
    bad = d.copy()
    i = bad.index[(bad.mkt_uid == "m0") & (bad.month == "2022-06-01")][0]
    bad.loc[i, "e_obs"] += np.log(0.15)
    bad.loc[i + 1, "e_obs"] -= np.log(0.15)
    contaminated = noise_from_autocov(autocovariances(bad).loc["rice"])
    trimmed = noise_from_autocov(autocovariances(bad, trim=0.01).loc["rice"])
    assert contaminated["rel_naive"] < clean["rel_naive"] - 0.1       # one error moves the estimate (0.56 -> 0.40)
    assert abs(trimmed["rel_naive"] - clean["rel_naive"]) < 0.1       # trimming restores it
