import numpy as np
import pandas as pd
import pytest

from rtfp.metrics import local_components, local_stats, pigeonhole_weights, weighted_local_stats


def test_local_stats_on_exact_half_scaling():
    rng = np.random.default_rng(0)
    e_obs = rng.normal(0, 0.05, 500)
    s = local_stats(0.5 * e_obs, e_obs)
    assert s["corr"] == pytest.approx(1.0)
    assert s["beta_rev"] == pytest.approx(0.5)
    assert s["beta_mz"] == pytest.approx(2.0)
    assert s["lam"] == pytest.approx(0.25)
    assert s["beta_rev"] == pytest.approx(s["beta_mz"] * s["lam"])  # identity check


def test_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50), rng.normal(size=50)
    w = rng.integers(0, 4, 50)
    dup_a, dup_b = np.repeat(a, w), np.repeat(b, w)
    ws = weighted_local_stats(a, b, w)
    ds = local_stats(dup_a, dup_b)
    for k in ["corr", "beta_mz", "beta_rev", "lam"]:
        assert ws[k] == pytest.approx(ds[k])


def test_own_mu_falls_back_to_published_mean_when_nothing_observed():
    p = pd.DataFrame({"item": "rice", "month": pd.to_datetime(["2025-01-01"] * 2), "dlog_pub": [0.10, 0.02],
                      "dlog_truth": [0.12, 0.00], "mu_obs_vfinal": [np.nan, 0.01], "mu_pub_vfinal": [0.05, 0.03],
                      "mu_obs_latest": [0.04, 0.04]})
    e = local_components(p, mode="own_mu")
    assert e.e_pub.tolist() == pytest.approx([0.05, 0.01])
    assert e.e_obs.tolist() == pytest.approx([0.08, -0.04])


def test_within_month_demeaning_gives_zero_means_and_drops_singletons():
    p = pd.DataFrame({"item": "rice", "month": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-02-01"]),
                      "dlog_pub": [0.1, 0.3, 0.5], "dlog_truth": [0.0, 0.2, 0.9]})
    e = local_components(p, mode="within_month")
    assert len(e) == 2
    assert e.e_pub.sum() == pytest.approx(0) and e.e_obs.sum() == pytest.approx(0)


def test_pigeonhole_weights_are_product_of_market_and_month_draw_counts():
    df = pd.DataFrame({"mkt_uid": ["a", "a", "b"], "month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-01-01"])})
    rng = np.random.default_rng(3)
    w = pigeonhole_weights(df, rng, markets=["a", "b"], months=list(df.month.unique()))
    assert w.dtype.kind in "iu" and len(w) == 3 and (w >= 0).all()
    # deterministic given the seed
    w2 = pigeonhole_weights(df, np.random.default_rng(3), markets=["a", "b"], months=list(df.month.unique()))
    assert (w == w2).all()


def test_level_stats_known_values():
    from rtfp.metrics import level_stats
    y = np.array([100.0, 100.0, 100.0, 100.0])
    c = np.array([110.0, 90.0, 100.0, 100.0])
    s = level_stats(c, y)
    assert s["mape"] == pytest.approx(5.0)          # (10+10+0+0)/4 percent
    assert s["median_ape"] == pytest.approx(5.0)       # median of {10, 10, 0, 0}
    assert s["share_err_gt10"] == pytest.approx(0.0)  # exactly 10% is not > 10%
    assert s["bias_logpct"] == pytest.approx(100 * np.mean(np.log(c / y)))


def test_level_mz_slope_month_demeaned_is_one_for_perfect_cross_section():
    from rtfp.metrics import level_mz
    df = pd.DataFrame({"item": "rice", "month": pd.to_datetime(["2025-01-01"] * 3 + ["2025-02-01"] * 3),
                       "c": [100, 110, 120, 200, 220, 240.0]})
    df["y"] = df.c * np.where(df.month == "2025-01-01", 1.0, 1.1)  # month-common error only
    assert level_mz(df, demean="month")["slope"] == pytest.approx(1.0)


def test_fe_ols_recovers_coefficient_with_two_way_clustered_se():
    from rtfp.metrics import fe_ols
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({"mkt_uid": rng.integers(0, 60, n).astype(str), "fe": rng.integers(0, 30, n).astype(str),
                       "x": rng.normal(size=n)})
    df["month"] = df.fe
    df["y"] = 2.0 * df.x + df.fe.astype(int) * 0.1 + rng.normal(size=n)
    r = fe_ols(df, "y", ["x"], fe="fe", clusters=["mkt_uid", "month"])
    assert r.loc["x", "coef"] == pytest.approx(2.0, abs=0.1)
    assert 0 < r.loc["x", "se"] < 0.1 and r.loc["x", "n"] == n


def test_fe_ols_flags_regressor_absorbed_by_fixed_effects():
    from rtfp.metrics import fe_ols
    rng = np.random.default_rng(8)
    n = 600
    df = pd.DataFrame({"mkt_uid": rng.integers(0, 30, n).astype(str), "fe": rng.integers(0, 20, n).astype(str),
                       "x": rng.normal(size=n)})
    df["month"] = df.fe
    df["const_in_fe"] = df.fe.astype(int) * 1.5  # constant within each FE group
    df["y"] = df.x + rng.normal(size=n)
    r = fe_ols(df, "y", ["x", "const_in_fe"], fe="fe", clusters=["mkt_uid", "month"])
    assert r.loc["const_in_fe", "absorbed"] and np.isnan(r.loc["const_in_fe", "coef"])
    assert not r.loc["x", "absorbed"] and r.loc["x", "coef"] == pytest.approx(1.0, abs=0.15)


def test_persistent_spike_flags():
    from rtfp.metrics import persistent_spikes
    # truth log levels around t: jump of +15% at t; at t+1 keeps 10% (>= half) -> spike; reversal -> not
    df = pd.DataFrame({"dlog_truth": [0.15, 0.15, -0.15, 0.05],
                       "dlog_truth_next": [-0.05, -0.10, 0.0, 0.0],   # change from t to t+1
                       "dlog_pub": [0.06, 0.20, -0.01, 0.0]})
    s = persistent_spikes(df, jump=0.10, keep=0.5, moved=0.05)
    assert s.is_spike.tolist() == [True, False, True, False]   # 0.15-0.10=0.05 < half of 0.15 -> not persistent
    assert s.caught.tolist()[0] and not s.caught.tolist()[2]    # same-direction move of >= 5%
