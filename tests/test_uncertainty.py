import numpy as np
import pandas as pd
import pytest

from rtfp.uncertainty import conformal_quantile, cqr_intervals, mondrian_thresholds, time_blocks


def test_conformal_quantile_is_the_finite_sample_order_statistic():
    scores = np.arange(1, 20)  # n = 19
    # alpha = 0.2: k = ceil((n+1)(1-alpha)) = ceil(16) = 16 -> 16th smallest = 16
    assert conformal_quantile(scores, alpha=0.2) == 16
    # alpha too small for n: k > n -> infinite threshold (honest: cannot guarantee coverage)
    assert np.isinf(conformal_quantile(np.arange(1, 5), alpha=0.05))


def test_cqr_interval_widens_quantile_band_by_threshold():
    lo, hi = cqr_intervals(np.array([-0.1]), np.array([0.1]), np.array([0.05]))
    assert lo[0] == pytest.approx(-0.15) and hi[0] == pytest.approx(0.15)


def test_mondrian_uses_group_threshold_and_falls_back_to_pooled_when_small():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 200 + ["b"] * 5)
    scores = np.r_[rng.normal(0, 1, 200), rng.normal(10, 1, 5)]
    th = mondrian_thresholds(scores, groups, alpha=0.2, min_group=30)
    assert th["a"] == pytest.approx(conformal_quantile(scores[:200], 0.2))
    assert th["b"] == pytest.approx(conformal_quantile(scores, 0.2))  # pooled fallback


def test_time_blocks_split_by_vintage_without_leakage():
    v = pd.Series(pd.to_datetime(["2024-01-01"] * 3 + ["2025-01-01"] * 3 + ["2026-01-01"] * 4))
    b = time_blocks(v, t1="2024-06-01", t2="2025-06-01")
    assert b.tolist() == ["train"] * 3 + ["calib"] * 3 + ["test"] * 4


def test_split_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(1)
    cover = []
    for _ in range(300):
        r_cal, r_test = rng.standard_t(4, 400), rng.standard_t(4, 400)
        q_lo, q_hi = -np.ones(400) * 0.5, np.ones(400) * 0.5
        scores = np.maximum(q_lo - r_cal, r_cal - q_hi)
        th = conformal_quantile(scores, alpha=0.2)
        lo, hi = cqr_intervals(q_lo, q_hi, np.full(400, th))
        cover.append(((r_test >= lo) & (r_test <= hi)).mean())
    assert np.mean(cover) == pytest.approx(0.80, abs=0.01)


def test_obs_share_counts_previous_months_only():
    from rtfp.uncertainty import obs_share
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    s = pd.DataFrame({"vintage": pd.Timestamp("2025-01-01"), "mkt_uid": "a", "item": "rice", "month": months,
                      "obs": [True, False, True, True, False, False]})
    out = obs_share(s, window=3).set_index("month").obs_share
    assert np.isnan(out.iloc[0])                    # no previous months
    assert out.iloc[1] == pytest.approx(1.0)        # previous: [T]
    assert out.iloc[3] == pytest.approx(2 / 3)      # previous 3: [T, F, T]
    assert out.iloc[5] == pytest.approx(2 / 3)      # previous 3: [T, T, F]


def test_group_empirical_quantiles_with_pooled_fallback():
    from rtfp.uncertainty import group_quantiles
    r = np.r_[np.linspace(-1, 1, 101), np.linspace(-10, 10, 5)]
    g = np.array(["a"] * 101 + ["b"] * 5)
    q = group_quantiles(r, g, levels=[0.1, 0.9], min_group=30)
    assert q["a"][0.1] == pytest.approx(np.quantile(r[:101], 0.1))
    assert q["b"][0.9] == pytest.approx(np.quantile(r, 0.9))  # small group falls back to pooled
    assert q["__pooled__"][0.1] == pytest.approx(np.quantile(r, 0.1))


def test_aci_alpha_update_moves_against_miscoverage():
    from rtfp.uncertainty import aci_next_alpha
    # target alpha 0.2; last step missed 50% -> alpha must fall (intervals widen); missed 0% -> alpha rises
    assert aci_next_alpha(0.2, target=0.2, err=0.5, gamma=0.05) == pytest.approx(0.2 + 0.05 * (0.2 - 0.5))
    assert aci_next_alpha(0.2, target=0.2, err=0.0, gamma=0.05) == pytest.approx(0.21)
    assert aci_next_alpha(0.001, target=0.2, err=1.0, gamma=0.5) == pytest.approx(1e-6)  # clipped, stays valid


def test_aci_restores_coverage_under_a_variance_shift():
    from rtfp.uncertainty import aci_sequential
    rng = np.random.default_rng(4)
    calib = rng.normal(0, 1, 500)
    batches = [rng.normal(0, 1 if k < 5 else 2, 200) for k in range(30)]   # error variance doubles at batch 5
    static, adaptive = aci_sequential(np.abs(calib), [np.abs(b) for b in batches], target=0.2, gamma=0.05)
    late = slice(15, 30)
    assert np.mean(static[late]) < 0.6                     # static thresholds badly under-cover after the shift
    assert np.mean(adaptive[late]) == pytest.approx(0.8, abs=0.06)
