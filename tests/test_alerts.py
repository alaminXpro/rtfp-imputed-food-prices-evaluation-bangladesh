"""ALPS (WFP 2014) and IPA (FAO GIEWS, Baquedano 2015; UN SDG 2.c.1 metadata 2024) on hand-computed toy series.

Expected values are computed here from first principles with plain arithmetic, independently of the module.
"""
import math

import numpy as np
import pandas as pd
import pytest

from rtfp.alerts import (agreement, alps_class, alps_insample, alps_realtime, compound_growth, ipa_class,
                         ipa_realtime, log_volatility, pca_gamma, weighted_mean_sd)


def _months(start, n):
    return pd.date_range(start, periods=n, freq="MS")


def _alps_toy():
    """36 months, price = 50 + 0.2*trend + month effect + eps, with eps orthogonal to trend and month dummies.

    eps is +1, -2, +1 in March 2015, 2016, 2017 and 0 elsewhere: it sums to zero within March and
    sum(T * eps) = T0 - 2(T0 + 12) + (T0 + 24) = 0, so the OLS residuals equal eps exactly.
    """
    m = _months("2015-01-01", 36)
    trend = np.arange(36)
    month_effect = np.array([0, 1, 3, 2, 5, 4, 0, -1, -2, 1, 2, 3], float)[m.month - 1]
    eps = np.zeros(36)
    eps[[2, 14, 26]] = [1.0, -2.0, 1.0]
    return m, 50 + 0.2 * trend + month_effect + eps, eps


def test_alps_insample_equals_residual_over_residual_sd():
    m, p, eps = _alps_toy()
    z = alps_insample(p, m)
    sd = math.sqrt((1 + 4 + 1) / 35)  # Stata egen sd(): denominator n - 1
    assert z == pytest.approx(eps / sd, abs=1e-9)
    assert z[26] == pytest.approx(math.sqrt(35 / 6))  # 2.415 -> Crisis
    assert list(alps_class(z[[2, 14, 26, 0]])) == ["Crisis", "Normal", "Crisis", "Normal"]


def test_alps_realtime_appends_candidates_to_history_and_refits_trend():
    m, p, _ = _alps_toy()
    z = alps_realtime(p[:-1], m, np.array([p[-1], p[-1] + 3.0]))
    assert z[0] == pytest.approx(alps_insample(p, m)[-1])
    p2 = p.copy()
    p2[-1] += 3.0
    assert z[1] == pytest.approx(alps_insample(p2, m)[-1])


def test_alps_requires_36_observations():
    m, p, _ = _alps_toy()
    assert np.isnan(alps_realtime(p[:34], m[:35], np.array([p[34]]))[0])
    assert np.isfinite(alps_realtime(p[:35], m[:36], np.array([p[35]]))[0])


def test_alps_thresholds_follow_table_1_of_the_guidance_note():
    z = np.array([-3.0, 0.2499, 0.25, 0.999, 1.0, 1.999, 2.0, np.nan])
    assert list(alps_class(z)) == ["Normal", "Normal", "Stress", "Stress", "Alert", "Alert", "Crisis", ""]


def test_compound_growth_and_log_volatility():
    p = np.array([100.0, 100.0, 100.0, 133.1])
    assert compound_growth(p, 3)[3] == pytest.approx(0.1)
    assert np.isnan(compound_growth(p, 3)[:3]).all()
    L = math.log(1.331)
    # log differences 0, 0, L: sample SD = sqrt(((L/3)^2 * 2 + (2L/3)^2) / 2) = L / sqrt(3)
    assert log_volatility(p, 3)[3] == pytest.approx(L / math.sqrt(3))
    p13 = 100 * 1.01 ** np.arange(13)
    assert compound_growth(p13, 12)[12] == pytest.approx(0.01)


def test_weighted_mean_and_sd_follow_equations_4_and_5():
    m, s = weighted_mean_sd(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]))
    mean = (1 * 1 + 2 * 2 + 3 * 4) / 6
    num = 1 * (1 - mean) ** 2 + 2 * (2 - mean) ** 2 + 3 * (4 - mean) ** 2
    assert m == pytest.approx(mean)
    assert s == pytest.approx(math.sqrt(num / (6 * (3 - 1) / 3)))


def test_pca_gamma_is_largest_eigenvalue_share():
    q = np.array([1.0, -1.0, 1.0, -1.0])
    assert pca_gamma(q, np.array([1.0, -1.0, -1.0, 1.0])) == pytest.approx(0.5)   # equal variances, cov 0
    assert pca_gamma(q, 0.5 * q) == pytest.approx(1.0)                             # rank one
    assert pca_gamma(q, np.zeros(4)) == pytest.approx(1.0)


def _ipa_toy():
    """61 months, Dec 2015 - Dec 2020, price 100 except Dec 2018 = 110 and Dec 2020 = 121."""
    m = _months("2015-12-01", 61)
    p = np.full(61, 100.0)
    p[(m.year == 2018) & (m.month == 12)] = 110.0
    p[-1] = 121.0
    return m, p


def _z(vals):
    w = np.arange(1, len(vals) + 1, dtype=float)  # increasing time weights, oldest year = 1
    mean = (w * vals).sum() / w.sum()
    sd = math.sqrt((w * (vals - mean) ** 2).sum() / (w.sum() * (len(vals) - 1) / len(vals)))
    return (vals[-1] - mean) / sd


def test_ipa_sdg_variant_on_toy_series():
    m, p = _ipa_toy()
    # December values, years 2016..2020 (Dec 2015 has no lagged price)
    q = np.array([0, 0, 1.1 ** (1 / 3) - 1, 0, 1.21 ** (1 / 3) - 1])
    a = np.array([0, 0, 1.1 ** (1 / 12) - 1, (1 / 1.1) ** (1 / 12) - 1, 1.21 ** (1 / 12) - 1])
    expected = 0.4 * _z(q) + 0.6 * _z(a)
    got = ipa_realtime(p[:-1], m, np.array([p[-1]]), variant="sdg")[0]
    assert got == pytest.approx(expected)
    assert ipa_class(np.array([got]))[0] == ("Alert" if expected >= 1 else "Watch" if expected >= 0.5 else "Normal")


def test_ipa_baquedano_variant_applies_volatility_adjustment_and_pca_weight():
    m, p = _ipa_toy()
    lp = np.log(p)

    def vol(i, k):
        return np.std(np.diff(lp[i - k:i + 1]), ddof=1)

    dec = [i for i in range(len(p)) if m[i].month == 12 and i >= 12]
    vq = np.array([((p[i] / p[i - 3]) ** (1 / 3) - 1) * (1 - vol(i, 3)) for i in dec])
    va = np.array([((p[i] / p[i - 12]) ** (1 / 12) - 1) * (1 - vol(i, 12)) for i in dec])
    allq = np.array([((p[i] / p[i - 3]) ** (1 / 3) - 1) * (1 - vol(i, 3)) for i in range(12, len(p))])
    alla = np.array([((p[i] / p[i - 12]) ** (1 / 12) - 1) * (1 - vol(i, 12)) for i in range(12, len(p))])
    c = np.cov(allq, alla)
    lam = ((c[0, 0] + c[1, 1]) + math.sqrt((c[0, 0] + c[1, 1]) ** 2 - 4 * (c[0, 0] * c[1, 1] - c[0, 1] ** 2))) / 2
    g = lam / (c[0, 0] + c[1, 1])
    expected = g * _z(vq) + (1 - g) * _z(va)
    got = ipa_realtime(p[:-1], m, np.array([p[-1]]), variant="baquedano")[0]
    assert got == pytest.approx(expected)


def test_ipa_needs_five_years_for_the_month():
    m, p = _ipa_toy()
    # with only Dec 2016..2019 history (4 years incl. the current), the indicator is not computed
    assert np.isnan(ipa_realtime(p[:48], m[:49], np.array([p[48]]), variant="sdg")[0])


def test_ipa_thresholds():
    assert list(ipa_class(np.array([-2.0, 0.49, 0.5, 0.99, 1.0, np.nan]))) == ["Normal", "Normal", "Watch", "Watch",
                                                                               "Alert", ""]


def test_agreement_on_a_hand_counted_table():
    ref = np.array([1, 1, 1, 0, 0, 0, 0, 0], bool)
    pred = np.array([1, 1, 0, 1, 0, 0, 0, 0], bool)
    a = agreement(ref, pred)
    po, pe = 6 / 8, (3 / 8) * (3 / 8) + (5 / 8) * (5 / 8)
    assert (a["tp"], a["fn"], a["fp"], a["tn"]) == (2, 1, 1, 4)
    assert a["recall"] == pytest.approx(2 / 3) and a["precision"] == pytest.approx(2 / 3)
    assert a["kappa"] == pytest.approx((po - pe) / (1 - pe))


def test_agreement_integer_weights_equal_duplicated_rows():
    rng = np.random.default_rng(3)
    ref, pred = rng.random(60) < 0.3, rng.random(60) < 0.4
    w = rng.integers(0, 4, 60)
    a, b = agreement(ref, pred, w), agreement(np.repeat(ref, w), np.repeat(pred, w))
    for k in ["recall", "precision", "kappa"]:
        assert a[k] == pytest.approx(b[k])


def test_agreement_without_reference_positives_is_undefined_not_zero():
    a = agreement(np.zeros(5, bool), np.zeros(5, bool))
    assert np.isnan(a["recall"]) and np.isnan(a["precision"])
