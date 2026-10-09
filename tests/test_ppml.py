"""PPML with dummy fixed effects and one-/two-way cluster-robust SEs; the switching-market design (methodology D3)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance as sw

from rtfp.ppml import cluster_vcov, drop_separated, lincom, noise_benchmark, ppml, wald

ONE = {"market": ["mkt_uid"]}
TWO = {"market": ["mkt_uid"], "twoway": ["mkt_uid", "month"]}


def _panel(n_mkt=40, n_month=40, beta=-0.4, zero_share=0.3, seed=0, sd_a=0.0, sd_b=0.0):
    """y = exp(alpha_m + gamma_t + (beta + a_m + b_t) D) * W, W >= 0 with exact zeros and E[W] = 1."""
    rng = np.random.default_rng(seed)
    mk, mo = np.repeat(np.arange(n_mkt), n_month), np.tile(np.arange(n_month), n_mkt)
    alpha, gamma = rng.normal(0, 0.3, n_mkt), rng.normal(0, 0.3, n_month)
    a, b = rng.normal(0, sd_a, n_mkt), rng.normal(0, sd_b, n_month)
    d = (rng.random(len(mk)) < 0.5).astype(float)
    w = (rng.random(len(mk)) > zero_share) * rng.gamma(2.0, 1 / (2 * (1 - zero_share)), len(mk))
    y = np.exp(alpha[mk] + gamma[mo] + (beta + a[mk] + b[mo]) * d) * w
    return pd.DataFrame({"y": y, "imputed": d, "mkt_uid": [f"m{i}" for i in mk], "month": mo})


def _sm_fit(df):
    X = pd.concat([df[["imputed"]], pd.get_dummies(df.mkt_uid, drop_first=True, dtype=float),
                   pd.get_dummies(df.month, prefix="t", drop_first=True, dtype=float)], axis=1)
    return sm.GLM(df.y.to_numpy(), sm.add_constant(X).to_numpy(), family=sm.families.Poisson()).fit(tol=1e-12)


# --- estimator ---------------------------------------------------------------------------------------------------

def test_recovers_coefficient_with_exact_zeros_in_outcome():
    df = _panel(n_mkt=60, n_month=60, beta=-0.4, zero_share=0.3, seed=1)
    assert (df.y == 0).mean() > 0.25
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=ONE)
    assert r.converged
    assert r.coef["imputed"] == pytest.approx(-0.4, abs=0.06)
    assert np.isfinite(r.se["market"]["imputed"])


def test_point_estimate_equals_statsmodels_dummy_glm():
    df = _panel(seed=2)
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"])
    assert r.coef["imputed"] == pytest.approx(_sm_fit(df).params[1], abs=1e-8)


def test_coefficient_is_invariant_to_outcome_scale():
    df = _panel(seed=3)
    a = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"]).coef["imputed"]
    b = ppml(df.assign(y=100 * df.y), "y", ["imputed"], fe=["mkt_uid", "month"]).coef["imputed"]
    assert a == pytest.approx(b, abs=1e-8)


def test_fixed_effect_group_with_all_zero_outcomes_is_dropped():
    df = _panel(seed=4)
    df.loc[df.mkt_uid == "m0", "y"] = 0.0  # separated: its FE would go to -inf
    kept, n_dropped = drop_separated(df, "y", ["mkt_uid", "month"])
    assert n_dropped == (df.mkt_uid == "m0").sum() and "m0" not in set(kept.mkt_uid)
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=ONE)
    assert r.n_dropped == n_dropped and r.n == len(df) - n_dropped
    ref = ppml(df[df.mkt_uid != "m0"], "y", ["imputed"], fe=["mkt_uid", "month"])
    assert r.coef["imputed"] == pytest.approx(ref.coef["imputed"], abs=1e-8)


def test_separation_drop_applies_to_every_fixed_effect_dimension():
    # market m0 sums to zero (rows 0, 1) and month 0 sums to zero (rows 0, 2); both groups go
    df = pd.DataFrame({"y": [0, 0, 0, 1, 2, 3, 4], "mkt_uid": ["m0", "m0", "m1", "m1", "m2", "m2", "m1"],
                       "month": [0, 1, 0, 1, 1, 2, 2]}, dtype=object).astype({"y": float})
    kept, n_dropped = drop_separated(df, "y", ["mkt_uid", "month"])
    assert n_dropped == 3
    assert sorted(kept.index) == [3, 4, 5, 6]


def test_collinear_fixed_effects_are_handled():
    # market x item and month x item FE in a pooled two-item model: the dummy sets are collinear per item
    a = _panel(seed=5).assign(item="a")
    b = _panel(seed=6).assign(item="b")
    df = pd.concat([a, b], ignore_index=True)
    df["mi"], df["ti"] = df.mkt_uid + df["item"], df.month.astype(str) + df["item"]
    r = ppml(df, "y", ["imputed"], fe=["mi", "ti"], clusters=ONE)
    assert r.converged and np.isfinite(r.coef["imputed"]) and np.isfinite(r.se["market"]["imputed"])


def test_regressor_absorbed_by_fixed_effects_raises():
    df = _panel(seed=7)
    df["imputed"] = (df.mkt_uid.str[1:].astype(int) % 2).astype(float)  # constant within market
    with pytest.raises(ValueError, match="absorbed"):
        ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"])


def test_integer_frequency_weights_equal_duplicated_rows():
    df = _panel(n_mkt=20, n_month=20, seed=8)
    w = np.random.default_rng(9).integers(0, 3, len(df))
    a = ppml(df.assign(w=w), "y", ["imputed"], fe=["mkt_uid", "month"], weights="w").coef["imputed"]
    dup = df.loc[df.index.repeat(w)]
    b = ppml(dup, "y", ["imputed"], fe=["mkt_uid", "month"]).coef["imputed"]
    assert a == pytest.approx(b, abs=1e-7)


# --- cluster-robust variance -------------------------------------------------------------------------------------

def test_oneway_cluster_vcov_matches_statsmodels_without_correction():
    df = _panel(seed=10)
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=ONE, correction=False)
    g = pd.factorize(df.mkt_uid)[0]
    ref = sw.cov_cluster(_sm_fit(df), g, use_correction=False)[1, 1]
    assert r.vcov["market"].loc["imputed", "imputed"] == pytest.approx(ref, rel=1e-6)


def test_twoway_cluster_vcov_matches_statsmodels_cgm_without_correction():
    df = _panel(seed=11, sd_b=0.3)
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=TWO, correction=False)
    g1, g2 = pd.factorize(df.mkt_uid)[0], df.month.to_numpy()
    ref = sw.cov_cluster_2groups(_sm_fit(df), g1, group2=g2, use_correction=False)[0][1, 1]
    assert r.vcov["twoway"].loc["imputed", "imputed"] == pytest.approx(ref, rel=1e-6)


def test_cr1_correction_is_g_over_g_minus_one_per_dimension():
    rng = np.random.default_rng(12)
    psi = rng.normal(size=(30, 2))
    g1, g2 = np.repeat(np.arange(6), 5), np.tile(np.arange(5), 6)
    raw = [cluster_vcov(psi, [g], correction=False) for g in (g1, g2, g1 * 10 + g2)]
    two = cluster_vcov(psi, [g1, g2], correction=True)
    assert two == pytest.approx(raw[0] * 6 / 5 + raw[1] * 5 / 4 - raw[2] * 30 / 29)
    assert cluster_vcov(psi, [g1], correction=True) == pytest.approx(raw[0] * 6 / 5)


def test_twoway_equals_oneway_when_second_dimension_is_unique_per_row():
    rng = np.random.default_rng(13)
    psi = rng.normal(size=(40, 1))
    g1 = np.repeat(np.arange(8), 5)
    assert cluster_vcov(psi, [g1, np.arange(40)]) == pytest.approx(cluster_vcov(psi, [g1]))


def test_twoway_is_symmetric_in_the_two_dimensions():
    rng = np.random.default_rng(14)
    psi = rng.normal(size=(30, 2))
    g1, g2 = rng.integers(0, 5, 30), rng.integers(0, 7, 30)
    assert cluster_vcov(psi, [g1, g2]) == pytest.approx(cluster_vcov(psi, [g2, g1]))


def test_twoway_se_matches_monte_carlo_sd_when_month_shocks_exist():
    """Effects vary by market and by month: one-way (market) SEs are too small, CGM two-way SEs are right."""
    est, se1, se2 = [], [], []
    for rep in range(200):
        df = _panel(n_mkt=40, n_month=40, seed=1000 + rep, sd_a=0.3, sd_b=0.45)
        r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=TWO)
        est.append(r.coef["imputed"])
        se1.append(r.se["market"]["imputed"])
        se2.append(r.se["twoway"]["imputed"])
    mc_sd = np.std(est, ddof=1)
    assert 0.85 < np.mean(se2) / mc_sd < 1.15
    assert np.mean(se1) / mc_sd < 0.85


def test_lincom_of_results_on_disjoint_clusters_adds_variances():
    a = ppml(_panel(seed=15), "y", ["imputed"], fe=["mkt_uid", "month"], clusters=ONE, correction=False)
    dfb = _panel(seed=16)
    dfb["mkt_uid"] = "b" + dfb.mkt_uid
    b = ppml(dfb, "y", ["imputed"], fe=["mkt_uid", "month"], clusters=ONE, correction=False)
    est, se = lincom([a, b], [0.5, 0.5], "imputed", ["mkt_uid"], correction=False)
    assert est == pytest.approx(0.5 * a.coef["imputed"] + 0.5 * b.coef["imputed"])
    assert se == pytest.approx(0.5 * np.sqrt(a.se["market"]["imputed"] ** 2 + b.se["market"]["imputed"] ** 2))


def test_lincom_of_one_result_reproduces_its_own_se():
    a = ppml(_panel(seed=17), "y", ["imputed"], fe=["mkt_uid", "month"], clusters=TWO)
    for name, cols in TWO.items():
        _, se = lincom([a], [1.0], "imputed", cols)
        assert se == pytest.approx(a.se[name]["imputed"])


def test_wald_statistic_known_value():
    from scipy.stats import chi2
    stat, df, p = wald(np.array([1.0, 2.0]), np.eye(2))
    assert (stat, df) == (pytest.approx(5.0), 2)
    assert p == pytest.approx(chi2.sf(5.0, 2))


# --- benchmark (methodology D3) ----------------------------------------------------------------------------------

def test_noise_benchmark_values():
    assert noise_benchmark(0.4) == pytest.approx(0.5 * np.log(0.4))
    assert noise_benchmark(0.4, power=2) == pytest.approx(np.log(0.4))
    for bad in (0.0, -0.1, 1.2, np.nan):
        assert np.isnan(noise_benchmark(bad))


@pytest.mark.parametrize("power", [1, 2])
def test_noise_removing_imputer_hits_the_benchmark_under_gaussian_changes(power):
    """True local change e ~ N(0, s_mt^2) with log-additive market and month scales; observed = e + noise with
    reliability rel; a noise-removing imputer reports e itself. PPML of |e|^power on 'imputed' -> power/2 log rel."""
    rng = np.random.default_rng(20 + power)
    n_mkt, n_month, rel = 80, 80, 0.4
    mk, mo = np.repeat(np.arange(n_mkt), n_month), np.tile(np.arange(n_month), n_mkt)
    scale = np.exp(rng.normal(0, 0.3, n_mkt)[mk] + rng.normal(0, 0.3, n_month)[mo])
    e = rng.normal(0, 1, len(mk)) * scale
    noise = rng.normal(0, np.sqrt(1 / rel - 1), len(mk)) * scale
    imp = (rng.random(len(mk)) < 0.5).astype(float)
    val = np.where(imp == 1, e, e + noise)
    df = pd.DataFrame({"y": np.abs(val) ** power, "imputed": imp, "mkt_uid": mk.astype(str), "month": mo})
    r = ppml(df, "y", ["imputed"], fe=["mkt_uid", "month"])
    assert r.coef["imputed"] == pytest.approx(noise_benchmark(rel, power=power), abs=0.04 * power)


# --- switching-market sample (which cells enter) -----------------------------------------------------------------

def _toy_panel():
    """Market A: O O O O I I I O over 2019-11..2020-06 (2020-05 has no observed change anywhere -> mu undefined);
    'Dhaka Division' and market C (imputed throughout) never enter."""
    months = pd.date_range("2019-11-01", periods=8, freq="MS")
    rows = []
    for name, uid, pattern in [("A Sadar", "a", "OOOOIIIO"), ("Dhaka Division", "dd", "OOOOIIIO"),
                               ("C Sadar", "c", "IIIIIIII")]:
        for i, (m, s) in enumerate(zip(months, pattern)):
            rows.append({"mkt_name": name, "mkt_uid": uid, "item": "rice", "month": m, "obs": s == "O",
                         "e_obs": 0.01 * (i + 1) if (s == "O" and i > 0 and pattern[i - 1] == "O") else np.nan,
                         "e_pub": np.nan if m == pd.Timestamp("2020-05-01") else -0.002 * (i + 1)})
    return pd.DataFrame(rows)


def test_classify_cells_enters_only_both_observed_and_both_imputed_changes_of_switching_markets():
    from rtfp.ppml import classify_cells

    c = classify_cells(_toy_panel(), start="2020-01-01")
    assert c.month.min() == pd.Timestamp("2020-01-01")
    a = c[c.mkt_uid == "a"].set_index("month")
    assert a.cell_type.tolist() == ["OO", "OO", "OI", "II", "II", "IO"]
    assert a.reason.tolist() == ["enters", "enters", "mixed_change", "enters", "no_outcome", "mixed_change"]
    ent = a[a.enters]
    assert ent.imputed.tolist() == [0.0, 0.0, 1.0]
    # observed cells use the survey change e_obs, imputed cells the published change e_pub
    assert ent.abs_e.tolist() == pytest.approx([0.03, 0.04, 0.012])
    assert (c[c.mkt_uid == "dd"].reason == "division_market").all()
    assert set(c[c.mkt_uid == "c"].reason) <= {"not_switching", "no_outcome"}
    assert (c[c.mkt_uid == "c"].reason == "not_switching").any()


def test_classify_cells_mixed_option_adds_imputed_after_observed_cells():
    from rtfp.ppml import classify_cells

    c = classify_cells(_toy_panel(), start="2020-01-01", mixed=True).set_index(["mkt_uid", "month"])
    row = c.loc[("a", pd.Timestamp("2020-03-01"))]
    assert row.enters and row.imputed == 1.0 and row.abs_e == pytest.approx(0.010)
    assert c.loc[("a", pd.Timestamp("2020-06-01"))].reason == "mixed_change"  # observed after imputed: never


def test_classify_cells_c_based_observed_side_uses_published_change():
    from rtfp.ppml import classify_cells

    c = classify_cells(_toy_panel(), start="2020-01-01", obs_side="c").set_index(["mkt_uid", "month"])
    assert c.loc[("a", pd.Timestamp("2020-01-01"))].abs_e == pytest.approx(0.006)


def test_classify_cells_rule_column_excludes_cells_before_switching_check():
    from rtfp.ppml import classify_cells

    p = _toy_panel()
    p["ok"] = ~((p.mkt_uid == "a") & (p.month == pd.Timestamp("2020-04-01")))
    c = classify_cells(p, start="2020-01-01", cell_ok="ok")
    a = c[c.mkt_uid == "a"].set_index("month")
    assert a.loc[pd.Timestamp("2020-04-01")].reason == "excluded_by_rule"
    # market A has no imputed cell left -> no longer a switching market
    assert a.loc[pd.Timestamp("2020-01-01")].reason == "not_switching"


# --- event windows around spell boundaries -----------------------------------------------------------------------

def _status(pattern, start="2020-01-01"):
    months = pd.date_range(start, periods=len(pattern), freq="MS")
    return pd.DataFrame({"mkt_uid": "a", "item": "rice", "month": months, "obs": [s == "O" for s in pattern]})


def test_event_windows_are_bounded_by_the_adjacent_spells():
    from rtfp.ppml import event_windows

    st = _status("OOOIIOOOOO")  # obs->imp at index 3; imp->obs at index 5
    ev = event_windows(st, window=6)
    m = st.month
    oi = ev[ev.direction == "obs_to_imp"].set_index("month").k
    assert oi.to_dict() == {m[0]: -3, m[1]: -2, m[2]: -1, m[3]: 0, m[4]: 1}
    io = ev[ev.direction == "imp_to_obs"].set_index("month").k
    assert io.to_dict() == {m[3]: -2, m[4]: -1, m[5]: 0, m[6]: 1, m[7]: 2, m[8]: 3, m[9]: 4}
    assert ev.groupby("direction").event_id.nunique().to_dict() == {"imp_to_obs": 1, "obs_to_imp": 1}
    assert not ev.balanced.any()


def test_event_windows_truncate_at_window_and_flag_balanced_events():
    from rtfp.ppml import event_windows

    st = _status("OOOOIIIIOO")
    ev = event_windows(st, window=2)
    oi = ev[ev.direction == "obs_to_imp"]
    assert sorted(oi.k) == [-2, -1, 0, 1, 2]
    assert oi.balanced.all()  # >= 2 months before and >= 3 months from the switch
    io = ev[ev.direction == "imp_to_obs"]
    assert sorted(io.k) == [-2, -1, 0, 1] and not io.balanced.any()


def test_event_windows_do_not_bridge_a_gap_in_months():
    from rtfp.ppml import event_windows

    st = _status("OOOIII").drop(index=3)  # month index 3 missing: the I spell starts after a gap
    assert event_windows(st, window=6).empty


# --- cluster-robust mean -----------------------------------------------------------------------------------------

def test_cluster_mean_with_singleton_clusters_is_the_usual_standard_error():
    from rtfp.ppml import cluster_mean

    x = np.array([1.0, 2.0, 4.0, 7.0])
    m, se = cluster_mean(x, np.arange(4))
    assert m == pytest.approx(3.5) and se == pytest.approx(x.std(ddof=1) / 2)


def test_cluster_mean_accounts_for_within_cluster_correlation():
    from rtfp.ppml import cluster_mean

    x = np.array([1.0, 1.0, 3.0, 3.0])
    _, se_c = cluster_mean(x, np.array([0, 0, 1, 1]))
    _, se_i = cluster_mean(x, np.arange(4))
    # two clusters of identical values: CR1 = sqrt(2/1 * (2^2 + 2^2) / 16) = 1
    assert se_c == pytest.approx(1.0) and se_c > se_i


# --- invariants on the real latest-release panel (skipped without data) -------------------------------------------

_PANEL = "data/interim/panel_latest.parquet"


@pytest.mark.skipif(not __import__("os").path.exists(_PANEL), reason="panel not built")
def test_real_switcher_sample_matches_direct_panel_lookups():
    from rtfp.ppml import classify_cells

    p = pd.read_parquet(_PANEL)
    p["item"] = p["item"].astype(str)
    e = classify_cells(p, start="2020-01-01").query("enters")
    look = p.set_index(["mkt_uid", "item", "month"])
    key_t = list(zip(e.mkt_uid, e["item"], e.month))
    key_p = list(zip(e.mkt_uid, e["item"], e.month - pd.offsets.MonthBegin(1)))
    obs_t, obs_p = look.obs.loc[key_t].to_numpy(), look.obs.loc[key_p].to_numpy()
    assert (obs_t == (e.imputed == 0).to_numpy()).all() and (obs_t == obs_p).all()  # OO or II only
    want = np.where(obs_t, look.e_obs.loc[key_t].abs(), look.e_pub.loc[key_t].abs())
    assert e.abs_e.to_numpy() == pytest.approx(want)
    assert not e.mkt_name.str.endswith(" Division").any() and e.month.min() >= pd.Timestamp("2020-01-01")
    g = e.groupby(["mkt_uid", "item"]).imputed.agg(["min", "max"])
    assert (g["min"] == 0).all() and (g["max"] == 1).all()
