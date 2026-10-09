import numpy as np
import pandas as pd
import pytest

from rtfp.panel import ITEMS, build_panel

MONTHS = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])


def _wide(markets):
    """markets: {name: (lat, lon, {item: [(y, c), ...per month]})}; items not given are all-missing."""
    rows = []
    for name, (lat, lon, series) in markets.items():
        for i, mth in enumerate(MONTHS):
            r = {"mkt_name": name, "adm1_name": "A", "adm2_name": "B", "lat": lat, "lon": lon,
                 "geo_id": "g_" + name, "price_date": mth.strftime("%Y-%m-%d"),
                 "data_coverage": 50.0, "data_coverage_recent": 50.0}
            for it in ITEMS:
                y, c = series.get(it, [(np.nan, 100.0)] * len(MONTHS))[i]
                r.update({it: y, f"c_{it}": c, f"o_{it}": c, f"h_{it}": c, f"l_{it}": c, f"trust_{it}": 8})
            rows.append(r)
    return pd.DataFrame(rows)


def _panel(markets):
    return build_panel(_wide(markets)).set_index(["mkt_uid", "month", "item"]).sort_index()


def test_key_is_unique_and_aggregate_row_dropped():
    w = _wide({"A": (23.0, 90.0, {}), "Market Average": (np.nan, np.nan, {})})
    p = build_panel(w)
    assert not p.duplicated(["mkt_uid", "month", "item"]).any()
    assert "Market Average" not in set(p.mkt_name)


def test_observed_change_uses_survey_value_not_published_estimate():
    s = {"rice": [(100, 100), (110, 105), (np.nan, 105), (np.nan, 105)]}
    p = _panel({"A": (23.0, 90.0, s)})
    row = p.loc[("m_a", MONTHS[1], "rice")]
    assert row.dlog_obs == pytest.approx(np.log(110 / 100))
    assert row.dlog_pub == pytest.approx(np.log(105 / 100))


def test_observed_change_needs_both_months_observed():
    s = {"rice": [(100, 100), (np.nan, 101), (120, 120), (130, 130)]}
    p = _panel({"A": (23.0, 90.0, s)})
    assert np.isnan(p.loc[("m_a", MONTHS[2], "rice")].dlog_obs)
    assert p.loc[("m_a", MONTHS[3], "rice")].dlog_obs == pytest.approx(np.log(130 / 120))


def test_month_mean_uses_observed_cells_only():
    obs = {"rice": [(100, 100), (110, 110), (np.nan, 110), (np.nan, 110)]}
    imp = {"rice": [(np.nan, 100), (np.nan, 200), (np.nan, 200), (np.nan, 200)]}  # huge imputed jump
    p = _panel({"A": (23.0, 90.0, obs), "B": (23.1, 90.1, imp)})
    assert p.loc[("m_a", MONTHS[1], "rice")].mu_obs == pytest.approx(np.log(1.10))
    assert p.loc[("m_b", MONTHS[1], "rice")].e_pub == pytest.approx(np.log(2.0) - np.log(1.10))


def test_distance_zero_iff_observed_and_nan_when_nothing_observed():
    obs = {"rice": [(100, 100), (101, 101), (np.nan, 101), (np.nan, 101)]}
    imp = {"rice": [(np.nan, 100)] * 4}
    p = _panel({"A": (23.0, 90.0, obs), "B": (23.0, 91.0, imp)})
    assert p.loc[("m_a", MONTHS[0], "rice")].dist_obs_km == 0
    d = p.loc[("m_b", MONTHS[0], "rice")].dist_obs_km
    assert 95 < d < 110  # ~102 km per degree of longitude at 23N
    assert np.isnan(p.loc[("m_b", MONTHS[2], "rice")].dist_obs_km)
    assert p.loc[("m_b", MONTHS[0], "rice")].n_obs_month == 1


def test_gap_counts_months_since_last_observation():
    s = {"rice": [(np.nan, 100), (100, 100), (np.nan, 100), (np.nan, 100)]}
    p = _panel({"A": (23.0, 90.0, s)})
    assert np.isnan(p.loc[("m_a", MONTHS[0], "rice")].gap)  # never observed before
    assert p.loc[("m_a", MONTHS[1], "rice")].gap == 0
    assert p.loc[("m_a", MONTHS[3], "rice")].gap == 2
