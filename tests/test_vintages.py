"""Hold-out definition (methodology M1). Toy stacks: one row per (vintage, market, month, item)."""
import numpy as np
import pandas as pd
import pytest

from rtfp.vintages import build_change_pairs, build_holdout

V = pd.to_datetime(["2025-01-10", "2025-02-10", "2025-03-10"])
T = pd.Timestamp("2024-12-01")


def _row(v, m, y, c, t=T, item="rice", n_obs=1):
    return {"vintage": v, "mkt_uid": m, "month": t, "item": item, "y": y, "c": c, "h": c, "l": c,
            "trust": 8, "gap": 0, "dist_obs_km": 0.0 if not np.isnan(y) else 10.0, "n_obs_month": n_obs,
            "obs": not np.isnan(y)}


def _stack(rows):
    return pd.DataFrame(rows)


def _h(rows):
    return build_holdout(_stack(rows)).set_index(["mkt_uid", "month", "item"])


def test_imputed_then_observed_enters_holdout_with_final_and_first_values():
    h = _h([_row(V[0], "m_a", np.nan, 100.0), _row(V[1], "m_a", np.nan, 105.0), _row(V[2], "m_a", 110.0, 110.0)])
    r = h.loc[("m_a", T, "rice")]
    assert r.c_first == 100.0 and r.c_final == 105.0 and r.y_truth == 110.0
    assert r.v_final == V[1] and r.v_truth == V[2]


def test_observed_imputed_observed_is_toggle():
    h = _h([_row(V[0], "m_a", 100.0, 100.0), _row(V[1], "m_a", np.nan, 100.0), _row(V[2], "m_a", 100.0, 100.0)])
    assert h.loc[("m_a", T, "rice")].stratum == "toggle"


def test_no_market_observed_in_imputing_vintage_is_forecast():
    h = _h([_row(V[0], "m_a", np.nan, 100.0, n_obs=0), _row(V[1], "m_a", 101.0, 101.0)])
    assert h.loc[("m_a", T, "rice")].stratum == "forecast"


def test_other_markets_observed_in_imputing_vintage_is_cross_sectional():
    h = _h([_row(V[0], "m_a", np.nan, 100.0, n_obs=5), _row(V[1], "m_a", 101.0, 101.0)])
    assert h.loc[("m_a", T, "rice")].stratum == "cross_sectional"


def test_toggle_takes_precedence_over_forecast():
    h = _h([_row(V[0], "m_a", 100.0, 100.0, n_obs=1), _row(V[1], "m_a", np.nan, 100.0, n_obs=0),
            _row(V[2], "m_a", 100.0, 100.0)])
    assert h.loc[("m_a", T, "rice")].stratum == "toggle"


def test_never_observed_and_always_observed_cells_are_excluded():
    h = _h([_row(V[0], "m_a", np.nan, 100.0), _row(V[1], "m_a", np.nan, 100.0),
            _row(V[0], "m_b", 90.0, 90.0), _row(V[1], "m_b", 90.0, 90.0)])
    assert len(h) == 0


def test_vintage_without_the_item_does_not_count():
    # rice absent from V[1]: the cell is imputed in V[0] and observed in V[2]; V[1] simply has no row.
    h = _h([_row(V[0], "m_a", np.nan, 100.0), _row(V[2], "m_a", 100.0, 100.0)])
    r = h.loc[("m_a", T, "rice")]
    assert r.v_final == V[0] and r.stratum == "cross_sectional"


def test_change_pair_types():
    t0, t1 = pd.Timestamp("2024-11-01"), T
    rows = [
        # market a: t1 imputed in V[0] and observed in V[1]; t0 imputed in V[0] -> both_imputed
        _row(V[0], "m_a", np.nan, 100.0, t=t0), _row(V[0], "m_a", np.nan, 102.0, t=t1),
        _row(V[1], "m_a", 99.0, 99.0, t=t0), _row(V[1], "m_a", 103.0, 103.0, t=t1),
        # market b: t1 imputed in V[0], t0 observed in V[0] -> imputed_t_observed_prev
        _row(V[0], "m_b", 80.0, 80.0, t=t0), _row(V[0], "m_b", np.nan, 81.0, t=t1),
        _row(V[1], "m_b", 80.0, 80.0, t=t0), _row(V[1], "m_b", 84.0, 84.0, t=t1),
    ]
    stack = _stack(rows)
    pairs = build_change_pairs(build_holdout(stack), stack).set_index("mkt_uid")
    assert pairs.loc["m_a", "pair_type"] == "both_imputed"
    assert pairs.loc["m_a", "dlog_pub"] == pytest.approx(np.log(102 / 100))
    assert pairs.loc["m_a", "dlog_truth"] == pytest.approx(np.log(103 / 99))
    assert pairs.loc["m_b", "pair_type"] == "imputed_t_observed_prev"


def test_change_pair_needs_previous_month_in_the_same_vintage():
    rows = [_row(V[0], "m_a", np.nan, 102.0), _row(V[1], "m_a", 103.0, 103.0)]  # no t-1 row in V[0]
    stack = _stack(rows)
    assert len(build_change_pairs(build_holdout(stack), stack)) == 0


def test_truth_not_observed_in_latest_vintage_is_unconfirmed():
    h = _h([_row(V[0], "m_a", np.nan, 100.0), _row(V[1], "m_a", 101.0, 101.0), _row(V[2], "m_a", np.nan, 101.0)])
    assert h.loc[("m_a", T, "rice")].stratum == "unconfirmed"


def test_late_arrival_beyond_max_lag_is_backfill():
    old = pd.Timestamp("2023-06-01")  # 19 months before V[1]
    h = _h([_row(V[0], "m_a", np.nan, 100.0, t=old, n_obs=3), _row(V[1], "m_a", 101.0, 101.0, t=old)])
    assert h.loc[("m_a", old, "rice")].stratum == "backfill"


def test_evaluation_truth_is_the_latest_value():
    h = _h([_row(V[0], "m_a", np.nan, 100.0), _row(V[1], "m_a", 101.0, 101.0), _row(V[2], "m_a", 102.0, 102.0)])
    r = h.loc[("m_a", T, "rice")]
    assert r.y_truth == 101.0 and r.y_eval == 102.0 and r.truth_revised


def test_truth_comes_from_latest_vintage_that_contains_the_item():
    # rice imputed in V0, observed in V1; V2 (overall latest) has no rice rows but has wheat flour.
    rows = [_row(V[0], "m_a", np.nan, 100.0), _row(V[1], "m_a", 101.0, 101.0),
            _row(V[2], "m_a", 50.0, 50.0, item="wheat_flour")]
    h = _h(rows)
    r = h.loc[("m_a", T, "rice")]
    assert r.stratum != "unconfirmed" and r.y_eval == 101.0


def test_build_stack_refuses_a_market_name_that_jumps_location(tmp_path):
    from rtfp.vintages import build_stack

    def write(v, lat):
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        row = {"mkt_name": "X Sadar", "adm1_name": "A", "adm2_name": "B", "lat": lat, "lon": 90.0, "geo_id": "g",
               "price_date": "2024-12-01", "data_coverage": 1.0, "data_coverage_recent": 1.0}
        for it in ["rice", "wheat_flour", "oil", "lentils"]:
            row.update({it: 1.0, f"c_{it}": 1.0, f"o_{it}": 1.0, f"h_{it}": 1.0, f"l_{it}": 1.0, f"trust_{it}": 8})
        pd.DataFrame([row]).to_csv(d / f"BGD_RTFP_mkt_2007_{v}.csv", index=False)

    write("2025-01-10", 23.0)
    write("2025-02-10", 24.0)  # ~111 km away
    with pytest.raises(ValueError, match="X Sadar"):
        build_stack(str(tmp_path))


def test_build_stack_drop_mode_excludes_the_jumping_market_rows(tmp_path):
    from rtfp.vintages import build_stack

    def write(v, rows):
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        out = []
        for name, lat in rows:
            r = {"mkt_name": name, "adm1_name": "A", "adm2_name": "B", "lat": lat, "lon": 90.0, "geo_id": "g" + name,
                 "price_date": "2024-12-01", "data_coverage": 1.0, "data_coverage_recent": 1.0}
            for it in ["rice", "wheat_flour", "oil", "lentils"]:
                r.update({it: 1.0, f"c_{it}": 1.0, f"o_{it}": 1.0, f"h_{it}": 1.0, f"l_{it}": 1.0, f"trust_{it}": 8})
            out.append(r)
        pd.DataFrame(out).to_csv(d / f"BGD_RTFP_mkt_2007_{v}.csv", index=False)

    write("2025-01-10", [("Jumper", 23.0), ("Stayer", 22.0)])
    write("2025-02-10", [("Jumper", 25.0), ("Stayer", 22.0)])
    s = build_stack(str(tmp_path), on_drift="drop")
    assert set(s.mkt_uid) == {"m_stayer"}
    assert s.attrs["dropped_markets"] == ["Jumper"]


def test_dropped_market_does_not_enter_month_means(tmp_path):
    from rtfp.vintages import build_stack

    def write(v, rows):
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        out = []
        for name, lat, months in rows:
            for mth, y in months:
                r = {"mkt_name": name, "adm1_name": "A", "adm2_name": "B", "lat": lat, "lon": 90.0, "geo_id": "g" + name,
                     "price_date": mth, "data_coverage": 1.0, "data_coverage_recent": 1.0}
                for it in ["rice", "wheat_flour", "oil", "lentils"]:
                    r.update({it: y, f"c_{it}": y, f"o_{it}": y, f"h_{it}": y, f"l_{it}": y, f"trust_{it}": 8})
                out.append(r)
        pd.DataFrame(out).to_csv(d / f"BGD_RTFP_mkt_2007_{v}.csv", index=False)

    months_j = [("2024-11-01", 100.0), ("2024-12-01", 200.0)]   # jumper doubles its price
    months_s = [("2024-11-01", 100.0), ("2024-12-01", 110.0)]
    write("2025-01-10", [("Jumper", 23.0, months_j), ("Stayer", 22.0, months_s)])
    write("2025-02-10", [("Jumper", 25.0, months_j), ("Stayer", 22.0, months_s)])
    s = build_stack(str(tmp_path), on_drift="drop")
    dec = s[(s.month == "2024-12-01") & (s["item"] == "rice")]
    assert np.allclose(dec.mu_obs, np.log(1.1))  # only the stayer's change, not the jumper's log(2)
    assert (dec.n_obs_month == 1).all()


def test_build_stack_ignores_vintages_after_as_of(tmp_path, monkeypatch):
    from rtfp.vintages import build_stack

    def write(v):
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        r = {"mkt_name": "S", "adm1_name": "A", "adm2_name": "B", "lat": 22.0, "lon": 90.0, "geo_id": "g",
             "price_date": "2024-12-01", "data_coverage": 1.0, "data_coverage_recent": 1.0}
        for it in ["rice", "wheat_flour", "oil", "lentils"]:
            r.update({it: 1.0, f"c_{it}": 1.0, f"o_{it}": 1.0, f"h_{it}": 1.0, f"l_{it}": 1.0, f"trust_{it}": 8})
        pd.DataFrame([r]).to_csv(d / f"BGD_RTFP_mkt_2007_{v}.csv", index=False)

    for v in ["2025-01-10", "2025-02-10", "2025-03-10"]:
        write(v)
    monkeypatch.setenv("RTFP_AS_OF", "2025-02-10")
    s = build_stack(str(tmp_path))
    assert s.vintage.max() == pd.Timestamp("2025-02-10") and s.vintage.nunique() == 2
