import pandas as pd
import pytest

from rtfp.crosswalk import AGGREGATE_NAMES, build_crosswalk


def _vintage(v, rows):
    return pd.DataFrame(rows, columns=["mkt_name", "adm1_name", "adm2_name", "lat", "lon", "geo_id"]).assign(vintage=v)


def test_rekeyed_geo_ids_map_to_one_stable_uid():
    a = _vintage("2024-03-11", [["Bogra Sadar", "Rajshahi", "Bogra", 24.85, 89.37, "gid_1"]])
    b = _vintage("2024-03-18", [["Bogra Sadar", "Rajshahi", "Bogra", 24.851, 89.372, "gid_999"]])
    xw = build_crosswalk(pd.concat([a, b]))
    assert xw.loc[xw.geo_id == "gid_1", "mkt_uid"].item() == xw.loc[xw.geo_id == "gid_999", "mkt_uid"].item()


def test_distinct_markets_get_distinct_uids():
    a = _vintage("2024-03-11", [["Bogra Sadar", "Rajshahi", "Bogra", 24.85, 89.37, "gid_1"],
                                ["Dhaka Sadar", "Dhaka", "Dhaka", 23.81, 90.41, "gid_2"]])
    xw = build_crosswalk(a)
    assert xw.mkt_uid.nunique() == 2


def test_aggregate_row_is_flagged_not_treated_as_market():
    a = _vintage("2024-03-11", [["Market Average", None, None, float("nan"), float("nan"), "gid_avg"],
                                ["Dhaka Sadar", "Dhaka", "Dhaka", 23.81, 90.41, "gid_2"]])
    xw = build_crosswalk(a)
    assert "Market Average" in AGGREGATE_NAMES
    assert xw.loc[xw.mkt_name == "Market Average", "is_aggregate"].item()
    assert not xw.loc[xw.mkt_name == "Dhaka Sadar", "is_aggregate"].item()


def test_same_name_far_apart_raises_instead_of_silently_merging():
    a = _vintage("2024-03-11", [["Sadar Bazar", "Dhaka", "Dhaka", 23.81, 90.41, "gid_1"]])
    b = _vintage("2024-03-18", [["Sadar Bazar", "Dhaka", "Dhaka", 22.30, 91.80, "gid_2"]])
    with pytest.raises(ValueError, match="Sadar Bazar"):
        build_crosswalk(pd.concat([a, b]), max_drift_km=5)


def test_one_row_per_vintage_and_geo_id():
    a = _vintage("2024-03-11", [["Bogra Sadar", "Rajshahi", "Bogra", 24.85, 89.37, "gid_1"]])
    xw = build_crosswalk(pd.concat([a, a]))
    assert len(xw) == 1


def test_drop_mode_removes_only_the_market_that_jumps_and_reports_it():
    a = _vintage("2024-03-11", [["Hola", "X", "Y", 1.0, 40.0, "g1"], ["Garissa", "X", "Z", -0.45, 39.6, "g2"]])
    b = _vintage("2024-03-18", [["Hola", "X", "Y", -2.0, 40.0, "g3"], ["Garissa", "X", "Z", -0.45, 39.6, "g4"]])
    xw = build_crosswalk(pd.concat([a, b]), max_drift_km=3, on_drift="drop")
    assert set(xw.mkt_name) == {"Garissa"}
    assert xw.attrs["dropped_markets"] == ["Hola"]
