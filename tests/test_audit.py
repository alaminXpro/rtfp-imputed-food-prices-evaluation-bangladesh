import numpy as np
import pandas as pd

from rtfp.audit import HDX_COMMODITY, audit_cells, match_markets


def _panel(rows):
    return pd.DataFrame(rows, columns=["mkt_uid", "mkt_name", "lat", "lon", "month", "item", "y", "c", "obs"]).assign(
        month=lambda d: pd.to_datetime(d.month))


def _hdx(rows):
    return pd.DataFrame(rows, columns=["market", "market_id", "latitude", "longitude", "date", "commodity",
                                       "pricetype", "priceflag", "price"])


def test_commodity_map_covers_the_four_items():
    assert set(HDX_COMMODITY.values()) == {"rice", "wheat_flour", "oil", "lentils"}


def test_markets_match_by_name_then_by_coordinates():
    pm = pd.DataFrame({"mkt_uid": ["m_a", "m_b", "m_c"], "mkt_name": ["A", "B old", "C"],
                       "lat": [23.0, 24.0, 25.0], "lon": [90.0, 90.0, 90.0]})
    hm = pd.DataFrame({"market": ["A", "B new", "Z"], "market_id": [1, 2, 3],
                       "latitude": [23.0, 24.01, 21.0], "longitude": [90.0, 90.0, 92.0]})
    m = match_markets(pm, hm, max_km=5).set_index("mkt_uid")
    assert m.loc["m_a", "market_id"] == 1 and m.loc["m_a", "how"] == "name"
    assert m.loc["m_b", "market_id"] == 2 and m.loc["m_b", "how"] == "coords"
    assert pd.isna(m.loc["m_c", "market_id"]) and m.loc["m_c", "how"] == "none"


def test_cell_status_match_mismatch_and_missing():
    p = _panel([["m_a", "A", 23, 90, "2024-01-01", "rice", 50.0, 50.0, True],
                ["m_a", "A", 23, 90, "2024-02-01", "rice", 50.0, 50.0, True],
                ["m_a", "A", 23, 90, "2024-03-01", "rice", 50.0, 50.0, True],
                ["m_a", "A", 23, 90, "2024-04-01", "rice", np.nan, 50.0, False]])
    h = _hdx([["A", 1, 23, 90, "2024-01-15", "Rice (coarse)", "Retail", "actual", 50.2],
              ["A", 1, 23, 90, "2024-02-15", "Rice (coarse)", "Retail", "aggregate", 60.0],
              ["A", 1, 23, 90, "2024-04-15", "Rice (coarse)", "Retail", "actual", 55.0],
              ["A", 1, 23, 90, "2024-01-15", "Rice (coarse)", "Wholesale", "actual", 4000.0]])
    mm = pd.DataFrame({"mkt_uid": ["m_a"], "market_id": [1], "how": ["name"]})
    a = audit_cells(p, h, mm, tol=0.01).set_index("month")
    assert a.loc["2024-01-01", "status"] == "match"
    assert a.loc["2024-01-01", "priceflag"] == "actual"
    assert a.loc["2024-02-01", "status"] == "value_mismatch"
    assert a.loc["2024-03-01", "status"] == "no_hdx_row"
    assert a.loc["2024-04-01", "status"] == "hdx_only"  # in HDX but not observed in RTFP
