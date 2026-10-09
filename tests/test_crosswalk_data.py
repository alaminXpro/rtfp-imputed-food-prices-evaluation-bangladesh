"""Assertions on the real RTFP vintages. Skipped when the raw data are not present."""
import os

import pytest

from rtfp.crosswalk import build_crosswalk, load_market_rows

VDIR = "data/raw/rtfp_vintages"
pytestmark = pytest.mark.skipif(not os.path.isdir(VDIR), reason="raw vintages not downloaded")


@pytest.fixture(scope="module")
def xw():
    return build_crosswalk(load_market_rows(VDIR))


def test_every_vintage_resolves_without_error_and_has_one_aggregate_row(xw):
    per_v = xw.groupby("vintage").is_aggregate.sum()
    assert len(per_v) == 83 and (per_v == 1).all()


def test_latest_vintage_has_110_real_markets(xw):
    latest = xw[xw.vintage == xw.vintage.max()]
    assert (~latest.is_aggregate).sum() == 110


def test_known_rekeyings_map_one_to_one(xw):
    for old, new in [("2024-03-11", "2024-03-18"), ("2025-03-24", "2025-03-31")]:
        a = xw[(xw.vintage == old) & ~xw.is_aggregate].set_index("mkt_uid").geo_id
        b = xw[(xw.vintage == new) & ~xw.is_aggregate].set_index("mkt_uid").geo_id
        assert set(a.index) == set(b.index)
        assert (a != b.reindex(a.index)).mean() > 0.9  # ids really changed
