"""Invariants of the latest-vintage panel against the raw CSV. Skipped without raw data."""
import os

import numpy as np
import pandas as pd
import pytest

from rtfp.download import latest_rtfp_csv
from rtfp.panel import ITEMS, load_panel

VDIR = "data/raw/rtfp_vintages"
pytestmark = pytest.mark.skipif(not os.path.isdir(VDIR), reason="raw vintages not downloaded")


@pytest.fixture(scope="module")
def raw_and_panel():
    src = latest_rtfp_csv(VDIR)
    raw = pd.read_csv(src)
    return raw[raw.mkt_name != "Market Average"], load_panel(str(src))


def test_observed_counts_match_raw_csv(raw_and_panel):
    raw, p = raw_and_panel
    for it in ITEMS:
        assert p[(p.item == it)].obs.sum() == raw[it].notna().sum(), it


def test_published_values_carried_exactly(raw_and_panel):
    raw, p = raw_and_panel
    r = raw.assign(month=pd.to_datetime(raw.price_date)).set_index(["mkt_name", "month"])
    for it in ITEMS:
        q = p[p.item == it].set_index(["mkt_name", "month"])
        assert np.allclose(q.c, r.loc[q.index, f"c_{it}"], equal_nan=True), it
        assert np.allclose(q.y, r.loc[q.index, it], equal_nan=True), it


def test_balanced_panel_110_markets(raw_and_panel):
    _, p = raw_and_panel
    assert p.mkt_uid.nunique() == 110
    assert p.groupby(["item", "mkt_uid"]).size().nunique() == 1


def test_distance_zero_iff_observed(raw_and_panel):
    _, p = raw_and_panel
    has = p.dist_obs_km.notna()
    assert ((p.dist_obs_km[has] == 0) == p.obs[has]).all()


def test_observed_change_only_when_both_months_observed(raw_and_panel):
    _, p = raw_and_panel
    assert not (p.dlog_obs.notna() & ~p.obs).any()
