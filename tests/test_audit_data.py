"""Re-checks the Task 1.2 gate on whatever latest vintage is present. Skipped without data."""
import glob
import os

import pandas as pd
import pytest

from rtfp.audit import HDX_COMMODITY, audit_cells, match_markets

PANEL = "data/interim/panel_latest.parquet"
pytestmark = pytest.mark.skipif(not os.path.exists(PANEL), reason="panel not built")


def test_plain_columns_equal_hdx_retail_values_one_to_one():
    p = pd.read_parquet(PANEL)
    from rtfp.download import hdx_file

    h = pd.read_csv(hdx_file("data/raw/hdx_wfp"))
    a = audit_cells(p, h, match_markets(p, h))
    obs = a[a.obs]
    assert (obs.status == "match").mean() >= 0.90  # registered gate
    assert (a.status == "hdx_only").sum() == 0
    hw = h[(h.pricetype == "Retail") & h.commodity.isin(HDX_COMMODITY)]
    hw = hw[pd.to_datetime(hw.date).between(p.month.min(), p.month.max() + pd.offsets.MonthEnd(1))]
    assert len(hw) == len(obs)
