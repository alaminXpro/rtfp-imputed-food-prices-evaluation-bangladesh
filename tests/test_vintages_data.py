"""Invariants of the real hold-out, re-checked by direct lookups in the stack. Skipped without data."""
import os

import numpy as np
import pandas as pd
import pytest

D = "data/interim"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{D}/holdout.parquet"), reason="hold-out not built")
KEY = ["mkt_uid", "month", "item"]


@pytest.fixture(scope="module")
def data():
    s = pd.read_parquet(f"{D}/vintages.parquet")
    s["item"] = s["item"].astype(str)
    h = pd.read_parquet(f"{D}/holdout.parquet")
    h["item"] = h["item"].astype(str)
    return s.set_index(["vintage"] + KEY).sort_index(), h


def test_stack_has_83_vintages_and_110_markets(data):
    s, _ = data
    assert s.index.get_level_values("vintage").nunique() == 83
    assert s.index.get_level_values("mkt_uid").nunique() == 110


def test_every_holdout_cell_is_imputed_at_v_final_and_observed_at_v_truth(data):
    s, h = data
    f = s.loc[list(zip(h.v_final, h.mkt_uid, h.month, h.item))]
    t = s.loc[list(zip(h.v_truth, h.mkt_uid, h.month, h.item))]
    assert (~f.obs.to_numpy()).all() and f.c.notna().all()
    assert t.obs.to_numpy().all()
    assert np.allclose(t.y.to_numpy(), h.y_truth.to_numpy())
    assert np.allclose(f.c.to_numpy(), h.c_final.to_numpy())
    assert (h.v_final < h.v_truth).all() and (h.v_first <= h.v_final).all()


def test_no_imputed_vintage_strictly_between_v_final_and_v_truth(data):
    s, h = data
    sr = s.reset_index()
    m = sr.merge(h[KEY + ["v_final", "v_truth"]], on=KEY)
    between = m[(m.vintage > m.v_final) & (m.vintage < m.v_truth)]
    assert between.obs.sum() == 0  # nothing observed in between (v_truth is the first observation)
    assert (~between.obs & between.c.notna()).sum() == 0  # and nothing imputed (v_final is the last)


def test_stratum_rules_hold(data):
    s, h = data
    sr = s.reset_index()
    m = sr.merge(h[KEY + ["v_final", "stratum"]], on=KEY)
    obs_before = m[(m.vintage < m.v_final) & m.obs].groupby(KEY).size().rename("n").reset_index()
    hh = h.merge(obs_before, on=KEY, how="left")
    assert (hh[hh.stratum == "toggle"].n > 0).all()
    assert hh[hh.stratum != "toggle"].n.isna().all()
    assert (h[h.stratum == "forecast"].n_obs_month_final == 0).all()
    assert (h[h.stratum == "cross_sectional"].n_obs_month_final > 0).all()


def test_evaluation_truth_equals_latest_vintage_and_hdx_feed(data):
    s, h = data
    latest = s.index.get_level_values("vintage").max()
    conf = h[h.y_eval.notna()]
    lv = s.loc[list(zip([latest] * len(conf), conf.mkt_uid, conf.month, conf.item))]
    assert lv.obs.to_numpy().all() and np.allclose(lv.y.to_numpy(), conf.y_eval.to_numpy())
    assert (h[h.y_eval.isna()].stratum.isin(["toggle", "unconfirmed"])).all()


def test_clean_strata_are_recent_arrivals(data):
    _, h = data
    clean = h[h.stratum.isin(["forecast", "cross_sectional"])]
    assert (clean.lag_truth_m <= 6).all()
    assert clean.y_eval.notna().all()
