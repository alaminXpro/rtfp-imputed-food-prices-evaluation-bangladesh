"""Tests for analysis/figures.py: pure helpers on toy data, and an end-to-end run into a temp directory."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "analysis" / "figures.py"


@pytest.fixture(scope="module")
def fig():
    spec = importlib.util.spec_from_file_location("figures_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cells(rows):
    """Toy clean hold-out cells: rows of (mkt_uid, item, stratum, n_cells)."""
    out = []
    for mkt, item, stratum, n in rows:
        for i in range(n):
            out.append({"mkt_uid": mkt, "item": item, "stratum": stratum,
                        "month": pd.Timestamp("2024-01-01") + pd.DateOffset(months=i + (0 if stratum == "forecast" else 20))})
    return pd.DataFrame(out)


def test_example_series_rule_most_clean_cells_then_most_cross_sectional(fig):
    h = _cells([("m_b", "oil", "forecast", 5), ("m_b", "oil", "cross_sectional", 1),
                ("m_a", "rice", "forecast", 4), ("m_a", "rice", "cross_sectional", 2),   # ties m_b oil on 6 total
                ("m_c", "lentils", "cross_sectional", 3),
                ("m_d", "lentils", "cross_sectional", 3)])                                # ties m_c on 3 cs
    picks = fig.pick_example_series(h)
    # (a) most clean cells: m_a rice and m_b oil tie at 6; more cross-sectional cells wins -> m_a rice
    assert (picks[0]["mkt_uid"], picks[0]["item"]) == ("m_a", "rice")
    # (b) most cross-sectional cells: m_c and m_d tie at 3 cs and 3 total; alphabetical -> m_c lentils
    assert (picks[1]["mkt_uid"], picks[1]["item"]) == ("m_c", "lentils")
    assert all("rule" in p for p in picks)


def test_example_series_rule_second_pick_differs_from_first(fig):
    h = _cells([("m_a", "rice", "cross_sectional", 9), ("m_b", "oil", "cross_sectional", 2)])
    picks = fig.pick_example_series(h)
    assert (picks[0]["mkt_uid"], picks[1]["mkt_uid"]) == ("m_a", "m_b")


def test_revision_paths_count_releases_back_from_v_final(fig):
    v = pd.to_datetime(["2025-01-06", "2025-01-20", "2025-02-03", "2025-02-17", "2025-03-03"])
    key = {"mkt_uid": "m_a", "item": "rice", "month": pd.Timestamp("2025-01-01")}
    stack = pd.DataFrame([
        {**key, "vintage": v[0], "c": 100.0, "obs": False},
        {**key, "vintage": v[1], "c": 104.0, "obs": False},
        # v[2] has no row: the item is absent from that release
        {**key, "vintage": v[3], "c": 110.0, "obs": True},   # v_truth: observed
        {**key, "vintage": v[4], "c": 111.0, "obs": True},   # after the truth arrived: ignored
    ])
    hold = pd.DataFrame([{**key, "stratum": "forecast", "v_final": v[1], "v_truth": v[3], "y_eval": 110.0}])
    p = fig.revision_paths(stack, hold).sort_values("k")
    assert p.k.tolist() == [1, 2]
    assert p.vintage.tolist() == [v[1], v[0]]
    assert np.allclose(p.err_logpct, [100 * np.log(104 / 110), 100 * np.log(100 / 110)])


def test_path_summary_drops_thin_k_and_reports_quartiles(fig):
    paths = pd.DataFrame({"item": "oil", "stratum": "forecast", "k": [1] * 5 + [2] * 2,
                          "err_logpct": [-2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 6.0]})
    s = fig.path_summary(paths, min_n=3)
    assert s.k.tolist() == [1]
    row = s.iloc[0]
    assert (row["n"], row["median"], row["q25"], row["q75"]) == (5, 0.0, -1.0, 1.0)


def test_gap_and_trust_bins(fig):
    gap = pd.Series([1, 2, 3, 4, 6, 7, 12, 13, 52, np.nan])
    got = fig.gap_bin(gap)
    assert got.iloc[:-1].astype(str).tolist() == ["1", "2", "3", "4-6", "4-6", "7-12", "7-12", "13+", "13+"]
    assert pd.isna(got.iloc[-1])  # no earlier survey value: gap undefined, excluded from the gap panel
    trust = pd.Series([5.8, 6.0, 7.9, 8.0, 9.9, 10.0])
    assert fig.trust_bin(trust).astype(str).tolist() == ["<6", "6.0-7.9", "6.0-7.9", "8.0-9.9", "8.0-9.9", "10"]


def test_wilson_interval_matches_known_value(fig):
    lo, hi = fig.wilson(0.5, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


DATA_READY = all((ROOT / p).exists() for p in [
    "data/interim/holdout.parquet", "data/interim/change_pairs.parquet", "data/interim/vintages.parquet",
    "data/interim/panel_latest.parquet", "outputs/tables/T2_levels.csv", "outputs/tables/T3_local_changes.csv",
    "outputs/tables/T4b_error_by_trust.csv", "outputs/tables/T7_uncertainty.csv",
    "outputs/tables/T7b_coverage_by_month.csv"])


def _mtimes(dirs):
    return {str(p): p.stat().st_mtime_ns for d in dirs for p in (ROOT / d).glob("*") if p.is_file()}


@pytest.mark.skipif(not DATA_READY, reason="interim data or tables not built")
def test_figures_script_writes_every_expected_file(tmp_path, fig):
    fig_dir, notes = tmp_path / "figures", tmp_path / "notes" / "FIGURES.md"
    before = _mtimes(["data/interim", "outputs/tables"])
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--fig-dir", str(fig_dir), "--notes", str(notes),
                        "--boot", "20"], cwd=ROOT, capture_output=True, text=True, timeout=900,
                       env={**os.environ, "MPLBACKEND": "Agg"})
    assert r.returncode == 0, r.stderr[-3000:]
    expected = [fig_dir / f"{stem}.{ext}" for stem in fig.FIGURE_STEMS for ext in ("png", "pdf", "csv")]
    missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
    assert not missing, missing
    assert notes.exists() and notes.stat().st_size > 0
    text = notes.read_text()
    for i, stem in enumerate(fig.FIGURE_STEMS, start=1):
        sec = text.split(f"## Figure {i}")[1].split("## Figure")[0]
        assert stem in sec and "**Source.**" in sec and "**N.**" in sec, stem
    # PNGs are 300 dpi
    from PIL import Image
    for stem in fig.FIGURE_STEMS:
        dpi = Image.open(fig_dir / f"{stem}.png").info.get("dpi")
        assert dpi is not None and round(dpi[0]) == 300, (stem, dpi)
    # inputs are read-only
    assert _mtimes(["data/interim", "outputs/tables"]) == before
