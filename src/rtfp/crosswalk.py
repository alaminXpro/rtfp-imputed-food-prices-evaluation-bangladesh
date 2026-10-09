"""Stable market identifiers across RTFP vintages.

geo_id is re-keyed between vintages (e.g. 2024-03-18, 2025-03-31), while market names and
coordinates are stable. A market's stable id (`mkt_uid`) is therefore derived from its name, and
we fail loudly if the same name ever points to locations further apart than `max_drift_km`.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

AGGREGATE_NAMES = {"Market Average"}
VINTAGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.csv$")
MARKET_COLS = ["mkt_name", "adm1_name", "adm2_name", "lat", "lon", "geo_id"]


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _uid(name: str) -> str:
    return "m_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def build_crosswalk(market_rows: pd.DataFrame, max_drift_km: float = 3.0) -> pd.DataFrame:
    """One row per (vintage, geo_id) with a stable `mkt_uid` and an `is_aggregate` flag.

    Raises ValueError if a market name spans locations more than `max_drift_km` apart.
    """
    xw = market_rows[["vintage"] + MARKET_COLS].drop_duplicates(["vintage", "geo_id"]).copy()
    xw["is_aggregate"] = xw.mkt_name.isin(AGGREGATE_NAMES)
    xw["mkt_uid"] = xw.mkt_name.map(_uid)

    real = xw[~xw.is_aggregate].dropna(subset=["lat", "lon"])
    for name, g in real.groupby("mkt_name"):
        pts = g[["lat", "lon"]].drop_duplicates().to_numpy()
        if len(pts) < 2:
            continue
        d = _haversine_km(pts[:, None, 0], pts[:, None, 1], pts[None, :, 0], pts[None, :, 1])
        if d.max() > max_drift_km:
            raise ValueError(f"market name {name!r} spans {d.max():.1f} km across vintages")
    if xw.groupby("vintage").mkt_uid.apply(lambda s: s.duplicated().any()).any():
        raise ValueError("two geo_ids share one market name within a vintage")
    return xw.sort_values(["vintage", "mkt_uid"]).reset_index(drop=True)


def vintage_of(path: str) -> str:
    return VINTAGE_RE.search(os.path.basename(path)).group(1)


def load_market_rows(vintage_dir: str) -> pd.DataFrame:
    """Distinct market rows of every vintage CSV under `vintage_dir`."""
    frames = []
    for f in glob.glob(os.path.join(vintage_dir, "BGD_RTFP_mkt_*", "*.csv")):
        d = pd.read_csv(f, usecols=MARKET_COLS).drop_duplicates()
        frames.append(d.assign(vintage=vintage_of(f)))
    return pd.concat(frames, ignore_index=True)
