"""Bounded replication of the Bangladesh pipeline in other RTFP countries (Task 6.1; analysis plan section 5;
methodology D6).

Step 1 (selection, before any country's results are computed):
    candidates   per-country RTFP market datasets ``{ISO3}_2021_RTFP_v02_M`` listed in the all-country catalogue
                 metadata (WLD_2021_RTFP_v02_M) or found by the catalogue search, other than Bangladesh
    rule         >= 50 archived market-file vintages (``*_RTFP_mkt_*``), >= 30 markets in the latest vintage
                 (excluding 'Market Average'), plain survey columns present; top five by markets, ties broken
                 by number of vintages (then ISO3, for determinism)
    'plain survey columns present' = at least one item has both a plain and a ``c_`` column in the latest
                 vintage and at least one plain value is non-missing (a header with no values carries no
                 survey values; both flags are recorded).

Step 2 (per selected country) re-uses the Bangladesh modules unchanged: rtfp.vintages (stack, hold-out,
change pairs), rtfp.noise (reliability), rtfp.metrics (levels, local changes, spikes), rtfp.audit (raw
columns vs the WFP/HDX feed).
"""
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.crosswalk import AGGREGATE_NAMES, MARKET_COLS, build_crosswalk, vintage_of
from rtfp.download import RTFP_LISTING, parse_file_listing, sha256_file
from rtfp.metrics import (level_mz, level_stats, local_components, persistent_spikes, pigeonhole_weights,
                          weighted_local_stats)
from rtfp.noise import autocovariances, noise_from_autocov
from rtfp.vintages import build_change_pairs, build_holdout, build_stack, latest_rows

STUDY_RE = re.compile(r"^([A-Z]{3})_2021_RTFP_v02_M$")
MIN_VINTAGES, MIN_MARKETS, N_SELECT = 50, 30, 5
EXCLUDE = ("BGD",)
WLD_METADATA = "https://microdata.worldbank.org/metadata/export/4483/json"
CATALOG_SEARCH = "https://microdata.worldbank.org/index.php/api/catalog/search?sk=RTFP&ps=200"
MANIFEST_FIELDS = ["source", "iso3", "filename", "url", "vintage", "downloaded_at", "sha256"]
USER_AGENT = "rtfp-replication/0.1 (academic research; polite: >=1 s between requests)"


# --------------------------------------------------------------------------- polite downloads

def _urlopen_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class PoliteFetcher:
    """Sequential HTTP GETs whose start times are at least `min_interval` seconds apart."""

    def __init__(self, min_interval: float = 1.0, clock=time.monotonic, sleep=time.sleep, opener=_urlopen_bytes):
        self.min_interval, self.clock, self.sleep, self.opener = min_interval, clock, sleep, opener
        self._last = None
        self.n_requests = 0

    def get_bytes(self, url: str) -> bytes:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()
        self.n_requests += 1
        return self.opener(url)

    def get_json(self, url: str) -> dict:
        return json.loads(self.get_bytes(url))


def read_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["source"], r["iso3"], r["filename"]): r for r in csv.DictReader(fh)}


def write_manifest(path: Path, rows: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def fetch_files(rows: list[dict], dest: Path, fetcher: PoliteFetcher, manifest: dict, iso3: str,
                source: str = "rtfp_other") -> list[Path]:
    """Download each row's url to dest/filename unless present; record or verify its sha256 in `manifest`."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        path = dest / r["filename"]
        key = (source, iso3, r["filename"])
        if not path.exists():
            data = fetcher.get_bytes(r["url"])
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(data)
            tmp.rename(path)
        digest = sha256_file(path)
        if key in manifest:
            if manifest[key]["sha256"] != digest:
                raise ValueError(f"sha256 of {path} differs from the manifest")
        else:
            manifest[key] = {"source": source, "iso3": iso3, "filename": r["filename"], "url": r["url"],
                             "vintage": r.get("vintage", ""),
                             "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"), "sha256": digest}
        out.append(path)
    return out


# --------------------------------------------------------------------------- step 1: selection

def candidates_from_catalog(wld_meta: dict, search: dict | None = None) -> pd.DataFrame:
    """Per-country RTFP market studies: union of the WLD metadata nation list and the catalogue search."""
    rows = {}
    for n in wld_meta["study_desc"]["study_info"].get("nation", []):
        iso = n["abbreviation"]
        rows[iso] = {"iso3": iso, "country": n["name"], "idno": f"{iso}_2021_RTFP_v02_M",
                     "in_wld_metadata": True, "in_catalog_search": False}
    res = (search or {}).get("result", search or {})
    for r in res.get("rows", []):
        m = STUDY_RE.match(r.get("idno", ""))
        if not m or m.group(1) == "WLD":
            continue
        iso = m.group(1)
        row = rows.setdefault(iso, {"iso3": iso, "country": r.get("nation", iso), "idno": r["idno"],
                                    "in_wld_metadata": False, "in_catalog_search": False})
        row["in_catalog_search"] = True
    return pd.DataFrame(list(rows.values())).sort_values("iso3").reset_index(drop=True)


def listing_summary(listing: dict) -> dict:
    """Distinct market-file vintages in a Microdata download listing, and the newest file."""
    files = {r["vintage"]: r for r in parse_file_listing(listing)}
    if not files:
        return {"n_vintages": 0, "first_vintage": None, "latest_vintage": None, "latest_filename": None,
                "latest_url": None}
    vs = sorted(files)
    return {"n_vintages": len(vs), "first_vintage": vs[0], "latest_vintage": vs[-1],
            "latest_filename": files[vs[-1]]["filename"], "latest_url": files[vs[-1]]["url"]}


def plain_items(columns) -> list[str]:
    """Items with both a plain (survey) column and a published ``c_`` column."""
    cols = set(columns)
    return sorted(c[2:] for c in cols if c.startswith("c_") and c[2:] in cols)


def latest_vintage_summary(df: pd.DataFrame) -> dict:
    """Markets (excluding 'Market Average') and plain-column presence in one vintage's wide table."""
    real = df[~df.mkt_name.isin(AGGREGATE_NAMES)]
    items = plain_items(df.columns)
    nonempty = [it for it in items if real[it].notna().any()]
    return {"n_markets": int(real.mkt_name.nunique()),
            "items_with_c": ";".join(sorted(c[2:] for c in df.columns if c.startswith("c_"))),
            "plain_items": ";".join(items), "plain_items_nonempty": ";".join(nonempty),
            "n_observed_cells": int(sum(real[it].notna().sum() for it in items)),
            "has_plain_columns": bool(items), "has_plain_values": bool(nonempty)}


def apply_selection_rule(c: pd.DataFrame, min_vintages: int = MIN_VINTAGES, min_markets: int = MIN_MARKETS,
                         n_select: int = N_SELECT, exclude=EXCLUDE) -> pd.DataFrame:
    """Add eligible / reason / rank / selected columns (analysis plan section 5)."""
    s = c.copy()
    reasons = []
    for r in s.itertuples(index=False):
        why = []
        if r.iso3 in exclude:
            why.append("excluded: bangladesh (the primary study)")
        if not r.n_vintages >= min_vintages:
            why.append(f"vintages {r.n_vintages} < {min_vintages}")
        if not r.n_markets >= min_markets:
            why.append(f"markets {r.n_markets} < {min_markets}")
        if not (bool(r.has_plain_columns) and bool(r.has_plain_values)):
            why.append("no plain survey columns" if not bool(r.has_plain_columns) else "plain columns empty")
        reasons.append("; ".join(why))
    s["reason"] = reasons
    s["eligible"] = s.reason == ""
    order = s[s.eligible].sort_values(["n_markets", "n_vintages", "iso3"], ascending=[False, False, True])
    s["rank"] = pd.Series(np.arange(1, len(order) + 1), index=order.index).reindex(s.index)
    s["selected"] = s["rank"].le(n_select).fillna(False).astype(bool)
    s.loc[s.eligible & ~s.selected, "reason"] = "eligible, not in top " + str(n_select)
    s.loc[s.selected, "reason"] = "selected"
    return s


def read_vintage_zip(path: Path, **read_csv_kw) -> pd.DataFrame:
    """Read the single CSV member of a downloaded RTFP market-file zip."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one CSV member, found {members}")
        with z.open(members[0]) as fh:
            return pd.read_csv(fh, **read_csv_kw)


def survey_candidates(cands: pd.DataFrame, fetcher: PoliteFetcher, raw_dir: Path, manifest: dict) -> pd.DataFrame:
    """For each candidate: market-file vintages from its listing, then markets and plain columns in its latest file.

    The listing JSON is kept under raw_dir/_listings/<ISO3>.json and the latest zip under raw_dir/<ISO3>/.
    """
    raw_dir = Path(raw_dir)
    (raw_dir / "_listings").mkdir(parents=True, exist_ok=True)
    out = []
    for r in cands.itertuples(index=False):
        raw = fetcher.get_bytes(RTFP_LISTING.format(idno=r.idno))
        (raw_dir / "_listings" / f"{r.iso3}.json").write_bytes(raw)
        ls = listing_summary(json.loads(raw))
        row = {**r._asdict(), **ls, "n_markets": np.nan, "has_plain_columns": False, "has_plain_values": False}
        if ls["n_vintages"]:
            [path] = fetch_files([{"filename": ls["latest_filename"], "url": ls["latest_url"],
                                   "vintage": ls["latest_vintage"]}], raw_dir / r.iso3, fetcher, manifest, r.iso3)
            row.update(latest_vintage_summary(read_vintage_zip(path, low_memory=False)))
            row["latest_sha256"] = manifest[("rtfp_other", r.iso3, ls["latest_filename"])]["sha256"]
        out.append(row)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- step 2: per-country pipeline

def country_vintage_rows(listing: dict, up_to: str | None = None) -> list[dict]:
    """Market-file rows of a listing, one per vintage, oldest first; optionally only vintages <= `up_to`."""
    rows = {r["vintage"]: r for r in parse_file_listing(listing)}
    return [rows[v] for v in sorted(rows) if up_to is None or v <= up_to]


CLEAN = ("forecast", "cross_sectional")
STRATA = ("toggle", "unconfirmed", "backfill", "forecast", "cross_sectional")


def noise_reliability(latest: pd.DataFrame, years: int = 6) -> dict:
    """Reliability of observed local changes in the latest vintage containing one item (methodology D2).

    Variant A: all markets ('Market Average' is never a market), price months from `years` years before the
    vintage month. Variant B: A without exact-zero changes (stale quotes; the staleness-aware variant).
    """
    v = pd.Timestamp(latest.vintage.max())
    since = v.to_period("M").to_timestamp() - pd.DateOffset(months=12 * years)
    d = latest[latest.month >= since].assign(e_obs=lambda x: x.dlog_obs - x.mu_obs).dropna(subset=["e_obs"])
    d = d[["mkt_uid", "item", "month", "dlog_obs", "e_obs"]].assign(item=lambda x: x["item"].astype(str))
    if d["item"].nunique() > 1:
        raise ValueError("noise_reliability expects one item")
    out = {}
    for name, x in (("A", d), ("B", d[d.dlog_obs != 0])):
        g = autocovariances(x) if len(x) else pd.DataFrame()
        base = {"variant": name, "since": since, "n_changes": int(len(x))}
        if g.empty or not {"g0", "g1", "g2", "g3"} <= set(g.columns) or g.iloc[0][["g0", "g1", "g2", "g3"]].isna().any():
            out[name] = {**base, "rel_naive": np.nan, "rel_ar1": np.nan, "phi": np.nan, "ar1_identified": False}
        else:
            out[name] = {**base, **noise_from_autocov(g.iloc[0])}
    return out


def rho_star(e: pd.DataFrame, rel: dict, min_n: int = 10) -> dict:
    """Noise-corrected local correlation corr / sqrt(rel), each reliability on its own population.

    A (naive, AR(1)) -> all pairs; B (naive, AR(1)) -> pairs whose truth change is non-zero. A combination is
    credible if 0 < rel <= 1. Conservative = the largest credible value (as Gate G1, analysis/t1_noise_gate.py).
    """
    pops = {"A": e, "B": e[e.dlog_truth != 0]}
    out = {"n_update_only": int(len(pops["B"]))}
    for var, pop in pops.items():
        corr = weighted_local_stats(pop.e_pub, pop.e_obs, np.ones(len(pop)))["corr"] if len(pop) >= min_n else np.nan
        if var == "B":
            out["corr_update_only"] = corr
        for est in ("naive", "ar1"):
            r = rel.get(var, {}).get(f"rel_{est}", np.nan)
            ok = r == r and 0 < r <= 1
            out[f"rho_star_{var}_{est}"] = corr / np.sqrt(r) if ok else np.nan
    vals = [out[f"rho_star_{v}_{k}"] for v in ("A", "B") for k in ("naive", "ar1")]
    vals = [x for x in vals if np.isfinite(x)]
    out["rho_star_conservative"] = max(vals) if vals else np.nan
    return out


HEADER_RENAMES = {"DATES": "price_date"}  # older releases (e.g. Myanmar to 2024-01-22) name the month column DATES


def _csv_member(z) -> str:
    [member] = [n for n in z.namelist() if n.lower().endswith(".csv")]
    return member


def _harmonised(cols: list) -> list:
    return [HEADER_RENAMES[c] if c in HEADER_RENAMES and HEADER_RENAMES[c] not in cols else c for c in cols]


def extract_vintages(zip_paths, work_dir: Path) -> tuple[dict, list]:
    """Extract each zip's CSV to work_dir/full/<stem>/<stem>.csv (the layout rtfp.vintages.build_stack reads).

    Data bytes are copied unchanged. Only the header line may change: an old-schema column named in
    HEADER_RENAMES (same content: first-of-month dates) is renamed when the new name is absent.
    Returns ({stem: header columns}, [stems whose header was renamed]).
    """
    import io
    import shutil
    import zipfile

    out, renamed = {}, []
    for zp in zip_paths:
        zp = Path(zp)
        stem = zp.name.removesuffix(".zip")
        dest = Path(work_dir) / "full" / stem / f"{stem}.csv"
        with zipfile.ZipFile(zp) as z, z.open(_csv_member(z)) as src:
            head = src.readline()
            raw_cols = next(csv.reader(io.StringIO(head.decode("utf-8-sig"))))
            cols = _harmonised(raw_cols)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if cols != raw_cols:
                    buf = io.StringIO()
                    csv.writer(buf, lineterminator="").writerow(cols)
                    head = buf.getvalue().encode() + (b"\r\n" if head.endswith(b"\r\n") else b"\n")
                with open(dest.with_suffix(".part"), "wb") as dst:
                    dst.write(head)
                    shutil.copyfileobj(src, dst, 1 << 20)
                dest.with_suffix(".part").rename(dest)
        with open(dest, newline="", encoding="utf-8-sig") as fh:
            out[stem] = next(csv.reader(fh))
        if out[stem] != raw_cols:
            renamed.append(stem)
    return out, renamed


def check_renamed_dates(zip_paths) -> dict:
    """For vintages with the old 'DATES' header: is every value the first of the month given by year/month?"""
    bad, n = [], 0
    for zp in zip_paths:
        d = read_vintage_zip(zp, usecols=lambda c: c in ("DATES", "price_date", "year", "month"))
        if "DATES" not in d.columns or "price_date" in d.columns:
            continue
        n += 1
        t = pd.to_datetime(d.DATES, format="%Y-%m-%d", errors="coerce")
        if not ((t.dt.year == d.year) & (t.dt.month == d.month) & (t.dt.day == 1)).all():
            bad.append(Path(zp).name)
    return {"n_with_dates": n, "n_inconsistent": len(bad), "inconsistent": bad}


def item_vintage_dir(work_dir: Path, item: str, headers: dict) -> Path:
    """work_dir/items/<item>/ with links to the extracted vintages that publish the item (``c_<item>``).

    Vintages lacking the item simply have no rows for it (as in rtfp.vintages); build_stack needs at least one
    of the requested items in every file it reads, hence one directory per item.
    """
    import shutil

    d = Path(work_dir) / "items" / item
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for stem, cols in sorted(headers.items(), key=lambda kv: vintage_of(kv[0] + ".csv")):
        if f"c_{item}" in cols:
            (d / stem).symlink_to((Path(work_dir) / "full" / stem).resolve(), target_is_directory=True)
    return d


def _boot_levels(hc: pd.DataFrame, h: pd.DataFrame, B: int, seed: int) -> dict:
    """Two-way (market x month) pigeonhole bootstrap of MAPE and bias, clusters from the hold-out (as T2)."""
    markets, months = sorted(set(h.mkt_uid)), sorted(set(h.month))
    ape, lr = np.abs(hc.c_final / hc.y_eval - 1).to_numpy(), np.log(hc.c_final / hc.y_eval).to_numpy()
    mape, bias = [], []
    for b in range(B):
        w = pigeonhole_weights(hc, np.random.default_rng(seed + b), markets, months)
        if w.sum() == 0:
            continue
        mape.append(100 * np.average(ape, weights=w))
        bias.append(100 * np.average(lr, weights=w))
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else np.nan
    return {"mape_lo": q(mape, 0.025), "mape_hi": q(mape, 0.975), "bias_lo": q(bias, 0.025), "bias_hi": q(bias, 0.975)}


def _boot_corr(x: pd.DataFrame, markets, months, B: int, seed: int, min_n: int) -> tuple:
    """Two-way pigeonhole bootstrap interval of corr(e_pub, e_obs); clusters from the latest panel (as T3)."""
    cs = []
    for b in range(B):
        w = pigeonhole_weights(x, np.random.default_rng(seed + b), markets, months)
        if (w > 0).sum() >= min_n:
            cs.append(weighted_local_stats(x.e_pub[w > 0], x.e_obs[w > 0], w[w > 0])["corr"])
    return (float(np.nanquantile(cs, 0.025)), float(np.nanquantile(cs, 0.975))) if cs else (np.nan, np.nan)


def item_metrics(stack: pd.DataFrame, B: int = 500, seed: int = 20261012, noise_years: int = 6,
                 min_pairs: int = 10) -> tuple[dict, list, dict]:
    """Core Bangladesh metrics (Tables A2, 1, 2, 3, 3b) for one item's stacked vintages.

    Returns (summary row, noise rows, {'holdout', 'pairs'}).
    """
    h = build_holdout(stack)
    pairs = build_change_pairs(h, stack)
    latest = latest_rows(stack)
    for d in (h, pairs, latest):
        d["item"] = d["item"].astype(str)
    vint = stack.vintage.drop_duplicates().sort_values()
    row = {"n_vintages_item": int(len(vint)), "first_vintage_item": vint.iloc[0].date().isoformat(),
           "latest_vintage_item": vint.iloc[-1].date().isoformat(),
           "n_markets_latest": int(latest.mkt_uid.nunique()), "n_observed_latest": int(latest.obs.sum())}
    counts = h.stratum.value_counts()
    for st in STRATA:
        row[f"cells_{st}"] = int(counts.get(st, 0))
    hc = h[h.stratum.isin(CLEAN) & h.y_eval.notna() & h.c_final.notna()]
    row["cells_clean"] = int(len(hc))
    row["clean_month_min"] = hc.month.min().date().isoformat() if len(hc) else None
    row["clean_month_max"] = hc.month.max().date().isoformat() if len(hc) else None

    # levels (Table 2): c_final against the latest-vintage survey value, clean strata pooled
    if len(hc):
        ls = level_stats(hc.c_final, hc.y_eval)
        row.update({"n_level": ls["n"], "mape": ls["mape"], "median_ape": ls["median_ape"],
                    "bias_logpct": ls["bias_logpct"], "share_err_gt10": ls["share_err_gt10"],
                    "mz_slope_month_demeaned": level_mz(hc.rename(columns={"c_final": "c", "y_eval": "y"}))["slope"],
                    **_boot_levels(hc, h, B, seed)})
        for st in CLEAN:
            hs = hc[hc.stratum == st]
            row[f"mape_{st}"] = level_stats(hs.c_final, hs.y_eval)["mape"] if len(hs) else np.nan
            row[f"bias_logpct_{st}"] = level_stats(hs.c_final, hs.y_eval)["bias_logpct"] if len(hs) else np.nan

    # noise reliability (Table 1, variants A and B)
    rel = noise_reliability(latest, years=noise_years)
    for var, r in rel.items():
        row.update({f"noise_n_changes_{var}": r["n_changes"], f"rel_{var}_naive": r["rel_naive"],
                    f"rel_{var}_ar1": r["rel_ar1"]})
    row["noise_since"] = rel["A"]["since"].date().isoformat()

    # local changes (Table 3): own-release mu, both-imputed pairs, by clean stratum
    markets, months = sorted(set(latest.mkt_uid)), sorted(set(latest.month))
    for i, st in enumerate(CLEAN):
        g = pairs[(pairs.stratum == st) & (pairs.pair_type == "both_imputed")]
        e = local_components(g, mode="own_mu") if len(g) else g.assign(e_pub=[], e_obs=[])
        row[f"pairs_{st}"] = int(len(e))
        row[f"pairs_fallback_mu_{st}"] = int(e.mu_obs_vfinal.isna().sum()) if len(e) else 0
        keys = ["corr", "beta_mz", "lam", "corr_lo", "corr_hi", "rho_star_A_naive", "rho_star_A_ar1",
                "rho_star_B_naive", "rho_star_B_ar1", "rho_star_conservative", "n_update_only"]
        vals = dict.fromkeys(keys, np.nan)
        if len(e) >= min_pairs:
            s = weighted_local_stats(e.e_pub, e.e_obs, np.ones(len(e)))
            lo, hi = _boot_corr(e, markets, months, B, seed + 1000 * (i + 1), min_pairs)
            vals.update({"corr": s["corr"], "beta_mz": s["beta_mz"], "lam": s["lam"], "corr_lo": lo, "corr_hi": hi,
                         **rho_star(e, rel, min_n=min_pairs)})
        row.update({f"{k}_{st}": v for k, v in vals.items()})

    # spikes (Table 3b): persistent survey jumps > 10%, share caught by a same-direction published move >= 5%
    clean_pairs = pairs[pairs.stratum.isin(CLEAN)]
    nxt = latest[["mkt_uid", "item", "month", "dlog_obs"]].assign(month=lambda d: d.month - pd.offsets.MonthBegin(1))
    sp = clean_pairs.merge(nxt.rename(columns={"dlog_obs": "dlog_truth_next"}), on=["mkt_uid", "item", "month"],
                           how="left").dropna(subset=["dlog_truth_next"])
    sp = persistent_spikes(sp) if len(sp) else sp.assign(is_spike=[], caught=[])
    row.update({"spike_pairs_with_next": int(len(sp)), "spikes": int(sp.is_spike.sum()),
                "spikes_caught": int(sp.caught.sum())})
    row["spike_share_caught"] = row["spikes_caught"] / row["spikes"] if row["spikes"] else np.nan
    noise_rows = [{k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}
                  for r in rel.values()]
    return row, noise_rows, {"holdout": h, "pairs": pairs}


def crosswalk_check(work_dir: Path, headers: dict) -> tuple[bool, str]:
    """The Bangladesh market-identity guard (rtfp.crosswalk.build_crosswalk) over all vintages of a country."""
    frames = []
    for stem in headers:
        f = Path(work_dir) / "full" / stem / f"{stem}.csv"
        frames.append(pd.read_csv(f, usecols=MARKET_COLS).drop_duplicates().assign(vintage=vintage_of(str(f))))
    try:
        xw = build_crosswalk(pd.concat(frames, ignore_index=True))
    except ValueError as ex:
        return False, str(ex)
    return True, f"{xw[~xw.is_aggregate].mkt_uid.nunique()} markets across {len(headers)} vintages"


def link_extracted_vintages(src_dir: Path, work_dir: Path, prefix: str) -> dict:
    """Link already-extracted vintages (src_dir/<prefix>_RTFP_mkt_*/<stem>.csv) into work_dir/full/. Returns headers."""
    out = {}
    full = Path(work_dir) / "full"
    full.mkdir(parents=True, exist_ok=True)
    for d in sorted(Path(src_dir).glob(f"{prefix}_RTFP_mkt_*")):
        f = d / f"{d.name}.csv"
        if not f.exists():
            continue
        if not (full / d.name).exists():
            (full / d.name).symlink_to(d.resolve(), target_is_directory=True)
        with open(f, newline="", encoding="utf-8-sig") as fh:
            out[d.name] = next(csv.reader(fh))
    return out


def run_country(zip_paths, iso3: str, work_dir: Path, **kw) -> dict:
    """Extract one country's downloaded vintages and run the Bangladesh pipeline on them (see run_extracted)."""
    headers, renamed = extract_vintages(zip_paths, work_dir)
    log = kw.get("log", print)
    if renamed:
        log(f"{iso3}: header renamed {HEADER_RENAMES} in {len(renamed)} vintages ({renamed[0]} .. {renamed[-1]})")
    return {**run_extracted(work_dir, headers, iso3, **kw), "renamed_header_vintages": renamed}


def run_extracted(work_dir: Path, headers: dict, iso3: str, items=None, B: int = 500, seed: int = 20261012,
                  noise_years: int = 6, min_pairs: int = 10, save_dir: Path | None = None, log=print) -> dict:
    """Run the Bangladesh pipeline on extracted vintages in work_dir/full/, one item at a time (one stack per item).

    items default to those with both a plain and a c_ column in the latest vintage.
    """
    import gc
    import resource

    work_dir = Path(work_dir)
    latest = max(headers, key=lambda s: vintage_of(s + ".csv"))
    items = plain_items(headers[latest]) if items is None else list(items)
    ok, msg = crosswalk_check(work_dir, headers)
    log(f"{iso3}: {len(headers)} vintages, latest {latest}; items {items}; crosswalk {'ok' if ok else 'FAILED'}: {msg}")
    rows, noise = [], []
    for it in items:
        d = item_vintage_dir(work_dir, it, headers)
        base = {"iso3": iso3, "item": it}
        try:
            # replication: a market whose name jumps location across releases is dropped and reported
            stack = build_stack(str(d), items=[it], prefix=iso3, on_drift="drop")
            dropped = ";".join(stack.attrs.get("dropped_markets", []))
            row, nrows, extra = item_metrics(stack, B=B, seed=seed, noise_years=noise_years, min_pairs=min_pairs)
            row = {**base, **row, "status": "ok", "dropped_markets_location_jump": dropped}
            noise += [{**base, **r} for r in nrows]
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                for k, v in extra.items():
                    v.to_parquet(Path(save_dir) / f"{iso3}_{it}_{k}.parquet", index=False)
            del stack, extra
        except Exception as ex:  # recorded in the table and the log, never silently dropped
            row = {**base, "status": f"error: {type(ex).__name__}: {ex}"}
        gc.collect()
        row["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20)
        log(f"  {iso3} {it}: {row['status']}; clean cells {row.get('cells_clean')}; peak RSS {row['peak_rss_gb']:.1f} GB")
        rows.append(row)
    return {"summary": pd.DataFrame(rows), "noise": pd.DataFrame(noise), "crosswalk_ok": ok, "crosswalk_msg": msg,
            "items": items, "latest_csv": work_dir / "full" / latest / f"{latest}.csv", "n_vintages": len(headers)}


# --------------------------------------------------------------------------- WFP/HDX feed

HDX_SEARCH = ("https://data.humdata.org/api/3/action/package_search?fq=groups:{iso}%20AND%20organization:wfp"
              "&q=food%20prices&rows=20&fl=name,title")
HDX_SHOW = "https://data.humdata.org/api/3/action/package_show?id={name}"


def hdx_package_name(search: dict) -> str | None:
    """The country's WFP price package (``wfp-food-prices-for-<country>``) in an HDX package_search result."""
    names = [r.get("name", "") for r in search.get("result", {}).get("results", [])]
    hits = [n for n in names if n.startswith("wfp-food-prices-for-")]
    return hits[0] if hits else None


def hdx_price_resource(package: dict, iso3: str) -> dict | None:
    """URL and date of the ``wfp_food_prices_<iso3>.csv`` resource of an HDX package_show result."""
    want = f"wfp_food_prices_{iso3.lower()}.csv"
    for r in package.get("result", {}).get("resources", []):
        if r.get("url", "").endswith(want):
            v = (r.get("last_modified") or package["result"].get("last_modified", ""))[:10]
            return {"url": r["url"], "vintage": v, "filename": want.replace(".csv", f"_{v}.csv")}
    return None


def fetch_hdx(iso3: str, fetcher: PoliteFetcher, raw_dir: Path, manifest: dict, attempts: int = 3,
              wait: float = 30.0) -> tuple[Path | None, str | None]:
    """Download the country's WFP/HDX price feed to raw_dir/<ISO3>/hdx/. Returns (path, package name).

    Network failures (stalls, timeouts, HTTP errors) are retried `attempts` times; if all fail, (None, None) is
    returned so the caller records the audit as unavailable instead of losing the country's results.
    """
    import socket
    import urllib.error

    last = None
    for k in range(attempts):
        try:
            return _fetch_hdx_once(iso3, fetcher, raw_dir, manifest)
        except (socket.timeout, TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
            last = e
            if k + 1 < attempts:
                time.sleep(wait)
    print(f"  {iso3}: WFP/HDX feed unavailable after {attempts} attempts ({last!r}); audit skipped", flush=True)
    return None, None


def _fetch_hdx_once(iso3: str, fetcher: PoliteFetcher, raw_dir: Path, manifest: dict) -> tuple[Path | None, str | None]:
    name = hdx_package_name(fetcher.get_json(HDX_SEARCH.format(iso=iso3.lower())))
    if name is None:
        return None, None
    res = hdx_price_resource(fetcher.get_json(HDX_SHOW.format(name=name)), iso3)
    if res is None:
        return None, name
    [path] = fetch_files([res], Path(raw_dir) / iso3 / "hdx", fetcher, manifest, iso3, source="hdx_wfp")
    return path, name


# --------------------------------------------------------------------------- raw-column audit (Task 1.2 transfer)

AUDIT_GATE = 0.90  # Bangladesh stop-gate (Table A1): share of observed cells equal to the WFP/HDX value
AUDIT_STATUSES = ("match", "value_mismatch", "no_hdx_row", "hdx_only", "neither")


def clean_hdx(hdx: pd.DataFrame) -> pd.DataFrame:
    """Drop non-data rows (e.g. an HXL '#tag' row) and make price, market_id and coordinates numeric."""
    d = hdx[pd.to_datetime(hdx.date, errors="coerce", format="ISO8601").notna()].copy()
    for c in ("price", "market_id", "latitude", "longitude"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def _retail_months(hdx: pd.DataFrame) -> pd.DataFrame:
    h = hdx[hdx.pricetype == "Retail"].copy()
    h["month"] = pd.to_datetime(h.date).dt.to_period("M").dt.to_timestamp()
    return h


def choose_hdx_commodity(panel: pd.DataFrame, hdx: pd.DataFrame, market_map: pd.DataFrame,
                         tol: float = 0.01) -> pd.DataFrame:
    """For each RTFP item, the HDX Retail commodity x unit series with the most cells equal to y (|y/p - 1| <= tol).

    RTFP item names do not identify a WFP commodity variety, so the series is chosen from the data. Choosing the
    best of several candidates can only raise the match share; the runner-up is reported alongside.
    """
    h = _retail_months(hdx)
    o = panel[panel.obs][["mkt_uid", "month", "item", "y"]].merge(
        market_map[["mkt_uid", "market_id"]].dropna(), on="mkt_uid")
    j = o.merge(h[["market_id", "month", "commodity", "unit", "price"]], on=["market_id", "month"])
    j = j[np.abs(j.y / j.price - 1) <= tol].drop_duplicates(["item", "mkt_uid", "month", "commodity", "unit"])
    n = j.groupby(["item", "commodity", "unit"]).size().rename("n_exact").reset_index()
    out = []
    for it in sorted(panel["item"].astype(str).unique()):
        c = n[n["item"] == it].sort_values(["n_exact", "commodity", "unit"], ascending=[False, True, True])
        best = c.iloc[0] if len(c) else None
        second = c.iloc[1] if len(c) > 1 else None
        out.append({"item": it, "hdx_commodity": best.commodity if best is not None else np.nan,
                    "hdx_unit": best.unit if best is not None else np.nan,
                    "n_exact": int(best.n_exact) if best is not None else 0,
                    "runner_up": f"{second.commodity} [{second.unit}]" if second is not None else np.nan,
                    "runner_up_n_exact": int(second.n_exact) if second is not None else 0})
    return pd.DataFrame(out)


def _audit_cells_with_map(panel: pd.DataFrame, hdx: pd.DataFrame, market_map: pd.DataFrame, cmap: dict,
                          tol: float) -> pd.DataFrame:
    """rtfp.audit.audit_cells, unchanged, run with a country-specific commodity map in place of Bangladesh's."""
    from unittest import mock

    from rtfp import audit

    with mock.patch.dict(audit.HDX_COMMODITY, cmap, clear=True):
        return audit.audit_cells(panel, hdx, market_map, tol=tol)


def audit_country(panel: pd.DataFrame, hdx: pd.DataFrame, tol: float = 0.01, gate: float = AUDIT_GATE) -> tuple:
    """Raw-column audit of one country's latest vintage against its WFP/HDX feed. Returns (cells, per-item summary).

    Markets matched by rtfp.audit.match_markets (exact name, else nearest within 5 km); each item audited
    against its chosen series (choose_hdx_commodity). If the feed has several rows for one market-month of the
    series, a cell counts as matched when any of them matches (n_dup_rows reported).
    """
    from rtfp.audit import match_markets

    panel = panel.assign(item=panel["item"].astype(str))
    mm = match_markets(panel[["mkt_uid", "mkt_name", "lat", "lon"]],
                       hdx[["market", "market_id", "latitude", "longitude"]])
    ch = choose_hdx_commodity(panel, hdx, mm, tol)
    cells, rows = [], []
    for r in ch.itertuples(index=False):
        p = panel[panel["item"] == r.item]
        if isinstance(r.hdx_commodity, str):
            h = hdx[(hdx.commodity == r.hdx_commodity) & (hdx.unit == r.hdx_unit)]
            a = _audit_cells_with_map(p, h, mm, {r.hdx_commodity: r.item}, tol)
        else:
            a = _audit_cells_with_map(p, hdx.iloc[0:0], mm, {}, tol)
        key = ["mkt_uid", "month", "item"]
        n_dup = int(a.duplicated(key).sum())
        a = a.assign(_m=a.status.eq("match")).sort_values(key + ["_m"], ascending=[True] * 3 + [False])
        a = a.drop_duplicates(key).drop(columns="_m")
        k = a.status.value_counts()
        obs_n = int(k.get("match", 0) + k.get("value_mismatch", 0) + k.get("no_hdx_row", 0))
        in_feed = int(k.get("match", 0) + k.get("value_mismatch", 0))
        share = k.get("match", 0) / obs_n if obs_n else np.nan
        m = a[a.status == "match"]
        rows.append({"item": r.item, "hdx_commodity": r.hdx_commodity, "hdx_unit": r.hdx_unit,
                     "runner_up": r.runner_up, "runner_up_n_exact": r.runner_up_n_exact,
                     **{s: int(k.get(s, 0)) for s in AUDIT_STATUSES}, "rtfp_observed": obs_n,
                     "share_matched": share,
                     "share_hdx_in_rtfp": in_feed / (in_feed + k.get("hdx_only", 0)) if in_feed + k.get("hdx_only", 0) else np.nan,
                     "median_ratio_mismatch": a[a.status == "value_mismatch"].ratio_y_hdx.median(),
                     "share_aggregate_among_matched": (m.priceflag == "aggregate").mean() if len(m) else np.nan,
                     "n_dup_rows": n_dup, "audit_pass": bool(share >= gate) if share == share else False,
                     "markets_by_name": int((mm.how == "name").sum()), "markets_by_coords": int((mm.how == "coords").sum()),
                     "markets_unmatched": int((mm.how == "none").sum())})
        cells.append(a)
    return pd.concat(cells, ignore_index=True), pd.DataFrame(rows)
