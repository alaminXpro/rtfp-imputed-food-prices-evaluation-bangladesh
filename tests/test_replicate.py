"""Bounded replication driver (Task 6.1, analysis plan section 5). Synthetic inputs only."""
import numpy as np
import pandas as pd
import pytest

from rtfp import replicate as rp


# --------------------------------------------------------------------------- selection (step 1)

def _listing(iso, vintages, extra=()):
    files = [{"filename": f"{iso}_RTFP_mkt_2007_{v}.zip", "dcdate": v,
              "links": {"download": f"https://x/{iso}/{v}?filename={iso}_RTFP_mkt_2007_{v}.zip"}} for v in vintages]
    files += [{"filename": f, "dcdate": "2026-01-01", "links": {"download": "https://x/other"}} for f in extra]
    return {"status": "success", "files": files}


def test_candidates_union_of_metadata_nations_and_catalog_search_without_world_and_other_series():
    wld = {"study_desc": {"study_info": {"nation": [{"name": "Nigeria", "abbreviation": "NGA"},
                                                    {"name": "Bangladesh", "abbreviation": "BGD"}]}}}
    search = {"result": {"rows": [{"idno": "NGA_2021_RTFP_v02_M", "nation": "Nigeria"},
                                  {"idno": "YEM_2021_RTFP_v02_M", "nation": "Yemen, Rep."},
                                  {"idno": "YEM_2023_RTEP_v01_M", "nation": "Yemen, Rep."},
                                  {"idno": "WLD_2021_RTFP_v02_M", "nation": "World"}]}}
    c = rp.candidates_from_catalog(wld, search).set_index("iso3")
    assert sorted(c.index) == ["BGD", "NGA", "YEM"]
    assert c.loc["NGA", "idno"] == "NGA_2021_RTFP_v02_M"
    assert bool(c.loc["NGA", "in_wld_metadata"]) and bool(c.loc["NGA", "in_catalog_search"])
    assert not bool(c.loc["YEM", "in_wld_metadata"]) and c.loc["YEM", "country"] == "Yemen, Rep."


def test_listing_summary_counts_distinct_market_file_vintages_only():
    lst = _listing("NGA", ["2024-01-02", "2026-10-05", "2025-05-05"],
                   extra=["NGA_RTP_details_mkt_2007_2026-10-05.zip", "README.pdf"])
    lst["files"].append(dict(lst["files"][0]))  # a duplicated entry must not count twice
    s = rp.listing_summary(lst)
    assert s["n_vintages"] == 3
    assert s["first_vintage"] == "2024-01-02" and s["latest_vintage"] == "2026-10-05"
    assert s["latest_filename"] == "NGA_RTFP_mkt_2007_2026-10-05.zip"
    assert s["latest_url"].endswith("NGA_RTFP_mkt_2007_2026-10-05.zip")


def test_listing_summary_of_empty_listing_has_zero_vintages():
    s = rp.listing_summary({"status": "success", "files": []})
    assert s["n_vintages"] == 0 and s["latest_filename"] is None


def test_plain_items_need_both_plain_and_c_columns():
    cols = ["mkt_name", "price_date", "maize", "c_maize", "o_maize", "c_rice", "beans", "c_food_price_index"]
    assert rp.plain_items(cols) == ["maize"]


def test_latest_vintage_summary_counts_markets_without_market_average_and_flags_empty_plain_columns():
    df = pd.DataFrame({"mkt_name": ["A", "A", "B", "Market Average"],
                       "maize": [1.0, np.nan, np.nan, 2.0], "c_maize": [1.0, 1.1, 1.2, 1.3],
                       "rice": [np.nan] * 4, "c_rice": [2.0] * 4, "c_food_price_index": [1.0] * 4})
    s = rp.latest_vintage_summary(df)
    assert s["n_markets"] == 2
    assert s["items_with_c"] == "food_price_index;maize;rice"
    assert s["plain_items"] == "maize;rice"           # columns exist
    assert s["plain_items_nonempty"] == "maize"        # rice plain column has no survey value
    assert s["n_observed_cells"] == 1                  # Market Average row is not a market
    assert s["has_plain_columns"] and s["has_plain_values"]


def _cands(rows):
    return pd.DataFrame(rows, columns=["iso3", "n_vintages", "n_markets", "has_plain_columns", "has_plain_values"])


def test_selection_rule_thresholds_exclusion_and_top_five_by_markets_with_vintage_ties():
    c = _cands([
        ("BGD", 83, 110, True, True),   # excluded: Bangladesh
        ("AAA", 49, 500, True, True),   # too few vintages
        ("BBB", 60, 29, True, True),    # too few markets
        ("CCC", 60, 300, False, False),  # no plain columns
        ("DDD", 60, 200, True, True),
        ("EEE", 70, 200, True, True),   # ties DDD on markets, more vintages -> ranks first
        ("FFF", 50, 30, True, True),    # exactly at both thresholds -> eligible
        ("GGG", 55, 100, True, True),
        ("HHH", 90, 150, True, True),
        ("III", 99, 31, True, True),    # eligible but 6th by markets -> not selected
    ])
    s = rp.apply_selection_rule(c).set_index("iso3")
    assert not s.loc[["BGD", "AAA", "BBB", "CCC"], "eligible"].any()
    assert "bangladesh" in s.loc["BGD", "reason"] and "vintages" in s.loc["AAA", "reason"]
    assert "markets" in s.loc["BBB", "reason"] and "plain" in s.loc["CCC", "reason"]
    assert s.eligible.sum() == 6
    sel = s[s.selected].sort_values("rank")
    assert list(sel.index) == ["EEE", "DDD", "HHH", "GGG", "III"]
    assert not s.loc["FFF", "selected"] and s.loc["FFF", "rank"] == 6


def test_selection_rule_requires_plain_values_not_just_headers():
    c = _cands([("AAA", 60, 100, True, False), ("BBB", 60, 90, True, True)])
    s = rp.apply_selection_rule(c).set_index("iso3")
    assert not s.loc["AAA", "eligible"] and s.loc["BBB", "selected"]


# --------------------------------------------------------------------------- polite downloads

class _FakeNet:
    """Records request times on a fake clock; returns canned bytes per URL."""

    def __init__(self, payloads):
        self.t, self.calls, self.payloads = 0.0, [], payloads

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def opener(self, url):
        self.calls.append((url, self.t))
        self.t += 0.1  # the request itself takes 0.1 s
        return self.payloads[url]


def test_polite_fetcher_spaces_request_starts_by_at_least_the_minimum_interval():
    net = _FakeNet({"u1": b"{}", "u2": b"{\"a\": 1}", "u3": b"x"})
    f = rp.PoliteFetcher(min_interval=1.0, clock=net.clock, sleep=net.sleep, opener=net.opener)
    assert f.get_json("u1") == {} and f.get_json("u2") == {"a": 1}
    f.get_bytes("u3")
    starts = [t for _, t in net.calls]
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(starts, starts[1:]))


def test_fetch_files_downloads_missing_records_sha256_and_skips_existing(tmp_path):
    import hashlib
    rows = [{"filename": "NGA_RTFP_mkt_2007_2026-10-05.zip", "url": "u/a", "vintage": "2026-10-05"},
            {"filename": "NGA_RTFP_mkt_2007_2026-09-28.zip", "url": "u/b", "vintage": "2026-09-28"}]
    net = _FakeNet({"u/a": b"AAA", "u/b": b"BBB"})
    f = rp.PoliteFetcher(min_interval=1.0, clock=net.clock, sleep=net.sleep, opener=net.opener)
    manifest = {}
    paths = rp.fetch_files(rows, tmp_path / "NGA", f, manifest, iso3="NGA")
    assert [p.name for p in paths] == [r["filename"] for r in rows]
    assert (tmp_path / "NGA" / rows[0]["filename"]).read_bytes() == b"AAA"
    m = manifest[("rtfp_other", "NGA", rows[1]["filename"])]
    assert m["sha256"] == hashlib.sha256(b"BBB").hexdigest() and m["vintage"] == "2026-09-28"
    n_calls = len(net.calls)
    rp.fetch_files(rows, tmp_path / "NGA", f, manifest, iso3="NGA")
    assert len(net.calls) == n_calls  # idempotent: nothing fetched again


def test_fetch_files_refuses_a_file_whose_bytes_differ_from_the_manifest(tmp_path):
    rows = [{"filename": "X_RTFP_mkt_2007_2026-10-05.zip", "url": "u/a", "vintage": "2026-10-05"}]
    net = _FakeNet({"u/a": b"AAA"})
    f = rp.PoliteFetcher(min_interval=0.0, clock=net.clock, sleep=net.sleep, opener=net.opener)
    manifest = {}
    rp.fetch_files(rows, tmp_path, f, manifest, iso3="XXX")
    (tmp_path / rows[0]["filename"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        rp.fetch_files(rows, tmp_path, f, manifest, iso3="XXX")


def test_manifest_round_trip(tmp_path):
    m = {("rtfp_other", "NGA", "a.zip"): {"source": "rtfp_other", "iso3": "NGA", "filename": "a.zip", "url": "u",
                                          "vintage": "2026-10-05", "downloaded_at": "t", "sha256": "s"}}
    rp.write_manifest(tmp_path / "MANIFEST.csv", m)
    assert rp.read_manifest(tmp_path / "MANIFEST.csv") == m


def _zip_bytes(member: str, text: str) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, text)
    return buf.getvalue()


def test_read_vintage_zip_reads_the_csv_member(tmp_path):
    p = tmp_path / "NGA_RTFP_mkt_2007_2026-10-05.zip"
    p.write_bytes(_zip_bytes("NGA_RTFP_mkt_2007_2026-10-05.csv", "mkt_name,maize,c_maize\nA,1.5,1.5\n"))
    df = rp.read_vintage_zip(p)
    assert list(df.columns) == ["mkt_name", "maize", "c_maize"] and df.maize.iloc[0] == 1.5


def test_survey_candidates_one_row_per_candidate_with_listing_and_latest_file_facts(tmp_path):
    import json
    cands = pd.DataFrame({"iso3": ["NGA", "ZZZ"], "country": ["Nigeria", "Nowhere"],
                          "idno": ["NGA_2021_RTFP_v02_M", "ZZZ_2021_RTFP_v02_M"],
                          "in_wld_metadata": [True, False], "in_catalog_search": [True, True]})
    lst = _listing("NGA", ["2026-09-28", "2026-10-05"])
    latest_url = lst["files"][1]["links"]["download"]
    csv_text = "mkt_name,maize,c_maize\nA,1.5,1.5\nB,,1.6\nMarket Average,,1.55\n"
    net = _FakeNet({rp.RTFP_LISTING.format(idno="NGA_2021_RTFP_v02_M"): json.dumps(lst).encode(),
                    rp.RTFP_LISTING.format(idno="ZZZ_2021_RTFP_v02_M"): json.dumps({"files": []}).encode(),
                    latest_url: _zip_bytes("NGA_RTFP_mkt_2007_2026-10-05.csv", csv_text)})
    f = rp.PoliteFetcher(min_interval=0.0, clock=net.clock, sleep=net.sleep, opener=net.opener)
    manifest = {}
    s = rp.survey_candidates(cands, f, tmp_path, manifest).set_index("iso3")
    assert s.loc["NGA", "n_vintages"] == 2 and s.loc["NGA", "n_markets"] == 2
    assert s.loc["NGA", "latest_vintage"] == "2026-10-05" and bool(s.loc["NGA", "has_plain_values"])
    assert s.loc["NGA", "latest_sha256"] == manifest[("rtfp_other", "NGA", "NGA_RTFP_mkt_2007_2026-10-05.zip")]["sha256"]
    assert (tmp_path / "_listings" / "NGA.json").exists()
    assert s.loc["ZZZ", "n_vintages"] == 0 and pd.isna(s.loc["ZZZ", "n_markets"])
    assert not bool(s.loc["ZZZ", "has_plain_columns"])


def test_country_vintage_rows_are_unique_and_ordered_by_vintage():
    lst = _listing("MLI", ["2026-10-05", "2024-01-02", "2025-05-05"], extra=["MLI_RTP_details_mkt_2007_2026-10-05.zip"])
    lst["files"].append(dict(lst["files"][0]))
    rows = rp.country_vintage_rows(lst)
    assert [r["vintage"] for r in rows] == ["2024-01-02", "2025-05-05", "2026-10-05"]
    assert all("_RTFP_mkt_" in r["filename"] for r in rows)


def test_country_vintage_rows_can_stop_at_the_selection_snapshot_vintage():
    lst = _listing("MLI", ["2026-10-05", "2024-01-02", "2026-10-12"])
    rows = rp.country_vintage_rows(lst, up_to="2026-10-05")
    assert [r["vintage"] for r in rows] == ["2024-01-02", "2026-10-05"]


# --------------------------------------------------------------------------- noise and rho* (step 2)

def test_noise_reliability_variant_a_uses_recent_months_and_variant_b_drops_zero_changes():
    rng = np.random.default_rng(0)
    months = pd.date_range("2015-01-01", "2026-09-01", freq="MS")
    rows = []
    for m in range(30):
        lvl = np.cumsum(rng.normal(0, 0.03, len(months)))
        y = lvl + rng.normal(0, 0.02, len(months))
        d = np.diff(y, prepend=np.nan)
        d[rng.random(len(months)) < 0.2] = 0.0  # stale quotes
        rows.append(pd.DataFrame({"mkt_uid": f"m{m}", "item": "maize", "month": months, "dlog_obs": d}))
    latest = pd.concat(rows, ignore_index=True)
    latest["mu_obs"] = latest.groupby("month").dlog_obs.transform("mean")
    latest["vintage"] = pd.Timestamp("2026-10-05")
    r = rp.noise_reliability(latest, years=6)
    a, b = r["A"], r["B"]
    since = pd.Timestamp("2020-10-01")
    e = latest[latest.month >= since].dropna(subset=["dlog_obs"])
    assert a["n_changes"] == len(e) and a["since"] == since
    assert b["n_changes"] == int((e.dlog_obs != 0).sum())
    assert 0 < a["rel_naive"] < 1


def test_rho_star_conservative_is_largest_credible_matched_combination():
    rng = np.random.default_rng(1)
    n = 200
    e_obs = rng.normal(0, 1, n)
    e_obs[:40] = 0.0
    p = pd.DataFrame({"e_obs": e_obs, "e_pub": 0.5 * e_obs + rng.normal(0, 1, n)})
    p["dlog_truth"] = p.e_obs
    rel = {"A": {"rel_naive": 0.5, "rel_ar1": 1.2}, "B": {"rel_naive": 0.4, "rel_ar1": np.nan}}
    out = rp.rho_star(p, rel)
    corr_all = np.corrcoef(p.e_pub, p.e_obs)[0, 1]
    nz = p[p.dlog_truth != 0]
    corr_nz = np.corrcoef(nz.e_pub, nz.e_obs)[0, 1]
    assert out["rho_star_A_naive"] == pytest.approx(corr_all / np.sqrt(0.5))
    assert np.isnan(out["rho_star_A_ar1"])  # rel > 1 is not credible
    assert out["rho_star_B_naive"] == pytest.approx(corr_nz / np.sqrt(0.4))
    assert out["rho_star_conservative"] == pytest.approx(max(corr_all / np.sqrt(0.5), corr_nz / np.sqrt(0.4)))


# --------------------------------------------------------------------------- per-country orchestration on a toy vintage dir

TOY_V = ["2025-07-05", "2025-08-05", "2025-09-05"]
TOY_MONTHS = pd.date_range("2025-01-01", "2025-08-01", freq="MS")
TOY_ERR = {m: (0.05 if m % 2 == 0 else -0.03) for m in range(10)}


def _toy_truth(m, t):
    return 100 * (1 + 0.01 * m) * (1 + 0.02 * t) + ((m * 7 + t * 3) % 5)


def _toy_observed(v, item, m, t):
    """Which cells are observed in which vintage (t = 0..7 for 2025-01..2025-08)."""
    if item == "maize":
        if v == 0:
            return t <= 4 and not (m >= 6 and t in (3, 4))
        if v == 1:
            return t <= 5
        return True
    # beans exist only from the second vintage on: months 6-7 imputed in v1, observed in v2
    return t <= 5 if v == 1 else True


def _toy_csv(v):
    rows = []
    for m in range(10):
        for t, month in enumerate(TOY_MONTHS):
            r = {"ISO3": "TOY", "mkt_name": f"Market {m}", "adm1_name": "R", "adm2_name": f"D{m}",
                 "lat": 10 + 0.5 * m, "lon": 20.0, "geo_id": f"g{m}", "price_date": month.date().isoformat(),
                 "data_coverage": 50.0, "data_coverage_recent": 60.0}
            for item, scale in (("maize", 1.0), ("beans", 2.0)):
                if item == "beans" and v == 0:
                    continue
                y = scale * _toy_truth(m, t)
                obs = _toy_observed(v, item, m, t)
                c = y if obs else y * (1 + TOY_ERR[m])
                r.update({item: y if obs else np.nan, f"o_{item}": c, f"c_{item}": c, f"h_{item}": c * 1.1,
                          f"l_{item}": c * 0.9, f"trust_{item}": 8.0})
            rows.append(r)
    avg = dict(rows[-1], mkt_name="Market Average", geo_id="avg")
    rows.append(avg)
    return pd.DataFrame(rows).to_csv(index=False)


@pytest.fixture
def toy_zips(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    paths = []
    for i, v in enumerate(TOY_V):
        stem = f"TOY_RTFP_mkt_2007_{v}"
        p = raw / f"{stem}.zip"
        p.write_bytes(_zip_bytes(f"{stem}.csv", _toy_csv(i)))
        paths.append(p)
    return paths


def test_extract_vintages_and_item_dirs_keep_only_vintages_that_carry_the_item(toy_zips, tmp_path):
    headers, renamed = rp.extract_vintages(toy_zips, tmp_path / "work")
    assert renamed == []
    assert sorted(headers) == [f"TOY_RTFP_mkt_2007_{v}" for v in TOY_V]
    d = rp.item_vintage_dir(tmp_path / "work", "beans", headers)
    assert sorted(p.name for p in d.iterdir()) == [f"TOY_RTFP_mkt_2007_{v}" for v in TOY_V[1:]]
    assert (d / f"TOY_RTFP_mkt_2007_{TOY_V[1]}" / f"TOY_RTFP_mkt_2007_{TOY_V[1]}.csv").exists()


def test_run_country_on_toy_vintages_reproduces_hand_counts_and_level_errors(toy_zips, tmp_path):
    res = rp.run_country(toy_zips, "TOY", tmp_path / "work", B=20, seed=1)
    t = res["summary"].set_index("item")
    assert sorted(t.index) == ["beans", "maize"]
    mz = t.loc["maize"]
    assert mz.n_vintages_item == 3 and mz.n_markets_latest == 10  # 'Market Average' is not a market
    # hold-out by construction: markets 6-9 months 4-5 (cross-sectional); month 6 imputed in v0 only,
    # months 7-8 imputed in v0 and v1 with no market observed (forecast)
    assert (mz.cells_cross_sectional, mz.cells_forecast, mz.cells_toggle, mz.cells_backfill) == (8, 30, 0, 0)
    assert mz.cells_clean == 38
    errs = [TOY_ERR[m] for m in range(6, 10) for _ in (3, 4)] + [TOY_ERR[m] for m in range(10) for _ in (5, 6, 7)]
    assert mz.mape == pytest.approx(100 * np.mean(np.abs(errs)))
    assert mz.bias_logpct == pytest.approx(100 * np.mean(np.log1p(errs)))
    # both-imputed pairs: (4,5) for markets 6-9 cross-sectional; (5,6) markets 6-9 and (7,8) all markets forecast
    assert (mz.pairs_cross_sectional, mz.pairs_forecast) == (4, 14)
    assert np.isnan(mz.corr_cross_sectional)  # fewer than 10 pairs: not estimated
    assert np.isfinite(mz.corr_forecast)
    bn = t.loc["beans"]
    assert bn.n_vintages_item == 2 and bn.first_vintage_item == TOY_V[1]
    assert (bn.cells_forecast, bn.cells_cross_sectional) == (20, 0)
    assert res["crosswalk_ok"]
    assert set(res["noise"].variant) == {"A", "B"}


def test_extract_vintages_renames_old_dates_header_to_price_date_and_keeps_data_bytes(tmp_path):
    body = "2008-01-01,A,1.5,1.5\r\n2008-02-01,A,1.6,1.6\r\n"
    old = tmp_path / "TOY_RTFP_mkt_2008_2023-06-06.zip"
    old.write_bytes(_zip_bytes("TOY_RTFP_mkt_2008_2023-06-06.csv", "DATES,mkt_name,maize,c_maize\r\n" + body))
    headers, renamed = rp.extract_vintages([old], tmp_path / "work")
    out = (tmp_path / "work" / "full" / old.stem / f"{old.stem}.csv").read_bytes()
    assert headers[old.stem] == ["price_date", "mkt_name", "maize", "c_maize"]
    assert renamed == [old.stem]
    assert out == ("price_date,mkt_name,maize,c_maize\r\n" + body).encode()


# --------------------------------------------------------------------------- WFP/HDX feed and raw-column audit

def test_hdx_package_name_prefers_the_country_wfp_food_prices_package():
    search = {"result": {"results": [{"name": "wfp-food-prices"},
                                     {"name": "wfp-food-prices-for-democratic-republic-of-the-congo"},
                                     {"name": "global-market-monitor"}]}}
    assert rp.hdx_package_name(search) == "wfp-food-prices-for-democratic-republic-of-the-congo"
    assert rp.hdx_package_name({"result": {"results": [{"name": "wfp-food-prices"}]}}) is None


def test_hdx_price_resource_picks_the_country_price_csv_and_its_date():
    pkg = {"result": {"name": "wfp-food-prices-for-mali", "last_modified": "2026-10-04T16:16:33",
                      "resources": [{"url": "https://h/x/wfp_food_prices_mli.csv", "last_modified": "2026-10-04T16:16:33"},
                                    {"url": "https://h/y/wfp_markets_mli.csv", "last_modified": "2026-09-27T14:09:49"}]}}
    r = rp.hdx_price_resource(pkg, "MLI")
    assert r == {"url": "https://h/x/wfp_food_prices_mli.csv", "vintage": "2026-10-04",
                 "filename": "wfp_food_prices_mli_2026-10-04.csv"}


def test_fetch_hdx_downloads_the_feed_into_the_country_hdx_dir_or_reports_none(tmp_path):
    import json
    pkg = {"result": {"name": "wfp-food-prices-for-mali", "last_modified": "2026-10-04T00:00:00",
                      "resources": [{"url": "https://h/x/wfp_food_prices_mli.csv", "last_modified": "2026-10-04T00:00:00"}]}}
    net = _FakeNet({rp.HDX_SEARCH.format(iso="mli"): json.dumps({"result": {"results": [{"name": "wfp-food-prices-for-mali"}]}}).encode(),
                    rp.HDX_SHOW.format(name="wfp-food-prices-for-mali"): json.dumps(pkg).encode(),
                    "https://h/x/wfp_food_prices_mli.csv": b"date,market\n",
                    rp.HDX_SEARCH.format(iso="zzz"): json.dumps({"result": {"results": []}}).encode()})
    f = rp.PoliteFetcher(min_interval=0.0, clock=net.clock, sleep=net.sleep, opener=net.opener)
    manifest = {}
    path, name = rp.fetch_hdx("MLI", f, tmp_path, manifest)
    assert name == "wfp-food-prices-for-mali" and path == tmp_path / "MLI" / "hdx" / "wfp_food_prices_mli_2026-10-04.csv"
    assert ("hdx_wfp", "MLI", "wfp_food_prices_mli_2026-10-04.csv") in manifest
    assert rp.fetch_hdx("ZZZ", f, tmp_path, manifest) == (None, None)


def _audit_panel():
    rows = []
    for t, month in enumerate(pd.date_range("2025-01-01", periods=4, freq="MS")):
        for item, base in (("maize", 100.0), ("beans", 300.0)):
            y = base + t if t < 3 else np.nan
            rows.append({"mkt_uid": "m_a", "mkt_name": "A", "lat": 10.0, "lon": 20.0, "month": month, "item": item,
                         "y": y, "c": base + t, "obs": t < 3})
    return pd.DataFrame(rows)


def _audit_hdx():
    rows = [["#date", "#market", "#id", "#lat", "#lon", "#commodity", "#unit", "#type", "#flag", "#price"]]
    for t, d in enumerate(["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15"]):
        rows.append([d, "A", 7, 10.0, 20.0, "Maize (white)", "KG", "Retail", "actual", 100.0 + t])
        rows.append([d, "A", 7, 10.0, 20.0, "Maize (yellow)", "KG", "Retail", "actual", 101.0 if t == 1 else 90.0])
        rows.append([d, "A", 7, 10.0, 20.0, "Maize (white)", "100 KG", "Wholesale", "actual", 9000.0])
        rows.append([d, "A", 7, 10.0, 20.0, "Beans (red)", "KG", "Retail", "actual", 500.0])
    return pd.DataFrame(rows, columns=["date", "market", "market_id", "latitude", "longitude", "commodity", "unit",
                                       "pricetype", "priceflag", "price"])


def test_choose_hdx_commodity_takes_the_retail_series_with_most_exact_matches_and_reports_runner_up():
    p, h = _audit_panel(), rp.clean_hdx(_audit_hdx())
    mm = pd.DataFrame({"mkt_uid": ["m_a"], "market_id": [7], "how": ["name"], "km": [0.0]})
    ch = rp.choose_hdx_commodity(p, h, mm).set_index("item")
    assert (ch.loc["maize", "hdx_commodity"], ch.loc["maize", "hdx_unit"]) == ("Maize (white)", "KG")
    assert ch.loc["maize", "n_exact"] == 3 and ch.loc["maize", "runner_up"] == "Maize (yellow) [KG]"
    assert ch.loc["maize", "runner_up_n_exact"] == 1
    assert pd.isna(ch.loc["beans", "hdx_commodity"]) and ch.loc["beans", "n_exact"] == 0


def test_clean_hdx_drops_the_hxl_tag_row():
    assert len(rp.clean_hdx(_audit_hdx())) == 16


def test_audit_country_classifies_cells_with_rtfp_audit_rules_and_summarises_per_item():
    cells, summ = rp.audit_country(_audit_panel(), rp.clean_hdx(_audit_hdx()))
    s = summ.set_index("item")
    assert s.loc["maize", "match"] == 3 and s.loc["maize", "share_matched"] == 1.0
    assert s.loc["maize", "hdx_only"] == 1  # April: in the feed, not observed in RTFP
    assert s.loc["beans", "share_matched"] == 0.0 and s.loc["beans", "no_hdx_row"] == 3
    assert not s.loc["beans", "audit_pass"] and s.loc["maize", "audit_pass"]
    from rtfp.audit import HDX_COMMODITY
    assert set(HDX_COMMODITY.values()) == {"rice", "wheat_flour", "oil", "lentils"}  # Bangladesh map restored


def test_run_extracted_on_linked_vintage_folders_equals_run_country_on_zips(toy_zips, tmp_path):
    # an already-extracted archive laid out like data/raw/rtfp_vintages/<stem>/<stem>.csv
    rp.extract_vintages(toy_zips, tmp_path / "src")
    headers = rp.link_extracted_vintages(tmp_path / "src" / "full", tmp_path / "linked", prefix="TOY")
    assert sorted(headers) == [f"TOY_RTFP_mkt_2007_{v}" for v in TOY_V]
    a = rp.run_extracted(tmp_path / "linked", headers, "TOY", B=20, seed=1)["summary"]
    b = rp.run_country(toy_zips, "TOY", tmp_path / "work", B=20, seed=1)["summary"]
    cols = [c for c in a.columns if c != "peak_rss_gb"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_check_renamed_dates_flags_dates_that_disagree_with_year_and_month(tmp_path):
    good = tmp_path / "X_RTFP_mkt_2008_2023-06-06.zip"
    good.write_bytes(_zip_bytes("a.csv", "DATES,year,month,mkt_name\n2008-01-01,2008,1,A\n2008-02-01,2008,2,A\n"))
    bad = tmp_path / "X_RTFP_mkt_2008_2023-07-06.zip"
    bad.write_bytes(_zip_bytes("b.csv", "DATES,year,month,mkt_name\n2008-01-15,2008,1,A\n"))
    new = tmp_path / "X_RTFP_mkt_2008_2024-07-06.zip"
    new.write_bytes(_zip_bytes("c.csv", "price_date,year,month,mkt_name\n2008-01-01,2008,1,A\n"))
    r = rp.check_renamed_dates([good, bad, new])
    assert r == {"n_with_dates": 2, "n_inconsistent": 1, "inconsistent": [bad.name]}


def test_fetch_hdx_returns_none_after_retries_instead_of_crashing(tmp_path):
    import socket

    from rtfp import replicate as rp

    class Stalling:
        calls = 0

        def get_json(self, url):
            Stalling.calls += 1
            raise socket.timeout("read timed out")

    path, name = rp.fetch_hdx("COD", Stalling(), tmp_path, {}, attempts=3, wait=0)
    assert path is None and name is None
    assert Stalling.calls == 3
