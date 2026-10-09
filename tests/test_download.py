import hashlib

from rtfp.download import latest_rtfp_csv, parse_file_listing, sha256_file

LISTING = {
    "status": "success",
    "files": [
        {"filename": "BGD_RTFP_mkt_2007_2026-10-05.zip", "dcdate": "2026-10-05", "resource_id": 3,
         "links": {"download": "https://x/3?filename=BGD_RTFP_mkt_2007_2026-10-05.zip"}},
        {"filename": "BGD_RTP_details_mkt_2007_2026-10-05.zip", "dcdate": "2026-10-05", "resource_id": 2,
         "links": {"download": "https://x/2"}},
        {"filename": "README.pdf", "dcdate": "2026-10-05", "resource_id": 1, "links": {"download": "https://x/1"}},
    ],
}


def test_parse_file_listing_keeps_only_market_price_files():
    rows = parse_file_listing(LISTING)
    assert [r["filename"] for r in rows] == ["BGD_RTFP_mkt_2007_2026-10-05.zip"]
    assert rows[0]["url"].startswith("https://x/3")
    assert rows[0]["vintage"] == "2026-10-05"


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "a.csv"
    p.write_bytes(b"price,1\n")
    assert sha256_file(p) == hashlib.sha256(b"price,1\n").hexdigest()


def test_latest_rtfp_csv_picks_newest_vintage_by_date_not_name_order(tmp_path):
    for v in ["2025-03-03", "2026-10-05", "2024-12-30"]:
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        (d / f"BGD_RTFP_mkt_2007_{v}.csv").write_text("x\n")
    assert latest_rtfp_csv(tmp_path).name == "BGD_RTFP_mkt_2007_2026-10-05.csv"


def test_latest_rtfp_csv_respects_as_of_cutoff(tmp_path):
    for v in ["2026-09-28", "2026-10-05", "2026-10-12"]:
        d = tmp_path / f"BGD_RTFP_mkt_2007_{v}"
        d.mkdir()
        (d / f"BGD_RTFP_mkt_2007_{v}.csv").write_text("x\n")
    assert latest_rtfp_csv(tmp_path, as_of="2026-10-05").name == "BGD_RTFP_mkt_2007_2026-10-05.csv"


def test_as_of_comes_from_environment(monkeypatch):
    from rtfp.download import as_of_date
    monkeypatch.setenv("RTFP_AS_OF", "2026-10-05")
    assert as_of_date() == "2026-10-05"
    monkeypatch.delenv("RTFP_AS_OF")
    assert as_of_date() is None


def test_hdx_file_picks_newest_on_or_before_as_of(tmp_path, monkeypatch):
    from rtfp.download import hdx_file
    monkeypatch.delenv("RTFP_AS_OF", raising=False)
    for v in ["2026-09-01", "2026-10-04", "2026-10-20"]:
        (tmp_path / f"wfp_food_prices_bgd_{v}.csv").write_text("x\n")
    assert hdx_file(tmp_path, as_of="2026-10-05").name == "wfp_food_prices_bgd_2026-10-04.csv"
    assert hdx_file(tmp_path, as_of=None).name == "wfp_food_prices_bgd_2026-10-20.csv"
