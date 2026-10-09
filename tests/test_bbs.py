"""Tests for the BBS monthly CPI (national food index) loader.

Fixtures in tests/fixtures/bbs/ are excerpts of real BBS publications (non-commercial use, source acknowledged:
Bangladesh Bureau of Statistics, "Consumer Price Index (CPI), Inflation Rate and Wage Rate Index (WRI) in Bangladesh",
monthly release):
- cpi_2023-04_table1-1_page.pdf: page 7 of the April 2023 release (raw file e6d67a1ec98f4062997cb9568ed75546.pdf),
  the first release on the new base. Its Table 1.1 mixes bases: Feb'22..Mar'23 on 2005-06=100, Apr'23 on 2021-22=100.
- cpi_2026-09_tables1-1_1-3.xlsx: sheets "Table-1.1_CPI_Point to Point" and "Table-1.3_CPI_National" (values only,
  same cell positions) copied from the September 2026 Excel release (raw file 8cb1d4aa-7b1d-4236-ae48-d449afdd31be.xlsx).
"""
import csv
import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest

from rtfp.bbs import (
    MONTHLY_COLUMNS,
    base_for_month,
    build_monthly,
    parse_month_label,
    parse_pdf,
    parse_release_page,
    parse_table_text,
    parse_xlsx,
    within_base_log_change,
)

FIX = Path(__file__).parent / "fixtures" / "bbs"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "external" / "bbs"

# --- month labels and base rule -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Jan’22", "2022-01"),  # curly apostrophe
        ("Jan'15", "2015-01"),
        ("April'15", "2015-04"),
        ("Apri'16", "2016-04"),
        ("June'15", "2015-06"),
        ("Sept'19", "2019-09"),
        ("Janu'20", "2020-01"),
        ("Jnly'19", "2019-07"),  # BBS typo in the September 2020 release
        ("Jul'26", "2026-07"),
    ],
)
def test_parse_month_label_variants_seen_in_bbs_releases(label, expected):
    assert parse_month_label(label) == expected


@pytest.mark.parametrize("label", ["Foo'22", "Ju'22", "2022-01", "Jan 2022"])
def test_parse_month_label_rejects_unknown(label):
    with pytest.raises(ValueError):
        parse_month_label(label)


def test_base_for_month_switches_in_april_2023():
    assert base_for_month("2014-11") == "2005-06"
    assert base_for_month("2023-03") == "2005-06"
    assert base_for_month("2023-04") == "2021-22"
    assert base_for_month("2026-09") == "2021-22"


# --- text parser (PDF text layer) ------------------------------------------------------------------------------

OLD_TABLE = """Month: January 2016
Table 1. Consumer Price Index (CPI) And Inflation Rate
(Point To Point)
(2005-06=100)
CPI Classification 2012-13 2013-14 2014-15 2014-15 2015-16
Nov'14 Dec'14 Jan'15 Nov'15 Dec'15 Jan'16
NATIONAL
General index 181.73 195.08 207.58 206.86 207.78 210.12 219.38 220.45 222.87
Inflation 6.78 7.35 6.41 6.21 6.11 6.04 6.05 6.10 6.07
Food index 193.24 209.79 223.80 223.81 224.29 226.61 236.61 236.57 236.42
Inflation 5.22 8.56 6.68 6.44 5.86 6.07 5.72 5.48 4.33
Non-food index 166.97 176.23 186.79 185.14 186.62 188.98 197.29 199.78 205.50
Inflation 9.17 5.55 5.99 5.84 6.48 6.01 6.56 7.05 8.74
RURAL
General index 183.90 196.90 209.10 208.32 209.19 211.55 220.01 220.87 222.75
Inflation 6.14 7.07 6.20 6.05 5.89 5.81 5.61 5.58 5.29
Food index 192.14 207.72 221.02 221.00 221.47 223.73 232.04 232.02 231.86
Inflation 4.64 8.11 6.40 6.38 5.78 5.79 5.00 4.76 3.63
URBAN
Food index 195.91 214.85 230.56 230.65 231.16 233.62 247.76 247.67 247.55
Inflation 6.64 9.67 7.31 6.61 6.07 6.69 7.42 7.14 5.96
"""


def test_parse_table_text_takes_national_food_row_and_monthly_columns_only():
    df = parse_table_text(OLD_TABLE, source_file="x.pdf")
    assert list(df["month"]) == ["2014-11", "2014-12", "2015-01", "2015-11", "2015-12", "2016-01"]
    # national, not rural (221.00...) or urban (230.65...); annual columns (193.24, 209.79, 223.80) dropped
    assert list(df["food_index_national"]) == [223.81, 224.29, 226.61, 236.61, 236.57, 236.42]
    assert list(df["food_inflation_yoy_national"]) == [6.44, 5.86, 6.07, 5.72, 5.48, 4.33]
    assert set(df["base"]) == {"2005-06"}
    assert set(df["release_month"]) == {"2016-01"}
    assert set(df["source_file"]) == {"x.pdf"}


NO_SPACE_TABLE = """Table:1.1ConsumerPriceIndex(CPI)andInflationRate(PointtoPoint)
(BaseIndex2021-22=100)
CPIClassification 2023-24 2024-25 2025-26 2025-26 2026-27
Jul’25 Aug’25 Sep’25 Jul'26 Aug'26 Sep'26
1 2 3 4 5 6 7 8 9 10
NATIONAL
Generalindex 119.63 131.62 143.05 136.83 140.06 141.53 148.21 151.64 153.33
Inflation 9.73 10.03 8.68 8.55 8.29 8.36 8.32 8.26 8.34
Foodindex 120.30 133.16 143.90 138.36 143.88 145.54 148.26 153.98 156.04
Inflation 10.66 10.70 8.06 7.56 7.60 7.64 7.16 7.02 7.22
Non-foodindex 119.08 130.37 142.36 135.59 136.96 138.26 148.18 149.73 151.13
Inflation 8.86 9.47 9.20 9.38 8.90 8.98 9.28 9.32 9.30
RURAL
Foodindex 120.41 133.12 143.67 138.13 143.47 144.97 147.99 153.53 155.38
Inflation 10.68 10.56 7.93 7.36 7.50 7.54 7.14 7.01 7.18
"""


def test_parse_table_text_handles_layout_without_spaces():
    # The September 2026 PDF's text layer drops spaces between words ("Foodindex").
    df = parse_table_text(NO_SPACE_TABLE, source_file="sep26.pdf")
    assert list(df["month"]) == ["2025-07", "2025-08", "2025-09", "2026-07", "2026-08", "2026-09"]
    assert list(df["food_index_national"]) == [138.36, 143.88, 145.54, 148.26, 153.98, 156.04]
    assert list(df["food_inflation_yoy_national"]) == [7.56, 7.60, 7.64, 7.16, 7.02, 7.22]
    assert set(df["base"]) == {"2021-22"}
    assert set(df["release_month"]) == {"2026-09"}


def test_parse_table_text_values_on_line_after_label():
    # September 2020 release: "General index" label and its numbers sit on separate lines.
    text = OLD_TABLE.replace("Food index 193.24", "Food index\n193.24")
    df = parse_table_text(text, source_file="x.pdf")
    assert list(df["food_index_national"]) == [223.81, 224.29, 226.61, 236.61, 236.57, 236.42]
    assert list(df["food_inflation_yoy_national"]) == [6.44, 5.86, 6.07, 5.72, 5.48, 4.33]


def test_parse_table_text_rejects_wrong_number_of_values():
    text = OLD_TABLE.replace("223.81 224.29 226.61", "223.81 224.29")
    with pytest.raises(ValueError, match="values"):
        parse_table_text(text, source_file="x.pdf")


def test_parse_table_text_requires_inflation_row_after_food_row():
    text = OLD_TABLE.replace("Inflation 5.22 8.56", "Something 5.22 8.56")
    with pytest.raises(ValueError, match="Inflation"):
        parse_table_text(text, source_file="x.pdf")


def test_parse_table_text_rejects_old_base_month_when_page_declares_only_new_base():
    # A page that declares only 2021-22=100 but carries a pre-April-2023 month would mean BBS re-published old
    # months on the new base; the base rule would then be wrong, so the parser must refuse, not guess.
    text = NO_SPACE_TABLE.replace("Jul’25 Aug’25 Sep’25", "Jan’23 Feb’23 Mar’23")
    with pytest.raises(ValueError, match="2005-06"):
        parse_table_text(text, source_file="x.pdf")


def test_parse_table_text_accepts_interleaved_two_base_header():
    # January 2024 release text layer: "(Base Index 2005-06 (Base Index 2021-22\n=100) =100)".
    text = (OLD_TABLE.replace("(2005-06=100)", "(Base Index 2005-06 (Base Index 2021-22\n=100) =100)")
            .replace("Nov'14 Dec'14 Jan'15 Nov'15 Dec'15 Jan'16", "Nov’22 Dec’22 Jan’23 Nov’23 Dec’23 Jan’24")
            .replace("223.81 224.29 226.61 236.61 236.57 236.42", "360.75 356.86 359.40 120.04 117.48 118.30"))
    df = parse_table_text(text, source_file="jan24.pdf")
    assert list(df["month"]) == ["2022-11", "2022-12", "2023-01", "2023-11", "2023-12", "2024-01"]
    assert list(df["base"]) == ["2005-06"] * 3 + ["2021-22"] * 3
    assert list(df["food_index_national"]) == [360.75, 356.86, 359.40, 120.04, 117.48, 118.30]


def test_parse_table_text_rejects_mixed_table_whose_magnitudes_contradict_bases():
    # Old-base food levels (~300+) must exceed new-base levels (~100-160) in a mixed table.
    text = (OLD_TABLE.replace("(2005-06=100)", "(Base Index 2005-06=100) (Base Index 2021-22=100)")
            .replace("Nov'15 Dec'15 Jan'16", "Feb'23 Mar'23 Apr'23"))
    with pytest.raises(ValueError, match="magnitude"):
        parse_table_text(text, source_file="x.pdf")


def test_parse_table_text_requires_national_block():
    with pytest.raises(ValueError, match="NATIONAL"):
        parse_table_text(OLD_TABLE.replace("NATIONAL", "NATION"), source_file="x.pdf")


# --- real-file fixtures ----------------------------------------------------------------------------------------
# BBS states no licence, so these excerpts are kept out of git (see data/raw/LICENSES.md); skip when absent.
needs_fixtures = pytest.mark.skipif(not (FIX / "cpi_2023-04_table1-1_page.pdf").exists(),
                                    reason="BBS fixture files are not redistributed; run locally after download")


@needs_fixtures
def test_parse_pdf_fixture_mixed_base_release():
    df = parse_pdf(FIX / "cpi_2023-04_table1-1_page.pdf")
    assert list(df["month"]) == ["2022-02", "2022-03", "2022-04", "2023-02", "2023-03", "2023-04"]
    assert list(df["base"]) == ["2005-06"] * 5 + ["2021-22"]
    assert list(df["food_index_national"]) == [334.95, 337.43, 340.25, 362.17, 368.09, 111.26]
    assert list(df["food_inflation_yoy_national"]) == [6.22, 6.34, 6.24, 8.13, 9.09, 8.84]
    assert set(df["release_month"]) == {"2023-04"}
    assert set(df["source_file"]) == {"cpi_2023-04_table1-1_page.pdf"}


@needs_fixtures
def test_parse_xlsx_fixture_uses_point_to_point_sheet_national_food_row():
    df = parse_xlsx(FIX / "cpi_2026-09_tables1-1_1-3.xlsx")
    assert list(df["month"]) == ["2025-07", "2025-08", "2025-09", "2026-07", "2026-08", "2026-09"]
    assert set(df["base"]) == {"2021-22"}
    expected = [138.35793453982944, 143.87727678188043, 145.538190414969,
                148.26236949164223, 153.98426994474866, 156.04085599364507]
    assert df["food_index_national"].tolist() == pytest.approx(expected, abs=1e-12)
    assert df["food_inflation_yoy_national"].iloc[-1] == pytest.approx(7.216432710019352, abs=1e-12)
    assert set(df["release_month"]) == {"2026-09"}
    assert set(df["source_file"]) == {"cpi_2026-09_tables1-1_1-3.xlsx"}


# --- building the monthly series -------------------------------------------------------------------------------


def _obs(rows):
    cols = ["source_file", "release_month", "month", "base", "food_index_national", "food_inflation_yoy_national"]
    return pd.DataFrame(rows, columns=cols)


def test_build_monthly_takes_latest_release_and_prefers_xlsx_within_a_release():
    obs = _obs([
        ("a.pdf", "2024-11", "2024-11", "2021-22", 136.61, 13.80),
        ("b.pdf", "2024-12", "2024-11", "2021-22", 136.61, 13.80),
        ("b.xlsx", "2024-12", "2024-11", "2021-22", 136.6074, 13.8012),
        ("b.pdf", "2024-12", "2024-12", "2021-22", 132.65, 12.92),
    ])
    monthly, revisions = build_monthly(obs)
    assert list(monthly.columns) == MONTHLY_COLUMNS
    assert list(monthly["month"]) == ["2024-11", "2024-12"]
    assert list(monthly["source_file"]) == ["b.xlsx", "b.pdf"]
    assert monthly["food_index_national"].iloc[0] == 136.6074
    assert revisions.empty  # 136.6074 rounds to the PDF's 136.61


def test_build_monthly_reports_disagreement_between_releases():
    obs = _obs([
        ("a.pdf", "2020-01", "2020-01", "2005-06", 300.11, 5.12),
        ("b.pdf", "2021-01", "2020-01", "2005-06", 300.15, 5.12),
    ])
    monthly, revisions = build_monthly(obs)
    assert monthly["food_index_national"].iloc[0] == 300.15
    assert list(revisions["month"]) == ["2020-01"]
    assert revisions["field"].iloc[0] == "food_index_national"


def test_build_monthly_treats_rounded_and_full_precision_workbooks_as_agreeing():
    # The November 2024 workbook stores 2-decimal values; later workbooks store full precision.
    obs = _obs([
        ("nov24.xlsx", "2024-11", "2023-10", "2021-22", 123.90, 12.56),
        ("dec24.xlsx", "2024-12", "2023-10", "2021-22", 123.89609, 12.556359),
        ("dec24.xlsx", "2024-12", "2023-11", "2021-22", 120.044696, 10.763892),
        ("jan25.xlsx", "2025-01", "2023-11", "2021-22", 120.044697, 10.763892),  # beyond 1e-9: a revision
    ])
    monthly, revisions = build_monthly(obs)
    assert list(monthly["source_file"]) == ["dec24.xlsx", "jan25.xlsx"]
    assert list(revisions["month"]) == ["2023-11"]


def test_within_base_log_change_is_nan_across_base_break_and_gaps():
    monthly = pd.DataFrame({
        "month": ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-07"],
        "base": ["2005-06", "2005-06", "2005-06", "2021-22", "2021-22", "2021-22"],
        "food_index_national": [359.40, 362.17, 368.09, 111.26, 109.62, 112.74],
    })
    d = within_base_log_change(monthly, k=1)
    assert math.isnan(d.iloc[0])
    assert d.iloc[1] == pytest.approx(math.log(362.17 / 359.40))
    assert d.iloc[2] == pytest.approx(math.log(368.09 / 362.17))
    assert math.isnan(d.iloc[3])  # Mar'23 (2005-06) -> Apr'23 (2021-22): no common base
    assert d.iloc[4] == pytest.approx(math.log(109.62 / 111.26))
    assert math.isnan(d.iloc[5])  # Jun'23 missing
    d2 = within_base_log_change(monthly, k=2)
    assert d2.iloc[2] == pytest.approx(math.log(368.09 / 359.40))
    assert math.isnan(d2.iloc[4])  # May'23 vs Mar'23 crosses the base change


# --- release page ----------------------------------------------------------------------------------------------

PAGE = """<html><body>
<p><u>Monthly Release on Price &amp; Wage Statistics: (CPI and WRI)</u></p>
<figure class="table"><table><tbody>
<tr><td>Year-2025</td><td>PDF</td><td>Excel</td><td>Year-2026</td><td>PDF</td><td>Excel</td></tr>
<tr><td>January</td><td><a href="https://h/a.pdf">PDF</a></td><td><a href="https://h/a.xlsx">Excel</a></td>
    <td>January</td><td><a href="https://h/b.pdf">PDF</a></td><td><a href="https://h/b.xlsx">Excel </a></td></tr>
<tr><td>October</td><td><a href="https://h/c.pdf">PDF</a></td><td><a href="https://h/c.xlsx">Excel</a></td>
    <td></td><td></td><td></td></tr>
</tbody></table></figure>
<p><u>Monthly Release on Price &amp; Wage Statistics:</u></p>
<figure class="table"><table><tbody>
<tr><td> </td><td>Year: 2016</td><td>Year: 2017</td><td> </td></tr>
<tr><td> </td><td><p><a href="https://h/d.pdf">January </a></p><p><a href="https://h/e.pdf">February</a></p></td>
    <td><p><a href="https://h/f.pdf">January</a></p><p><a href="https://h/a.pdf">February</a></p></td></tr>
</tbody></table></figure>
<p><u>Industrial Production Statistics (IPS)</u></p>
<figure class="table"><table><tbody>
<tr><td>Year: 2019</td></tr><tr><td><p><a href="https://h/ips.pdf">January</a></p></td></tr>
</tbody></table></figure>
</body></html>"""


def test_parse_release_page_reads_both_table_layouts_and_skips_other_sections():
    links = parse_release_page(PAGE)
    got = [(r["page_year"], r["page_month"], r["format"], r["url"]) for r in links]
    assert got == [
        ("2025", "January", "pdf", "https://h/a.pdf"),
        ("2025", "January", "xlsx", "https://h/a.xlsx"),
        ("2026", "January", "pdf", "https://h/b.pdf"),
        ("2026", "January", "xlsx", "https://h/b.xlsx"),
        ("2025", "October", "pdf", "https://h/c.pdf"),
        ("2025", "October", "xlsx", "https://h/c.xlsx"),
        ("2016", "January", "pdf", "https://h/d.pdf"),
        ("2016", "February", "pdf", "https://h/e.pdf"),
        ("2017", "January", "pdf", "https://h/f.pdf"),
        ("2017", "February", "pdf", "https://h/a.pdf"),  # duplicate URL kept here; dedup is the caller's choice
    ]


# --- built data (skipped until `python -I -m rtfp.bbs build` has run) -------------------------------------------

needs_data = pytest.mark.skipif(not (DATA / "bbs_food_cpi_monthly.csv").exists(), reason="BBS data not built")


@needs_data
def test_data_monthly_schema_base_rule_and_no_gaps():
    m = pd.read_csv(DATA / "bbs_food_cpi_monthly.csv", dtype={"month": str, "base": str})
    assert list(m.columns) == MONTHLY_COLUMNS
    assert m["month"].is_unique and m["month"].is_monotonic_increasing
    assert all(m["base"] == m["month"].map(base_for_month))
    months = pd.PeriodIndex(m["month"], freq="M")
    assert len(months) == (months[-1] - months[0]).n + 1, "gap in monthly coverage"
    assert m[["food_index_national", "food_inflation_yoy_national"]].notna().all().all()


@needs_data
def test_data_published_yoy_matches_levels_where_both_months_share_a_base():
    m = pd.read_csv(DATA / "bbs_food_cpi_monthly.csv", dtype={"month": str, "base": str}).set_index("month")
    checked = 0
    for month, row in m.iterrows():
        prev = str(pd.Period(month, freq="M") - 12)
        if prev in m.index and m.at[prev, "base"] == row["base"]:
            implied = 100 * (row["food_index_national"] / m.at[prev, "food_index_national"] - 1)
            assert abs(implied - row["food_inflation_yoy_national"]) < 0.02, month
            checked += 1
    assert checked > 100


@needs_data
def test_data_fiscal_year_means_match_bbs_published_annual_food_index():
    # Independent check: BBS's own annual (July-June) food index in Table 1.6 "Historical Data" of the newest
    # workbook must equal the mean of our 12 monthly values for every complete fiscal year on the same base.
    import openpyxl

    with open(DATA / "MANIFEST.csv", newline="") as fh:
        xlsx = max((r for r in csv.DictReader(fh) if r["filename"].endswith(".xlsx")), key=lambda r: r["release_month"])
    wb = openpyxl.load_workbook(DATA / "raw" / xlsx["filename"], read_only=True, data_only=True)
    sheet = next(ws for ws in wb.worksheets if "Historical" in ws.title and "CPI" in ws.title)
    published, base = {}, None
    for row in sheet.iter_rows(values_only=True):
        label = str(row[0]).strip() if row[0] is not None else ""
        if label.startswith("Base:"):
            base = label.split(":")[1].split("=")[0].strip()
        elif len(label) == 7 and label[4] == "-" and isinstance(row[2], (int, float)):
            published[(base, label)] = float(row[2])
    wb.close()
    m = pd.read_csv(DATA / "bbs_food_cpi_monthly.csv", dtype={"month": str, "base": str})
    yr, mo = m["month"].str[:4].astype(int), m["month"].str[5:].astype(int)
    start = yr - (mo < 7)
    m["fy"] = start.astype(str) + "-" + ((start + 1) % 100).map("{:02d}".format)
    fy = m.groupby(["base", "fy"])["food_index_national"].agg(["size", "mean"])
    compared = 0
    for (b, f), r in fy[fy["size"] == 12].iterrows():
        assert (b, f) in published, (b, f)
        assert abs(r["mean"] - published[(b, f)]) < 0.01, (b, f, r["mean"], published[(b, f)])
        compared += 1
    assert compared >= 10


@needs_data
def test_data_manifest_hashes_match_raw_files():
    with open(DATA / "MANIFEST.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and list(rows[0]) == ["filename", "url", "release_month", "sha256"]
    for r in rows:
        h = hashlib.sha256((DATA / "raw" / r["filename"]).read_bytes()).hexdigest()
        assert h == r["sha256"], r["filename"]


@needs_data
def test_data_pdf_and_xlsx_of_the_same_release_agree_to_two_decimals():
    obs = pd.read_csv(DATA / "bbs_food_cpi_all_releases.csv", dtype={"month": str, "release_month": str})
    obs["fmt"] = obs["source_file"].str.rsplit(".", n=1).str[-1]
    both = obs.pivot_table(index=["release_month", "month"], columns="fmt", values="food_index_national").dropna()
    assert len(both) > 100
    assert ((both["pdf"] - both["xlsx"]).abs() <= 0.005 + 1e-9).all()
