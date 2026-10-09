"""TCB (Trading Corporation of Bangladesh) daily Dhaka retail price bulletins: scrape, parse, aggregate.

Source: https://tcb.gov.bd/pages/daily-rmps ("প্রতিদিনের খুচরা বাজার দর"). Each listing entry links one
.xlsx bulletin on Oracle object storage. The date printed on the listing is not trusted: the date used
for every price row is read from inside the file (the date heading the "today's price" columns), and the
listing date is kept alongside it in MANIFEST.csv so mismatches can be audited.

Outputs (in --out, default data/external/tcb):
  raw/*.xlsx            downloaded bulletins, named after their object-storage path
  raw/listing/*.html    the listing pages as fetched
  MANIFEST.csv          one row per listed file: filename,url,listing_date,file_date,sha256,... status
  tcb_daily.csv         date,item_bn,item_en,variant,unit,min,max,source_file
                        (item_bn and unit as printed, Unicode-NFC with whitespace collapsed; min/max in
                        Tk per printed unit; a 0 or blank price means "not quoted" and is not written)
  tcb_monthly.csv       month,item_en,variant,n_days,mean_min,mean_max,mean_mid
  duplicate_date_conflicts.csv   dates where two files with the same in-file date disagree (rows dropped)

Usage: python -I -m rtfp.tcb [--out data/external/tcb] [--offline]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import math
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import warnings
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

LISTING_URL = "https://tcb.gov.bd/pages/daily-rmps"
PAGE_SIZE = 100  # the site's own page-size selector accepts 10/20/50/100 via ?page_size=
USER_AGENT = ("rtfp-attenuation/0.1 (academic research on food price data quality; "
              "non-commercial; at most 1 request per second)")
MIN_INTERVAL_S = 1.1

DAILY_COLUMNS = ["date", "item_bn", "item_en", "variant", "unit", "min", "max", "source_file"]
MONTHLY_COLUMNS = ["month", "item_en", "variant", "n_days", "mean_min", "mean_max", "mean_mid"]
MANIFEST_COLUMNS = ["filename", "url", "listing_date", "file_date", "sha256", "listing_no",
                    "listing_date_comment", "date_header", "date_tarikh", "date_text", "date_conflict",
                    "status", "note"]

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_INVISIBLE = {**dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff")), ord("\u00a0"): " "}  # drop ZW chars/BOM; NBSP->space


class TCBParseError(ValueError):
    """The file opened as a workbook but the TCB price table could not be located or read."""


class InvalidTCBFile(TCBParseError):
    """The file is not a readable .xlsx workbook (HTML error page, truncated download, legacy .xls, ...)."""


# --------------------------------------------------------------------------- text helpers

def bn_digits(s: str) -> str:
    """Replace Bengali digits with ASCII digits."""
    return s.translate(_BN_DIGITS)


def _norm(s) -> str:
    """NFC-normalise, drop zero-width characters, NBSP to space, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", s).strip()


def _n(s: str) -> str:
    """NFC form of a pattern literal, so that e.g. 'ড়' written precomposed or decomposed compares equal."""
    return unicodedata.normalize("NFC", s)


# --------------------------------------------------------------------------- item mapping

_MASUR = re.compile(_n("ম[শস][ুূ]র"))
_GRAIN = [(_n("বড়"), "large_grain"), (_n("মাঝার"), "medium_grain"), (_n("ছোট"), "small_grain")]
_PACKED = [_n("প্যাকেট"), _n("প্যাঃ"), _n("বোতল")]


def map_item(label) -> tuple[str, str] | None:
    """Map a printed TCB item label to (item_en, variant), or None if it is not a target item.

    Targets: coarse rice, loose white atta, loose palm oil, loose super palm oil (kept separate),
    and masur lentil by grain size (large/medium/small kept separate).
    """
    s = _norm(label)
    if not s:
        return None
    packed = any(p in s for p in _PACKED)
    if _MASUR.search(s) and _n("ডাল") in s:
        found = [v for k, v in _GRAIN if k in s]
        return ("lentil_masur", found[0] if len(found) == 1 else "unspecified")
    if s.startswith(_n("চাল")) and _n("মোটা") in s:
        return ("rice_coarse", "none")
    if (s.startswith(_n("আটা")) and _n("ময়দা") not in s and not packed
            and (_n("খোলা") in s or _n("লুজ") in s) and _n("লাল") not in s):
        return ("wheat_flour_atta_loose", "none")
    if _n("পাম") in s and not packed:
        if _n("সুপার") in s:
            return ("palm_oil_super_loose", "none")
        if _n("লুজ") in s or _n("খোলা") in s:
            return ("palm_oil_loose", "none")
    return None


# --------------------------------------------------------------------------- prices and dates

def parse_price(v) -> float:
    """Return a positive price as float, else NaN (blank, dash, error string, or 0 = not quoted)."""
    if v is None or isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) and x > 0 else math.nan
    s = bn_digits(_norm(v)).replace(",", "")
    if re.fullmatch(r"\d+(\.\d+)?", s):
        x = float(s)
        return x if x > 0 else math.nan
    return math.nan


def _is_unreadable_price(v) -> bool:
    """A non-empty cell that is neither a number nor an obvious 'missing' marker."""
    if v is None or isinstance(v, (int, float, dt.date)):
        return False
    s = bn_digits(_norm(v))
    return bool(s) and s not in {"-", "—", "–"} and not s.startswith("#") and math.isnan(parse_price(v))


_DMY = re.compile(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})")
_MONTHS = [("জানু", 1), ("ফেব্রু", 2), ("মার্চ", 3), ("এপ্রি", 4), ("মে", 5), ("জুন", 6), ("জুলা", 7),
           ("আগ", 8), ("সেপ্টে", 9), ("অক্টো", 10), ("নভে", 11), ("ডিসে", 12)]
_TEXT_DATE = re.compile(r"(\d{1,2})\s+(\S+?)\s*,?\s+(\d{4})\s*" + _n("খ্রি"))


def _mk_date(y, m, d) -> dt.date | None:
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def _to_date(v) -> dt.date | None:
    """A date from a cell: datetime, Excel serial number, or 'dd-mm-yyyy' text (Bengali or ASCII digits)."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, (int, float)):
        if 36526 <= v <= 54789:  # 2000-01-01 .. 2050-01-01 in the 1900 date system
            return (dt.datetime(1899, 12, 30) + dt.timedelta(days=int(v))).date()
        return None
    m = _DMY.search(bn_digits(_norm(v)))
    return _mk_date(m.group(3), m.group(2), m.group(1)) if m else None


def _text_date(s: str) -> dt.date | None:
    """Parse '... ১৭ মে ২০২১ খ্রিঃ ...' (Gregorian date written in Bengali)."""
    m = _TEXT_DATE.search(bn_digits(_norm(s)))
    if not m:
        return None
    word = m.group(2)
    months = [num for prefix, num in _MONTHS if word.startswith(_n(prefix))]
    return _mk_date(m.group(3), months[0], m.group(1)) if len(months) == 1 else None


# --------------------------------------------------------------------------- one bulletin

def _sniff(path: Path) -> str:
    head = Path(path).read_bytes()[:8]
    if not head:
        return "empty file"
    if head.startswith(b"PK"):
        return "zip container"
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        return "OLE2 container (legacy .xls or similar)"
    if head.lstrip().lower().startswith((b"<!do", b"<htm", b"<?xm", b"<")):
        return "HTML/XML text"
    if head.startswith(b"%PDF"):
        return "PDF"
    return f"unknown signature {head!r}"


def _cell(rows, r, c):
    return rows[r][c] if 0 <= r < len(rows) and 0 <= c < len(rows[r]) else None


def _find(rows, pred, r0=0, r1=None):
    for r in range(r0, len(rows) if r1 is None else min(r1, len(rows))):
        for c, v in enumerate(rows[r]):
            if isinstance(v, str) and pred(_norm(v)):
                return r, c
    return None


def _locate_table(rows) -> tuple[int, int, int, list[int], tuple[int, int]]:
    """Find the bulletin table: header row, label column, today's-min column, week-ago columns, sub-header."""
    hit = _find(rows, lambda s: s.startswith(_n("পণ্যের নাম")))
    if not hit:
        raise TCBParseError("no 'পণ্যের নাম' header")
    hr, name_c = hit
    today = [c for c, v in enumerate(rows[hr]) if isinstance(v, str) and _n("অদ্যকার") in _norm(v)]
    if not today:
        raise TCBParseError("no 'অদ্যকার মূল্য' (today's price) column")
    lo_c = today[0]
    week = [c for c, v in enumerate(rows[hr]) if isinstance(v, str) and _n("সপ্তাহ") in _norm(v)]
    sub = _find(rows, lambda s: s.startswith(_n("সর্বনিম্ন")), hr + 1, hr + 4)
    if sub is None or sub[1] != lo_c or _n("সর্ব") not in _norm(_cell(rows, sub[0], lo_c + 1)):
        raise TCBParseError("min/max sub-header not under the today's-price column")
    return hr, name_c, lo_c, week, sub


_END_MARKERS = (_n("যে সকল"), _n("মন্তব্য"))


def read_tcb_file(path) -> dict:
    """Read one bulletin. Returns file_date (from inside the file), the three in-file date fields,
    date_conflict, warnings, and rows: DataFrame[item_bn, item_en, variant, unit, min, max, wk_min, wk_max].

    file_date = the date heading the "অদ্যকার মূল্য" (today's price) columns; if absent, the "তারিখঃ"
    field; if absent, the Gregorian date in the heading text. Raises InvalidTCBFile / TCBParseError.
    """
    path = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb = openpyxl.load_workbook(path, data_only=True)
    except Exception as e:  # noqa: BLE001 - any failure to open means "not a readable xlsx"
        raise InvalidTCBFile(f"{_sniff(path)}; {type(e).__name__}: {e}") from None

    tables, errors = [], []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        try:
            tables.append((ws.title, rows, _locate_table(rows)))
        except TCBParseError as e:
            errors.append(f"sheet {ws.title!r}: {e}")
    if not tables:
        raise TCBParseError("; ".join(errors) or "workbook has no sheets")
    sheet, rows, (hr, name_c, lo_c, week, sub) = tables[0]

    warns = [f"{len(tables)} sheets hold a price table; used {sheet!r}"] if len(tables) > 1 else []
    date_header = _to_date(_cell(rows, hr + 1, lo_c))
    date_tarikh = None
    t = _find(rows, lambda s: _n("তারিখ") in s, 0, hr)
    if t:
        tr, tc = t
        date_tarikh = _to_date(_norm(rows[tr][tc]).split(_n("তারিখ"), 1)[1])
        for c in range(tc + 1, len(rows[tr])):
            if date_tarikh is None and rows[tr][c] not in (None, ""):
                date_tarikh = _to_date(rows[tr][c])
                break
    date_text = None
    for r in range(hr):
        for v in rows[r]:
            if isinstance(v, str) and date_text is None:
                date_text = _text_date(v)
    found = {k: v for k, v in [("header", date_header), ("tarikh", date_tarikh), ("text", date_text)] if v}
    if not found:
        raise TCBParseError("no date found inside the file")
    source = next(iter(found))
    file_date = found[source]
    conflict = len(set(found.values())) > 1

    out, seen = [], set()
    for r in range(sub[0] + 1, len(rows)):
        label = _cell(rows, r, name_c)
        if not isinstance(label, str):
            continue
        s = _norm(label)
        if s.startswith(_END_MARKERS):
            break
        key = map_item(s)
        if key is None:
            continue
        if key in seen:
            warns.append(f"duplicate row for {key} at row {r + 1}; kept the first")
            continue
        seen.add(key)
        raw_lo, raw_hi = _cell(rows, r, lo_c), _cell(rows, r, lo_c + 1)
        for v in (raw_lo, raw_hi):
            if _is_unreadable_price(v):
                warns.append(f"unreadable price {v!r} for {key}")
        wk_lo = _cell(rows, r, week[0]) if week else None
        wk_hi = _cell(rows, r, week[0] + 1) if week else None
        unit = _cell(rows, r, name_c + 1)
        out.append({"item_bn": s, "item_en": key[0], "variant": key[1],
                    "unit": _norm(unit) if unit is not None else "",
                    "min": parse_price(raw_lo), "max": parse_price(raw_hi),
                    "wk_min": parse_price(wk_lo), "wk_max": parse_price(wk_hi)})
    cols = ["item_bn", "item_en", "variant", "unit", "min", "max", "wk_min", "wk_max"]
    df = pd.DataFrame(out, columns=cols).astype({c: float for c in ["min", "max", "wk_min", "wk_max"]})
    return {"file_date": file_date, "date_header": date_header, "date_tarikh": date_tarikh,
            "date_text": date_text, "date_source": source, "date_conflict": conflict,
            "sheet": sheet, "warnings": warns, "rows": df}


# --------------------------------------------------------------------------- many bulletins

def parse_many(paths) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Parse files; return (daily, files, problems).

    daily: one row per (date, item_en, variant) with a price; date is the in-file date.
    files: per readable file, its in-file dates and row counts.
    problems: unreadable files (status 'invalid' / 'parse_error') and dates where two files disagree
    ('duplicate_date_conflict'; all conflicting rows for that date and item are dropped).
    Rows quoted as blank/0 for both min and max are not written to daily.
    """
    frames, files, problems = [], [], []
    for p in sorted(map(Path, paths), key=lambda p: p.name):
        try:
            f = read_tcb_file(p)
        except InvalidTCBFile as e:
            problems.append({"filename": p.name, "status": "invalid", "reason": str(e)})
            continue
        except TCBParseError as e:
            problems.append({"filename": p.name, "status": "parse_error", "reason": str(e)})
            continue
        rows = f["rows"]
        quoted = rows[rows["min"].notna() | rows["max"].notna()].copy()
        files.append({"filename": p.name, **{k: f[k] for k in
                      ["file_date", "date_header", "date_tarikh", "date_text", "date_source", "date_conflict"]},
                      "n_items": len(rows), "n_quoted": len(quoted), "warnings": "; ".join(f["warnings"])})
        quoted.insert(0, "date", f["file_date"])
        quoted["source_file"] = p.name
        frames.append(quoted)
    if not frames:
        return pd.DataFrame(columns=DAILY_COLUMNS), files, problems
    d = pd.concat(frames, ignore_index=True)
    key = ["date", "item_en", "variant"]
    vals = d[["unit", "min", "max"]].astype(str).agg("|".join, axis=1)
    n_versions = vals.groupby([d[k] for k in key]).transform("nunique")
    for (date, item, var), g in d[n_versions > 1].groupby(key):
        problems.append({"filename": "|".join(g.source_file), "status": "duplicate_date_conflict",
                         "reason": f"{date} {item} {var}: " + "; ".join(
                             f"{r.source_file}={r.min:g}-{r.max:g} {r.unit}" for r in g.itertuples())})
    d = d[n_versions == 1].drop_duplicates(key, keep="first")
    d = d.sort_values(key).reset_index(drop=True)
    return d[DAILY_COLUMNS], files, problems


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month means of daily min, max and midpoint per (item_en, variant); variants not merged.

    n_days counts days with at least one of min/max; mean_mid uses only days with both.
    Raises if an (item_en, variant) is quoted in more than one unit (means would mix units).
    """
    d = daily.copy()
    d["unit_key"] = d["unit"].astype(str).str.replace(r"\s+", "", regex=True)
    units = d.groupby(["item_en", "variant"])["unit_key"].nunique()
    if (units > 1).any():
        raise ValueError(f"more than one unit for {units[units > 1].index.tolist()}")
    d = d[d["min"].notna() | d["max"].notna()]
    d["month"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m")
    d["mid"] = (d["min"] + d["max"]) / 2
    g = d.groupby(["month", "item_en", "variant"], sort=True)
    m = g.agg(n_days=("date", "nunique"), mean_min=("min", "mean"), mean_max=("max", "mean"),
              mean_mid=("mid", "mean")).reset_index()
    return m[MONTHLY_COLUMNS]


# --------------------------------------------------------------------------- listing pages

class _ListingParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self._row, self._col, self.text = [], None, None, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = {"cells": {}, "hrefs": [], "comment_dates": []}
        elif tag == "td" and self._row is not None and a.get("data-column"):
            self._col = a["data-column"]
            self._row["cells"][self._col] = ""
        elif tag == "a" and self._row is not None and self._col == "files" and a.get("href"):
            self._row["hrefs"].append(a["href"])

    def handle_endtag(self, tag):
        if tag == "td":
            self._col = None
        elif tag == "tr" and self._row is not None:
            if "id" in self._row["cells"]:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        self.text.append(data)
        if self._row is not None and self._col:
            self._row["cells"][self._col] += data

    def handle_comment(self, data):
        if self._row is not None:
            m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", data)
            if m:
                self._row["comment_dates"].append(m.groups())


def filename_from_url(url: str) -> str:
    """Local name for a bulletin: its object path after 'office-tcb/', '/' -> '_', unsafe chars -> '_'."""
    path = urllib.parse.urlsplit(url).path
    m = re.search(r"/office-tcb/(.+)$", path)
    name = (m.group(1) if m else path.rsplit("/", 1)[-1]).replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name or hashlib.sha256(url.encode()).hexdigest()[:16]


def parse_listing_page(html: str) -> tuple[list[dict], int | None]:
    """Return (entries, total entry count reported by the page). One entry per linked file."""
    p = _ListingParser()
    p.feed(html)
    text = _norm("".join(p.text))
    m = re.search(_n("মোট") + r"\s*([0-9০-৯]+)\s*" + _n("এন্ট্রি"), text)
    total = int(bn_digits(m.group(1))) if m else None
    out = []
    for r in p.rows:
        c = r["cells"]
        ld = _to_date(c.get("publish_date", ""))
        cd = r["comment_dates"][-1] if r["comment_dates"] else None  # (M, D, YYYY)
        cdate = _mk_date(cd[2], cd[0], cd[1]) if cd else None
        base = {"listing_no": int(bn_digits(_norm(c["id"])) or 0),
                "title": _norm(c.get("title")), "publish_time": _norm(c.get("publish_time")),
                "listing_date": ld.isoformat() if ld else "",
                "listing_date_comment": cdate.isoformat() if cdate else ""}
        hrefs = r["hrefs"] or [""]
        for h in hrefs:
            out.append({**base, "url": h, "filename": filename_from_url(h) if h else ""})
    return out, total


# --------------------------------------------------------------------------- network

# tcb.gov.bd sends only its leaf certificate, without the intermediate, so OpenSSL-based clients cannot
# build the chain ("unable to get local issuer certificate"); browsers and macOS curl fetch it via AIA.
# This is that public intermediate, downloaded 2026-10-09 from the leaf's AIA URL
# http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt and checked with `openssl verify`
# against the system root "Sectigo Public Server Authentication Root R46". It is only an aid to chain
# building: tls_context() disables partial-chain acceptance, so every chain must still end at a system root.
SECTIGO_DV_R36_SHA256 = "8c54c334b66ba4e426772af4a3f9136c19a1aec729fdb28c535c07a5a4ef22e0"
SECTIGO_DV_R36_PEM = """\
-----BEGIN CERTIFICATE-----
MIIGTDCCBDSgAwIBAgIQOXpmzCdWNi4NqofKbqvjsTANBgkqhkiG9w0BAQwFADBf
MQswCQYDVQQGEwJHQjEYMBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTYwNAYDVQQD
Ey1TZWN0aWdvIFB1YmxpYyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gUm9vdCBSNDYw
HhcNMjEwMzIyMDAwMDAwWhcNMzYwMzIxMjM1OTU5WjBgMQswCQYDVQQGEwJHQjEY
MBYGA1UEChMPU2VjdGlnbyBMaW1pdGVkMTcwNQYDVQQDEy5TZWN0aWdvIFB1Ymxp
YyBTZXJ2ZXIgQXV0aGVudGljYXRpb24gQ0EgRFYgUjM2MIIBojANBgkqhkiG9w0B
AQEFAAOCAY8AMIIBigKCAYEAljZf2HIz7+SPUPQCQObZYcrxLTHYdf1ZtMRe7Yeq
RPSwygz16qJ9cAWtWNTcuICc++p8Dct7zNGxCpqmEtqifO7NvuB5dEVexXn9RFFH
12Hm+NtPRQgXIFjx6MSJcNWuVO3XGE57L1mHlcQYj+g4hny90aFh2SCZCDEVkAja
EMMfYPKuCjHuuF+bzHFb/9gV8P9+ekcHENF2nR1efGWSKwnfG5RawlkaQDpRtZTm
M64TIsv/r7cyFO4nSjs1jLdXYdz5q3a4L0NoabZfbdxVb+CUEHfB0bpulZQtH1Rv
38e/lIdP7OTTIlZh6OYL6NhxP8So0/sht/4J9mqIGxRFc0/pC8suja+wcIUna0HB
pXKfXTKpzgis+zmXDL06ASJf5E4A2/m+Hp6b84sfPAwQ766rI65mh50S0Di9E3Pn
2WcaJc+PILsBmYpgtmgWTR9eV9otfKRUBfzHUHcVgarub/XluEpRlTtZudU5xbFN
xx/DgMrXLUAPaI60fZ6wA+PTAgMBAAGjggGBMIIBfTAfBgNVHSMEGDAWgBRWc1hk
lfmSGrASKgRieaFAFYghSTAdBgNVHQ4EFgQUaMASFhgOr872h6YyV6NGUV3LBycw
DgYDVR0PAQH/BAQDAgGGMBIGA1UdEwEB/wQIMAYBAf8CAQAwHQYDVR0lBBYwFAYI
KwYBBQUHAwEGCCsGAQUFBwMCMBsGA1UdIAQUMBIwBgYEVR0gADAIBgZngQwBAgEw
VAYDVR0fBE0wSzBJoEegRYZDaHR0cDovL2NybC5zZWN0aWdvLmNvbS9TZWN0aWdv
UHVibGljU2VydmVyQXV0aGVudGljYXRpb25Sb290UjQ2LmNybDCBhAYIKwYBBQUH
AQEEeDB2ME8GCCsGAQUFBzAChkNodHRwOi8vY3J0LnNlY3RpZ28uY29tL1NlY3Rp
Z29QdWJsaWNTZXJ2ZXJBdXRoZW50aWNhdGlvblJvb3RSNDYucDdjMCMGCCsGAQUF
BzABhhdodHRwOi8vb2NzcC5zZWN0aWdvLmNvbTANBgkqhkiG9w0BAQwFAAOCAgEA
YtOC9Fy+TqECFw40IospI92kLGgoSZGPOSQXMBqmsGWZUQ7rux7cj1du6d9rD6C8
ze1B2eQjkrGkIL/OF1s7vSmgYVafsRoZd/IHUrkoQvX8FZwUsmPu7amgBfaY3g+d
q1x0jNGKb6I6Bzdl6LgMD9qxp+3i7GQOnd9J8LFSietY6Z4jUBzVoOoz8iAU84OF
h2HhAuiPw1ai0VnY38RTI+8kepGWVfGxfBWzwH9uIjeooIeaosVFvE8cmYUB4TSH
5dUyD0jHct2+8ceKEtIoFU/FfHq/mDaVnvcDCZXtIgitdMFQdMZaVehmObyhRdDD
4NQCs0gaI9AAgFj4L9QtkARzhQLNyRf87Kln+YU0lgCGr9HLg3rGO8q+Y4ppLsOd
unQZ6ZxPNGIfOApbPVf5hCe58EZwiWdHIMn9lPP6+F404y8NNugbQixBber+x536
WrZhFZLjEkhp7fFXf9r32rNPfb74X/U90Bdy4lzp3+X1ukh1BuMxA/EEhDoTOS3l
7ABvc7BYSQubQ2490OcdkIzUh3ZwDrakMVrbaTxUM2p24N6dB+ns2zptWCva6jzW
r8IWKIMxzxLPv5Kt3ePKcUdvkBU/smqujSczTzzSjIoR5QqQA6lN1ZRSnuHIWCvh
JEltkYnTAH41QJ6SAWO66GrrUESwN/cgZzL4JLEqz1Y=
-----END CERTIFICATE-----
"""


def tls_context() -> ssl.SSLContext:
    """Default verifying context plus the pinned Sectigo intermediate; partial chains not accepted."""
    der = ssl.PEM_cert_to_DER_cert(SECTIGO_DV_R36_PEM)
    if hashlib.sha256(der).hexdigest() != SECTIGO_DV_R36_SHA256:
        raise RuntimeError("embedded intermediate certificate does not match its pinned fingerprint")
    ctx = ssl.create_default_context()
    ctx.load_verify_locations(cadata=SECTIGO_DV_R36_PEM)
    if hasattr(ssl, "VERIFY_X509_PARTIAL_CHAIN"):
        ctx.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


class PoliteFetcher:
    """Sequential GETs with a minimum interval between requests and a descriptive User-Agent."""

    def __init__(self, min_interval: float = MIN_INTERVAL_S, user_agent: str = USER_AGENT, retries: int = 3):
        self.min_interval, self.ua, self.retries, self._last = min_interval, user_agent, retries, 0.0
        self.ctx = tls_context()

    def get(self, url: str) -> bytes:
        for attempt in range(self.retries):
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.ua})
                with urllib.request.urlopen(req, timeout=90, context=self.ctx) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                if e.code < 500 or attempt == self.retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
            time.sleep(5 * (attempt + 1))
        raise RuntimeError("unreachable")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_listing(fetcher: PoliteFetcher, listing_dir: Path) -> tuple[list[dict], int]:
    listing_dir.mkdir(parents=True, exist_ok=True)
    entries, total, page = [], None, 1
    while True:
        html = fetcher.get(f"{LISTING_URL}?page={page}&page_size={PAGE_SIZE}").decode("utf-8")
        (listing_dir / f"listing_page_{page:03d}.html").write_text(html, encoding="utf-8")
        rows, t = parse_listing_page(html)
        total = total or t
        entries += rows
        if not rows or total is None or page * PAGE_SIZE >= total:
            break
        page += 1
    return entries, total


def read_saved_listing(listing_dir: Path) -> tuple[list[dict], int]:
    entries, total = [], None
    for f in sorted(listing_dir.glob("listing_page_*.html")):
        rows, t = parse_listing_page(f.read_text(encoding="utf-8"))
        total = total or t
        entries += rows
    return entries, total


def download_files(entries: list[dict], raw_dir: Path, fetcher: PoliteFetcher) -> dict:
    """Download each linked file once (skips files already on disk). Returns {filename: error}."""
    errors = {}
    for e in {e["filename"]: e for e in entries if e["url"]}.values():
        dest = raw_dir / e["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            data = fetcher.get(e["url"])
        except Exception as ex:  # noqa: BLE001 - record and continue
            errors[e["filename"]] = f"{type(ex).__name__}: {ex}"
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
    return errors


def assemble(entries: list[dict], raw_dir: Path, dl_errors: dict | None = None
             ) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    """Parse the downloaded files behind listing entries. Returns (manifest rows, daily, problems).

    Manifest rows keep the listing date and the in-file date side by side; the daily rows are dated from
    the file content only. Notes flag a listing date that differs from the in-file date, a missing listing
    date, and files whose bytes duplicate another listed file.
    """
    dl_errors = dl_errors or {}
    by_file: dict[str, list[dict]] = {}
    for e in entries:
        by_file.setdefault(e["filename"], []).append(e)
    paths = [raw_dir / f for f in by_file if f and f not in dl_errors and (raw_dir / f).exists()]
    daily, files, problems = parse_many(paths)
    finfo = {f["filename"]: f for f in files}
    pinfo = {p["filename"]: p for p in problems if p["status"] in ("invalid", "parse_error")}
    hashes = {f: sha256_file(raw_dir / f) for f in by_file if f and (raw_dir / f).exists()}
    first_with_hash: dict[str, str] = {}
    for f in sorted(hashes, key=lambda f: min(e["listing_no"] for e in by_file[f])):
        first_with_hash.setdefault(hashes[f], f)

    def iso(d):
        return d.isoformat() if d else ""

    manifest = []
    for fname, es in by_file.items():
        f, p = finfo.get(fname), pinfo.get(fname)
        status = ("no_file_link" if not fname else "download_error" if fname in dl_errors
                  else p["status"] if p else "ok")
        notes = [n for n in [dl_errors.get(fname), p["reason"] if p else "", f["warnings"] if f else ""] if n]
        listing_dates = [e["listing_date"] for e in es]
        if f and any(ld and ld != iso(f["file_date"]) for ld in listing_dates):
            notes.append("listing date differs from in-file date")
        if not all(listing_dates):
            notes.append("no listing date on the site")
        h = hashes.get(fname, "")
        if h and first_with_hash[h] != fname:
            notes.append(f"same bytes as {first_with_hash[h]}")
        manifest.append({
            "filename": fname, "url": es[0]["url"], "listing_date": ";".join(listing_dates),
            "file_date": iso(f["file_date"]) if f else "", "sha256": h,
            "listing_no": ";".join(str(e["listing_no"]) for e in es),
            "listing_date_comment": ";".join(e["listing_date_comment"] for e in es),
            "date_header": iso(f["date_header"]) if f else "", "date_tarikh": iso(f["date_tarikh"]) if f else "",
            "date_text": iso(f["date_text"]) if f else "", "date_conflict": f["date_conflict"] if f else "",
            "status": status, "note": "; ".join(notes)})
    manifest.sort(key=lambda r: int(r["listing_no"].split(";")[0] or 0))
    return manifest, daily, problems


def build(out: Path, offline: bool = False) -> dict:
    raw_dir, listing_dir = out / "raw", out / "raw" / "listing"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if offline:
        entries, total = read_saved_listing(listing_dir)
        dl_errors = {}
    else:
        fetcher = PoliteFetcher()
        entries, total = fetch_listing(fetcher, listing_dir)
        dl_errors = download_files(entries, raw_dir, fetcher)
    manifest, daily, problems = assemble(entries, raw_dir, dl_errors)
    with open(out / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(manifest)
    daily.to_csv(out / "tcb_daily.csv", index=False)
    monthly_summary(daily).to_csv(out / "tcb_monthly.csv", index=False)
    conflicts = [p for p in problems if p["status"] == "duplicate_date_conflict"]
    pd.DataFrame(conflicts, columns=["filename", "status", "reason"]).to_csv(
        out / "duplicate_date_conflicts.csv", index=False)
    return {"listing_total": total, "entries": len(entries), "files": len(manifest), "manifest": manifest,
            "daily": daily, "problems": problems}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/external/tcb", type=Path)
    ap.add_argument("--offline", action="store_true", help="re-parse saved listing pages and files only")
    a = ap.parse_args(argv)
    res = build(a.out, offline=a.offline)
    st = pd.Series([m["status"] for m in res["manifest"]]).value_counts()
    print(f"listing total reported: {res['listing_total']}; entries parsed: {res['entries']}; "
          f"distinct files: {res['files']}")
    print("status counts:", st.to_dict())
    d = res["daily"]
    if len(d):
        print(f"daily rows: {len(d)}; dates {d.date.min()} .. {d.date.max()}")
        print(d.groupby(["item_en", "variant"]).date.nunique().to_string())


if __name__ == "__main__":
    sys.exit(main())
