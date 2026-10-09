"""Download RTFP vintages (World Bank Microdata API) and the WFP/HDX feed; keep a sha256 manifest.

Usage: python -m rtfp.download [--root data/raw]
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

RTFP_LISTING = "https://microdata.worldbank.org/index.php/api/downloads/{idno}/files?type=data"
HDX_PACKAGE = "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-bangladesh"
VINTAGE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
MANIFEST_FIELDS = ["source", "filename", "url", "vintage", "downloaded_at", "sha256"]


def parse_file_listing(listing: dict) -> list[dict]:
    """Keep only market-level price files (``*_RTFP_mkt_*``) from a Microdata download listing."""
    rows = []
    for f in listing.get("files", []):
        name = f["filename"]
        if "_RTFP_mkt_" not in name:
            continue
        m = VINTAGE_RE.search(name)
        rows.append({"filename": name, "url": f["links"]["download"], "vintage": m.group(1) if m else f["dcdate"]})
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_rtfp_csv(vintage_dir: Path) -> Path:
    """Return the CSV of the newest vintage, ordered by the date in the file name."""
    csvs = list(Path(vintage_dir).glob("BGD_RTFP_mkt_*/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no RTFP vintages under {vintage_dir}")
    return max(csvs, key=lambda p: VINTAGE_RE.search(p.name).group(1))


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {(r["source"], r["filename"]): r for r in csv.DictReader(fh)}


def _write_manifest(path: Path, rows: dict) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])


def download_rtfp(root: Path, manifest: dict, idno: str = "BGD_2021_RTFP_v02_M") -> None:
    dest = root / "rtfp_vintages"
    dest.mkdir(parents=True, exist_ok=True)
    for f in parse_file_listing(_fetch_json(RTFP_LISTING.format(idno=idno))):
        stem = f["filename"].removesuffix(".zip")
        csv_path = dest / stem / f"{stem}.csv"
        if not csv_path.exists():
            zpath = dest / f["filename"]
            urllib.request.urlretrieve(f["url"], zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest / stem)
            zpath.unlink()
        key = ("rtfp", csv_path.name)
        if key not in manifest:
            manifest[key] = {"source": "rtfp", "filename": csv_path.name, "url": f["url"], "vintage": f["vintage"],
                             "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
                             "sha256": sha256_file(csv_path)}


def download_hdx(root: Path, manifest: dict) -> None:
    dest = root / "hdx_wfp"
    dest.mkdir(parents=True, exist_ok=True)
    pkg = _fetch_json(HDX_PACKAGE)["result"]
    vintage = pkg["last_modified"][:10]
    res = next(r for r in pkg["resources"] if r["url"].endswith("wfp_food_prices_bgd.csv"))
    path = dest / f"wfp_food_prices_bgd_{vintage}.csv"
    if not path.exists():
        urllib.request.urlretrieve(res["url"], path)
    key = ("hdx_wfp", path.name)
    if key not in manifest:
        manifest[key] = {"source": "hdx_wfp", "filename": path.name, "url": res["url"], "vintage": vintage,
                         "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"), "sha256": sha256_file(path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw", type=Path)
    args = ap.parse_args()
    mpath = args.root / "MANIFEST.csv"
    manifest = _read_manifest(mpath)
    download_rtfp(args.root, manifest)
    download_hdx(args.root, manifest)
    _write_manifest(mpath, manifest)
    print(f"manifest rows: {len(manifest)}; latest RTFP: {latest_rtfp_csv(args.root / 'rtfp_vintages').name}")


if __name__ == "__main__":
    main()
