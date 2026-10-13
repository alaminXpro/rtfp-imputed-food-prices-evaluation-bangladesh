# How much local price information do ML-imputed food prices carry?

Code and derived data for an independent, release-history evaluation of the World Bank's Real-Time Food Prices (RTFP) for Bangladesh: 110 markets, four items (rice, wheat flour, palm oil, lentils), monthly from 2007, judged against survey values that arrived after each release.

Independent research. Not affiliated with or endorsed by the World Bank, WFP, the Government of Bangladesh, BBS or TCB.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
make data       # RTFP Bangladesh vintages + WFP/HDX feed, with sha256 manifest (data/raw/MANIFEST.csv)
make external   # BBS monthly CPI and TCB daily Dhaka prices (raw files downloaded, not redistributed)
make all        # every table, results note and figure (outputs/tables, outputs/results_notes, outputs/figures)
make replication  # five-country replication, Table 9 (downloads ~2.3 GB; run separately)
make test       # test suite
```

Python 3.14 was used (`requirements.txt` pins every package). `make all` took about 10 minutes on a laptop with single-threaded BLAS (set in the Makefile to keep the machine cool); `make replication` takes longer, mostly downloading.

**Data cutoff.** The Makefile sets `RTFP_AS_OF=2026-10-05`: releases, WFP/HDX files, BBS months and TCB months after that date are ignored, so the outputs reproduce exactly after newer releases appear. Override with `make all RTFP_AS_OF=YYYY-MM-DD`.

## Layout

| Path | Contents |
|---|---|
| `src/rtfp/` | library: download, crosswalk, panel, vintages (hold-out), audit, noise, metrics, ppml, uncertainty, simulate, alerts, replicate, bbs, tcb |
| `analysis/` | one script per output table; `notes.py` generates results notes from tables; `figures.py` the figures; `t9_replication.py` the replication |
| `outputs/figures/` | figures with generated captions (`FIGURES.md`) |
| `tests/` | unit tests on toy data and invariant tests on the real data (skipped when data are absent) |
| `outputs/tables/` | derived tables (CC BY 4.0) |
| `outputs/results_notes/` | one note per table: claim, numbers, method, caveats (generated, not hand-written) |
| `docs/methodology.md` | design specification; `docs/analysis_plan.md` analysis plan as registered on OSF (both files are frozen at the registered version) |
| `data/raw/LICENSES.md` | terms of each source and what may be redistributed |

## Data versions

The analysis uses the RTFP Bangladesh release of 2026-10-05 (the `RTFP_AS_OF` cutoff) and all 83 archived releases from 2024-01-25 to 2026-10-05. File hashes are in `data/raw/MANIFEST.csv`.

## Data sources and attribution

Full terms, quoted from each source, are in `data/raw/LICENSES.md`.

- **World Bank RTFP.** The World Bank: Monthly food price estimates by product and market: World Food Programme (via HDX) and FAO FPMA. Cite each release with its own version string, e.g. Andrée, B. P. J. (2021). *Monthly food price estimates by product and market* (Version 2026-10-05). BGD_2021_RTFP_v02_M. Washington, DC: World Bank Microdata Library. https://doi.org/10.48529/2ZH0-JF55. Releases used: 2024-01-25 to 2026-10-05. Licensed CC BY 4.0 with the World Bank's additional terms: https://www.worldbank.org/ext/en/legal/terms-conditions/datasets. The World Bank does not endorse this work.
- **WFP food prices (HDX).** World Food Programme, "Bangladesh - Food Prices", https://data.humdata.org/dataset/wfp-food-prices-for-bangladesh, licensed CC BY-IGO 3.0 (http://creativecommons.org/licenses/by/3.0/igo/legalcode). We transformed these data (matched to RTFP markets, converted to monthly log changes and errors); derived files are our changes, not WFP's. Raw files keep CC BY-IGO 3.0.
- **BBS food CPI.** Bangladesh Bureau of Statistics, *Monthly Release on Price & Wage Statistics*, national food index. No licence is stated; the publications print "This publication or any portion thereof cannot be copied or reproduced for any commercial purpose. Data therein can, however, be used and published with acknowledgement of their sources" (`data/raw/LICENSES.md` §3). On that basis the extracted monthly food index and the Table 7n series are in `data/third_party/bbs/`, acknowledged to BBS and **not** covered by this repository's MIT or CC BY 4.0 licences (no commercial reproduction; see that folder's README). The raw BBS PDF and Excel files are not redistributed; `make external` downloads them.
- **TCB Dhaka retail prices.** TCB states no licence or terms of use (§4); download scripts only, raw files processed locally (`make external`); Table A5 reports only summary statistics computed from TCB's daily bulletins, with TCB acknowledged as the source. Short BBS/TCB excerpts used as parser test values are in `tests/` (see `LICENSE`).
- **Replication countries.** RTFP releases for five other countries (World Bank Microdata Library) and their WFP/HDX feeds; only download listings, catalogue snapshots and manifests are in the repository.

## Status

Exploratory results up to release 2026-10-05.

**Registration.** The analysis plan (`docs/analysis_plan.md`) and design specification (`docs/methodology.md`) were registered on OSF on 2026-10-13: https://doi.org/10.17605/OSF.IO/3TJWN (SHA-256 of both files and the source commit `b69e994` are given in the registration). The prospective confirmatory tests (H1, H1f, H2) use only releases published after that date, once the minimum sample in §2 of the plan is reached; they are not part of the results here.
