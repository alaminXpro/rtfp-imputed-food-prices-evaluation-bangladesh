# How much local price information do ML-imputed food prices carry?

Code and derived data for an independent, release-history evaluation of the World Bank's Real-Time Food Prices (RTFP) for Bangladesh: 110 markets, four items (rice, wheat flour, palm oil, lentils), monthly from 2007, judged against survey values that arrived after each release.

Independent research. Not affiliated with or endorsed by the World Bank, WFP, the Government of Bangladesh, BBS or TCB.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
make data       # RTFP Bangladesh vintages + WFP/HDX feed, with sha256 manifest (data/raw/MANIFEST.csv)
make external   # BBS monthly CPI and TCB daily Dhaka prices (no licence stated: downloaded, not redistributed)
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
| `docs/methodology.md` | design specification; `docs/analysis_plan.md` registered analysis plan (draft) |
| `data/raw/LICENSES.md` | terms of each source and what may be redistributed |

## Data versions

The analysis uses the RTFP Bangladesh release of 2026-10-05 (the `RTFP_AS_OF` cutoff) and all 83 archived releases from 2024-01-25 to 2026-10-05. File hashes are in `data/raw/MANIFEST.csv`.

## Data sources and attribution

Full terms, quoted from each source, are in `data/raw/LICENSES.md`.

- **World Bank RTFP.** The World Bank: Monthly food price estimates by product and market: World Food Programme (via HDX) and FAO FPMA. Cite each release with its own version string, e.g. Andrée, B. P. J. (2021). *Monthly food price estimates by product and market* (Version 2026-10-05). BGD_2021_RTFP_v02_M. Washington, DC: World Bank Microdata Library. https://doi.org/10.48529/2ZH0-JF55. Releases used: 2024-01-25 to 2026-10-05. Licensed CC BY 4.0 with the World Bank's additional terms: https://www.worldbank.org/ext/en/legal/terms-conditions/datasets. The World Bank does not endorse this work.
- **WFP food prices (HDX).** World Food Programme, "Bangladesh - Food Prices", https://data.humdata.org/dataset/wfp-food-prices-for-bangladesh, licensed CC BY-IGO 3.0 (http://creativecommons.org/licenses/by/3.0/igo/legalcode). We transformed these data (matched to RTFP markets, converted to monthly log changes and errors); derived files are our changes, not WFP's. Raw files keep CC BY-IGO 3.0.
- **BBS food CPI and TCB Dhaka retail prices.** No licence is stated. We provide download scripts only; the BBS- and TCB-derived series are rebuilt locally (`make external`) and are not in this repository. The published tables report summary statistics from them with acknowledgement of the source, as the BBS publications permit.
- **Replication countries.** RTFP releases for five other countries (World Bank Microdata Library) and their WFP/HDX feeds; only download listings, catalogue snapshots and manifests are in the repository.

## Status

Exploratory results up to release 2026-10-05. Confirmatory tests (H1, H2 in `docs/analysis_plan.md`) will use releases published after registration.
