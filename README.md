# How much local price information do ML-imputed food prices carry?

Code and derived data for an independent, release-history evaluation of the World Bank's Real-Time Food Prices (RTFP) for Bangladesh: 110 markets, four items (rice, wheat flour, palm oil, lentils), monthly from 2007, judged against survey values that arrived after each release.

Independent research. Not affiliated with or endorsed by the World Bank, WFP, the Government of Bangladesh, BBS or TCB.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
make data       # RTFP Bangladesh vintages + WFP/HDX feed, with sha256 manifest (data/raw/MANIFEST.csv)
make external   # BBS monthly CPI and TCB daily Dhaka prices (no licence stated: downloaded, not redistributed)
make all        # every table in outputs/tables and every note in outputs/results_notes
make test       # test suite
```

Python 3.14 was used (`requirements.txt` pins every package). A full `make all` runs in a few minutes on a laptop.

## Layout

| Path | Contents |
|---|---|
| `src/rtfp/` | library: download, crosswalk, panel, vintages (hold-out), audit, noise, metrics, uncertainty, bbs, tcb |
| `analysis/` | one script per output table; `notes.py` generates results notes from tables |
| `tests/` | unit tests on toy data and invariant tests on the real data (skipped when data are absent) |
| `outputs/tables/` | derived tables (CC BY 4.0) |
| `outputs/results_notes/` | one note per table: claim, numbers, method, caveats (generated, not hand-written) |
| `docs/methodology.md` | design specification; `docs/analysis_plan.md` registered analysis plan (draft) |
| `data/raw/LICENSES.md` | terms of each source and what may be redistributed |

## Data versions

The analysis uses the newest RTFP Bangladesh release present in `data/raw/rtfp_vintages/` (at the time of writing 2026-10-05) and all 83 archived releases from 2024-01-25. File hashes are in `data/raw/MANIFEST.csv`.

## Status

Exploratory results up to release 2026-10-05. Confirmatory tests (H1, H2 in `docs/analysis_plan.md`) will use releases published after registration.
