# Bangladesh Bureau of Statistics (BBS) data — third-party terms

**Source:** Bangladesh Bureau of Statistics, *Monthly Release on Price & Wage Statistics (CPI and WRI)*, national
food index, releases 2014-11 to 2026-09, downloaded from
https://bbs.gov.bd/pages/static-pages/6922de7a933eb65569e1ae8f on 2026-10-09. The file and SHA-256 hash of every
release used are in `data/external/bbs/MANIFEST.csv`; the release each month was read from is in the `source_file`
column.

**Terms.** These files are **not** covered by this repository's MIT or CC BY 4.0 licences. BBS states no licence.
Its publications print this notice (June 2026 monthly release, PDF page 5):

> This publication or any portion thereof cannot be copied or reproduced for any commercial purpose. Data therein
> can, however, be used and published with acknowledgement of their sources.

The files are published here on that basis: with acknowledgement of BBS as the source, for non-commercial research
use. Anyone reusing them must acknowledge BBS and must not reproduce them for commercial purposes. Details and the
other BBS documents checked are in `data/raw/LICENSES.md` §3.

| File | Contents |
|---|---|
| `bbs_food_cpi_monthly.csv` | One row per month: `month`, CPI `base` (2005-06 = 100 to 2023-03; 2021-22 = 100 from 2023-04), `food_index_national`, `food_inflation_yoy_national` (%), `source_file` (BBS release the value was read from). Up to the data cutoff `RTFP_AS_OF`. |
| `T7n_bbs_national_series.csv` | Monthly series behind Table 7n: RTFP published and survey-only national index changes (our computation from World Bank RTFP data) merged with BBS within-base log changes `d1` (1 month) and `d3` (3 months). |

**Changes made:** values were transcribed from the BBS releases by `src/rtfp/bbs.py` (one value per month from the
latest release printing it) and, for the second file, converted to within-base log changes by
`analysis/f7_bbs_national.py`. No splice across the 2023 rebasing. This is not an official BBS product, and BBS has
not endorsed this work.

**Regenerate:** `make external && make all` (writes both files).
