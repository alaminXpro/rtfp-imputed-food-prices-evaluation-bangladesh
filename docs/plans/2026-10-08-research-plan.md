# Variance Attenuation in ML-Imputed Food Prices — Research Engineering Plan

> Scope: everything except manuscript writing (data, engineering, statistics, figures, reproducibility).
> Owner: Md Alamin (engineering/statistics). Co-author: writing. Handoff format in §8.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the evidence base (tables, figures, numbers, methods facts) showing whether and how much the World Bank's ML-imputed Real-Time Food Prices (RTFP) attenuate price-change variance in Bangladesh, what that costs downstream, and how to fix it.

**Approach:** Use the dataset's own observed-value columns to separate observed from imputed cells. Identify attenuation with three designs that rule out "observed markets are just more volatile": (1) within-market/within-month fixed effects, (2) a *natural hold-out* from 83 archived RTFP vintages (cells first imputed, later observed), (3) a masking experiment with a replica imputer. Then quantify consequences (alerts, market integration, local shocks, forecast intervals) and test a variance-restoring remedy.

**Tech stack:** Python 3.12, pandas, numpy, statsmodels, linearmodels (panel FE), scikit-learn (replica imputer), lightgbm (forecast consequence), matplotlib; git; pytest for data-integrity tests.

## Global constraints

- Primary dataset: RTFP Bangladesh **version 2026-10-05** (`BGD_RTFP_mkt_2007_2026-10-05.csv`, 26,418 rows, 111 markets, 2007-01 to 2026-10). Before the final run, re-pull and switch to whatever is newest; record the version in every output.
- Commodities: rice (coarse), wheat flour, oil (palm), lentils (masur). Unit of analysis: market × month × commodity.
- Observed cell = non-missing raw column (`rice`, `wheat_flour`, `oil`, `lentils`). Imputed cell = raw missing and `c_<item>` present.
- Main outcome: monthly log change `Δp = log c_t − log c_{t−1}`.
- Every number handed to the co-author must come from a script in `analysis/`, never from a notebook cell.
- Confirmatory tests (H1–H5) are fixed in §2 **before** they are run (Task 0.3). Anything else is labelled exploratory.

## Review focus (failure modes most likely to sink the paper)

1. **Selection:** observed markets (e.g. Cox's Bazar, the camp markets) are inherently more volatile. Handled by Tasks 2.2 and 3.1–3.2. If attenuation disappears there, pivot (Gate G1).
2. **Raw column semantics:** the "raw = observed" reading is inferred, not documented. `c_` matches raw in only about 90% of observed cells. Handled by Task 1.2 and the optional help-desk email (§9).
3. **Vintage artefacts:** markets were renamed or added (74 → 77 → 111 markets), so "imputed then observed" may really be a re-keyed market. Handled by Task 3.1 (stable `geo_id` keys, drop renamed markets).
4. **Composition over time:** observed cells cluster in 2020+, which is a high-inflation period. Pooled variance comparisons are therefore meaningless; always use month fixed effects or within-month ratios.
5. **Circular "ground truth":** DAM feeds WFP, which feeds RTFP, and Dhaka is observed throughout. External checks must use only cells that never entered the model (Task 5.x).
6. **"Imputation just removes noise":** observed prices contain measurement noise, so part of the lower variance may be a *good* thing. Handled by Task 2.4 (λ vs λ*). This is the objection most likely to come from a strong reviewer.
7. **Wrong diagnostic:** a Mincer–Zarnowitz slope ≈ 1 does *not* rule out attenuation. Use the reverse slope and variance ratios on idiosyncratic changes (`docs/methodology.md` Proposition 2).

---

## 1. What we already have (done 2026-10-08)

| Item | Location | Status |
|---|---|---|
| 83 RTFP Bangladesh vintages, 2024-01-25 → 2026-10-05 | `data/raw/rtfp_vintages/` | downloaded |
| WFP/HDX Bangladesh prices (the RTFP input feed), 2026-10-04 | `data/raw/hdx_wfp/` | downloaded |
| Quick-look attenuation test | `analysis/01_attenuation_quicklook.py` | within-month SD ratio, imputed ÷ observed = 0.43–0.55; imputed smoother in 90–92% of months |
| Natural hold-out check | (scratch) | ~1,100–1,450 imputed-then-observed cells per item |

## 2. Research questions and pre-specified hypotheses

- **H1 (existence):** Conditional on market and month fixed effects, imputed cells have smaller idiosyncratic price changes `|e_mt|` than observed cells. `e_mt = Δp_mt − mean_t(Δp)` (national month mean).
- **H2 (natural hold-out):** For cells imputed in an earlier vintage and observed later, the variance of `e_mt` in the imputed version is lower than in the later observed version of the *same* cell.
- **H3 (mechanism):** Attenuation grows with distance to the nearest observed market, gap length (months since the market's last observation) and low trust score. It is largest for `spatially_interpolated = 1` markets.
- **H4 (co-movement):** Pairwise correlations of Δp are higher for imputed–imputed market pairs than for observed–observed pairs at the same distance, so market-integration estimates are inflated.
- **H5 (alerts):** Price-spike alerts (WFP ALPS and FAO IPA) computed from imputed values have lower recall against alerts computed from observed values.
- **Exploratory:** a local-shock event study (August 2024 floods), forecast-interval coverage, the remedy, and the multi-country extension.

---

## 3. Repository layout

```
DA Paper/
├── data/
│   ├── raw/                 # never edited; downloaded files + MANIFEST.csv (url, date, sha256)
│   ├── interim/             # harmonised panels (parquet)
│   └── external/            # TCB xlsx, DAM pdf/csv (Phase 5)
├── src/rtfp/
│   ├── download.py          # pull latest + all vintages via World Bank API; write manifest
│   ├── panel.py             # build market×month×item panel with flags & covariates
│   ├── vintages.py          # stack vintages; detect imputed→observed cells
│   ├── metrics.py           # idiosyncratic change, variance ratios, spike rates, ALPS, IPA
│   ├── replica_imputer.py   # RTFP-like chained-equation imputer for masking experiment
│   └── remedy.py            # variance-restoring post-processing
├── analysis/                # one script per table/figure: 10_*.py … 70_*.py
├── tests/                   # pytest data-integrity tests
├── outputs/{tables,figures,results_notes}/
├── docs/plans/              # this file
└── requirements.txt
```

---

## Phase 0: Setup and data freeze (week 1)

### Task 0.1: Project scaffolding and reproducibility
- [ ] `git init`; add a `.gitignore` covering `data/raw/`, `data/interim/` and `.venv/`. Commit the code only, never the data.
- [ ] Write `requirements.txt` with exact versions (`pip freeze`) and commit it.
- [ ] Write `src/rtfp/download.py`:
  - query `https://microdata.worldbank.org/index.php/api/downloads/BGD_2021_RTFP_v02_M/files?type=data`
  - download every `*_RTFP_mkt_*.zip`, `*_details_*` and `*_ticker_info_*` file
  - write `data/raw/MANIFEST.csv` (filename, URL, download timestamp, sha256)
- [ ] Do the same for the HDX WFP file: `https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-bangladesh`.
- **Check:** re-running `download.py` reproduces identical sha256 values for existing files.

### Task 0.2: Data licences and provenance
- [ ] Record the licence terms in `data/raw/LICENSES.md`:
  - RTFP: World Bank terms of use (attribution required).
  - HDX/WFP: read the licence on the HDX page.
  - TCB/DAM: public government data; note the source URL.
- **Deliverable to co-author:** the data-availability statement facts.

### Task 0.3: Freeze the analysis plan
- [ ] Copy §2 (hypotheses), plus the test and threshold choices from Phases 2–4, into `docs/analysis_plan.md`.
- [ ] Commit it, and optionally post it on OSF as a time-stamped pre-registration.
- **Why:** reviewers trust confirmatory tests fixed before results are seen. It costs half a day.

## Phase 1: Build the panels (weeks 1–2)

### Task 1.1: Market × month × item panel (`src/rtfp/panel.py`)
- [ ] Reshape the latest vintage to long format, one row per (geo_id, month, item). Keep these columns:
  - raw price and `c_`, `o_`, `h_`, `l_`
  - `trust_<item>`, `spatially_interpolated`, `data_coverage`, `data_coverage_recent`, `start_dense_data`, `last_survey_point`
  - `lat`, `lon`, `adm1`, `adm2`
- [ ] Add derived columns:
  - `obs` (bool)
  - `dlog`
  - `e` (idiosyncratic change)
  - `gap_len` (months since the last observed value in that market and item)
  - `dist_obs_km` (haversine distance to the nearest market observed in the same month)
  - `n_obs_month` (number of markets observed that month)
- [ ] Save to `data/interim/panel_latest.parquet`.
- **Tests (`tests/test_panel.py`):**
  - `(geo_id, month, item)` is unique
  - `dlog` is NaN only at each series' first month
  - `dist_obs_km == 0` whenever `obs` is True
  - row count = 26,418 × 4 items minus all-missing rows

### Task 1.2: Audit the "raw = observed" assumption
- [ ] Match RTFP observed cells to HDX WFP Retail rows (by market name or nearest lat/lon within 5 km, month and commodity).
- [ ] Report the share that match, and the price ratio raw ÷ HDX.
- [ ] Characterise the ~10% of observed cells where `c_ ≠ raw`: size, direction, whether they are outliers (the RTFP metadata says outliers are adjusted by "non-parametric density estimation").
- **Output:** `outputs/tables/T_A1_observed_flag_audit.csv` plus a results note.
- **Check:** at least 90% of raw values are found in HDX; if not, stop and resolve before Phase 2.

### Task 1.3: Vintage panel (`src/rtfp/vintages.py`)
- [ ] Stack all 83 vintages into `data/interim/vintages.parquet` with columns (vintage_date, geo_id, month, item, raw, c_, trust). Key on `geo_id`, not `mkt_name`.
- [ ] For each cell, record `first_vintage`, `status_by_vintage` and `first_obs_vintage`.
- [ ] Define the **hold-out set:** cells imputed in vintage v and observed in a later vintage v′, excluding markets whose `geo_id` changed.
- **Output:** `T_A2_holdout_counts.csv` (count by item and year).
- **Check:** counts are of the same order as the quick check (~1,100–1,450 per item).

## Phase 2: Core attenuation evidence (weeks 2–3)

### Task 2.0: Theory simulation (verifies Propositions 1–6 before touching real data)
- [ ] Write `src/rtfp/simulate.py`, which generates synthetic panels:
  - true prices = common AR(1) + local AR(1) + seasonality
  - observed = true + white noise ν
  - a random missingness pattern
- [ ] Impute with conditional-mean and stochastic imputers. Check numerically:
  - Proposition 1: λ_common ≈ 1 and λ_idio ≈ R²
  - Proposition 2: MZ slope ≈ 1 while the reverse slope = λ
  - Proposition 4: imputed–imputed correlation inflation
  - Proposition 6: the MA(1) estimator recovers Var(ν)
- **Tests (`tests/test_simulate.py`):** each check passes within a tolerance of ±0.05 over 200 simulated panels.
- **Output:** **Appendix Table A0**. Gives the co-author a verified theory section.

### Task 2.1: Descriptive attenuation (H1, descriptive)
- [ ] Compute the within-month SD ratio of `e` (imputed ÷ observed) per item; bootstrap the 95% CI by resampling markets 1,000 times.
- [ ] Compute these shares for imputed vs observed cells:
  - spike shares, |Δp| > 5%, 10% and 20%
  - kurtosis
  - share of zero or near-zero changes
- [ ] Decompose variance: Var(Δp) = Var(common_t) + Var(e), separately for imputed and observed.
- **Outputs:**
  - **Table 2** (descriptives by item × status)
  - **Fig 1** (example market: observed points vs imputed line)
  - **Fig 2** (distribution of `e`, observed vs imputed, by item)

### Task 2.2: Fixed-effects test robust to selection (H1, confirmatory)
- [ ] Regression: `log|e_mt| = β·imputed_mt + α_m + γ_t + ε`, per item and pooled with item×month FE. Use standard errors clustered by market, with Driscoll–Kraay as a robustness check.
- [ ] Alternatives: a Brown–Forsythe variance test within each month, and a quantile regression of `|e|` at the 50th and 90th percentiles.
- [ ] **Switchers-only sample:** markets with both observed and imputed spells, 2020 onward.
- **Output:** **Table 3**.
- **Gate G1:**
  - If β < 0 and significant in the switchers sample for at least 3 of 4 items, proceed.
  - If not, the attenuation is mostly selection. Pivot the paper to "external validation and revision analysis" (Phases 3.1, 5, 6 still apply).

### Task 2.3: Mechanism (H3)
- [ ] Regress `log|e|` on `dist_obs_km`, `gap_len`, `trust`, `spatially_interpolated` and `n_obs_month`, imputed cells only, with month FE.
- [ ] Plot the binned relationship between attenuation and distance to the nearest observed market.
- **Outputs:** **Table 4** and **Fig 3** (attenuation vs distance and gap length).

### Task 2.4: Noise vs signal decomposition (answers "imputation just removes noise")
- [ ] Estimate measurement-noise variance in observed prices with the MA(1) autocovariance estimator: Var(ν) = −Cov(Δy_t, Δy_{t−1}), computed on consecutive observed runs per market and item.
- [ ] Cross-check for Dhaka with two sources, TCB monthly vs the WFP/DAM observed series: the covariance of two independent measurements gives the signal variance.
- [ ] Report raw λ and signal λ* = Var(Δp̂) / (Var(Δy) − 2Var(ν)).
- **Output:** **Table 4b**. If λ* is close to 1, attenuation is mostly noise removal and the framing must change. This is part of Gate G1.

## Phase 3: Causal-strength designs (weeks 3–5)

### Task 3.1: Natural hold-out from vintages (H2, the headline result)
- [ ] For hold-out cells, compare the imputed value from the last vintage *before* the cell became observed with the observed value from the latest vintage. Report:
  - level error (MAPE, bias)
  - change error (MAE of Δp)
  - Mincer–Zarnowitz regression of observed on imputed. **Expected slope ≈ 1:** a conditional-mean imputer is calibrated, which is why standard validation passes (Proposition 2 in `docs/methodology.md`).
  - **Attenuation diagnostics:** the *reverse* slope (imputed `e` on observed `e`) and the variance ratio Var(e_imp)/Var(e_obs) on the *same* cells. Both are expected below 1 (Proposition 1).
- [ ] Bland–Altman plot.
- [ ] **Published-band check:** report whether the later-observed price falls inside the imputed cell's `[l_, h_]` band, and compare band width in imputed vs observed cells. The quick look (`analysis/02_holdout_quicklook.py`) found:
  - the band has the same width (~4–6%) in both, so it does not widen where data are missing
  - it covers the later truth in only 53–59% of imputed cells, vs 92–94% for observed cells
  - Frame this as "the band is an intra-month range, not an uncertainty interval". Do not claim the World Bank presents it as a confidence interval.
- [ ] Revision path: how the imputed value moved across vintages before the observation arrived.
- **Outputs:** **Table 5**, **Fig 4** (Mincer–Zarnowitz scatter with slope) and **Fig 5** (revision paths).
- **Check:** recompute on a subsample excluding the 2026 market expansion (77 → 111) to rule out artefacts.

### Task 3.2: Masking experiment with a replica imputer (mechanism under our control)
- [ ] `src/rtfp/replica_imputer.py`: a chained-equation imputer close to the published RTFP design:
  - **starting values:** seasonal interpolation for gaps of ≤3 months, inverse-distance weighting otherwise
  - **predictors:** other items' log prices, market FE, month dummies, coordinates
  - **learners:** elastic net and gradient-boosted trees (Cubist substitute)
  - M = 5 chains, 8 iterations, final = ensemble mean
- [ ] On fully observed 2021–2026 segments, randomly mask 20% of observed cells (stratified by market), impute, and compare the variance of imputed vs true `e`. Repeat 50 times.
- [ ] Run three arms:
  - ensemble mean, as RTFP does
  - single stochastic draw
  - Rubin-style multiple imputation with residual noise
- **Output:** **Table 6**. This shows that ensemble averaging itself causes the attenuation and that stochastic draws avoid it.
- **Note:** this validates the mechanism, not RTFP itself. Task 3.1 is the RTFP evidence.

## Phase 4: Consequences (weeks 5–7)

### Task 4.1: Early-warning alerts (H5)
- [ ] Implement **ALPS** (WFP 2014 technical note): deviation of the price from the seasonal trend, scaled by its SD; thresholds 0.25 / 1 / 2.
- [ ] Implement **IPA** (FAO, Baquedano): weighted within-year and across-year compound growth rates.
- [ ] On hold-out cells (Task 3.1) and masked cells (Task 3.2), compare alerts computed from imputed values with alerts computed from observed values: recall, precision and Cohen's κ.
- **Outputs:** **Table 7** and **Fig 6** (alerts missed by imputation, on a map or timeline).

### Task 4.2: Co-movement and market integration (H4)
- [ ] Pairwise correlations of Δp for observed–observed, imputed–imputed and observed–imputed market pairs, binned by distance.
- [ ] Price-transmission half-life (error-correction model of each market against Dhaka), estimated on observed-only vs all cells.
- **Outputs:** **Fig 7** (correlation vs distance by pair type) and **Table 8** (half-lives).

### Task 4.3: Local shock event study (exploratory)
- [ ] Event: the August 2024 eastern floods (Feni, Noakhali, Cumilla and others; confirm the affected-district list from an official source).
- [ ] Difference-in-differences of Δp, affected vs unaffected districts, estimated separately on observed and on imputed cells.
- **Hypothesis:** the local price response is muted in imputed cells.
- **Output:** **Fig 8**.

### Task 4.4: Forecast-interval consequence (reuses the lab project)
- [ ] Train the existing LightGBM quantile + conformal pipeline on all cells. Report interval coverage separately on observed and on imputed test cells.
- **Expected:** under-coverage on observed cells, because the intervals were calibrated on smoothed data.
- **Output:** **Table 9**.

## Phase 5: External checks (weeks 6–8, optional, gated)

### Task 5.1: TCB Dhaka range vs RTFP high/low (validates the modelled intra-month range)
- [ ] Scrape the TCB daily archive (`https://tcb.gov.bd/pages/daily-rmps`, from May 2021, `?page=N`, xlsx files). Take the date from *inside* each file, because listing dates are sometimes wrong.
- [ ] Parse coarse rice, atta (loose), palm oil (loose) and masur lentils, keeping min and max.
- [ ] Aggregate to monthly open/high/low/close (OHLC) and compare with RTFP Dhaka `o_/h_/l_/c_`.
- **Note:** Dhaka is an observed market, so this checks RTFP's modelled *range*, not its imputation.
- **Output:** **Table 10**.

### Task 5.2: DAM district prices for never-input cells (gated)
- [ ] Probe `https://market.dam.gov.bd/market_daily_price_report` (POST form, PDF export). Find the earliest available date and whether district retail prices exist for markets and months **absent** from the HDX feed.
- **Gate G2:** go ahead only if at least 300 such cells are obtainable within 1 week of effort. Otherwise drop the task.

### Task 5.3: BBS food CPI benchmark (independent of DAM/WFP)
- [ ] Download the BBS monthly CPI Excel releases from `https://bbs.gov.bd/pages/static-pages/6922de7a933eb65569e1ae8f` (2016 onward; national/rural/urban; COICOP 01 Food). Note the base change to 2021-22 = 100 and splice the series.
- [ ] Compare RTFP's national food price index (mean of `c_food_price_index` across markets) with BBS food CPI: correlation of 12-month inflation, m/m variance ratio, turning points.
- **Expected:** close agreement, because the common component is preserved (Proposition 1). This shows our critique is specific to *local* signals, not a blanket "RTFP is wrong".
- **Limit:** BBS publishes no item- or market-level prices, so it cannot validate imputed cells.
- **Output:** **Fig 10**.

## Phase 6: Remedy, robustness and generalisation (weeks 7–9)

### Task 6.1: Vintage-calibrated uncertainty layer (`src/rtfp/uncertainty.py`), the remedy
We do **not** build a replacement imputer. The hold-out shows local moves are mostly not predictable from the available information, so no imputer can recover them. The remedy instead tells users how uncertain each imputed cell is, working on the *published* files.
- [ ] **Error model:** on hold-out cells (Task 1.3), fit quantile LightGBM for the log error `log y − log ĉ` at the 5th, 10th, 50th, 90th and 95th percentiles. Features:
  - `gap_len`, `dist_obs_km`, `trust_<item>`, `data_coverage`, `data_coverage_recent`, `spatially_interpolated`
  - item, calendar month
  - vintage age (vintage date − price month)
- [ ] **Conformal calibration:** conformalized quantile regression (Romano et al. 2019), Mondrian by gap-length × distance bins. Because hold-out cells (markets observed later) are not exchangeable with never-observed cells, add weighted conformal under covariate shift (Tibshirani, Barber, Candès & Ramdas 2019). Weights come from a classifier that separates hold-out cells from never-observed cells.
- [ ] **Path simulation:** for each market, draw M = 200 paths `ĉ · exp(r)`, where the residual `r` has the calibrated marginal quantiles, AR(1) persistence over time and spatial correlation decaying with distance (both estimated on hold-out residuals). The path mean stays at the RTFP value, so level accuracy is unchanged.
- [ ] **Probabilistic alerts:** ALPS/IPA computed on every path gives P(alert) per market-month.
- [ ] **Evaluation:** time-blocked split of hold-out cells (calibrate on earlier vintages, test on later ones). Report:
  - 80% and 90% interval coverage and width
  - CRPS
  - Brier score of P(alert)
  - alert recall at matched precision
  - Baselines:
    1. the published `[l_, h_]` band
    2. a fixed-width interval
    3. a global Gaussian
    4. replica-imputer multiple-imputation draws (Task 3.2)
    5. constrained-Bayes rescaling (Ghosh 1992)
- [ ] **Novelty framing:** conformal intervals for missing entries already exist (Gui, Barber & Ma, *Conformalized Matrix Completion*, NeurIPS 2023). What's new: calibration on real later-arriving truth from release history, path simulation for probabilistic early warning, and an operational dataset.
- **Outputs:** **Table 11** (coverage, width, CRPS, Brier and recall vs baselines) and **Fig 11** (one market: published band vs calibrated interval vs simulated paths).
- **Optional follow-up paper (not in this one):** an information-augmented imputer adding rainfall/flood extent, daily TCB/DAM prices and the exchange rate. First test cheaply whether those predict hold-out errors.

### Task 6.2: Robustness battery
- [ ] Alternative definitions of `e`: division-month mean instead of national.
- [ ] Monthly vs 3-month changes; excluding 2007–2019; excluding the camp markets; using `inflation_<item>` instead of `dlog`.
- [ ] Earliest vs latest vintage.
- **Output:** **Appendix tables A3–A8**.

### Task 6.3: Multi-country generalisation (strong upgrade, recommended)
- [ ] Download the all-country file (`WLD_2021_RTFP_v02_M`: 40 countries, about 3,300 markets).
- [ ] Re-run Tasks 2.1–2.2 per country.
- **Output:** **Fig 9** (country forest plot of the attenuation coefficient).
- **Why:** turns "a Bangladesh case" into "a property of the dataset", which opens the door to Food Policy-level venues.
- **Check:** confirm the raw columns exist in the all-country file first.

## Phase 7: Reproducibility package (week 9)

- [ ] Add a `Makefile` or `run_all.sh` that rebuilds every table and figure from `data/raw/` in one command.
- [ ] Write a README covering data sources, versions, sha256 values and run time.
- [ ] Add `CITATION.cff` (authors, ORCIDs, title, licence) and `LICENSE` (MIT for code, CC BY 4.0 for derived data) to the repo root. Zenodo and GitHub both read them for citation metadata.
- [ ] Connect the GitHub repo to Zenodo (GitHub–Zenodo integration). Every tagged release (`v1.0-submission`, `v1.1-revision`, …) is then archived automatically with its own version DOI, plus one concept DOI that always resolves to the latest version.
- [ ] Each release bundles:
  1. code
  2. derived panels and hold-out sets
  3. `MANIFEST.csv` with RTFP vintage dates and sha256
  4. a zipped static build of the website (`site/out/`), so the interactive results survive even if hosting disappears
  5. the download script for raw World Bank files. Include the raw files themselves only if Task 0.2 confirms the licence allows redistribution.
- [ ] The paper's data-availability statement cites the Zenodo concept DOI. The live site is linked as a supplementary resource, not as the archive.
- [ ] **Double-anonymous review:** a Zenodo record shows author names. Before submission, reserve the DOI and keep the record unpublished, and give reviewers an anonymised code mirror (anonymous.4open.science). Publish the Zenodo record at acceptance.

## Phase 8: Companion website, a "living audit" (weeks 8–10)

Hosting roles:
- **Vercel** is the live site.
- **GitHub** holds the source.
- **Zenodo** is the permanent, citable archive (Phase 7).

GitHub Pages isn't needed. The static build can be moved there at any time if Vercel ever becomes unsuitable.

### Task 8.1: Interactive site (Next.js static export on Vercel)
- [ ] `site/` is a Next.js app with `output: 'export'`, so it builds to plain static files in `site/out/`. There are no server functions, so nothing can break during review, and the same build can be zipped into Zenodo.
- [ ] Charts: Plotly. Maps: MapLibre. Data: pre-computed JSON from `outputs/site_data/`, copied into `site/public/data/` at build time.
- [ ] Deploy through the Vercel GitHub integration: every push to `main` redeploys. The Hobby plan is for non-commercial use, which covers academic research.
- [ ] Pages:
  1. **Market explorer:** observed points vs imputed line, with trust score
  2. **Attenuation map:** λ by market
  3. **Alert audit:** alerts missed by imputation
  4. **Vintage viewer:** how imputations revised as data arrived
  5. **Methods & downloads**
- [ ] `analysis/80_site_data.py` exports the JSON from the same pipeline as the paper.
- [ ] Add a clear disclaimer: independent research, not affiliated with the World Bank, WFP or the Government of Bangladesh. No institutional logos.

### Task 8.2: Monthly auto-update (GitHub Actions)
- [ ] A monthly cron job:
  1. runs `download.py`
  2. detects a new RTFP vintage
  3. re-runs the hold-out (D3) on newly observed cells
  4. commits the refreshed `outputs/site_data/` JSON to `main`, which triggers the Vercel redeploy
- [ ] The job does **not** create a Zenodo release. Releases are cut by hand at submission, revision and acceptance, so each DOI matches a version of the paper.
- **Why:** every month adds new out-of-sample hold-out cells. That is a *prospective* evaluation that continues after submission, and it can be cited in revisions.
- **During double-anonymous review:** keep author names off the Vercel site (no names in pages, metadata or the repo shown on the site), and share code through an anonymised mirror (e.g. anonymous.4open.science). Add author names after acceptance.

## Where Colab Pro is used

The core paper is statistics-heavy, not GPU-heavy; most tasks run on a laptop. Use Colab for:
- Task 3.2 at scale (50 replications × 3–4 arms; × 40 countries if Task 6.3 runs)
- the optional generative-imputer arm (CSDI, a diffusion model for time-series imputation; GPU)
- Task 6.3, multi-country re-runs

Keep every Colab notebook a thin wrapper that calls `src/rtfp/`, so results stay reproducible.

---

## 8. Handoff to the co-author

For every table or figure, deliver a matching `outputs/results_notes/<ID>.md` containing:
1. **Claim:** one sentence.
2. **Key numbers:** with CI and N.
3. **Method facts:** data, sample, test, software.
4. **Caveats.**
5. **The file paths** of the table or figure.

A one-page `outputs/results_notes/SUMMARY.md` stays updated with the current headline numbers. The co-author writes only from these notes, so every number in the manuscript traces back to a script.

## 9. Non-writing to-dos

- [ ] **Decide with the co-author:** target venue (PLOS ONE first, Food Policy if Task 6.3 works) and author order (CRediT roles). Do it before Phase 4, because venue word limits decide how many consequence analyses go in the main text.
- [ ] **Optional email to the World Bank Microdata help desk:** ask for confirmation that the raw item columns are the observed survey values, and how the outlier adjustment works. Ask only factual questions; don't share results before submission.
- [ ] **Optional OSF pre-registration** of `docs/analysis_plan.md` (Task 0.3).
- [ ] **Literature reference list for the co-author:** van Buuren 2026; Ratledge et al. 2022; Proctor, Carleton & Sum 2023; Angelopoulos et al. 2023; Adewopo et al. 2025; Andrée 2021/2023/2026; WFP ALPS 2014; FAO IPA; Emediegwu & Rogna 2024.

## 10. Timeline and gates

| Week | Work | Gate |
|---|---|---|
| 1 | Phase 0 and Tasks 1.1–1.2 | Raw-flag audit passes (≥90% match) |
| 2 | Task 1.3, Phase 2 | **G1:** attenuation survives market and month fixed effects |
| 3–5 | Phase 3 (hold-out + masking) | Headline result fixed; co-author starts drafting |
| 5–7 | Phase 4 consequences | — |
| 6–8 | Phase 5 external (optional) | **G2:** DAM has ≥300 never-input cells |
| 7–9 | Phase 6 remedy, robustness, multi-country | — |
| 9 | Phase 7 reproducibility; final re-pull of latest vintage and full re-run | Submission-ready |
| 8–10 | Phase 8 companion site (Next.js on Vercel) + monthly auto-update | Live before submission (anonymised link); Zenodo DOI reserved |
