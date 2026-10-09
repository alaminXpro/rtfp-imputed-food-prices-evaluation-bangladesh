# Research Engineering Plan — Independent Release-History Evaluation of RTFP

> Scope: everything except manuscript writing (data, engineering, statistics, figures, reproducibility).
> Owner: Md Alamin (engineering/statistics). Co-author: writing. Handoff format in §5.
> Revision 3 (2026-10-09). Spec: `docs/methodology.md` (revision 3). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure how much local price information the World Bank's ML-imputed Real-Time Food Prices (RTFP) carry for Bangladesh, judged against survey values that arrived after each release; show what that implies for early warning and analysis; and release calibrated uncertainty for imputed cells within the hold-out's support.

**Approach:** Resolve markets through a name+coordinate crosswalk; build a hold-out of cells imputed in one release and surveyed in a later one, split into toggle / forecast / cross-sectional strata; estimate a measurement-noise band before anything else; report level and local-change metrics with two-way cluster bootstrap CIs; register the confirmatory tests and confirm them on releases published after registration.

**Tech stack:** Python 3.12, pandas, numpy, statsmodels, linearmodels, pyfixest (PPML), scikit-learn, lightgbm, matplotlib; pytest; git; Zenodo. Colab only for Task 6.1 and the appendix masking study.

## Global constraints

- Primary dataset: newest RTFP Bangladesh vintage (currently 2026-10-05). Re-pull before the final run; record the version in every output.
- Items: rice, wheat flour, palm oil, lentils. Unit: market × month × item.
- **Market key = internal `mkt_uid` from the crosswalk (Task 1.3).** geo_id is re-keyed on 2024-03-18 and 2025-03-31 and must not be used across vintages.
- Observed side uses `log y` (plain column); imputed side uses `log ĉ` (`c_`). Evaluation truth for a hold-out cell = y in the latest vintage (equals the WFP feed); first-observed value kept for revisions.
- Local component: each side demeaned by its own vintage's observed-cell month mean; pooled versions reported alongside.
- **One hold-out definition** with strata toggle > unconfirmed > backfill (>6 months) > forecast > cross-sectional and pair types (methodology M1). Headline = cross-sectional stratum, both-imputed pairs.
- Rice is absent from 42 of 83 vintages (and 2024-01-25 has rice survey values but no published rice estimates): vintage-dependent quantities use only vintages containing the item.
- Every number handed to the co-author comes from a script in `analysis/`.
- Confirmatory H1–H2 use only releases published after registration (methodology M4).

## Review focus (failure modes most likely to sink the paper)

1. **Noise.** Survey changes contain noise, stale quotes and averaged values. Task 2.1 produces a ρ* band, not a single number, and gates the project.
2. **Pipeline toggling.** Hundreds of cells per item flip observed↔imputed between consecutive releases; they are not late arrivals. Toggle stratum is excluded from the headline.
3. **Market re-keying.** geo_id changed twice; a geo_id-keyed hold-out silently drops most early vintages.
4. **National-error contamination.** In forecast cells no market was observed, so "local" errors include national nowcast error. Within-month demeaning on each side; forecast stratum reported separately.
5. **Raw-column semantics.** "Plain column = survey value" is inferred. Task 1.2 stops the project below 90% match.
6. **Over-claiming.** Headline stated as a measured share with CI; "first" softened; b and η not separately claimed; producer-friendly tone.
7. **Small confirmatory sample.** Releases add cells irregularly (zero in some weeks). State the expected post-registration N before testing.

---

## 1. Current state (2026-10-09)

| Item | Location | Status |
|---|---|---|
| 83 RTFP Bangladesh vintages, 2024-01-25 → 2026-10-05 | `data/raw/rtfp_vintages/` | downloaded |
| WFP/HDX feed, 2026-10-04 | `data/raw/hdx_wfp/` | downloaded |
| Quick looks (exploratory; superseded; not quotable) | `analysis/01_*`, `analysis/02_*` | within-month SD ratio 0.43–0.55; hold-out level MAPE 3–4%; corr of monthly changes 0.28–0.52; local β_MZ 0.10–0.26, λ 0.13–0.50. Known defects: geo_id keying, toggle cells pooled, common μ from latest vintage incl. imputed cells, c_ used as observed series. |
| Git repo | `main` (docs), `work/phase0-1` (code) | created |

## 2. Repository layout

```
DA Paper/
├── data/raw/            # downloads + MANIFEST.csv (never edited)
├── data/interim/        # crosswalk, panel_latest, vintages, holdout (parquet)
├── data/external/       # BBS xlsx, TCB xlsx
├── src/rtfp/
│   ├── download.py      # RTFP vintages, HDX feed, manifest
│   ├── crosswalk.py     # market name+coords → mkt_uid across re-keyings
│   ├── panel.py         # latest-vintage panel with flags and covariates
│   ├── vintages.py      # stacked vintages; hold-out H, strata, pair types
│   ├── noise.py         # measurement-noise band (D2)
│   ├── metrics.py       # level and local-change metrics; two-way bootstrap
│   ├── alerts.py        # ALPS, IPA
│   ├── simulate.py      # framework simulation and alert benchmark
│   ├── uncertainty.py   # quantile model + conformal calibration (D5)
│   └── external.py      # BBS and TCB loaders
├── analysis/            # one script per output: t1_noise.py, t2_levels.py, …
├── tests/
├── outputs/{tables,figures,results_notes}/
├── docs/{methodology.md, analysis_plan.md, plans/}
└── CITATION.cff, LICENSE, requirements.txt, pyproject.toml, Makefile
```

## 3. Phases and tasks

### Phase 0: Setup (week 1)

#### Task 0.1: Reproducible download and manifest
- [ ] `src/rtfp/download.py`: Microdata listing `.../api/downloads/{idno}/files?type=data`; download new `*_RTFP_mkt_*` zips; extract; manifest (source, filename, url, vintage, downloaded_at, sha256). HDX package `wfp-food-prices-for-bangladesh`. Idempotent.
- [ ] `requirements.txt` (pinned), `pyproject.toml`, `Makefile` target `data`.
- **Tests:** listing parser keeps only market files; sha256 helper; newest-vintage picker orders by date.

#### Task 0.2: Licences
- [ ] `data/raw/LICENSES.md`: RTFP terms (attribution), HDX/WFP licence, BBS and TCB (public government data). Decide what the Zenodo bundle may contain.

### Phase 1: Data (weeks 2–5)

#### Task 1.1: Latest-vintage panel (`src/rtfp/panel.py`)
- [ ] Long format (market, month, item): y, `c_, o_, h_, l_`, trust, data coverage fields, lat/lon, adm1/adm2; join WFP `priceflag` and source from HDX.
- [ ] Derived: obs, Δ on each side, μ (observed-only), e, gap, dist_obs, n_obs_month.
- **Tests:** key unique; Δ NaN only at series start; dist_obs = 0 iff observed; μ uses observed cells only (toy frame); observed side uses y not c_.

#### Task 1.2: Raw-column audit (stop-gate)
- [ ] Match observed cells to HDX Retail rows (name or lat/lon within 5 km; month; commodity). Share matched; ratio y ÷ HDX; priceflag distribution.
- [ ] Characterise cells where c_ ≠ y (~10%).
- **Output:** Table A1 + results note. **< 90% matched → stop and report.**

#### Task 1.3: Crosswalk, vintages and hold-out (`crosswalk.py`, `vintages.py`)
- [ ] Crosswalk: mkt_name + lat/lon tolerance → mkt_uid across all vintages. **Test:** the re-keyings of 2024-03-18 and 2025-03-31 map 1:1; unmatched markets listed.
- [ ] Stack vintages → `vintages.parquet` (vintage, mkt_uid, month, item, y, ĉ, h_, l_, trust) with per-vintage item availability.
- [x] H with ĉ_first, ĉ_final, truth = latest-vintage y; strata toggle > unconfirmed > backfill > forecast > cross-sectional; pair types. (Done 2026-10-09; Tables A2–A3.)
- [ ] Covariate balance H vs never-observed cells.
- **Outputs:** Table A2 (H counts by item × stratum × year; change-pair counts by pair type), Table A3 (balance).
- **Tests:** toy vintages → expected membership, strata precedence, toggle detection, pair typing, first/final values.

#### Task 1.4: External data (`external.py`)
- [ ] BBS: locate and download the monthly CPI Excel series 2016 →; document the 2005-06 → 2021-22 splice; check whether a rice sub-index exists.
- [ ] TCB: scrape `https://tcb.gov.bd/pages/daily-rmps` (May 2021 →, `?page=N`, xlsx; date from inside each file); coarse rice, loose atta, loose palm oil, masur; monthly series. Confirm TCB is absent from WFP's source list. Month counts for Dhaka Sadar overlap per item.
- **Tests:** parser on two saved sample files; date taken from file content, not listing.

### Phase 2: Noise gate, registration, headline (weeks 6–9)

#### Task 2.1: Measurement-noise band (`noise.py`) — Gate G1
- [ ] Autocovariances γ_Δ(1..3) of observed local changes on consecutive observed runs; φ = γ_Δ(3)/γ_Δ(2); Var(ν) = −γ_Δ(1) + γ_Δ(2)/φ; naive Var(ν) = −γ_Δ(1); staleness-aware variant (changes conditional on a quote update); excluding `priceflag = aggregate`.
- [ ] rel per item under each estimator; ρ* band on the clean hold-out from Task 1.3 (cross-sectional stratum, both-imputed pairs).
- [ ] Dhaka Sadar vs TCB two-source check (supporting).
- **Outputs:** Table 1 (noise band). **Gate G1:** pivot if ≥ 2 of 4 items have ρ* ≥ 0.5 at the conservative rel. Record outcome.
- **Tests:** estimators recover known Var(ν) and φ on simulated AR(1)+noise series (shared with Task 3.1 generator).

#### Task 2.R: Registration
- [ ] `docs/analysis_plan.md`: H1–H2 (prospective), H3–H4 (registered, existing data), descriptive checks, thresholds, hold-out definition, strata, G1 outcome and both branches, D6 country-selection criteria, expected post-registration N (min 6 months, ≥ 150 cells per item), disclosure of all exploratory work.
- [ ] Commit; post on OSF; record date.

#### Task 2.2: Hold-out levels and local changes (`metrics.py`) — headline
- [ ] Levels: MAPE, median APE, bias, level MZ (market-FE; month-demeaned), Bland–Altman, Adewopo-style level correlations.
- [ ] Local changes: corr, β_MZ, λ, β_rev (identity check), ρ* band; within-month and pooled; by stratum × pair type × first/final.
- [ ] Spikes with the persistence definition (methodology D1.3).
- [ ] Two-way cluster bootstrap with rel resampled jointly; Holm across items.
- **Outputs:** Table 2 (levels), Table 3 (local changes), Fig 1 (example market), Fig 2 (e_obs vs ê scatter), Fig 3 (revision paths).

#### Task 2.3: Error vs covariates
- [ ] |level error| and |ê − e_obs| on gap, dist_obs, trust, coverage, vintage age; month FE; two-way clustered SEs.
- **Outputs:** Table 4, Fig 4. Sub-result: does the trust score predict external error?

#### Task 2.4: Switching markets
- [ ] PPML of |e| on imputed + market×item FE + month×item FE; two-way clustered SEs; benchmark log sqrt(rel).
- [ ] Event-time plot −6..+6 with pre-trends.
- **Outputs:** Table 5, Fig 5.

### Phase 3: Framework simulation (week 10)

#### Task 3.1: `simulate.py`
- [ ] Panels: common AR(1) + local AR(1) + seasonality; observed = true + ν, with stale-quote and averaging variants; realistic missingness.
- [ ] Four imputer types: calibrated conditional mean; IDW/seasonal fill; chain mean with residual noise; outlier smoothing.
- [ ] Report which reproduces the empirical (ρ*, λ, β_MZ) triple from Task 2.2; verify noise estimators and disattenuation; produce the alert benchmark for Task 4.1.
- **Tests:** calibrated case gives β_MZ ≈ 1 within ±0.05 over 200 panels; Var(ν) estimator unbiased within tolerance under AR(1)+white noise.
- **Output:** Table A4.

### Phase 4: Consequences and safe use (weeks 11–12)

#### Task 4.1: Alerts (`alerts.py`)
- [ ] ALPS (WFP 2014; 0.25/1/2) and IPA (FAO) from ĉ and from y on H; recall, precision, κ with CIs; "local event" definition; power calculation; comparison with the Task 3.1 benchmark.
- **Outputs:** Table 6, Fig 6.

#### Task 4.2: Regressor attenuation
- [ ] Report β_MZ as the coefficient multiplier; if a suitable published RTFP-based regression is found, re-estimate it on observed-only cells (worked example).
- **Output:** paragraph + Table 6b if the worked example exists.

#### Task 4.3: National check vs BBS (descriptive)
- [ ] Equal-weight log index from c_ (all cells) and from observed cells only, vs BBS food CPI (and rice sub-index if available); m/m and 3-month changes; Newey–West SEs.
- **Output:** Fig 7.

#### Task 4.4: Safe-use table
- [ ] Script assembling Tables 2–6 and Fig 7 into the supported / unsupported uses table.
- **Output:** Table 8.

### Phase 5: Calibrated uncertainty (weeks 13–15)

#### Task 5.1: `uncertainty.py`
- [ ] Quantile LightGBM on log(y/ĉ); features gap, dist_obs, trust, coverage, item, month, vintage age, stratum.
- [ ] Three-way time-blocked split by first-observed vintage (train ≤ T1, calibrate (T1, T2], test > T2); CQR, Mondrian by gap × distance; weighted conformal within support only.
- [ ] Support restriction and flags; coverage pooled and by month; width; CRPS; baselines.
- **Outputs:** Table 7, Fig 8, `outputs/tables/rtfp_bgd_calibrated_quantiles.csv`.

### Phase 6: Bounded replication (weeks 16–18, gated, Colab)

#### Task 6.1
- [ ] Go only if Phases 1–2 finished by week 10. Select 3–5 per-country RTFP market datasets by the registered criteria; download; crosswalk; audit; Tasks 1.3, 2.1, 2.2 core metrics.
- **Output:** Fig 9 (forest plot of ρ* bands), Table 9.

### Phase 7: Reproducibility, confirmation, handoff (weeks 19–20)

- [ ] `make all` rebuilds every output from `data/raw/`.
- [ ] `CITATION.cff`, `LICENSE` (MIT code, CC BY 4.0 derived data), README with versions, sha256s, run time.
- [ ] Re-pull newest vintage; confirmatory H1–H2 if the stated minimum N is reached, otherwise report the registered test as pending with current N.
- [ ] GitHub–Zenodo integration; tag `v1.0-submission`; DOI reserved, released on acceptance; anonymised code mirror for review.
- [ ] Bundle: code, derived panels, H, crosswalk, calibrated quantiles, manifest; raw World Bank files only if Task 0.2 allows.

### Appendix tasks (if time)

- **A-mask** Replica masking study (Colab): chained-equation replica, 20% masking × 50 reps; arms: ensemble mean, single draw, proper MI.
- **A-robust** Robustness battery: c_-based observed side; division-level μ; 3-month changes; excluding 2007–2019, camp markets, and aggregate quotes; `inflation_<k>`; earliest vs latest vintage. Table A5.

### Deferred to a follow-up paper

Path simulation and probabilistic alerts; information-augmented imputer; August 2024 flood DiD; forecast-interval coverage; companion website and monthly auto-updating test.

## 4. Outputs index

| ID | Content | Task |
|---|---|---|
| Table 1 | Noise band and ρ* band | 2.1 |
| Table 2 | Level metrics | 2.2 |
| Table 3 | Local-change metrics | 2.2 |
| Table 4 | Error vs covariates | 2.3 |
| Table 5 | Switching-market PPML | 2.4 |
| Table 6 | Alert agreement | 4.1 |
| Table 7 | Calibrated uncertainty vs baselines | 5.1 |
| Table 8 | Safe-use table | 4.4 |
| Table 9 | Replication | 6.1 |
| A1–A5 | Raw audit; H counts; balance; simulation; robustness | 1.2, 1.3, 1.3, 3.1, A-robust |
| Fig 1–9 | Example market; scatter; revisions; covariates; event time; alerts; BBS; intervals; forest plot | as above |

## 5. Handoff to the co-author

For every table or figure, `outputs/results_notes/<ID>.md`: claim (one sentence); key numbers with CI and N; method facts; caveats; file paths. `SUMMARY.md` holds the current headline numbers. The co-author writes only from these notes.

## 6. Non-writing to-dos

- [ ] Default venue PLOS ONE; revisit Food Policy after Task 6.1. Agree author order (CRediT) before Phase 4.
- [ ] OSF registration (Task 2.R).
- [ ] Optional factual email to the World Bank Microdata help desk: plain item columns = survey values? outlier adjustment? pipeline toggling between releases? No results shared.
- [ ] Reference list: Andrée 2021/2023/2026; Adewopo et al. 2025; Rubin 1987; van Buuren 2018/2026; Gui, Barber & Ma 2023; Tibshirani et al. 2019; Romano et al. 2019; Gneiting & Raftery 2007; Santos Silva & Tenreyro 2006; WFP ALPS 2014; FAO IPA; Emediegwu & Rogna 2024.

## 7. Timeline and gates (18–20 weeks alongside study)

| Weeks | Work | Gate |
|---|---|---|
| 1 | Phase 0 | — |
| 2–5 | Tasks 1.1–1.4 | Raw-column audit ≥ 90%; crosswalk 1:1 |
| 6 | Task 2.1 | **G1**: fewer than 2 of 4 items with ρ* ≥ 0.5 |
| 6 | Task 2.R registration | posted before any post-registration release is analysed |
| 7–9 | Tasks 2.2–2.4 | headline fixed; co-author starts drafting |
| 10 | Task 3.1 | Phase 6 go/no-go |
| 11–12 | Phase 4 | — |
| 13–15 | Phase 5 | — |
| 16–18 | Phase 6 (if go) | — |
| 19–20 | Phase 7 | submission-ready |
