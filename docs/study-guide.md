# Study Guide — Topics the Paper Uses

> Order = study order. Each topic lists why the paper needs it, the plan task it feeds, search terms for YouTube/Google, and a free reference where I know one.
> Tiers: **A** = you must be able to explain it to a reviewer; **B** = you must be able to run it and read the output; **C** = skim, look up when needed.
> I did not verify individual YouTube videos. The search terms and channel names are starting points; pick the clearest lecture you find.

## Part 1 — Foundations (tier A, start here)

| # | Topic | Why the paper needs it | Feeds | Search terms |
|---|---|---|---|---|
| 1 | **Variance, covariance, correlation, R²** | Every result is a variance ratio or a correlation | all | "variance covariance correlation explained", "what R squared really means" |
| 2 | **Conditional expectation `E[Y\|X]`** | The whole theory: a best-guess imputer *is* a conditional mean | M2 | "conditional expectation intuition", "law of total variance" |
| 3 | **Law of total variance** | `Var(Y) = E[Var(Y\|X)] + Var(E[Y\|X])` is Proposition 1: predicting by the conditional mean throws away the first term | M2 Prop 1 | "law of total variance proof intuition" |
| 4 | **Regression to the mean / shrinkage** | Why predictions are less spread out than reality | M2 | "regression to the mean explained", "shrinkage estimator James-Stein" |
| 5 | **Log prices and log changes** | Outcome is `Δp = log p_t − log p_{t−1}` (approx. % change) | all | "log returns vs percentage change", "log difference approximation" |

## Part 2 — Missing data and imputation (tier A)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 6 | **Missing data mechanisms: MCAR, MAR, MNAR** | Observed markets are *not* random (selection objection) | Task 2.2 | "MCAR MAR MNAR explained" |
| 7 | **Single vs multiple imputation; Rubin's rules** | RTFP publishes one averaged series; proper imputation keeps the spread | Task 3.2, 6.1 | "multiple imputation Rubin's rules", "multiple imputation by chained equations MICE" |
| 8 | **MICE / chained equations** | RTFP's method is a MICE adaptation (Andrée 2021) | Task 3.2 | "MICE algorithm step by step" |
| 9 | **Why mean-type imputation shrinks variance** | The central mechanism of the paper | M2 | "regression imputation underestimates variance", "imputation bias variance" |
| 10 | **Ensemble averaging** | RTFP averages M chains; averaging shrinks variance further | Task 3.2 | "bagging variance reduction ensemble mean" |

Free reference: Stef van Buuren, *Flexible Imputation of Missing Data* (online book, chapters 1–3). Also cite van Buuren (2026), arXiv 2605.03733, which makes the shrinkage point directly.

## Part 3 — Measurement error (tier A, the subtle part)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 11 | **Classical vs Berkson measurement error** | Proposition 5: imputed values used as a regressor (Berkson, safe) vs as an outcome for a local shock (unsafe) | M2 Prop 5 | "classical measurement error attenuation bias", "Berkson error vs classical error" |
| 12 | **Attenuation bias** | Same word as "variance attenuation" but a different thing; you must keep them apart | M2 | "attenuation bias regression measurement error" |
| 13 | **Signal vs noise; autocovariance fingerprint of noise** | Answers "imputation just removes noise": noise makes consecutive changes negatively correlated | Task 2.4 | "Roll model bid-ask bounce autocovariance", "MA(1) process first difference of noise" |

Reference: Roll (1984) uses the same trick for bid-ask bounce. Wooldridge, *Introductory Econometrics*, measurement-error section.

## Part 4 — Time series basics (tier A/B)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 14 | **AR(1), MA(1), autocorrelation (ACF)** | Simulation (Task 2.0) and the noise estimator both use them | 2.0, 2.4 | "AR MA ARMA models explained", "autocorrelation function ACF" |
| 15 | **Stationarity, seasonality, differencing** | Why we use changes, not levels | all | "stationarity differencing explained" |
| 16 | **Kalman filter / smoother (concept only)** | RTFP fills short gaps with a seasonal Kalman smoother | context | "Kalman filter intuition" (tier C) |
| 17 | **Spurious correlation of trending series** | Why level correlations of 0.94–0.99 prove little | M2 Prop 2 | "spurious regression trending time series" |

Free reference: Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., otexts.com/fpp3).

## Part 5 — Panel econometrics (tier A for 18–20, B for 21–23)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 18 | **Fixed effects (market, month)** | The main answer to "observed markets are more volatile" | Task 2.2 | "fixed effects panel data explained", "two-way fixed effects" |
| 19 | **Clustered standard errors** | Markets are correlated over time and each other | Task 2.2 | "clustered standard errors explained", "when to cluster" |
| 20 | **Identification: what a "switchers" design does** | Compare the same market when imputed vs observed | Task 2.2 | "within estimator fixed effects intuition" |
| 21 | **Driscoll–Kraay, two-way clustering** | Robustness for spatial and time correlation | 2.2 | "Driscoll Kraay standard errors" |
| 22 | **Quantile regression** | Test the median and the 90th percentile of `\|e\|`, not just the mean | 2.2 | "quantile regression explained" |
| 23 | **Variance tests (Brown–Forsythe, Levene)** | Test equality of spread across groups | 2.2 | "Levene test Brown-Forsythe" |

References: Angrist & Pischke, *Mostly Harmless Econometrics*; Nick Huntington-Klein, *The Effect* (free online, with lectures); Cameron, Gelbach & Miller (2008) on clustering.

## Part 6 — Resampling and multiple testing (tier B)

| # | Topic | Why | Search terms |
|---|---|---|---|
| 24 | **Bootstrap; cluster bootstrap; wild cluster bootstrap; block bootstrap** | Confidence intervals for variance ratios, which have no simple formula | "bootstrap confidence interval StatQuest", "wild cluster bootstrap" |
| 25 | **Multiple testing: Holm vs Bonferroni vs FDR** | 4 items × several hypotheses | "Holm-Bonferroni correction explained" |
| 26 | **Pre-registration; confirmatory vs exploratory** | Credibility of the results | "pre-registration OSF explained" |

## Part 7 — Agreement and calibration metrics (tier A for 27–28)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 27 | **Mincer–Zarnowitz regression** | The standard "is the prediction calibrated" test; we show it passes (slope ≈ 1) while attenuation is present | Task 3.1 | "Mincer Zarnowitz regression forecast efficiency" |
| 28 | **Reverse regression and why its slope = λ** | The diagnostic the standard validation misses | M2 Prop 2 | "reverse regression slope R squared" |
| 29 | **Bland–Altman plot** | Agreement between two measurement methods | 3.1 | "Bland Altman plot explained" |
| 30 | **MAE, MAPE, RMSE, bias** | Level accuracy numbers | 3.1 | "forecast accuracy metrics MAE MAPE RMSE" |

## Part 8 — Probabilistic evaluation (tier B)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 31 | **Proper scoring rules; CRPS** | Scores a whole distribution, so the remedy isn't judged only by average error | Task 6.1 | "CRPS continuous ranked probability score explained", "proper scoring rules" |
| 32 | **Prediction intervals and coverage** | Task 4.4: intervals trained on smooth data under-cover real prices | Task 4.4 | "prediction interval coverage calibration" |
| 33 | **Quantile regression with LightGBM; conformal prediction (split, Mondrian)** | Your existing lab pipeline | Task 4.4 | "conformal prediction tutorial", "LightGBM quantile regression" |

Key paper: Gneiting & Raftery (2007), *Strictly proper scoring rules, prediction, and estimation*, JASA.

## Part 9 — Machine learning pieces (tier B)

| # | Topic | Why | Search terms |
|---|---|---|---|
| 34 | **Elastic net (lasso + ridge)** | One of RTFP's two base learners; used in the replica imputer | "elastic net regularization StatQuest" |
| 35 | **Gradient boosting; Cubist (model trees)** | Cubist is RTFP's other learner; we use boosted trees as the stand-in | "gradient boosting StatQuest", "Cubist model tree rules" |
| 36 | **Cross-validation, and why random CV hides problems in time series and space** | RTFP's reported CV score 0.85 | "time series cross validation", "spatial cross validation" |
| 37 | **Inverse distance weighting; spatial autocorrelation (Tobler's law)** | RTFP's fallback for sparse markets; also explains the distance effect | "inverse distance weighting", "Tobler first law of geography", "Moran's I" |
| 38 | **Diffusion models for time-series imputation (CSDI)** | Optional GPU arm | "CSDI time series imputation diffusion" (tier C) |

## Part 10 — Market integration and price transmission (tier B, domain)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 39 | **Law of one price; spatial market integration** | Proposition 4: imputation makes markets look more integrated | Task 4.2 | "spatial market integration law of one price", "food price transmission" |
| 40 | **Cointegration and error-correction models; half-life of adjustment** | How fast price gaps close, which imputation biases | 4.2 | "cointegration error correction model explained", "half-life mean reversion" |
| 41 | **Threshold / asymmetric transmission** | Context for Bangladesh rice integration papers | context | "threshold cointegration price transmission" |

Context papers: Ravallion (1986); Goletti, Ahmed & Farid (1995); the 2022 *Agricultural and Food Economics* paper on rice in Bangladesh.

## Part 11 — Causal design for the flood event (tier B)

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 42 | **Difference-in-differences; parallel trends** | August 2024 flood effect, estimated separately on observed and imputed cells | Task 4.3 | "difference in differences explained", "parallel trends assumption" |
| 43 | **Event study design; pre-trends** | Check no divergence before the flood | 4.3 | "event study regression leads and lags" |

Free references: Scott Cunningham, *Causal Inference: The Mixtape*; Huntington-Klein, *The Effect*. A single event with one treatment date means you do **not** need the newer staggered-DiD literature, but be able to say why.

## Part 12 — Food security and early warning (tier A for 44–46: this is the paper's "so what")

| # | Topic | Why | Feeds | Search terms |
|---|---|---|---|---|
| 44 | **WFP ALPS (Alert for Price Spikes)** | Operational alert we replicate | Task 4.1 | "WFP Alert for Price Spikes ALPS technical guidance note" |
| 45 | **FAO IPA (Indicator of Price Anomalies)** | Second alert; SDG indicator 2.c.1 | 4.1 | "FAO indicator of price anomalies IPA SDG 2.c.1" |
| 46 | **How RTFP is built and used** | You must know Andrée 2021 (PRWP 9886) and Andrée & Pape 2023 (PRWP 10559) in detail | all | "Estimating Food Price Inflation from Partial Surveys" |
| 47 | **Precision, recall, F1, Cohen's κ for alerts** | Measuring missed alerts | 4.1 | "precision recall F1 explained", "Cohen's kappa" |
| 48 | **Food price data ecosystem: WFP, FAO GIEWS, FEWS NET, DAM, TCB, BBS CPI** | Knowing who collects what, and who feeds whom (circularity) | 1.2, 5.x | read the data pages themselves |
| 49 | **Consumer price index (CPI) basics; base year; COICOP** | The BBS benchmark | Task 5.3 | "how CPI is calculated", "COICOP classification" |

## Part 13 — Generalising across countries (tier B)

| # | Topic | Why | Search terms |
|---|---|---|---|
| 50 | **Meta-analysis: forest plot, fixed vs random effects, heterogeneity (I², τ²)** | Combining the effect across ~40 countries | "meta analysis forest plot random effects" |
| 51 | **Meta-regression** | Does attenuation depend on a country's data coverage? | "meta-regression explained" |

## Part 14 — Research practice and engineering (tier B)

| # | Topic | Why | Search terms |
|---|---|---|---|
| 52 | **Reproducible research: git, environments, one-command rebuild, manifests/hashes** | Reviewer trust | "reproducible research workflow Python" |
| 53 | **FAIR data, data licences, DOIs, Zenodo** | Archive and citation | "Zenodo GitHub integration DOI", "FAIR data principles" |
| 54 | **Python stack: pandas, statsmodels, linearmodels, scikit-learn, matplotlib** | All analysis | "linearmodels PanelOLS tutorial", "statsmodels quantile regression" |
| 55 | **Next.js static export, Plotly, MapLibre, GitHub Actions cron, Vercel** | Companion site | "Next.js static export", "GitHub Actions scheduled workflow", "MapLibre GL JS tutorial" |

## Suggested study order (about 5 weeks, 1–2 hours a day)

| Week | Parts | Goal |
|---|---|---|
| 1 | 1, 2, 3 | Explain, without notes, why a good imputer still shrinks variance, and how that differs from measurement error |
| 2 | 4, 5 | Run a fixed-effects regression with clustered errors and read it |
| 3 | 6, 7, 8 | Know which metrics hide attenuation (MZ) and which expose it (reverse slope, variance ratio, CRPS) |
| 4 | 9, 10, 11 | Understand the replica imputer, market integration, and the flood design |
| 5 | 12, 13, 14 | Read Andrée 2021 and 2023, ALPS and IPA; skim meta-analysis; set up the repo |

You can start coding Phase 0–1 and the simulation (Task 2.0) in week 2; the simulation is also the best way to learn Parts 1–4 by doing.

## Sentences you should be able to say fluently (reviewer-proof)

1. "A conditional-mean imputer is unbiased and calibrated, yet it removes the variance that the observed information cannot explain."
2. "Our estimate is the share of local price variation lost, after removing measurement noise."
3. "Cross-validated R² and level correlations cannot detect this, because they are dominated by trend and between-market variation."
4. "Imputed prices are safe as regressors but unsafe as outcomes for local shocks."
5. "We never treat DAM-fed cells as ground truth."

## Core reading list (read in this order)

1. Andrée (2021), PRWP 9886, *Estimating Food Price Inflation from Partial Surveys*
2. Andrée & Pape (2023), PRWP 10559
3. Adewopo, Andrée, Peter, Solano-Hermosilla & Micale (2025), *PLOS ONE* 20(4): e0320720
4. van Buuren (2026), arXiv 2605.03733; and *Flexible Imputation of Missing Data*, ch. 1–3
5. Ratledge et al. (2022), *Nature* 611: 491–495
6. Proctor, Carleton & Sum (2023), NBER WP 30861
7. Angelopoulos et al. (2023), *Science* 382: 669–674 (prediction-powered inference)
8. Gneiting & Raftery (2007), JASA (proper scoring rules)
9. WFP (2014), ALPS Technical Guidance Note; Baquedano (FAO), IPA
10. Andrée (2026), arXiv 2603.02898
