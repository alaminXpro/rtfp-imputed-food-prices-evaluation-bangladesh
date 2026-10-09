# Methodology Design — Variance Attenuation in ML-Imputed Food Prices

> Design specification for the Methods section, not manuscript prose. Every element maps to a task in `docs/plans/2026-10-08-research-plan.md`.
> Notation is fixed here, so the manuscript, code and results notes all use the same symbols.

## M0. Contribution statement (what makes this non-naive)

1. **Theory:** a formal account of why conditional-mean (ensemble) imputation preserves national price signals but attenuates local ones.
   - It also shows why the standard validation metrics used for RTFP (cross-validated R² on levels, level correlations, Mincer–Zarnowitz) are *structurally blind* to this attenuation: "validated yet attenuated".
2. **Identification:** three designs that rule out the obvious objection that observed markets are simply more volatile.
   - within-market/within-month fixed effects
   - a natural hold-out from 83 archived data vintages (the World Bank's *own* model judged against later-arriving truth)
   - a controlled masking experiment
3. **Signal vs noise:** we separate how much of the "lost" variance is measurement noise (removing it is good) from how much is genuine local price signal (removing it is bad).
4. **Consequences:** measured with operational indicators that agencies actually use:
   - WFP ALPS and FAO IPA alerts
   - market-integration estimates
   - local shock impacts
   - forecast-interval coverage
5. **A usage taxonomy:** which uses of imputed prices are safe (national trends; use as a regressor) and which are unsafe (local spikes, spatial integration, local shock outcomes), each backed by a proposition and a test.
6. **A remedy:** distribution-preserving imputation, evaluated with proper scoring rules (CRPS), not only MAE.
7. **Generalisation:** the core test repeated on all ~40 RTFP countries.
8. **Credibility:** pre-registered confirmatory hypotheses and a fully reproducible pipeline.

## M1. Data and notation

- Markets `m = 1..M` (Bangladesh: 111), months `t` (2007-01 to 2026-10), items `k` ∈ {rice, wheat flour, palm oil, lentils}.
- True log retail price `p_mtk`. Published RTFP estimate `p̂_mtk = log c_<k>`.
- Observation indicator `O_mtk = 1` if the raw item column is non-missing. The observed price is `y_mtk = p_mtk + ν_mtk`, where `ν` is measurement or sampling noise.
- Price change `Δp_mtk = p_mtk − p_m,t−1,k`. Common component `μ_tk` = cross-market mean of Δp in month t. Idiosyncratic component `e_mtk = Δp_mtk − μ_tk`.
- RTFP-supplied covariates: `trust_k` (1–10), `spatially_interpolated` (0/1), `data_coverage`, `data_coverage_recent`.
- Derived covariates: `dist_obs` (km to the nearest market observed in month t) and `gap` (months since the market's last observation).
- **Vintages** `v = 1..83` (2024-01-25 → 2026-10-05). The hold-out set H is the cells with `O^(v) = 0` in an earlier vintage and `O^(v′) = 1` in a later one.
- **External sources:**
  - TCB Dhaka daily min/max (from May 2021)
  - BBS monthly CPI, national/rural/urban and COICOP food group (collected independently of DAM/WFP)
  - DAM district prices (optional, gated)

## M2. Theoretical framework (Section 3 of the paper)

Assume the imputation approximates the conditional mean given the observed information set 𝓘: `p̂ = E[p | 𝓘]`. RTFP averages M chained-equation chains, which approaches this as M grows. Write `p = p̂ + u`, with `E[u | 𝓘] = 0`. This is a **Berkson-type** error: the error is uncorrelated with the *prediction*, not with the truth.

**Proposition 1 (attenuation):**
- `Var(Δp̂) = Var(Δp) − Var(Δu) ≤ Var(Δp)`.
- The attenuation factor `λ = Var(Δp̂)/Var(Δp) = R²_Δ` is the share of price changes predictable from 𝓘.
- Decompose into common and idiosyncratic parts: `λ_common ≈ 1`, because observed markets reveal μ_t, while `λ_idio = R²` of local deviations, which is small.
- **Prediction:** pooled variance is roughly preserved, but within-month dispersion is strongly attenuated. This matches the quick-look result (ratio 0.43–0.55).

**Proposition 2 (calibration-blindness):**
- For `p̂ = E[p|𝓘]`, regressing truth on the prediction gives slope 1 (Mincer–Zarnowitz efficiency holds).
- Level correlation `corr(p, p̂)` is dominated by between-market and trend variation, so it is high.
- The *reverse* regression slope (p̂ on p) equals λ.
- **Implication:** the metrics used to validate RTFP so far cannot detect attenuation. The right diagnostics are variance ratios and reverse-regression slopes on **idiosyncratic changes**.
- *This is why the paper is not a re-validation exercise.*
- **Empirical update (hold-out quick look, 2026-10-09):**
  - Levels behave as Proposition 2 predicts: MAPE 3.1–3.8%, bias ≈ 0.
  - Local changes do not. The MZ slope of observed `e` on imputed `e` is 0.10–0.26, not ≈ 1; the reverse slope is 0.03–0.08; the variance ratio is 0.13–0.50.
  - So the imputed local deviations are not merely shrunk: they are only weakly related to what actually happened. Proposition 2 therefore needs a second case, where the imputer is *not* the conditional mean for the local component. Possible reasons: model misspecification, outlier smoothing, or vintage-specific noise.
  - Revisit after Task 2.4 (noise correction can raise these slopes) and Task 2.0 (simulate a misspecified imputer).

**Proposition 3 (alerts):**
- An alert fires when a standardised deviation exceeds a threshold c.
- With **absolute** thresholds (e.g. fixed % growth, loss-optimised thresholds as in Yemen), recall falls with λ.
- With **series-standardised** thresholds (ALPS uses each series' own residual SD), uniform scaling would cancel. Attenuation is not uniform, though: idiosyncratic spikes shrink relative to common seasonal swings.
- **Prediction:** recall losses concentrate in *local* events, while national events are preserved. Test both threshold types.

**Proposition 4 (co-movement inflation):**
- `corr(Δp̂_i, Δp̂_j) = Var(μ) / sqrt((Var(μ)+λ_i σ²_i)(Var(μ)+λ_j σ²_j))` rises as λ falls.
- Imputed–imputed pairs therefore look more integrated than they are. Price-transmission speed is overstated (half-lives too short).

**Proposition 5 (downstream regressions):**
- **(a) Imputed price used as a regressor:** Berkson error does not bias a linear slope; it costs only precision. This is a *safe* use, and the paper should say so.
- **(b) Imputed price used as an outcome** for a local shock Z (flood, conflict) that is not in 𝓘: the coefficient is attenuated by the share of Z's effect predictable from 𝓘. This is an *unsafe* use.

**Proposition 6 (noise vs signal):**
- Observed changes contain noise: `Var(Δy) = Var(Δp) + 2Var(ν)` when ν is serially uncorrelated.
- Noise leaves an MA(1) fingerprint: `Cov(Δy_t, Δy_{t−1}) = −Var(ν)`. This identifies Var(ν), Roll-style.
- Signal attenuation is therefore `λ* = Var(Δp̂) / (Var(Δy) − 2Var(ν))`. Only the part of λ below what noise removal explains counts as lost information.

**Simulation check:** each proposition is verified on synthetic panels with known p, ν and 𝓘 before being applied to real data (Task 2.0).

## M3. Empirical designs

**D1 — Descriptive attenuation (Task 2.1):**
- within-month SD ratio of `e`, imputed ÷ observed
- variance decomposition into common and idiosyncratic parts
- tail metrics: share of |Δp| > 5/10/20%, kurtosis
- 95% CIs from a market-cluster bootstrap with B = 1,000

**D2 — Fixed-effects test robust to selection (Task 2.2, confirmatory H1):**
```
log|e_mtk| = β·(1 − O_mtk) + α_mk + γ_tk + X'δ + ε
```
- Fixed effects: market × item and month × item.
- Standard errors: two-way clustered by market and month; Driscoll–Kraay as a robustness check.
- Restricted sample: **switchers only**, i.e. markets with both observed and imputed spells, from 2020 onward.
- Complements: Brown–Forsythe test within each month, and quantile regressions at the 50th and 90th percentiles.

**D3 — Natural hold-out from vintages (Task 3.1, confirmatory H2, headline):**
- For cells in H, compare the last pre-observation imputation `p̂^(v)` with the later observation `y^(v′)`.
- Report:
  1. level accuracy: MAPE, bias, Bland–Altman
  2. MZ slope (expected ≈ 1, illustrating Proposition 2)
  3. **reverse slope and variance ratio of e (expected < 1, Proposition 1)**
  4. revision paths across vintages
- **Artefact control:** key on `geo_id`, drop renamed markets, and re-estimate excluding the 2026 market expansion.

**D4 — Masking experiment (Task 3.2):**
- Build a replica of the RTFP imputer (chained equations; elastic net + boosted trees; IDW/seasonal starting values; M = 5).
- On dense 2021–2026 segments, mask 20% of observed cells, stratified, with R = 50 replications.
- Arms:
  1. ensemble mean, as RTFP does
  2. single stochastic draw
  3. proper multiple imputation, M draws with noise
  4. *(optional, GPU)* a generative imputer, CSDI
- Metrics: MAE, λ, alert recall, CRPS.
- Shows that the *mean-type output* causes attenuation, and the MAE-vs-distribution trade-off.

**D5 — Noise decomposition (Task 2.4):**
- MA(1) autocovariance estimator of Var(ν) per market and item.
- Cross-check against a two-source method: TCB vs WFP/DAM for Dhaka, where the covariance of two independent measures gives the signal variance.
- Report λ (raw) and λ* (signal).

**D6 — Consequences (Phase 4):**
- **Alerts:** ALPS (WFP 2014) and IPA (FAO). On H and masked cells, report recall, precision and κ of imputed-based vs observed-based alerts. Use persistence-filtered alert definitions (≥ 2 months) so noise-driven alerts in observed data don't inflate "misses".
- **Integration:** pairwise correlations by pair type × distance bin. Error-correction half-lives (observed-only vs all cells).
- **Local shock:** August 2024 eastern-flood difference-in-differences. Effect estimated on observed vs imputed cells (tests Proposition 5b).
- **Forecasting:** LightGBM quantile + conformal intervals trained on all cells. Coverage on observed vs imputed test cells.

**D7 — Remedy: a vintage-calibrated uncertainty layer (Task 6.1):**
- **No replacement imputer.** Proposition 1 and the hold-out results show local moves are largely outside 𝓘, so no imputer recovers them. The remedy quantifies uncertainty on the *published* values.
- **Error model:**
  - quantile LightGBM for `log y − log p̂` on H
  - features: `gap`, `dist_obs`, `trust`, coverage, `spatially_interpolated`, item, month, vintage age
- **Calibration:**
  - conformalized quantile regression, Mondrian by gap × distance
  - weighted conformal under covariate shift, because H is not exchangeable with never-observed cells
- **Path simulation:**
  - M = 200 paths per market, with residuals that are AR(1) over time and spatially correlated
  - the mean is held at the RTFP value
  - each path feeds ALPS/IPA, giving P(alert)
- **Evaluation:**
  - time-blocked split of H (calibrate on earlier vintages, test on later)
  - metrics: coverage, width, CRPS, Brier score of P(alert), alert recall
- **Baselines:**
  - the published `[l, h]` band
  - a fixed-width interval
  - a global Gaussian
  - replica multiple-imputation draws (D4)
  - constrained Bayes (Ghosh 1992)
- **Closest prior method:** Gui, Barber & Ma (2023), *Conformalized Matrix Completion*. Our additions are real later-arriving truth for calibration, a path simulation for probabilistic early warning, and an operational dataset.

**D8 — External benchmarks:**
- **BBS:** compare the 12-month inflation and the m/m variance of RTFP's national food price index with BBS food CPI. BBS is independent of DAM/WFP; this checks the *common* component, which Proposition 1 says should be preserved.
- **TCB:** compare monthly OHLC from daily min/max with RTFP Dhaka `o_/h_/l_/c_`, validating the modelled intra-month range.
- **DAM:** district prices for cells never input to RTFP, only if Gate G2 passes.

**D9 — Generalisation (Task 6.3):** repeat D1–D2 for every RTFP country; show a forest plot of β and λ; run a meta-regression of λ on country data coverage.

## M4. Inference and reporting standards

- Confirmatory hypotheses H1–H5 are pre-registered (`docs/analysis_plan.md`). Everything else is labelled exploratory.
- **Multiple testing:** Holm correction across 4 items × confirmatory hypotheses.
- **Uncertainty:**
  - market-cluster bootstrap for ratios
  - wild cluster bootstrap where clusters are few (per-country runs)
  - moving-block bootstrap for time-series metrics
- **Effect sizes:** always reported with 95% CIs and N. No p-value-only claims.
- **Robustness:** national vs division-month definition of μ; 1- vs 3-month changes; excluding 2007–2019 and the camp markets; using `inflation_<k>`; earliest vs latest vintage.
- **Reproducibility:** one-command rebuild, pinned environment, data manifest with sha256, Zenodo DOI.

## M5. Anticipated reviewer objections → where answered

| Objection | Answer |
|---|---|
| "Observed markets are just more volatile." | D2 (FE, switchers), D3 (same cells), D4 (random masking) |
| "Imputation just removes noise; that is good." | D5 / Proposition 6: λ vs λ*; only signal loss is criticised |
| "RTFP is validated (R² 0.85, r = 0.94–0.99)." | Proposition 2: those metrics are blind to attenuation; D3 shows MZ ≈ 1 *and* λ < 1 on the same cells |
| "DAM feeds RTFP, so this is circular." | We never use DAM-fed cells as truth: H is later observations, BBS is independent, DAM only for never-input cells |
| "One country." | D9, all RTFP countries |
| "So what?" | D6 consequences on operational alerts, integration and local shock estimates, plus the safe/unsafe taxonomy |
| "Criticism without a fix." | D7 remedy evaluated with CRPS |
