# Methodology Design — What ML-Imputed Food Prices Can and Cannot Tell Early-Warning Systems

> Design specification for the paper, not manuscript prose.
> Revision 3 (2026-10-09), after a three-lens review with adversarial verification that inspected the data files. Supersedes revisions 1–2.

**Working title (chosen after Task 2.2):** *How much local price information do ML-imputed food prices carry? An independent, release-history evaluation of the World Bank's RTFP.*

## M0. Core contributions

1. **To our knowledge, the first evaluation of RTFP without producer involvement, and the first to use its archived release history.** Market-months that were imputed in one release and surveyed in a later one let us judge the producer's model against data it did not have, on price levels *and* on local month-to-month changes. We position this against Adewopo et al. (2025, producer co-authored, Nigeria, level correlations) and reproduce their level metrics on our sample. Headline, stated as a measured share with CI after Task 2.2: *imputed cells track X% of local month-to-month variation, bounded for measurement noise in the survey data.*
2. **Why standard validation metrics cannot see this.** Cross-validated R², level correlations and level Mincer–Zarnowitz tests are dominated by trend and between-market differences. We show, on the same cells, level metrics that pass alongside local-change metrics that do not, and state exactly which quantities the hold-out identifies (M2). The statistics are textbook (Rubin 1987; van Buuren 2018); the contribution is the measurement and the mapping to metrics in use.
3. **The boundary of safe use.** National index vs BBS food CPI (independent source), descriptive. Local uses: ALPS/IPA alert agreement on hold-out cells against a simulation benchmark of what a noise-removing imputer would achieve; coefficient attenuation when imputed local changes are used as a regressor (equals β_MZ, measured directly). Assembled into one evidence-backed table of supported and unsupported uses.
4. **Calibrated uncertainty for imputed cells inside the hold-out's support**, released as a join-able table; cells outside the support are flagged, not covered. Known technique (conformalized quantile regression; Gui, Barber & Ma 2023 for missing entries) applied with real later-arriving survey values.
5. **A bounded replication in 3–5 other RTFP countries**, selected by pre-stated data criteria before their results are seen (gated, Task 6.1).
6. **Registered, prospective confirmation and full reproducibility.** H1–H2 are confirmed on cells first surveyed in releases published after registration, with the expected sample size stated in advance; everything else is labelled. One-command rebuild; Zenodo DOI reserved at submission, released on acceptance.

## M1. Data and notation

### Data facts the design must respect (found in review)
- **geo_id is re-keyed** on 2024-03-18 (all), 2025-03-31 (all but 2; coordinates also shift by up to 1.25 km) and 2026-01-13 (3 ids); one geo_id moves from 'Rangpur Municipality Market' to 'Rangpur Sadar'. Market names never change, so markets are resolved by name to a stable `mkt_uid`, with a guard that fails if a name moves more than 3 km (Task 1.3). geo_id is never used as a key across vintages.
- **Pipeline toggling:** since 2024-11-04, releases come from one of two pipelines in an irregular order (runs of up to 6). 'Reduced' releases hold about a third fewer pre-2023 survey values (lentils ~1,250–1,500 vs 1,976). In 7 releases (mostly 2025-10-27) some survey values also *differ* from the latest release (798 cells; median 7.6%, max 31%). Cells that drop and return form the toggle stratum.
- **Rice is absent from 42 of 83 vintages**; 2024-01-25 has rice survey values but no published rice estimates. Vintage-dependent quantities, including the evaluation truth, use the latest vintage containing the item.
- **Stale and aggregated quotes:** 14–33% of consecutive observed changes are exactly zero; about half of all observed cells, and 69% of the 2020-08+ non-Division noise sample, carry WFP `priceflag = aggregate`. `priceflag` is joined from the WFP feed (Task 1.2).
- **Gross entry errors exist in the survey feed:** e.g. Patuakhali Sadar 2025-03 lentils Tk 17.3 between 112.5 and 91.4 (flagged 'actual'). One such error dominates autocovariance-based noise estimates, so a trimmed estimator is reported alongside.
- `spatially_interpolated` is 0 on every row of the latest vintage and is not used.
- **Co-located markets:** 'Rajshahi Division'/'Rajshahi Sadar' and 'Rangpur Municipality Market'/'Rangpur Sadar' share coordinates in 25 vintages, so `dist_obs` can be 0 for an imputed cell. The six '… Division' series are observed only 2007-01 to 2020-07; too little overlap with district markets to tell whether they are averages. They never supply clean hold-out truth (2023-09+); noise is estimated on non-Division markets from 2020-08 (Division series and 2007– as sensitivity).
- Months with zero or near-zero observed markets exist (e.g. 2025-11, 2026-01, 2026-09/10).

### Notation
- Markets `m` (`mkt_uid`), months `t`, items `k` ∈ {rice, wheat flour, palm oil, lentils}, vintages `v`.
- Survey value `y_mtk` = plain item column when present (assumption audited in Task 1.2). Published estimate `ĉ_mtk = c_<k>`.
- **Observed side uses `log y`; imputed side uses `log ĉ`.** (c_ differs from y in ~10% of observed cells due to outlier adjustment; c_-based variants go to the robustness battery.)
- `Δ` = monthly log change. `μ_tk^(v)` = mean Δ over cells *observed in vintage v*, month t, item k. Local component `e = Δ − μ`, each side demeaned with its own vintage's μ. Pooled (common μ) versions reported alongside to show the size of the national-error term.
- Derived: `gap` (months since the market's last observation in v), `dist_obs` (km to nearest market observed in t in v), `n_obs_month`.

### Hold-out H (Task 1.3)
A cell (mkt_uid, t, k) enters H at its first observed vintage `v_truth` that follows an imputed vintage; `v_final` is the last imputed vintage before it (the estimate judged); `ĉ_first` is its earliest imputed value. **Evaluation truth = y in the latest vintage** (`y_eval`), which equals the WFP/HDX feed exactly (Table A1); the first-observed value is kept for revision statistics (Table A2d). *Revised 2026-10-09: ~1,000 values that appeared in single releases are absent from both the latest release and the WFP feed.*

**Strata (mutually exclusive, precedence top-down; v = v_final):**
1. **Toggle:** the survey value appeared in a vintage before v (releases alternate between 'full' and 'reduced' pipelines that drop and restore values). Not a late arrival; reported separately.
2. **Unconfirmed:** the value that appeared is not observed in the latest vintage, hence not in the WFP feed. Excluded.
3. **Backfill:** arrived more than 6 months after the price month (sensitivity 3 and 12). Historical additions; reported separately.
4. **Forecast:** zero markets observed for item k in month t in v. Real-time nowcast: the imputing release came a median of 1 month after the price month.
5. **Cross-sectional:** month t partly observed in v. The model knew μ_t; the clean test of *local* information and the headline stratum.

Clean strata (4–5) hold 5,768 cells, price months 2023-09 to 2026-08 (Table A2). Results therefore describe recent imputations; never-observed cells are mostly 2007–2019 history.

**Change pairs** (t−1, t), both ĉ from the same vintage:
- *both-imputed* (headline);
- *imputed-t / observed-(t−1)* (one-step nowcast case, reported separately; shares ν_{t−1} with the truth);
- *observed-t / imputed-(t−1)*: excluded.

### External sources
BBS monthly CPI, national food index only (2005-06 base, to 2023-03: ILO group 'Food, beverage & tobacco'; 2021-22 base, from 2023-04: COICOP 01 'Food and non-alcoholic beverages'); changes computed within one base, no splice; TCB Dhaka daily min/max (May 2021 →), compared with **Dhaka Sadar** only (the one Dhaka market with substantial observed coverage; rice ends 2024-12); HDX/WFP feed with source and priceflag fields.

## M2. Framework (paper Section 2)

Observed local change `e_obs = e + ν_Δ` (ν_Δ measurement noise in the change). Imputed local change `ê = b·s + η`, where `s = E[e|𝓘]` is the part predictable from the model's information set and `η` is model noise independent of `e` and ν.

**What the hold-out identifies** (after Var(ν_Δ) from D2):
- `Var(ê)`, `Cov(ê, e_obs) = Cov(ê, e)`, `Var(e)`. Hence:
  - `β_MZ = Cov(ê, e)/Var(ê)` — the calibration slope; equals the coefficient attenuation factor when ê replaces e as a regressor.
  - `ρ* = Cov(ê, e)/sqrt(Var(ê)·Var(e))` — share of local variation tracked is ρ*².
  - the composite `b·Var(s)`.
- **Not identified:** b and Var(η) separately. Two bounding readings: (i) `η = 0` — the imputer carries only signal but mis-scaled by b; (ii) `b = 1` — the imputer is calibrated in its signal and the excess variance is noise. Under (i) ρ*² = Var(s)/Var(e); under (ii) ρ*² is a lower bound on it. We report both readings and claim only what holds under either.
- Identity used as a check: `β_rev = β_MZ · λ`, with `λ = Var(ê)/Var(e_obs)`.

**Expectations, each tested:**
- *Levels:* small MAPE, no bias, level MZ slope ≈ 1 (market-FE and month-demeaned versions). Equivalence tested with TOST.
- *National signal:* RTFP's national index tracks BBS food inflation about as well as an observed-only index (descriptive).
- *Local information:* ρ*² well below 1 in the cross-sectional stratum.
- *Regressor use:* coefficients on imputed local changes are attenuated by β_MZ (measured directly in D1).
- *Alerts:* ALPS/IPA agreement with survey-based alerts is lower than the simulation benchmark of a noise-removing, signal-preserving imputer (D4).

**Simulation (Task 3.1):** synthetic panels (common AR(1) + local AR(1) + seasonality; observed = true + ν with stale-quote and averaging variants; realistic missingness). Four imputer types: calibrated conditional mean; IDW/seasonal fill; chain mean with residual chain noise; outlier smoothing. Targets: the empirical (ρ*, λ, β_MZ) triple with its CI from Task 2.2. Also checks the noise estimators and the disattenuation.

Language: *framework*, *expectations*, *measurement* — no *propositions*, *identification* or *causal*.

## M3. Empirical designs

**D1 — Hold-out evaluation (Task 2.2, headline).** By item × stratum × pair type × first/final:
1. *Levels:* MAPE, median APE, bias, level MZ slope (market-FE; month-demeaned), Bland–Altman; Adewopo-style level correlations for comparability.
2. *Local changes:* corr(ê, e_obs), β_MZ, λ, β_rev, ρ* (with the D2 band). Within-month×item demeaned and pooled versions.
3. *Spikes:* a survey spike is |Δy_t| > 10% whose level shift persists (≥ 50% of the jump retained at t+1, same sign), with (t, t+1) observed in v′ (conditioning stated). Share of spikes where ê moved ≥ 5% in the same direction.
4. *Error vs covariates:* |level error| and |ê − e_obs| on gap, dist_obs, trust, vintage age (RTFP's `data_coverage` is one value per release for all markets, so it is not a cell covariate); month FE; two-way clustered SEs. Does the trust score predict external error?
5. *Revision paths* across vintages containing the item.
Inference: two-way (market × month) cluster bootstrap, B = 1,000, with rel resampled jointly so ρ*'s CI includes noise-estimate uncertainty. Holm across items.
- **Implemented (2026-10-09), deviations:** Tables 2, 3 and 9 use B = 500 pigeonhole draws; the ρ* columns of Table 3 carry no CI (rel held fixed); joint resampling of rel is done only in the Gate G1 bootstrap (Table 1c, B = 500) and in Table 5 (B = 1,000). No Holm correction in the exploratory Tables 2–3 and 9 (no hypothesis is tested there). Known gross survey entry errors (e.g. Patuakhali Sadar 2025-03) remain in the evaluation truth of Tables 2–4 and 6; only the noise estimate has a sensitivity variant without it (A_no_patuakhali).

**D2 — Measurement-noise band (Task 2.1, runs before registration; Gate G1).**
- Model: observed local level `y = ℓ + ν`, with `ℓ` AR(1) (persistence φ, variance V) and ν white noise. Autocovariances of observed local changes:
  - `γ_Δ(1) = −Var(ν) − V(1−φ)²`
  - `γ_Δ(k) = −V(1−φ)²·φ^(k−1)` for k ≥ 2
- Estimators: `φ = γ_Δ(3)/γ_Δ(2)`; `Var(ν) = −γ_Δ(1) + γ_Δ(2)/φ`. The naive white-noise estimator `Var(ν) = −γ_Δ(1)` ignores mean reversion and overstates Var(ν) under this model; both are reported. Derivation in the appendix. *Result (Task 3.1 simulation, A4):* verified under white survey noise (naive above truth, AR(1) below where identified); under stale-quote and averaging noise both estimators understate Var(ν), so the naive estimator is not a guaranteed bound outside the white-noise model.
- Reliability of observed local changes: `rel = Var(Δℓ)/(Var(Δℓ) + 2·Var(ν))`, with `Var(Δℓ) = 2V(1−φ)`.
- Staleness-aware variant: estimate on changes conditional on a quote update; stale quotes treated as a separate error component.
- Dhaka Sadar vs TCB two-source check: supporting sanity check only (one market, short overlap; month counts in Table A5), after verifying that TCB is not among WFP's listed sources.
- Output: **ρ* reported as a band** over rel ∈ [lowest credible estimate, 1], plus the point estimate under each estimator.
- **Gate G1 (single threshold, exhaustive):** using the clean hold-out from Task 1.3 and the *conservative* end of the band, pivot if **≥ 2 of 4 items** have ρ* ≥ 0.5 (ρ*² ≥ 0.25) in the cross-sectional stratum. Pivot = validation/revision study with the uncertainty table as lead contribution. The same 0.5 threshold is H1.
- **Implementation clarification (2026-10-09, after independent audit):** each reliability is applied to truth pairs of the population it was estimated on (A → all pairs; staleness-aware B → pairs whose truth change is non-zero); a combination is credible if 0 < rel ≤ 1. The first implementation mixed populations (B's reliability on all pairs); it is reported as `literal_v1`. **Outcome: PROCEED (1 of 4 items; lentils 0.61, wheat flour 0.496); literal_v1 = PIVOT (2 of 4); bootstrap share PIVOT 0.35.** Tables T1–T1d.

**D3 — Within-market comparison on switching markets (Task 2.4, support).**
- Sample: markets with both observed and imputed spells, 2020 →.
- PPML (Poisson pseudo-ML, log link; handles exact zeros and many FE; Santos Silva & Tenreyro 2006) of |e| on `imputed` + market×item FE + month×item FE; two-way clustered SEs.
- Benchmark-relative hypothesis: coefficient on `imputed` below `log sqrt(rel)` — the value a noise-removing, signal-preserving imputer would produce under Gaussian e. Below it means more than noise was removed.
- Event-time plot (−6..+6 months around spell boundaries) with pre-boundary trends shown.
- Covariate balance of H vs never-observed cells comes from Task 1.3 (Table A3).
- **Implemented (2026-10-09):** Table 5 / `T5.md`, exploratory. Benchmarks from A_main and A_trim1 reliabilities; bootstrap B = 1,000 re-estimates rel and the coefficient on the same draw. The Gaussian benchmark is fragile: the distribution-shape term log(shape_imp/shape_obs) is as large as the coefficients, so the squared-change specification (no Gaussian assumption) is reported alongside.

**D4 — Consequences (Tasks 4.1–4.2).**
- *Alerts:* ALPS (WFP 2014) and IPA (FAO) from ĉ vs from y on H cells; recall, precision, κ with CIs. Benchmark: recall/precision a calibrated, noise-removing imputer achieves against noisy survey alerts, from the Task 3.1 simulation calibrated to estimated Var(e), Var(ν) and the empirical seasonal structure. "Local event" = survey alert in a market while < 25% of markets alert that month. Power calculation on expected alert counts precedes any confirmatory use.
  - *Implemented (2026-10-09):* Table 6 / `T6.md`, exploratory. The "local event" filter (< 25% of markets alerting) was **not** implemented; all survey alerts are evaluated. Benchmarks are base-rate matched: a noise-removing oracle (the benchmark named above) and a value with no local information; the T8 rule uses the latter. Power from bootstrap SEs (Table T6_alerts_power).
- *Regressor attenuation:* β_MZ from D1 reported as the multiplier on coefficients estimated with imputed local changes; one worked example re-estimated on observed-only cells if a suitable published RTFP regression exists.
- *National (descriptive):* equal-weight log index of c_ across all markets and items vs (a) the observed-only index, both against BBS food CPI and, where available, the BBS rice sub-index; m/m and 3-month changes; Newey–West / block-bootstrap SEs. *As implemented:* no splice; changes are computed within one CPI base (2005-06 base to 2023-03, 2021-22 base after).
  - *Result (2026-10-09):* BBS food CPI is essentially uncorrelated with both the published and the survey-only four-staple index, so it cannot validate the national signal; imputation is not the cause (the survey-only index is equally uncorrelated); the basket difference is a plausible but untested explanation; BBS publishes no rice sub-index. The published national index tracks the survey-only index (corr 0.81–0.87), partly mechanically. See `outputs/results_notes/T7n.md`.
- *Safe-use table:* assembled by script from D1–D4 outputs (Table 8).

**D5 — Calibrated uncertainty (Task 5.1).**
- Target: log(y/ĉ), i.e. the value the survey will report (latent-price intervals would be wider).
- Quantile LightGBM at 5/10/50/90/95%; features: gap, dist_obs, trust, market-level observed share up to v_final (computed by us; RTFP's `data_coverage` is release-level), item, month, vintage age, stratum.
- **Three-way time-blocked split** by first-observed vintage: train ≤ T1, calibrate (T1, T2], test > T2.
- Conformalized quantile regression (Romano et al. 2019), Mondrian by gap × distance (*as implemented:* stratum × months since last report, since distance is undefined for forecast cells). Weighted conformal (Tibshirani et al. 2019) only within the covariate support of H.
- **Support restriction:** intervals released only for cells with gap ≤ max(H), dist_obs ≤ max(H), and markets with ≥ 1 observation; others flagged "no calibrated interval".
- Metrics: 80/90% coverage (pooled and by month, with dispersion), width, CRPS. Baselines: published [l_, h_] band; fixed width; global Gaussian.
- **Implemented (2026-10-09):** quantile model is scikit-learn HistGradientBoosting (quantile loss), not LightGBM (system library unavailable). Four-way time split (train 60% | model selection 10% | calibration 10% | test 20%); candidates unconditional quantiles, group-empirical quantiles (stratum × gap), quantile GB; chosen on the selection block: **unconditional**. Static Mondrian calibration under-covers later releases, so an online variant is added: one pooled conformal threshold, recomputed on a calibration pool that grows with each test release, at a level updated by adaptive conformal inference (Gibbs & Candès 2021, γ = 0.05). It reaches near-nominal pooled coverage. The decomposition in `T7.md` shows that this comes from the pooled threshold and the growing pool; the adaptive-level step adds essentially nothing, and with one update per test release (a handful of releases; count in `T7.md`) ACI's long-run guarantee does not apply. The released interval is the same multiplicative band for every inside-support cell (unconditional quantiles were selected). Numbers: `T7.md` (generated). Covariates do not sharpen intervals. Results note `T7.md`.

**D6 — Bounded replication (Task 6.1, gated).** 3–5 per-country RTFP market datasets chosen by criteria stated in the registration (≥ 50 vintages; ≥ 30 markets; plain columns present; not Bangladesh). Repeat Tasks 1.3, 2.1, 2.2 core metrics. Forest plot of ρ* bands. Go only if Bangladesh Phases 1–2 finish on schedule.
- **Implemented (2026-10-09):** Table 9 / `T9.md`, exploratory; five countries chosen by the rule from the catalogue snapshot. Pipeline changes for other countries: markets whose name moves > 3 km across vintages are dropped (`on_drift="drop"`, listed per country) and excluded before own-release μ is computed; ρ* > 1 is not counted as a success; scale mismatches between RTFP and WFP/HDX are tabulated (T9d). No forest plot yet; ρ* intervals not reported (rel held fixed).

**Task 3.1 simulation (framework check).** *Implemented (2026-10-09):* Table A4 / `A4.md`. Panels calibrated to Bangladesh; four stylized imputers; three survey-noise models. A calibrated conditional mean is ruled out mainly by λ; the naive noise estimator bounds Var(ν) from above only under white noise (see D2).

**Appendix designs.** Replica-imputer masking study; robustness battery (c_-based observed side; division-level μ; 3-month changes; excluding 2007–2019 and camp markets; `inflation_<k>`; earliest vs latest vintage; excluding `priceflag = aggregate`).

## M4. Registration and inference

- Order: Tasks 1.1–1.3 and 2.1 (noise band and G1) run on existing data, *then* registration (Task 2.R). Registration records the G1 outcome and pre-specifies both branches.
- **Prospective confirmatory hypotheses** (sample: cells first observed in vintages published after registration; cross-sectional and forecast strata; expected N stated in the registration, minimum 6 post-registration months and ≥ 150 cells per item before testing):
  - **H1 (local information):** ρ* < 0.5 at the conservative rel, per item, one-sided, Holm-corrected; both-imputed pairs, cross-sectional stratum. If that stratum does not reach the minimum N, H1 is reported as pending (no forecast-stratum substitute).
  - **H1f (nowcast local information, separate):** the same test on forecast-stratum pairs, reported under its own label.
  - **H2 (levels):** level MZ slope within [0.9, 1.1] and |bias| < 1%, per item, TOST with two-way clustered SEs.
- **Defined on existing data and already computed before registration, so exploratory** (2026-10-09; see `docs/analysis_plan.md` §1): H3 (D3 benchmark-relative PPML, Table 5), H4 (D4 alert agreement vs simulation benchmark, Table 6 with power). Originally intended as registered-before-analysis; that ordering did not hold.
- **Descriptive:** national index vs BBS.
- Effect sizes with 95% CIs and N everywhere; no p-value-only claims. All quick-look results disclosed as exploratory.

## M5. Anticipated objections → where answered

| Objection | Answer |
|---|---|
| "It's just noise removal, and it's intended." | D2 reports ρ* as a band over credible noise levels; D4's benchmark is what a noise-removing imputer would achieve; the producer's own statements on smoothing are quoted; framing is "what the data supports". |
| "Observed and imputed cells differ." | D1 compares the same cells; strata separate forecast, cross-sectional and toggle cases; D3 compares the same market across spells; covariate balance (Table A3). |
| "RTFP is validated (R² 0.85; r 0.94–0.99 in Nigeria)." | D1 reproduces level metrics on H alongside local-change metrics on the same cells; M2 states why the former cannot detect the latter. |
| "DAM feeds RTFP, so this is circular." | Truth in H is a survey value that was absent from the vintage that produced the estimate; toggle cells, whose value appeared earlier, are reported separately. The test is against the producer's own target variable. Errors shared by the survey and the model cannot be detected this way; BBS is the only independent source and is used for the national check. |
| "Textbook result." | Agreed and cited; the contribution is the external measurement on an operational dataset, the metrics that hide it, and the consequences. |
| "Registered after seeing data." | Confirmatory H1–H2 use only post-registration releases; everything else labelled. |
| "One country." | D6 bounded replication. |
| "Criticism without a fix." | D5, with honest novelty framing and support limits. |

## M6. Deferred to a follow-up paper

Path simulation and probabilistic alerts; information-augmented imputer (rainfall, flood extent, daily TCB/DAM, exchange rate); August 2024 flood DiD; forecast-interval coverage; companion website (Next.js on Vercel) and the monthly auto-updating prospective test.
