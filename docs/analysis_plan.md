# Registered Analysis Plan (draft for OSF)

**Project:** An independent, release-history evaluation of the World Bank's ML-imputed Real-Time Food Prices (RTFP), Bangladesh.
**Status:** DRAFT, 2026-10-09. Not yet posted. The registration date is the date this file is posted on OSF; record it here when posted: `REGISTERED: ____-__-__ (OSF link: ____)`.
**Spec:** `docs/methodology.md` (revision 3 with the 2026-10-09 clarifications). Where this file and the spec differ, this file governs the confirmatory analysis.

## 1. Disclosure of work done before registration (exploratory)

Everything below was computed on RTFP releases up to **2026-10-05** and is exploratory. Each result is labelled as such wherever it appears.
- Quick looks `analysis/01_*`, `analysis/02_*` (superseded; known defects listed in the plan §1).
- Raw-column audit (Table A1): plain item columns equal the WFP/HDX retail values exactly, one-to-one, in the latest release.
- Hold-out construction and counts (Tables A2–A3); stratum definitions were revised after inspecting the data (toggle > unconfirmed > backfill > forecast > cross-sectional; backfill threshold of 6 months chosen after seeing the arrival-lag distribution).
- Measurement-noise band and Gate G1 (Table 1, T1b–T1d). Order of events: (1) the gate rule was committed before any gate code existed (git 86970eb, 2026-10-09 11:59); (2) the first implementation ("literal_v1") gave **PIVOT (2 of 4 items)**; (3) an audit found that it compared reliabilities and correlations from different pair populations, contrary to D2; (4) the D2-faithful implementation (populations matched) gives **PROCEED (1 of 4)**. Both results are reported (T1d); the change was made after seeing the first result, so the gate outcome is exploratory.
- Level accuracy (Table 2), local-change agreement and spikes (Table 3), error predictors (Table 4), first uncertainty model (Table 7).
- Also already computed on the same data, therefore **exploratory, not confirmatory**, despite being listed in §3: the within-market switcher analysis that H3 describes (Table 5, bootstrap B = 1000); the alert comparison and power calculation that H4 describes (Table 6, simulation benchmarks in Table A4); the national BBS comparison (T7n); the safe-use table (Table 8); the TCB two-source noise check (Table A5); and the five-country replication (Table 9). Their numbers may be cited only as exploratory.

## 2. Confirmatory sample (prospective)

- **Cells:** hold-out cells (methodology M1) whose `v_truth` (first release in which the survey value appears after imputation) is **published after the registration date**, in the clean strata (forecast, cross-sectional), with the survey value present in the latest release used for the confirmatory run.
- **Imputed values judged:** `c_final` from the last release before `v_truth`. Releases up to and including the registration date may serve as `v_final`.
- **Minimum before testing:** at least 6 calendar months of post-registration releases **and** at least 150 confirmatory cells per item for H2 and at least 60 both-imputed change pairs per item for H1. If not reached by the time of submission, report the registered tests as pending with the current N; do not test early.
- **Expected N (from the 2024–2026 release flow):** roughly 150–300 cells and 50–100 change pairs per item per 6 months, nearly all in the forecast stratum. H1 is defined on the cross-sectional stratum; if that stratum does not reach the minimum, H1 is reported as pending with its N and is **not** replaced by a forecast-stratum test. The forecast stratum is tested separately as H1f (below).

## 3. Hypotheses

**Prospective confirmatory (tested only on the §2 sample):**
- **H1 (local information).** For each item, the noise-corrected local correlation ρ* between imputed and observed local changes is below 0.5. Pairs: both-imputed, cross-sectional stratum. Local components: own-release μ (methodology M1). Noise correction: reliability of the confirmatory pairs' own population, estimated on the same release's non-Division markets ≥ 2020-08 (variant A, naive estimator) for all pairs, and the staleness-aware variant B for pairs with non-zero truth change; the larger credible ρ* is tested. One-sided test at 5% using the two-way (market × month) bootstrap; Holm correction across the four items.
- **H1f (nowcast local information; separate hypothesis).** The same test as H1 on forecast-stratum pairs (cells imputed beyond the market's last survey month). It answers a different question (how much local information a nowcast carries), is reported under its own label, and does not substitute for H1.
- **H2 (levels).** For each item, the month-demeaned level Mincer–Zarnowitz slope of log survey value on log imputed value lies within [0.9, 1.1], and the mean log error lies within ±1%. Two one-sided tests (TOST) at 5% with two-way clustered SEs; Holm across items.

**Defined on existing data, not prospective; already computed before registration (see §1), so reported as exploratory:**
- **H3 (within-market).** In markets with both observed and imputed spells since 2020, the PPML coefficient on `imputed` for |local change| is below log √rel (the value expected if imputation only removed noise).
- **H4 (alerts).** WFP ALPS and FAO IPA alert recall from imputed values, against alerts from survey values on hold-out cells, is below the recall a calibrated noise-removing imputer achieves in the Task 3.1 simulation. Confirmatory only if a power calculation on expected alert counts gives ≥ 80% power; otherwise descriptive.

**Descriptive:** RTFP national food index vs BBS food CPI; published-band coverage; error predictors; calibrated uncertainty intervals.

## 4. Exclusions and robustness (fixed now)

- Exclude: 'Market Average' row; toggle, unconfirmed and backfill strata from the evaluation sample; pairs whose own-release μ is undefined are kept with the published-mean fallback and reported with and without.
- Robustness (reported for every confirmatory number): within-month demeaning; trimmed (1%) noise estimate; excluding cells flagged `priceflag = aggregate`; c_-based observed side; backfill threshold 3 and 12 months.

## 5. Bounded replication (methodology D6)

Countries were selected by this rule, applied to the World Bank Microdata catalogue (snapshot in `data/raw/rtfp_other/_catalog/`) **before any of their results were computed** (Table T9_country_selection); the replication has since been run (Table 9) and is exploratory: RTFP market-level datasets other than Bangladesh with **≥ 50 archived market-file vintages, ≥ 30 markets in the latest vintage, and plain survey columns present**; take the five with the most markets (ties broken by number of vintages). Run the Bangladesh pipeline (crosswalk, hold-out, noise, Tables 1–3 core metrics) unchanged, except that the raw-column audit uses the country's WFP/HDX feed where available.

## 6. What would change the conclusions

- If H1 is rejected for at least two items (ρ* ≥ 0.5), the paper reports that imputed cells track a substantial share of local variation and drops the "limited local information" framing.
- If H2 fails, the level-accuracy statement is limited to what the data show (point estimates with CIs).
