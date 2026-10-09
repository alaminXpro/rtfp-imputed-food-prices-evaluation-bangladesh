# Registered Analysis Plan (draft for OSF)

**Project:** An independent, release-history evaluation of the World Bank's ML-imputed Real-Time Food Prices (RTFP), Bangladesh.
**Status:** DRAFT, 2026-10-09. Not yet posted. The registration date is the date this file is posted on OSF; record it here when posted: `REGISTERED: ____-__-__ (OSF link: ____)`.
**Spec:** `docs/methodology.md` (revision 3 with the 2026-10-09 clarifications). Where this file and the spec differ, this file governs the confirmatory analysis.

## 1. Disclosure of work done before registration (exploratory)

Everything below was computed on RTFP releases up to **2026-10-05** and is exploratory. Each result is labelled as such wherever it appears.
- Quick looks `analysis/01_*`, `analysis/02_*` (superseded; known defects listed in the plan §1).
- Raw-column audit (Table A1): plain item columns equal the WFP/HDX retail values exactly, one-to-one, in the latest release.
- Hold-out construction and counts (Tables A2–A3); stratum definitions were revised after inspecting the data (toggle > unconfirmed > backfill > forecast > cross-sectional; backfill threshold of 6 months chosen after seeing the arrival-lag distribution).
- Measurement-noise band and Gate G1 (Table 1, T1b–T1d). The gate rule was committed before the gate script existed (git 86970eb, 11:59); its first implementation mixed populations; the D2-faithful implementation gives **PROCEED (1 of 4 items)**; the literal first implementation gives PIVOT (2 of 4); both are reported.
- Level accuracy (Table 2), local-change agreement and spikes (Table 3), error predictors (Table 4), first uncertainty model (Table 7).

## 2. Confirmatory sample (prospective)

- **Cells:** hold-out cells (methodology M1) whose `v_truth` (first release in which the survey value appears after imputation) is **published after the registration date**, in the clean strata (forecast, cross-sectional), with the survey value present in the latest release used for the confirmatory run.
- **Imputed values judged:** `c_final` from the last release before `v_truth`. Releases up to and including the registration date may serve as `v_final`.
- **Minimum before testing:** at least 6 calendar months of post-registration releases **and** at least 150 confirmatory cells per item for H2 and at least 60 both-imputed change pairs per item for H1. If not reached by the time of submission, report the registered tests as pending with the current N; do not test early.
- **Expected N (from the 2024–2026 release flow):** roughly 150–300 cells and 50–100 change pairs per item per 6 months, nearly all in the forecast stratum. The cross-sectional stratum may not accrue enough pairs; if it does not, H1 is tested on the forecast stratum and labelled so.

## 3. Hypotheses

**Prospective confirmatory (tested only on the §2 sample):**
- **H1 (local information).** For each item, the noise-corrected local correlation ρ* between imputed and observed local changes is below 0.5. Pairs: both-imputed. Local components: own-release μ (methodology M1). Noise correction: reliability of the confirmatory pairs' own population, estimated on the same release's non-Division markets ≥ 2020-08 (variant A, naive estimator) for all pairs, and the staleness-aware variant B for pairs with non-zero truth change; the larger credible ρ* is tested. One-sided test at 5% using the two-way (market × month) bootstrap; Holm correction across the four items.
- **H2 (levels).** For each item, the month-demeaned level Mincer–Zarnowitz slope of log survey value on log imputed value lies within [0.9, 1.1], and the mean log error lies within ±1%. Two one-sided tests (TOST) at 5% with two-way clustered SEs; Holm across items.

**Registered before analysis, not prospective (existing data; reported as such):**
- **H3 (within-market).** In markets with both observed and imputed spells since 2020, the PPML coefficient on `imputed` for |local change| is below log √rel (the value expected if imputation only removed noise).
- **H4 (alerts).** WFP ALPS and FAO IPA alert recall from imputed values, against alerts from survey values on hold-out cells, is below the recall a calibrated noise-removing imputer achieves in the Task 3.1 simulation. Confirmatory only if a power calculation on expected alert counts gives ≥ 80% power; otherwise descriptive.

**Descriptive:** RTFP national food index vs BBS food CPI; published-band coverage; error predictors; calibrated uncertainty intervals.

## 4. Exclusions and robustness (fixed now)

- Exclude: 'Market Average' row; toggle, unconfirmed and backfill strata from the evaluation sample; pairs whose own-release μ is undefined are kept with the published-mean fallback and reported with and without.
- Robustness (reported for every confirmatory number): within-month demeaning; trimmed (1%) noise estimate; excluding cells flagged `priceflag = aggregate`; c_-based observed side; backfill threshold 3 and 12 months.

## 5. Bounded replication (methodology D6)

Countries are selected by this rule, applied to the World Bank Microdata catalogue **before any of their results are computed**: RTFP market-level datasets other than Bangladesh with **≥ 50 archived market-file vintages, ≥ 30 markets in the latest vintage, and plain survey columns present**; take the five with the most markets (ties broken by number of vintages). Run the Bangladesh pipeline (crosswalk, hold-out, noise, Tables 1–3 core metrics) unchanged, except that the raw-column audit uses the country's WFP/HDX feed where available.

## 6. What would change the conclusions

- If H1 is rejected for at least two items (ρ* ≥ 0.5), the paper reports that imputed cells track a substantial share of local variation and drops the "limited local information" framing.
- If H2 fails, the level-accuracy statement is limited to what the data show (point estimates with CIs).
