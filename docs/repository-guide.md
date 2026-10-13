# Repository guide: what was built, why, and what it found

This guide walks through the repository in the order the work was done. Every number below comes from the generated
results notes in `outputs/results_notes/` (data cutoff: RTFP release 2026-10-05). If a note and this guide ever
disagree, the note is right: it is regenerated from the tables; this guide is hand-written.

---

## 1. The question in one paragraph

The World Bank's **Real-Time Food Prices (RTFP)** dataset fills gaps in WFP's market price surveys with a machine-learning
model. For Bangladesh it publishes a price for every one of 110 markets × 4 items (rice, wheat flour, palm oil,
lentils) × every month since 2007, even where nobody surveyed a price. The producers validate it themselves. We ask:
**when RTFP imputes a price, how much *local* information does it carry** — that is, does it capture what happened in
*that* market that month, beyond the national trend? We answer it independently, using the dataset's own release
history, and we turn the answer into practical guidance (what the data can and cannot be used for) and calibrated
uncertainty intervals.

## 2. The key idea: a natural hold-out from 83 archived releases

RTFP is re-published roughly weekly; the World Bank Microdata Library keeps old releases. We downloaded **83 Bangladesh
releases (2024-01-25 to 2026-10-05)**. In an early release a cell may be *imputed* (no survey value yet); in a later
release the real survey value for that same cell appears. Comparing the earlier imputed value with the later survey
value is an **out-of-sample test the producer did not design** — no masking, no re-running their model.

- 18,184 cells were imputed in one release and observed in a later one.
- Only **5,768 are clean** late arrivals. The rest are pipeline artefacts (see §3), so we classify every cell into a
  *stratum* with precedence **toggle > unconfirmed > backfill > forecast > cross_sectional**:
  - *toggle*: the survey value was already in an earlier release, then disappeared and came back (not a real late arrival).
  - *unconfirmed*: the value that appeared later is not in the latest release (so not trustworthy as truth).
  - *backfill*: arrived more than 6 months after the price month (mostly old wheat-flour history).
  - *forecast*: in the imputing release, no market had a survey value for that item-month (a real-time nowcast).
  - *cross_sectional*: some markets were observed that month, this one was not (the headline case: can the model
    infer one market from its neighbours?).
- **Truth** = the survey value in the latest release, which equals WFP's published value exactly (Table A1).
- Clean cells cover price months **2023-09 to 2026-08** only, so results describe recent imputations.

Code: `src/rtfp/vintages.py` (`build_stack`, `build_holdout`, `build_change_pairs`); counts in `A2.md`.

## 3. Data facts we discovered (and had to design around)

| Fact | Consequence |
|---|---|
| `geo_id` market keys are re-numbered between releases (2024-03-18, 2025-03-31, 2026-01-13) | Markets are matched by **name** (`src/rtfp/crosswalk.py`), with a check that a name never moves > 3 km; it never does in Bangladesh |
| A "Market Average" row appears in every release | Excluded as an aggregate (110 real markets) |
| The pipeline alternates "full" and "reduced" releases; 7 releases carry different survey values | The *toggle* and *unconfirmed* strata; truth taken from the latest release |
| Rice is missing from 42 of 83 releases | Fewer rice pairs |
| `data_coverage` is one number per release, not per market | Not usable as a cell covariate |
| Plain item columns (`rice`, …) = WFP survey values exactly (17,304 of 17,304 cells); `c_<item>` = published estimate | Lets us separate survey from imputation (`A1.md`) |
| A gross entry error: Patuakhali Sadar 2025-03 (lentils 17.3 vs neighbours ~100) | Noise estimate reported with and without it |
| Six "… Division" series are observed only 2007–2020-07 | Excluded from noise estimation; cannot be truth |
| About half of survey values carry WFP's `priceflag = aggregate` | Survey values are themselves noisy |

## 4. The central difficulty: survey noise

The "truth" is a survey price, and survey prices are noisy (different vendor, grade, rounding, stale quotes). If the
imputed change correlates weakly with the survey change, that could be because the imputation misses local
information **or** because the survey change is mostly noise. So we estimate the noise:

- **Model:** observed local level = true local level (AR(1)) + white measurement noise ν.
- **Estimators** from autocovariances of monthly changes (`src/rtfp/noise.py`): *naive* Var(ν) = −γ(1), and an
  AR(1)-aware version. **Reliability** rel = share of the observed change variance that is real.
- **Disattenuated correlation** ρ* = corr / √rel: the correlation we would see against a noise-free truth.
- Reliability of survey monthly changes is low: 0.27–0.51 (naive, untrimmed) and 0.67–0.79 (1% most extreme changes
  trimmed). A two-source check in Dhaka against TCB daily prices (`A5.md`) gives 0.23–0.54, but its intervals are too
  wide to choose between those two.
- The simulation (§6, A4) showed the naive estimator overstates noise only if noise is white; with stale quotes both
  estimators understate it. So the naive estimator is not a guaranteed safe bound — stated in the notes.

## 5. The gate (G1): decide the paper's framing before looking further

Rule written down before any gate code: **pivot** to a different paper if ≥ 2 of 4 items have ρ* ≥ 0.5.

- First implementation ("literal_v1") → **PIVOT (2 of 4)**.
- An independent audit found it applied a reliability estimated on one population (non-zero changes) to another (all
  pairs). Corrected, with populations matched → **PROCEED (1 of 4)**: lentils 0.61, wheat flour 0.50, oil 0.42, rice
  0.15; bootstrap intervals are very wide; 35% of bootstrap draws say PIVOT.
- Both results are reported, with the order of events, in `T1.md` and `docs/analysis_plan.md`. The verdict is fragile,
  so the paper never says "uninformative"; it reports measured shares with intervals.

## 6. Results, table by table

Each script in `analysis/` writes one table to `outputs/tables/`; `analysis/notes.py` (and a few scripts themselves)
write a results note to `outputs/results_notes/` with claim, numbers, method, caveats. Notes are generated, never
hand-edited, so numbers cannot drift from the tables.

| Table / note | Question | Main result |
|---|---|---|
| **A1** `a1_observed_flag_audit.py` | Are plain columns the WFP survey values? | Yes, 100% match (gate passed) |
| **A2/A3** `a2_holdout_counts.py` | How big is the natural hold-out? | 5,768 clean cells; 74–117 both-imputed cross-sectional pairs per item; never-surveyed cells are mostly 2007–2019, so not evaluable |
| **T1** `t1_noise_gate.py` | Survey noise and Gate G1 | Reliability 0.27–0.51; PROCEED 1 of 4 (fragile; see §5) |
| **T2** `t2_levels.py` | Are imputed price *levels* accurate? | On average yes: mean error 2.9–6.5%, bias −1.5% to +1.6%. But cross-market calibration slope 0.57–1.11: imputed differences between markets are compressed vs the survey, yet not enough given how weakly they track it (7 of 8 cells) |
| **T3** `t3_local_changes.py` | Do imputed *local monthly changes* track real ones? | Weakly: corr 0.10–0.44 (own-release mean removed), −0.13–0.10 (within month); every 95% CI includes 0. Of 138 persistent survey spikes > 10%, the imputed series showed a ≥ 5% same-direction move for 29 (21%) |
| **T4** `t4_error_covariates.py` | What predicts imputation error? | Months since the market last reported (more months, more error); RTFP's trust score only weakly (86% of clean cells have trust 10) |
| **T5** `t5_switchers.py` | Within the same market, are imputed spells smoother than surveyed spells? | Yes: imputed local changes are 31–38% smaller (PPML, all CIs below 0). Whether that is more than noise removal depends on the noise estimate: significant in 0 of 4 items vs the untrimmed benchmark, 4 of 4 vs the trimmed one |
| **A4** `a4_simulation.py` | Framework check by simulation | A perfectly calibrated imputer is ruled out mainly by the variance ratio λ (6 of 8 cells), not by the calibration slope |
| **T6** `t6_alerts.py` | Do imputed prices raise the same early-warning alerts (WFP ALPS, FAO IPA)? | IPA recall 45–58%, not distinguishable from a simulated value with no local information, below one that keeps local signal; indicative only (power limited) |
| **T7** `t7_uncertainty.py` | Can we give honest uncertainty intervals? | The published high/low band covers the later survey value in only 59% of cases. Our calibrated interval covers 80.7% / 91.5% (80% / 90% nominal) on later releases; it is one fixed band (−5.4% / +6.4% at 80%) for every cell inside the hold-out's support; covariates did not help |
| **T7n** `f7_bbs_national.py` | Does RTFP's national signal match official food inflation (BBS)? | No positive association, but the survey-only index doesn't match either, so the cause is not imputation (basket difference plausible, untested). Published national index tracks the survey-only one (corr 0.81–0.87, partly mechanical) |
| **T8** `t8_safe_use.py` | What can users safely do with RTFP? | Supported: national movements, average price level (point estimates). Not supported: cross-market differences, the published band as uncertainty, market-specific monthly changes, local spike detection, local alerts; regressor use not shown valid; never-surveyed history not evaluable |
| **T9** `t9_replication.py` | Does it replicate elsewhere? | 5 countries chosen by a fixed rule (Myanmar, Kenya, Indonesia, DR Congo, Mali; 42 country-items): results vary; cross-sectional correlation median 0.21, CI excludes 0 for only 5 of 32 items |
| **Figures** `figures.py` | Six publication figures | Each caption generated and cross-checked against the tables (`FIGURES.md`) |

The interval file `outputs/tables/rtfp_bgd_calibrated_intervals_2026-10-05.csv` is a usable deliverable: an 80%/90%
interval for every imputed cell inside the hold-out's support (1,280 cells), flagged otherwise.

## 7. How the code is organised

```
src/rtfp/            reusable library (unit-tested)
  download.py        RTFP releases + WFP/HDX file, sha256 manifest, data cutoff (RTFP_AS_OF)
  crosswalk.py       stable market ids across re-keyed releases
  panel.py           latest-release panel: survey vs published, monthly log changes, local components
  vintages.py        stack of all releases, hold-out strata, change pairs
  audit.py           RTFP vs WFP/HDX cell match
  noise.py           autocovariance noise estimators
  metrics.py         level and local-change metrics, two-way pigeonhole bootstrap, FE OLS with two-way clusters
  ppml.py            Poisson pseudo-ML with fixed effects (T5)
  uncertainty.py     conformal quantiles, Mondrian groups, adaptive conformal (ACI)
  simulate.py        calibrated simulation of panels and imputers (A4, T6 benchmarks)
  alerts.py          ALPS and IPA indicators, formulas quoted from the source documents
  bbs.py, tcb.py     loaders for BBS CPI releases and TCB daily bulletins
  replicate.py       per-country driver for the replication
analysis/            one script per table/figure; notes.py writes results notes
tests/               270 tests: rules on toy data + invariants re-checked on the real data
outputs/tables/      every derived table (CC BY 4.0)
outputs/results_notes/  generated notes (the hand-over to the co-author)
outputs/figures/     figures + FIGURES.md captions
docs/                methodology.md (design spec), analysis_plan.md (draft OSF registration),
                     repository-guide.md (this guide)
data/raw/            downloaded sources (RTFP + HDX files not in git; MANIFEST.csv and LICENSES.md are)
data/external/       BBS and TCB downloads (only their manifests in git)
data/third_party/bbs/  BBS food CPI series, published under BBS's terms (not CC BY / MIT)
```

**Run order** (`Makefile`): `make data` → `make external` → `make all` (panel → stack → A1, A2, T1, T2, T3, T4, T7,
T7n, A5, T5, A4, T6, notes, T8, figures; about 10 minutes single-threaded) → `make replication` (separately; ~2.3 GB
download) → `make test`. `RTFP_AS_OF=2026-10-05` pins the data so outputs reproduce exactly after new releases appear.

## 8. How we made sure the claims are true

1. **Tests first** for every library function (written to fail, then made to pass), plus data tests that re-check
   invariants on the real files.
2. **Independent verification** of each major step by separate reviewer agents who re-computed numbers with their own
   code; several bugs were caught this way (below).
3. **No hand-typed numbers** in notes: every number is computed from a table when the note is generated.
4. **Clean rebuild**: the whole pipeline was rebuilt from an empty output folder in a separate checkout; all tables and
   notes came out byte-identical (latest check on commit `bad8186`).
5. **Adversarial review rounds** at the end: a whole-branch review (15 serious issues) and three further verification
   rounds (28, then 9 confirmed issues), each fixed and re-checked.

## 9. Mistakes found and corrected (the honest list)

- Early quick-look scripts used the wrong market key, pooled toggling releases, took means from the wrong release and
  counted "Market Average" as a market — all superseded.
- Gate G1 first mixed populations (literal PIVOT) — corrected and both reported.
- T3 read a stale gate file; an ad-hoc check's correlation included the month column — both caught before reporting.
- T5's first note came from a 2-draw smoke test; the 1,000-draw run changed the verdict (0 of 4 significant vs the
  untrimmed benchmark).
- Alert results were first overstated ("at or below"); now base-rate-matched and labelled indicative.
- T8 verdicts were first hand-typed; now computed from stated rules.
- The laptop overheated and Python crashed: a scipy Cholesky bug on this Mac plus too many parallel jobs. Fixed with
  numpy's Cholesky, 2 workers, single-threaded BLAS.
- Replication: a Kenyan market "moved" 376 km between releases (dropped and listed); dropped markets still leaked into
  monthly means (fixed); ρ* > 1 was counted as success (fixed).
- Final-review round: no data cutoff; BBS series under the wrong licence; several notes misdescribed results
  (decomposition of interval coverage, noise-estimator direction, alerts wording, cross-market slope). I also
  introduced two wrong explanations during fixing (the cross-market slope and the ACI guarantee); the next verification
  round caught both and they are corrected.

## 10. Rulings I made on your behalf (all logged)

Market id by name; truth = latest release; the two extra strata and the 6-month backfill threshold (chosen after seeing
the data, disclosed, with 3/12 sensitivity); noise estimated on the markets and period that supply truth; scikit-learn
gradient boosting instead of LightGBM; model choice on a separate selection block; online recalibration added to the
intervals; "drop and report" for location-jumping markets in the replication only; BBS series published under BBS terms
(your choice B).

## 11. Licensing in one line each

Code MIT; derived tables CC BY 4.0; RTFP data CC BY 4.0 + World Bank terms (cite each release, DOI
10.48529/2ZH0-JF55); WFP/HDX CC BY-IGO 3.0; BBS: no licence, used with acknowledgement and no commercial reproduction
(only the extracted series in `data/third_party/bbs/`); TCB: no licence, scripts only. Details: `data/raw/LICENSES.md`.

## 12. What is not done yet

- **OSF registration** of `docs/analysis_plan.md` (makes H1/H2 confirmatory on releases published afterwards).
- **GitHub push and Zenodo DOI** (CITATION.cff still has placeholders).
- Deferred minor items: a sensitivity check without known gross survey errors in T2–T4/T6; MANIFEST download times
  and hash re-verification; a full audit of the replication's dropped markets; licence check of replication sources.
- The paper text itself (your co-author), using `outputs/results_notes/` and `outputs/figures/`.

## 13. Suggested way to learn it yourself

1. Read `docs/methodology.md` M0–M1 (contributions, data facts, hold-out), then this guide's §2–§5.
2. Open `src/rtfp/vintages.py` and `tests/test_vintages.py` side by side: the hold-out logic on toy data.
3. Read the notes in this order: A1 → A2 → T1 → T2 → T3 → T5 → T7 → T8 → T9.
4. Re-run one step yourself, e.g. `RTFP_AS_OF=2026-10-05 .venv/bin/python -I analysis/t2_levels.py`, and compare with
   `outputs/tables/T2_levels.csv`.
