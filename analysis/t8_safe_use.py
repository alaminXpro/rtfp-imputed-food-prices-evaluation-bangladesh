"""Table 8: what RTFP Bangladesh imputed prices can and cannot support (Task 4.4).

Every row is assembled from a result table; the verdict rule is stated in the 'rule' column so the
co-author can see why each verdict follows from the numbers. Rows whose source table is missing are marked
'pending'.
"""
from pathlib import Path

import numpy as np
import pandas as pd

T = Path("outputs/tables")
N = Path("outputs/results_notes")


def rng(s, d=2):
    return f"{np.nanmin(s):.{d}f}–{np.nanmax(s):.{d}f}"


def main():
    rows = []
    from rtfp.download import as_of_date
    tm = pd.read_csv(T / "T_A2c_stratum_timing.csv").set_index("stratum").loc[["forecast", "cross_sectional"]]
    pm_lo, pm_hi = tm.price_month_min.min()[:7], tm.price_month_max.max()[:7]
    t2 = pd.read_csv(T / "T2_levels.csv")
    c2 = t2[(t2.estimate == "c_final") & t2.stratum.isin(["cross_sectional", "forecast"])]
    nat = pd.read_csv(T / "T7n_published_vs_survey_national.csv")
    t3 = pd.read_csv(T / "T3_local_changes.csv")
    h3 = t3[(t3.stratum == "cross_sectional") & (t3.pair_type == "both_imputed")]
    sp = pd.read_csv(T / "T3b_spikes.csv").set_index("item").loc["ALL"]
    t7 = pd.read_csv(T / "T7_uncertainty.csv")
    band = t7[t7.method == "published_band"].iloc[0]
    t4 = pd.read_csv(T / "T4_error_covariates.csv")
    tr = t4[(t4["sample"] == "pooled:univariate") & (t4.regressor == "trust_final")].iloc[0]
    a3 = pd.read_csv(T / "T_A3_covariate_balance.csv")
    never = a3[a3.group.str.contains("never observed")]
    tb = pd.read_csv(T / "T4b_error_by_trust.csv")
    share_low_trust = 1 - tb.n.iloc[-1] / tb.n.sum()  # last bin = trust exactly 10

    def verdict(ok, yes="supported", no="not supported"):
        return yes if bool(ok) else no

    rows.append({"use": "National monthly price movement (published index)",
                 "evidence": f"monthly corr with survey-only national index {rng(nat.corr_pub_vs_obs)}; slope {rng(nat.slope_obs_on_pub)} (T7n)",
                 "rule": "corr >= 0.8 and slope within 0.8-1.2 in every period",
                 "verdict": verdict((nat.corr_pub_vs_obs >= 0.8).all() and nat.slope_obs_on_pub.between(0.8, 1.2).all(),
                                    "supported (partly mechanical: observed cells enter both)"),
                 "source": "T7n_published_vs_survey_national.csv"})
    n_bias_ci = int(((c2.bias_lo < -2) | (c2.bias_hi > 2)).sum())
    ci_fail = (["|bias| > 2%"] if n_bias_ci else []) + (["MAPE > 10%"] if (c2.mape_hi > 10).any() else [])
    h2_bias = (c2.bias_logpct.abs() < 1)
    h2_slope = c2.mz_slope_month_demeaned.between(0.9, 1.1)
    rows.append({"use": "Average price level of a market-month (point estimate)",
                 "evidence": f"MAPE {rng(c2.mape, 1)}%, median {rng(c2.median_ape, 1)}%, bias {rng(c2.bias_logpct, 1)}%, cells off >10%: {rng(100 * c2.share_err_gt10, 0)}%; bias 95% CI extends past ±2% in {n_bias_ci} of {len(c2)} cells; MAPE CI upper bound up to {c2.mape_hi.max():.1f}% (T2)",
                 "rule": "MAPE <= 10% and |bias| <= 2% in every item x clean stratum (forecast, cross-sectional; toggle, unconfirmed and backfill excluded as in the analysis plan), point estimates",
                 "verdict": verdict((c2.mape <= 10).all() and (c2.bias_logpct.abs() <= 2).all(),
                                    "supported for the average level on point estimates" + (f" (CIs do not rule out {' or '.join(ci_fail)})" if ci_fail else "")),
                 "source": "T2_levels.csv"})
    rows.append({"use": "Cross-market price differences (draft-plan H2 criteria)",
                 "evidence": f"|bias| < 1% in {int(h2_bias.sum())} of {len(c2)} item x stratum cells; month-demeaned calibration slope in [0.9, 1.1] in {int(h2_slope.sum())} of {len(c2)} (range {rng(c2.mz_slope_month_demeaned)}; below 1 = a given imputed difference between markets corresponds on average to a smaller surveyed difference) (T2)",
                 "rule": "H2: |bias| < 1% and slope in [0.9, 1.1] in every item x stratum",
                 "verdict": verdict((h2_bias & h2_slope).all(), no=f"not supported (slope below 0.9 in {int((c2.mz_slope_month_demeaned < 0.9).sum())} of {len(c2)}: imputed cross-market differences only partly borne out by the survey)"),
                 "source": "T2_levels.csv"})
    rows.append({"use": "Published high/low band as an uncertainty range",
                 "evidence": f"later survey value inside the band in {100 * band.coverage:.0f}% of test cells (T7)",
                 "rule": "coverage >= 80%", "verdict": verdict(band.coverage >= 0.8), "source": "T7_uncertainty.csv"})
    rows.append({"use": "Market-specific month-to-month changes",
                 "evidence": f"corr {rng(h3[h3['mode'] == 'own_mu']['corr'])} (own-release mean removed), {rng(h3[h3['mode'] == 'within_month']['corr'])} (within month); 95% CI includes 0 in {int(((h3.corr_lo <= 0) & (h3.corr_hi >= 0)).sum())} of {len(h3)} item x definition rows (T3)",
                 "rule": "corr >= 0.5 with CI above 0 for >= 3 of 4 items (own-release mean)",
                 "verdict": verdict(((h3[h3["mode"] == "own_mu"]["corr"] >= 0.5) & (h3[h3["mode"] == "own_mu"].corr_lo > 0)).sum() >= 3),
                 "source": "T3_local_changes.csv"})
    rows.append({"use": "Detecting local price spikes (>10%, persistent)",
                 "evidence": f"{int(sp.caught)} of {int(sp.spikes)} survey spikes ({100 * sp.share_caught:.0f}%) shown by a same-direction published move >= 5% (T3b)",
                 "rule": "share caught >= 50%", "verdict": verdict(sp.share_caught >= 0.5), "source": "T3b_spikes.csv"})
    hb = h3[h3["mode"] == "own_mu"]
    bm = hb.beta_mz
    rows.append({"use": "Imputed local changes as a regressor",
                 "evidence": f"calibration slope (beta_MZ) {rng(bm)}; 95% CIs {', '.join(f'{r.item} [{r.beta_mz_lo:.2f}, {r.beta_mz_hi:.2f}]' for r in hb.itertuples())}: include 1 for {int(((hb.beta_mz_lo <= 1) & (hb.beta_mz_hi >= 1)).sum())} of 4 items and 0 for {int(((hb.beta_mz_lo <= 0) & (hb.beta_mz_hi >= 0)).sum())} of 4, so the attenuation factor is not pinned down (T3)",
                 "rule": "beta_MZ within 0.8-1.2 for >= 3 of 4 items",
                 "verdict": verdict(bm.between(0.8, 1.2).sum() >= 3, no="not shown to be valid (no reliable correction factor)"),
                 "source": "T3_local_changes.csv"})
    rows.append({"use": "Trust score as a guide to cell reliability",
                 "evidence": f"univariate {tr.coef:.2f} pp error per trust point (p = {tr.p:.3f}); {100 * (1 - share_low_trust):.0f}% of clean cells have trust 10 (T4)",
                 "rule": "p < 0.05 and >= 20% of clean cells below trust 10",
                 "verdict": verdict(tr.p < 0.05 and share_low_trust >= 0.2, no="limited"),
                 "source": "T4_error_covariates.csv"})
    rows.append({"use": "Imputed history for never-surveyed markets (mostly 2007–2019)",
                 "evidence": f"never-observed cells: median year {int(never.year_median.median())}, share >= 2020 {rng(never.share_year_ge_2020)}; hold-out covers {pm_lo}..{pm_hi} only (A3)",
                 "rule": f"median price year inside the hold-out window ({pm_lo}..{pm_hi})",
                 "verdict": verdict(never.year_median.median() >= int(pm_lo[:4]), yes="evaluable", no="not evaluable with this design"),
                 "source": "T_A3_covariate_balance.csv"})
    if (T / "T6_alerts.csv").exists():
        e6 = pd.read_csv(T / "T6_alerts.csv")
        z6 = pd.read_csv(T / "T6_alerts_zscores.csv")
        alps_rate = (z6[z6.indicator == "alps"].z_survey >= 1).mean()
        b6 = pd.read_csv(T / "T6_alerts_benchmark_matched.csv")
        ipa = e6[(e6["item"] == "ALL") & e6.indicator.str.startswith("ipa") & (e6.level == "alert")].set_index("indicator")
        com = b6[(b6.benchmark == "common_only") & (b6.level == "alert")].set_index("indicator").recall_matched
        orc = b6[(b6.benchmark == "oracle_noise_removed") & (b6.level == "alert")].set_index("indicator").recall_matched
        gain = (ipa.recall - com.reindex(ipa.index))
        rows.append({"use": "Early-warning alerts (FAO IPA; all alerts, D4 local-event filter not applied)",
                     "evidence": f"IPA alert recall {rng(ipa.recall)} vs {rng(com.reindex(ipa.index))} for a simulated value with no local information and {rng(orc.reindex(ipa.index))} for one that removes noise but keeps local signal, base rates matched (T6); ALPS: {100 * alps_rate:.0f}% of clean cells are already 'Alert or Crisis' on survey values (T6 Claim 2)",
                     "rule": "recall exceeds the no-local-information benchmark by >= 0.10 for both IPA variants",
                     "verdict": verdict((gain >= 0.10).all()), "source": "T6_alerts.csv, T6_alerts_benchmark_matched.csv, T6_alerts_zscores.csv"})
    else:
        rows.append({"use": "Early-warning alerts (FAO IPA)", "evidence": "", "rule": "", "verdict": "pending",
                     "source": "T6 (not yet produced)"})
    t = pd.DataFrame(rows)
    t.to_csv(T / "T8_safe_use.csv", index=False)
    lines = ["| Use | Evidence | Rule | Verdict |", "|---|---|---|---|"] + \
            [f"| {r.use} | {r.evidence} | {r.rule} | **{r.verdict}** |" for r in t.itertuples()]
    (N / "T8.md").write_text("# T8 — What the imputed prices can and cannot support (Task 4.4)\n\n"
                             "*Generated by `analysis/t8_safe_use.py` from the result tables; each verdict follows the stated rule.*\n\n"
                             + "\n".join(lines) + "\n\n**Caveat.** All rows are exploratory (pre-registration) and describe RTFP "
                             f"Bangladesh releases up to the data cutoff {as_of_date() or 'latest'}, evaluated on clean cells (forecast and cross-sectional strata) surveyed after imputation (price months {pm_lo} to {pm_hi}). "
                             "The rules are judgement thresholds chosen for readability, stated so readers can apply their own.\n")
    print(t[["use", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
