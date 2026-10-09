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
    rows.append({"use": "Price level of a market-month (point estimate)",
                 "evidence": f"MAPE {rng(c2.mape, 1)}%, median {rng(c2.median_ape, 1)}%, bias {rng(c2.bias_logpct, 1)}%, cells off >10%: {rng(100 * c2.share_err_gt10, 0)}% (T2)",
                 "rule": "MAPE <= 10% and |bias| <= 2% in every item x stratum",
                 "verdict": verdict((c2.mape <= 10).all() and (c2.bias_logpct.abs() <= 2).all(),
                                    "supported, with stated uncertainty (use calibrated intervals)"),
                 "source": "T2_levels.csv"})
    rows.append({"use": "Published high/low band as an uncertainty range",
                 "evidence": f"later survey value inside the band in {100 * band.coverage:.0f}% of test cells (T7)",
                 "rule": "coverage >= 80%", "verdict": verdict(band.coverage >= 0.8), "source": "T7_uncertainty.csv"})
    rows.append({"use": "Market-specific month-to-month changes",
                 "evidence": f"corr {rng(h3[h3['mode'] == 'own_mu']['corr'])} (own-release mean removed), {rng(h3[h3['mode'] == 'within_month']['corr'])} (within month); most 95% CIs include 0 (T3)",
                 "rule": "corr >= 0.5 with CI above 0 for >= 3 of 4 items (own-release mean)",
                 "verdict": verdict(((h3[h3["mode"] == "own_mu"]["corr"] >= 0.5) & (h3[h3["mode"] == "own_mu"].corr_lo > 0)).sum() >= 3),
                 "source": "T3_local_changes.csv"})
    rows.append({"use": "Detecting local price spikes (>10%, persistent)",
                 "evidence": f"{int(sp.caught)} of {int(sp.spikes)} survey spikes ({100 * sp.share_caught:.0f}%) shown by a same-direction published move >= 5% (T3b)",
                 "rule": "share caught >= 50%", "verdict": verdict(sp.share_caught >= 0.5), "source": "T3b_spikes.csv"})
    bm = h3[h3["mode"] == "own_mu"].beta_mz
    rows.append({"use": "Imputed local changes as a regressor",
                 "evidence": f"calibration slope (beta_MZ) {rng(bm)}: coefficients are multiplied by roughly this factor (T3)",
                 "rule": "beta_MZ within 0.8-1.2 for >= 3 of 4 items",
                 "verdict": verdict(bm.between(0.8, 1.2).sum() >= 3, no="not supported without correction"),
                 "source": "T3_local_changes.csv"})
    rows.append({"use": "Trust score as a guide to cell reliability",
                 "evidence": f"univariate {tr.coef:.2f} pp error per trust point (p = {tr.p:.3f}); almost all clean cells have trust 10 (T4)",
                 "rule": "p < 0.05 and >= 20% of clean cells below trust 10",
                 "verdict": verdict(tr.p < 0.05 and share_low_trust >= 0.2, no="limited"),
                 "source": "T4_error_covariates.csv"})
    rows.append({"use": "Imputed history for never-surveyed markets (mostly 2007–2019)",
                 "evidence": f"never-observed cells: median year {int(never.year_median.median())}, share >= 2020 {rng(never.share_year_ge_2020)}; hold-out covers 2023-09..2026-08 only (A3)",
                 "rule": "median price year inside the hold-out window (2023-09..2026-08)",
                 "verdict": verdict(never.year_median.median() >= 2023, yes="evaluable", no="not evaluable with this design"),
                 "source": "T_A3_covariate_balance.csv"})
    al = sorted(T.glob("T6_alerts*.csv"))
    if al:
        a = pd.read_csv(al[0])
        rows.append({"use": "Local early-warning alerts (ALPS/IPA)", "evidence": f"see {al[0].name}", "rule": "see T6 note",
                     "verdict": "see T6", "source": al[0].name})
    else:
        rows.append({"use": "Local early-warning alerts (ALPS/IPA)", "evidence": "", "rule": "", "verdict": "pending",
                     "source": "T6 (not yet produced)"})
    t = pd.DataFrame(rows)
    t.to_csv(T / "T8_safe_use.csv", index=False)
    lines = ["| Use | Evidence | Rule | Verdict |", "|---|---|---|---|"] + \
            [f"| {r.use} | {r.evidence} | {r.rule} | **{r.verdict}** |" for r in t.itertuples()]
    (N / "T8.md").write_text("# T8 — What the imputed prices can and cannot support (Task 4.4)\n\n"
                             "*Generated by `analysis/t8_safe_use.py` from the result tables; each verdict follows the stated rule.*\n\n"
                             + "\n".join(lines) + "\n\n**Caveat.** All rows are exploratory (pre-registration) and describe RTFP "
                             "Bangladesh releases 2024-01-25 to 2026-10-05, evaluated on cells surveyed after imputation (price months 2023-09 to 2026-08). "
                             "The rules are judgement thresholds chosen for readability, stated so readers can apply their own.\n")
    print(t[["use", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
