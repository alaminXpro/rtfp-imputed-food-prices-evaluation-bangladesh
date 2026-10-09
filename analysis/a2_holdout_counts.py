"""Tables A2 (hold-out counts) and A3 (covariate balance). Task 1.3.

Inputs:  data/interim/{vintages,holdout,change_pairs}.parquet, data/interim/panel_latest.parquet
Outputs: outputs/tables/T_A2_holdout_counts.csv, T_A2b_pair_counts.csv, T_A2c_stratum_timing.csv,
         T_A2d_truth_revisions.csv, T_A3_covariate_balance.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["mkt_uid", "month", "item"]
OUT = Path("outputs/tables")


def months_between(a, b):
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def main():
    h = pd.read_parquet("data/interim/holdout.parquet")
    pairs = pd.read_parquet("data/interim/change_pairs.parquet")
    latest = pd.read_parquet("data/interim/panel_latest.parquet")
    for d in (h, pairs, latest):
        d["item"] = d["item"].astype(str)
    OUT.mkdir(parents=True, exist_ok=True)

    # A2: counts by item x stratum x year of the price month
    a2 = h.assign(year=h.month.dt.year).pivot_table(index=["item", "year"], columns="stratum", values="mkt_uid",
                                                     aggfunc="size", fill_value=0)
    a2.to_csv(OUT / "T_A2_holdout_counts.csv")
    tot = h.pivot_table(index="item", columns="stratum", values="mkt_uid", aggfunc="size", fill_value=0)

    # A2b: change pairs by item x stratum x pair type
    a2b = pairs.pivot_table(index=["item", "stratum"], columns="pair_type", values="mkt_uid", aggfunc="size",
                            fill_value=0)
    a2b.to_csv(OUT / "T_A2b_pair_counts.csv")

    # A2c: timing — months from price month to the imputing vintage, and to the truth vintage
    h["lag_final_m"] = months_between(h.month, h.v_final)
    h["lag_truth_m"] = months_between(h.month, h.v_truth)
    h["wait_days"] = (h.v_truth - h.v_final).dt.days
    a2c = h.groupby(["stratum"]).agg(n=("mkt_uid", "size"),
                                     lag_final_median=("lag_final_m", "median"),
                                     lag_final_p10=("lag_final_m", lambda s: s.quantile(0.1)),
                                     lag_final_p90=("lag_final_m", lambda s: s.quantile(0.9)),
                                     wait_days_median=("wait_days", "median"),
                                     price_month_min=("month", "min"), price_month_max=("month", "max"))
    a2c.to_csv(OUT / "T_A2c_stratum_timing.csv")

    # A2d: is the first-observed value revised by the latest vintage? (confirmed cells only)
    conf = h[h.y_eval.notna()]
    a2d = conf.groupby(["item", "stratum"]).agg(n=("mkt_uid", "size"), share_revised=("truth_revised", "mean"))
    a2d.round(4).to_csv(OUT / "T_A2d_truth_revisions.csv")

    # A2e: sensitivity of stratum sizes to the backfill threshold
    from rtfp.vintages import build_holdout
    stack = pd.read_parquet("data/interim/vintages.parquet")
    sens = {k: build_holdout(stack, max_lag_months=k).assign(item=lambda d: d["item"].astype(str))
            .pivot_table(index="item", columns="stratum", values="mkt_uid", aggfunc="size", fill_value=0)
            for k in (3, 6, 12)}
    a2e = pd.concat(sens, names=["max_lag_months"])
    a2e.to_csv(OUT / "T_A2e_backfill_threshold_sensitivity.csv")

    # A3: balance — non-toggle hold-out cells (covariates in v_final) vs never-observed cells in the latest
    stack_obs = pd.read_parquet("data/interim/vintages.parquet", columns=KEY + ["obs"])
    stack_obs["item"] = stack_obs["item"].astype(str)
    ever = stack_obs.groupby(KEY).obs.any().rename("ever_obs").reset_index()
    never = latest.merge(ever, on=KEY, how="left")
    never = never[~never.ever_obs.fillna(False).astype(bool) & never.c.notna()]
    hn = h[h.stratum.isin(["forecast", "cross_sectional"])]
    def summ(df, gap, dist, trust, label):
        return pd.Series({"group": label, "n": len(df), "year_median": df.month.dt.year.median(),
                          "share_year_ge_2020": (df.month.dt.year >= 2020).mean(),
                          "gap_median": df[gap].median(), "gap_missing_share": df[gap].isna().mean(),
                          "dist_obs_km_median": df[dist].median(), "trust_median": df[trust].median(),
                          "data_coverage_median": df.data_coverage.median()})
    rows = []
    for it in sorted(h.item.unique()):
        rows.append(summ(hn[hn.item == it], "gap_final", "dist_obs_final", "trust_final", f"{it}: holdout (forecast+cross-sectional)"))
        rows.append(summ(never[never.item == it], "gap", "dist_obs_km", "trust", f"{it}: never observed"))
    a3 = pd.DataFrame(rows)
    a3.round(3).to_csv(OUT / "T_A3_covariate_balance.csv", index=False)

    pd.set_option("display.width", 250)
    print("HOLDOUT CELLS by item x stratum\n", tot, "\n")
    print("CHANGE PAIRS\n", a2b, "\n")
    print("TIMING\n", a2c, "\n")
    print("TRUTH REVISIONS\n", a2d.round(3), "\n")
    print("BACKFILL THRESHOLD SENSITIVITY\n", a2e, "\n")
    print("BALANCE\n", a3.round(3).to_string(index=False))


if __name__ == "__main__":
    main()


def _md(df):
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def write_note():
    t = lambda n: pd.read_csv(OUT / n)
    h = pd.read_parquet("data/interim/holdout.parquet")
    h["item"] = h["item"].astype(str)
    cells = h.pivot_table(index="item", columns="stratum", values="mkt_uid", aggfunc="size", fill_value=0)
    pairs = t("T_A2b_pair_counts.csv").set_index(["item", "stratum"])
    clean = h[h.stratum.isin(["forecast", "cross_sectional"])]
    vint = pd.read_parquet("data/interim/vintages.parquet", columns=["vintage"]).vintage
    note = f"""# A2/A3 — The natural hold-out (Task 1.3)

*Generated by `analysis/a2_holdout_counts.py`; do not edit numbers by hand.*

**Claim.** Across {vint.nunique()} RTFP Bangladesh releases ({vint.min().date()} to {vint.max().date()}), {len(h):,} market-month-item cells were imputed in one release and observed in a later one. Only {len(clean):,} of them are clean late arrivals usable for evaluation (forecast + cross-sectional strata); the rest are pipeline toggles, values that later vanished, or historical backfill.

**Cells by stratum** (precedence toggle > unconfirmed > backfill > forecast > cross_sectional):

{_md(cells)}

- *toggle*: the survey value was already in an earlier release, then dropped and restored. Not a late arrival.
- *unconfirmed*: the value that appeared is not in the latest release (and therefore not in the WFP feed, Table A1). Not usable as truth.
- *backfill*: arrived more than 6 months after the price month. Mostly wheat flour; consistent with the first release holding fewer pre-2023 wheat-flour survey values (2,084) than later full releases (2,548), though the cause is not documented.
- *forecast*: no market observed for that item-month in the imputing release (real-time nowcast: the imputing release came a median of 1 month after the price month).
- *cross_sectional*: some markets observed that item-month in the imputing release.

**Monthly change pairs** (both published values from the same imputing release; truth from the latest release):

{_md(pairs)}

**Clean strata cover only recent months:** price months {clean.month.min().date()} to {clean.month.max().date()}. Cells never observed in any release are mostly 2007–2019 history (Table A3), so hold-out results describe recent imputations and do not transfer automatically to the historical record.

**Truth revisions:** share of confirmed cells whose first-observed value differs (>0.1%) from the latest value — Table A2d (0–3.4% by item and stratum).

**Backfill threshold sensitivity** (3 / 6 / 12 months): Table A2e. Cross-sectional counts change modestly; wheat-flour backfill dominates at every threshold.

**Method.** `src/rtfp/vintages.py`; invariants tested in `tests/test_vintages.py` (rules on toy data) and `tests/test_vintages_data.py` (re-checked by direct lookups on real data).

**Caveats.**
- The headline sample is small: both-imputed cross-sectional pairs are under 120 per item. Confidence intervals will be wide; the forecast stratum is larger but answers a different question (real-time nowcasting).
- The 6-month backfill threshold is a judgement call made after looking at the arrival-lag distribution (natural break at 3–6 months); registered with 3 and 12 as sensitivity.
- Pipeline toggling between releases is documented here from the data; its cause is not known to us.

**Files.** `outputs/tables/T_A2_holdout_counts.csv`, `T_A2b_pair_counts.csv`, `T_A2c_stratum_timing.csv`, `T_A2d_truth_revisions.csv`, `T_A2e_backfill_threshold_sensitivity.csv`, `T_A3_covariate_balance.csv`.
"""
    Path("outputs/results_notes").mkdir(parents=True, exist_ok=True)
    (Path("outputs/results_notes") / "A2.md").write_text(note)


if __name__ == "__main__":
    write_note()
