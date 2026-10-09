"""Table 6: do imputed prices raise the same early-warning alerts as survey prices? (Task 4.1, methodology D4)

Empirical: for every clean hold-out cell (forecast + cross_sectional) the alert class at month t is computed twice,
appending to the SAME history (the latest release's published series for that market and item, months < t):
(a) the imputed value c_final, (b) the survey value y_eval. Survey-based classes are the reference.
Indicators (src/rtfp/alerts.py, formulas quoted from WFP 2014 and Baquedano 2015 / SDG 2.c.1):
  ALPS real-time (alert = Alert or Crisis, ALPS >= 1; any = not Normal, ALPS >= 0.25)
  IPA Baquedano and IPA SDG (alert = Price Alert >= 1; any = Watch or above >= 0.5)
Agreement: recall, precision, Cohen's kappa; two-way (market x month) pigeonhole bootstrap CIs.
Benchmark (simulation calibrated to Bangladesh, analysis/a4_simulation.calibrate): recall of survey alerts by
(i) an oracle that removes measurement noise but keeps every local signal, and (ii) a common-only value with no
local information, both natively and with simulated alert base rates matched to the data per item (quantile
threshold) and pooled with empirical item weights. Empirical recall near (ii) means imputed values carry little
local alert information. Power uses the bootstrap SE of recall (two-way clustering).
Power: number of survey alerts per item, and the recall difference detectable at 80% power.
"""
import os

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import runpy  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rtfp.alerts import agreement, alps_realtime, ipa_realtime  # noqa: E402
from rtfp.metrics import pigeonhole_weights  # noqa: E402
from rtfp.simulate import alert_benchmark  # noqa: E402

OUT = Path("outputs/tables")
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
B = int(os.environ.get("T6_BOOT", 500))
SEED = 20261013
SIM_PANELS = int(os.environ.get("T6_SIM_PANELS", 50))
SIM_EVAL_MONTHS = 12
THRESH = {"alps": {"alert": 1.0, "any": 0.25}, "ipa_baquedano": {"alert": 1.0, "any": 0.5},
          "ipa_sdg": {"alert": 1.0, "any": 0.5}}


def empirical_z(h, panel):
    series = {k: g.sort_values("month") for k, g in panel.groupby(["mkt_uid", "item"])}
    rows = []
    for r in h.itertuples(index=False):
        g = series[(r.mkt_uid, r.item)]
        hist = g[g.month < r.month]
        months = pd.DatetimeIndex(list(hist.month) + [r.month])
        cand = np.array([r.y_eval, r.c_final])
        z = {"alps": alps_realtime(hist.c.to_numpy(float), months, cand),
             "ipa_baquedano": ipa_realtime(hist.c.to_numpy(float), months, cand, variant="baquedano"),
             "ipa_sdg": ipa_realtime(hist.c.to_numpy(float), months, cand, variant="sdg")}
        for ind, v in z.items():
            rows.append((r.mkt_uid, r.item, r.month, r.stratum, ind, v[0], v[1]))
    return pd.DataFrame(rows, columns=["mkt_uid", "item", "month", "stratum", "indicator", "z_survey", "z_imputed"])


def agree_table(z, ref_col, pred_col, group_cols, w=None):
    out = []
    for keys, g in z.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ind = keys[group_cols.index("indicator")]
        ok = g[ref_col].notna() & g[pred_col].notna()
        g = g[ok]
        ww = None if w is None else w[g.index]
        for level, cut in THRESH[ind].items():
            a = agreement(g[ref_col] >= cut, g[pred_col] >= cut, ww)
            out.append({**dict(zip(group_cols, keys)), "level": level, **a})
    return pd.DataFrame(out)


def detectable_diff(n_pos, p0, power=0.8, alpha=0.05):
    """Smallest recall difference from p0 detectable with n_pos reference alerts (normal approximation)."""
    from scipy.stats import norm
    if n_pos <= 0:
        return np.nan
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return (za + zb) * np.sqrt(max(p0 * (1 - p0), 1e-6) / n_pos)


def main():
    h = pd.read_parquet("data/interim/holdout.parquet")
    h["item"] = h["item"].astype(str)
    h = h[h.stratum.isin(["forecast", "cross_sectional"]) & h.y_eval.notna()].copy()
    panel = pd.read_parquet("data/interim/panel_latest.parquet", columns=["mkt_uid", "item", "month", "c"])
    panel["item"] = panel["item"].astype(str)
    z = empirical_z(h, panel).reset_index(drop=True)

    emp = agree_table(z, "z_survey", "z_imputed", ["item", "indicator"])
    emp_all = agree_table(z.assign(item="ALL"), "z_survey", "z_imputed", ["item", "indicator"])
    emp = pd.concat([emp, emp_all], ignore_index=True)

    markets, months = sorted(set(panel.mkt_uid)), sorted(set(panel.month))
    boot = []
    for b in range(B):
        w = pd.Series(pigeonhole_weights(z, np.random.default_rng(SEED + b), markets, months), index=z.index)
        bt = agree_table(z.assign(item="ALL"), "z_survey", "z_imputed", ["item", "indicator"], w=w)
        bi = agree_table(z, "z_survey", "z_imputed", ["item", "indicator"], w=w)
        boot.append(pd.concat([bt, bi]).assign(b=b))
    bt = pd.concat(boot)
    ci = bt.groupby(["item", "indicator", "level"]).agg(
        recall_lo=("recall", lambda s: s.quantile(0.025)), recall_hi=("recall", lambda s: s.quantile(0.975)),
        precision_lo=("precision", lambda s: s.quantile(0.025)), precision_hi=("precision", lambda s: s.quantile(0.975)),
        kappa_lo=("kappa", lambda s: s.quantile(0.025)), kappa_hi=("kappa", lambda s: s.quantile(0.975))).reset_index()
    emp = emp.merge(ci, on=["item", "indicator", "level"], how="left")

    # simulation benchmark, calibrated per item
    a4 = runpy.run_path("analysis/a4_simulation.py", run_name="lib")
    t1 = pd.read_csv(OUT / "T1_noise_reliability.csv")
    t3 = pd.read_csv(OUT / "T3_local_changes.csv")
    full_panel = pd.read_parquet("data/interim/panel_latest.parquet")
    full_panel["item"] = full_panel["item"].astype(str)
    specs, _ = a4["calibrate"](full_panel, t1, t3)
    sims = []
    for it in ITEMS:
        for k in range(SIM_PANELS):
            rng = np.random.default_rng([SEED, ITEMS.index(it), k])
            s = alert_benchmark(specs[it], SIM_EVAL_MONTHS, rng, price_scale=1.0)
            sims.append(s.assign(item=it, panel=k))  # ALPS and IPA are scale-free
    sim = pd.concat(sims, ignore_index=True)
    bench = []
    for col, name in (("z_oracle", "oracle_noise_removed"), ("z_common", "common_only")):
        tb = agree_table(sim, "z_survey", col, ["item", "indicator"]).assign(benchmark=name)
        bench.append(tb)
    bench = pd.concat(bench, ignore_index=True)

    # base-rate-matched benchmark: in the simulation, set each item x indicator x level threshold at the quantile of
    # simulated survey z that reproduces the EMPIRICAL survey alert rate, apply it to survey, oracle and common-only;
    # pool recall over items with EMPIRICAL survey-alert weights (verifier recommendation)
    emp_items = emp[emp["item"] != "ALL"].set_index(["item", "indicator", "level"])
    matched, per_panel = [], []
    for (it, ind), g in sim.groupby(["item", "indicator"]):
        for level in THRESH[ind]:
            key = (it, ind, level)
            if key not in emp_items.index:
                continue
            rate = emp_items.loc[key, "n_ref_pos"] / emp_items.loc[key, "n"]
            gg = g.dropna(subset=["z_survey", "z_oracle", "z_common"])
            if rate <= 0 or len(gg) == 0:
                continue
            cut = np.quantile(gg.z_survey, 1 - rate)
            for col, name in (("z_oracle", "oracle_noise_removed"), ("z_common", "common_only")):
                a = agreement(gg.z_survey >= cut, gg[col] >= cut)
                matched.append({"benchmark": name, "item": it, "indicator": ind, "level": level, "recall": a["recall"],
                                "emp_weight": emp_items.loc[key, "n_ref_pos"], "sim_rate_matched": a["n_ref_pos"] / a["n"]})
                for k, gk in gg.groupby("panel"):  # same threshold, one panel at a time: Monte Carlo error
                    ak = agreement(gk.z_survey >= cut, gk[col] >= cut)
                    if ak["n_ref_pos"] > 0:
                        per_panel.append({"benchmark": name, "item": it, "indicator": ind, "level": level, "panel": k,
                                          "recall": ak["recall"], "emp_weight": emp_items.loc[key, "n_ref_pos"]})
    matched = pd.DataFrame(matched)
    mpool = (matched.assign(wr=matched.recall * matched.emp_weight).groupby(["benchmark", "indicator", "level"])
             .agg(wr=("wr", "sum"), w=("emp_weight", "sum")).reset_index())
    mpool["recall_matched"] = mpool.wr / mpool.w
    pp = pd.DataFrame(per_panel)
    pp = (pp.assign(wr=pp.recall * pp.emp_weight).groupby(["benchmark", "indicator", "level", "panel"])
          .agg(wr=("wr", "sum"), w=("emp_weight", "sum")).reset_index())
    pp["recall"] = pp.wr / pp.w
    mc = pp.groupby(["benchmark", "indicator", "level"]).recall.agg(
        recall_mc_se=lambda r: r.std(ddof=1) / np.sqrt(len(r)), n_panels_used="size").reset_index()
    mpool = mpool.merge(mc, on=["benchmark", "indicator", "level"], how="left")
    first = panel.month.min()
    hist_len = ((z.month.dt.year - first.year) * 12 + z.month.dt.month - first.month)
    meta = pd.DataFrame([{"bootstrap_draws": B, "sim_panels_per_item": SIM_PANELS, "sim_eval_months": SIM_EVAL_MONTHS,
                          "sim_window_start": a4["WINDOW_START"][:7], "sim_window_months": int(specs[ITEMS[0]].n_months),
                          "sim_history_months_min": int(specs[ITEMS[0]].n_months) - SIM_EVAL_MONTHS,
                          "sim_history_months_max": int(specs[ITEMS[0]].n_months) - 1,
                          "data_history_first_month": str(first.date())[:7], "data_history_months_min": int(hist_len.min()),
                          "data_history_months_max": int(hist_len.max())}])

    # power from the bootstrap SE of recall (accounts for two-way clustering), 80% power, two-sided 5%
    from scipy.stats import norm
    se = bt.groupby(["item", "indicator", "level"]).recall.std().rename("recall_boot_se").reset_index()
    power = emp[["item", "indicator", "level", "n_ref_pos", "recall"]].merge(se, on=["item", "indicator", "level"], how="left")
    power["detectable_recall_diff_80pct"] = (norm.ppf(0.975) + norm.ppf(0.8)) * power.recall_boot_se
    power["iid_detectable_diff_for_reference"] = [detectable_diff(n, p if p == p else 0.5)
                                                   for n, p in zip(power.n_ref_pos, power.recall)]
    OUT.mkdir(parents=True, exist_ok=True)
    z.to_csv(OUT / "T6_alerts_zscores.csv", index=False)
    emp.to_csv(OUT / "T6_alerts.csv", index=False)
    bench.to_csv(OUT / "T6_alerts_benchmark.csv", index=False)
    matched.to_csv(OUT / "T6_alerts_benchmark_matched_items.csv", index=False)
    mpool.to_csv(OUT / "T6_alerts_benchmark_matched.csv", index=False)
    power.to_csv(OUT / "T6_alerts_power.csv", index=False)
    meta.to_csv(OUT / "T6_alerts_meta.csv", index=False)
    pd.set_option("display.width", 230)
    cols = ["item", "indicator", "level", "n", "n_ref_pos", "n_pred_pos", "recall", "recall_lo", "recall_hi",
            "precision", "kappa"]
    print(emp[cols].round(3).to_string(index=False))
    print(bench[["benchmark", "item", "indicator", "level", "n", "n_ref_pos", "recall", "precision", "kappa"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
