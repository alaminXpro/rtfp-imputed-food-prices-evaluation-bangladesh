"""Table 1: measurement-noise reliability of observed local changes, and Gate G1 (Task 2.1).

Noise sample variants (latest release, observed local changes e_obs):
  A_main         non-Division markets, price months >= 2020-08 (the population that supplies hold-out truth)
  B_update_only  A, dropping exact-zero changes (stale quotes): the staleness-aware variant of methodology D2
  C_actual_only  A, only cells whose WFP priceflag is 'actual'
  D_all_2007     all markets incl. Division series, 2007 onward
  A_trim1        A with the 1% most extreme changes trimmed per item (robust to gross entry errors)
  A_no_patuakhali A without Patuakhali Sadar 2025-03..04 (one gross entry error found in audit)

Gate G1 (methodology D2), implemented faithfully (revision 2026-10-09 after audit):
  each reliability is applied to the truth pairs of the SAME population it was estimated on:
    A rel  -> all cross-sectional both-imputed pairs
    B rel  -> the subset whose truth change is non-zero (quote updated)
  rho* = corr / sqrt(rel); a combination is credible if 0 < rel <= 1. Conservative rho* per item = the
  largest credible rho*. PIVOT if >= 2 of 4 items have conservative rho* >= 0.5.
The first, literal implementation (B rel applied to all pairs, zero changes included) is reported as
'literal_v1' for transparency; the audit showed it mixes populations and biases rho* upward.
CIs: two-way (market x month) pigeonhole bootstrap; same draw for noise sample and pairs.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.metrics import local_components, pigeonhole_weights, weighted_local_stats
from rtfp.noise import autocovariances, noise_from_autocov

B = 500
SEED = 20261009
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
OUT = Path("outputs/tables")


def noise_samples(panel, audit):
    p = panel.merge(audit[["mkt_uid", "month", "item", "priceflag"]], on=["mkt_uid", "month", "item"], how="left")
    p = p[p.e_obs.notna()]
    is_div = p.mkt_name.str.endswith(" Division")
    a = p[~is_div & (p.month >= "2020-08-01")]
    pat = (a.mkt_name == "Patuakhali Sadar") & a.month.isin(pd.to_datetime(["2025-03-01", "2025-04-01"]))
    return {"A_main": (a, 0.0), "B_update_only": (a[a.dlog_obs != 0], 0.0),
            "C_actual_only": (a[a.priceflag == "actual"], 0.0), "D_all_2007": (p, 0.0),
            "A_trim1": (a, 0.01), "A_no_patuakhali": (a[~pat], 0.0)}


def rel_table(samples, w_by=None, only=None):
    rows = []
    for name, (d, trim) in samples.items():
        if only and name not in only:
            continue
        dd = d if w_by is None else d.assign(w=w_by(d))
        g = autocovariances(dd, trim=trim)
        for it in ITEMS:
            if it in g.index:
                rows.append({"variant": name, "item": it, "n_changes": int(g.loc[it, "n0"]), **noise_from_autocov(g.loc[it])})
    return pd.DataFrame(rows)


def pair_sets(pairs):
    cs = pairs[(pairs.stratum == "cross_sectional") & (pairs.pair_type == "both_imputed")]
    e = local_components(cs, mode="own_mu")
    fallback = e.mu_obs_vfinal.isna()
    return {"all": e, "update_only": e[e.dlog_truth != 0], "no_fallback": e[~fallback],
            "within_month": local_components(cs, mode="within_month")}


def combos(rt, ps, w=None):
    """rho* for each (reliability, pair population) combination; matched combos define the gate."""
    plan = [("A_naive", "A_main", "rel_naive", "all", True), ("A_ar1", "A_main", "rel_ar1", "all", True),
            ("B_naive", "B_update_only", "rel_naive", "update_only", True),
            ("B_ar1", "B_update_only", "rel_ar1", "update_only", True),
            ("literal_v1", "B_update_only", "rel_naive", "all", False),
            ("A_trim1", "A_trim1", "rel_naive", "all", False),
            ("A_no_patuakhali", "A_no_patuakhali", "rel_naive", "all", False),
            ("A_naive_no_fallback", "A_main", "rel_naive", "no_fallback", False),
            ("A_naive_within_month", "A_main", "rel_naive", "within_month", False),
            ("no_noise_correction", None, None, "all", False)]
    rows = []
    for name, var, col, pop, in_gate in plan:
        e = ps[pop]
        for it in ITEMS:
            x = e[e["item"] == it]
            ww = np.ones(len(x)) if w is None else w(x)
            sel = ww > 0
            if sel.sum() < 10:
                continue
            st = weighted_local_stats(x.e_pub.to_numpy()[sel], x.e_obs.to_numpy()[sel], ww[sel])
            rel = 1.0 if var is None else rt.set_index(["variant", "item"]).get(col, pd.Series(dtype=float)).get((var, it), np.nan)
            credible = bool(0 < rel <= 1) if rel == rel else False
            rows.append({"combo": name, "item": it, "pairs": pop, "n": st["n"], "corr": st["corr"], "rel": rel,
                         "rho_star": st["corr"] / np.sqrt(rel) if credible else np.nan, "credible": credible,
                         "in_gate": in_gate})
    return pd.DataFrame(rows)


def gate(cb):
    g = cb[cb.in_gate & cb.credible].groupby("item").rho_star.max()
    return g, int((g >= 0.5).sum())


def main():
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    audit = pd.read_parquet("data/interim/audit_cells.parquet")
    pairs = pd.read_parquet("data/interim/change_pairs.parquet")
    for d in (panel, audit, pairs):
        d["item"] = d["item"].astype(str)
    samples = noise_samples(panel, audit)
    rt = rel_table(samples)
    ps = pair_sets(pairs)
    cb = combos(rt, ps)
    g, n_over = gate(cb)
    lit = cb[cb.combo == "literal_v1"].set_index("item").rho_star
    n_lit = int((lit >= 0.5).sum())

    markets, months = sorted(set(panel.mkt_uid)), sorted(set(panel.month))
    boot = []
    for b in range(B):
        wfun = lambda d, sb=SEED + b: pigeonhole_weights(d, np.random.default_rng(sb), markets, months)
        rtb = rel_table(samples, w_by=wfun, only=("A_main", "B_update_only"))
        cbb = combos(rtb, ps, w=wfun)
        gb, nb = gate(cbb)
        boot.append({"b": b, "n_over": nb, **{f"rho_{k}": v for k, v in gb.items()}})
    bt = pd.DataFrame(boot)

    OUT.mkdir(parents=True, exist_ok=True)
    rt.to_csv(OUT / "T1_noise_reliability.csv", index=False)
    cb.to_csv(OUT / "T1b_gate_combinations.csv", index=False)
    summ = pd.DataFrame({"conservative_rho_star": g,
                         "boot_lo": [bt[f"rho_{it}"].quantile(0.025) for it in g.index],
                         "boot_hi": [bt[f"rho_{it}"].quantile(0.975) for it in g.index]})
    summ.to_csv(OUT / "T1c_gate_summary.csv")
    verdict = "PIVOT" if n_over >= 2 else "PROCEED"
    share_pivot = float((bt.n_over >= 2).mean())
    pd.Series({"items_over_0.5": n_over, "verdict": verdict, "literal_v1_items_over_0.5": n_lit,
               "literal_v1_verdict": "PIVOT" if n_lit >= 2 else "PROCEED",
               "bootstrap_share_pivot": share_pivot}).to_csv(OUT / "T1d_gate_verdict.csv")
    pd.set_option("display.width", 250)
    print(rt[["variant", "item", "n_changes", "g0", "var_nu_naive", "rel_naive", "phi", "rel_ar1"]].round(4).to_string(index=False))
    print(cb.round(3).to_string(index=False))
    print(summ.round(3).to_string())
    print(f"GATE G1 (D2-faithful): {n_over} of 4 items -> {verdict}; literal_v1: {n_lit} of 4; bootstrap share PIVOT {share_pivot:.3f}")


if __name__ == "__main__":
    main()
