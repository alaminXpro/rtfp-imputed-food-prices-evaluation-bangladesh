"""Table 1: measurement-noise reliability of observed local changes, and Gate G1 (Task 2.1).

Noise sample variants (latest release, observed local changes e_obs):
  A main          non-Division markets, price months >= 2020-08 (the population that supplies hold-out truth)
  B update_only   A, dropping changes that are exactly zero (stale quotes)
  C actual_only   A, only cells whose WFP priceflag is 'actual' (not 'aggregate')
  D all_2007      all markets incl. Division series, 2007 onward
Gate G1 (methodology D2): on cross-sectional, both-imputed change pairs, rho* = corr(e_pub, e_obs)/sqrt(rel)
at the conservative (lowest credible) rel from variants A and B; pivot if >= 2 of 4 items have rho* >= 0.5.
Uncertainty: two-way (market x month) pigeonhole bootstrap, the same draw applied to noise sample and pairs.
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
    return {"A_main": a, "B_update_only": a[a.dlog_obs != 0], "C_actual_only": a[a.priceflag == "actual"],
            "D_all_2007": p}


def rel_table(samples, w_by=None):
    rows = []
    for name, d in samples.items():
        dd = d if w_by is None else d.assign(w=w_by(d))
        g = autocovariances(dd)
        for it in ITEMS:
            if it not in g.index:
                continue
            e = noise_from_autocov(g.loc[it])
            rows.append({"variant": name, "item": it, "n_changes": int(g.loc[it, "n0"]), **e})
    return pd.DataFrame(rows)


def conservative_rel(rt):
    """Lowest credible reliability per item among variants A and B (naive and AR(1) estimators)."""
    sub = rt[rt.variant.isin(["A_main", "B_update_only"])]
    cand = pd.concat([sub[["item", "rel_naive"]].rename(columns={"rel_naive": "rel"}),
                      sub[["item", "rel_ar1"]].rename(columns={"rel_ar1": "rel"})])
    cand = cand[(cand.rel > 0) & (cand.rel <= 1)]
    return cand.groupby("item").rel.min()


def headline_pairs(pairs, stratum, mode):
    p = pairs[(pairs.stratum == stratum) & (pairs.pair_type == "both_imputed")]
    return local_components(p, mode=mode)


def main():
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    audit = pd.read_parquet("data/interim/audit_cells.parquet")
    pairs = pd.read_parquet("data/interim/change_pairs.parquet")
    for d in (panel, audit, pairs):
        d["item"] = d["item"].astype(str)
    samples = noise_samples(panel, audit)
    rt = rel_table(samples)
    rel_c = conservative_rel(rt)

    evals = {(s, m): headline_pairs(pairs, s, m) for s in ("cross_sectional", "forecast")
             for m in ("own_mu", "within_month")}
    point = []
    for (s, m), e in evals.items():
        for it in ITEMS:
            x = e[e["item"] == it]
            st = weighted_local_stats(x.e_pub, x.e_obs, np.ones(len(x)))
            point.append({"stratum": s, "mode": m, "item": it, **st, "rel_conservative": rel_c.get(it, np.nan),
                          "rho_star": st["corr"] / np.sqrt(rel_c.get(it, np.nan))})
    pt = pd.DataFrame(point)

    # bootstrap: same pigeonhole draw for noise sample (A, B) and pairs
    markets = sorted(set(panel.mkt_uid))
    months = sorted(set(panel.month))
    boot = []
    for b in range(B):
        # one seed per draw: every call below gets the identical market x month weights
        wfun = lambda d, sb=SEED + b: pigeonhole_weights(d, np.random.default_rng(sb), markets, months)
        rtb = rel_table({k: samples[k] for k in ("A_main", "B_update_only")}, w_by=wfun)
        relb = conservative_rel(rtb)
        for (s, m), e in evals.items():
            w = wfun(e)
            for it in ITEMS:
                sel = (e["item"] == it).to_numpy() & (w > 0)
                if sel.sum() < 10:
                    continue
                st = weighted_local_stats(e.e_pub.to_numpy()[sel], e.e_obs.to_numpy()[sel], w[sel])
                r = relb.get(it, np.nan)
                boot.append({"b": b, "stratum": s, "mode": m, "item": it, "corr": st["corr"], "beta_mz": st["beta_mz"],
                             "lam": st["lam"], "rel": r, "rho_star": st["corr"] / np.sqrt(r)})
    bt = pd.DataFrame(boot)
    ci = bt.groupby(["stratum", "mode", "item"]).agg(
        corr_lo=("corr", lambda s: s.quantile(0.025)), corr_hi=("corr", lambda s: s.quantile(0.975)),
        rho_lo=("rho_star", lambda s: s.quantile(0.025)), rho_hi=("rho_star", lambda s: s.quantile(0.975)),
        beta_mz_lo=("beta_mz", lambda s: s.quantile(0.025)), beta_mz_hi=("beta_mz", lambda s: s.quantile(0.975)),
        n_boot=("corr", "size")).reset_index()
    res = pt.merge(ci, on=["stratum", "mode", "item"], how="left")

    OUT.mkdir(parents=True, exist_ok=True)
    rt.to_csv(OUT / "T1_noise_reliability.csv", index=False)
    res.to_csv(OUT / "T1b_local_information_and_gate.csv", index=False)

    head = res[(res.stratum == "cross_sectional") & (res["mode"] == "own_mu")]
    n_over = int((head.rho_star >= 0.5).sum())
    gate = "PIVOT" if n_over >= 2 else "PROCEED"
    pd.set_option("display.width", 250)
    cols = ["variant", "item", "n_changes", "g0", "var_nu_naive", "rel_naive", "phi", "var_nu_ar1", "rel_ar1"]
    print(rt[cols].round(5).to_string(index=False))
    print(res[["stratum", "mode", "item", "n", "corr", "corr_lo", "corr_hi", "beta_mz", "lam", "rel_conservative",
               "rho_star", "rho_lo", "rho_hi"]].round(3).to_string(index=False))
    print(f"GATE G1: {n_over} of 4 items with rho* >= 0.5 (cross-sectional, both-imputed, own_mu) -> {gate}")
    return rt, res, gate, n_over


if __name__ == "__main__":
    main()
