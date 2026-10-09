"""Table 3: do imputed monthly changes track real local price movements? (Task 2.2, local part)

Pairs: hold-out change pairs (t-1, t), both published values from the imputing release; truth from the latest
release (WFP-confirmed). Reported by stratum x pair type x local-component definition:
  own_mu        each side minus its own release's observed-market mean change (methodology M1, primary)
  within_month  both sides demeaned within item x month over the evaluation sample
rho* is reported as a band: corr (no noise correction) .. corr / sqrt(rel), with rel from Table 1
(A_main naive, and the conservative minimum used by Gate G1).
Spikes (Table 3b): survey jumps > 10% whose level persists (>= half retained at t+1); share caught by a
same-direction published move of >= 5%.
CIs: two-way (market x month) pigeonhole bootstrap.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp.metrics import local_components, persistent_spikes, pigeonhole_weights, weighted_local_stats

B = 500
SEED = 20261011
OUT = Path("outputs/tables")
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]


def main():
    pairs = pd.read_parquet("data/interim/change_pairs.parquet")
    panel = pd.read_parquet("data/interim/panel_latest.parquet")
    for d in (pairs, panel):
        d["item"] = d["item"].astype(str)
    rel = pd.read_csv(OUT / "T1_noise_reliability.csv")
    rel_a = rel[rel.variant == "A_main"].set_index("item").rel_naive
    gate = pd.read_csv(OUT / "T1b_local_information_and_gate.csv")
    rel_c = gate.drop_duplicates("item").set_index("item").rel_conservative

    clean = pairs[pairs.stratum.isin(["cross_sectional", "forecast"])]
    markets, months = sorted(set(panel.mkt_uid)), sorted(set(panel.month))
    rows = []
    for (st, pt), g in clean.groupby(["stratum", "pair_type"]):
        for mode in ("own_mu", "within_month"):
            e = local_components(g, mode=mode)
            for it in ITEMS:
                x = e[e["item"] == it]
                if len(x) < 10:
                    continue
                s = weighted_local_stats(x.e_pub, x.e_obs, np.ones(len(x)))
                bs = []
                for b in range(B):
                    w = pigeonhole_weights(x, np.random.default_rng(SEED + b), markets, months)
                    if (w > 0).sum() >= 10:
                        bs.append(weighted_local_stats(x.e_pub[w > 0], x.e_obs[w > 0], w[w > 0]))
                bs = pd.DataFrame(bs)
                rows.append({"stratum": st, "pair_type": pt, "mode": mode, "item": it, **s,
                             "corr_lo": bs["corr"].quantile(0.025), "corr_hi": bs["corr"].quantile(0.975),
                             "beta_mz_lo": bs.beta_mz.quantile(0.025), "beta_mz_hi": bs.beta_mz.quantile(0.975),
                             "rho_star_relA": s["corr"] / np.sqrt(rel_a[it]),
                             "rho_star_conservative": s["corr"] / np.sqrt(rel_c[it])})
    t = pd.DataFrame(rows)

    # spikes: need the truth change from t to t+1 (both observed in the latest release)
    nxt = panel[["mkt_uid", "item", "month", "dlog_obs"]].assign(month=lambda d: d.month - pd.offsets.MonthBegin(1))
    sp = clean.merge(nxt.rename(columns={"dlog_obs": "dlog_truth_next"}), on=["mkt_uid", "item", "month"], how="left")
    sp = persistent_spikes(sp.dropna(subset=["dlog_truth_next"]))
    spikes = sp.groupby(["item"]).agg(pairs_with_next=("is_spike", "size"), spikes=("is_spike", "sum"),
                                      caught=("caught", "sum")).reset_index()
    spikes["share_caught"] = spikes.caught / spikes.spikes
    tot = spikes[["pairs_with_next", "spikes", "caught"]].sum()
    spikes.loc[len(spikes)] = {"item": "ALL", **tot.to_dict(), "share_caught": tot.caught / tot.spikes}

    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "T3_local_changes.csv", index=False)
    spikes.to_csv(OUT / "T3b_spikes.csv", index=False)
    pd.set_option("display.width", 250)
    print(t[["stratum", "pair_type", "mode", "item", "n", "corr", "corr_lo", "corr_hi", "beta_mz", "beta_mz_lo",
             "beta_mz_hi", "lam", "rho_star_relA", "rho_star_conservative"]].round(3).to_string(index=False))
    print(spikes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
