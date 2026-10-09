"""Quick look: natural hold-out from RTFP vintages.

A hold-out cell is a (market, month, item) that was imputed (raw missing, c_ present)
in some earlier vintage and is observed (raw present) in the latest vintage.
For each such cell we keep the imputation from the LAST vintage in which it was still
imputed, and compare it with the observed value in the latest vintage.

Usage: python analysis/02_holdout_quicklook.py data/raw/rtfp_vintages
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
KEY = ["geo_id", "price_date"]


def load(path):
    cols = KEY + ITEMS + [f"{p}_{i}" for i in ITEMS for p in ("c", "h", "l", "trust")]
    d = pd.read_csv(path, usecols=lambda c: c in cols)
    d["price_date"] = pd.to_datetime(d["price_date"])
    return d


def vintage_date(path):
    return pd.Timestamp(os.path.basename(path).split("_")[-1].replace(".csv", ""))


def main(base):
    files = sorted(glob.glob(os.path.join(base, "BGD_RTFP_mkt_*", "*.csv")), key=vintage_date)
    latest = load(files[-1]).set_index(KEY)

    # Last pre-observation imputation per cell: iterate oldest -> newest, overwrite.
    last_imp = {it: {} for it in ITEMS}
    for f in files[:-1]:
        v = vintage_date(f)
        d = load(f)
        for it in ITEMS:
            if it not in d.columns:
                continue
            m = d[it].isna() & d[f"c_{it}"].notna()
            sub = d.loc[m, KEY + [f"c_{it}", f"h_{it}", f"l_{it}", f"trust_{it}"]]
            for row in sub.itertuples(index=False):
                last_imp[it][(row[0], row[1])] = (v, row[2], row[3], row[4], row[5])

    # National mean log change per month in latest vintage (common component mu_t).
    lat = latest.reset_index().sort_values(KEY)
    mu = {}
    for it in ITEMS:
        lat[f"dl_{it}"] = lat.groupby("geo_id")[f"c_{it}"].transform(lambda s: np.log(s).diff())
        mu[it] = lat.groupby("price_date")[f"dl_{it}"].mean()

    out_rows, range_rows = [], []
    for it in ITEMS:
        imp = pd.DataFrame(
            [(k[0], k[1], *v) for k, v in last_imp[it].items()],
            columns=["geo_id", "price_date", "vintage", "c_imp", "h_imp", "l_imp", "trust_imp"],
        ).set_index(KEY)
        j = imp.join(latest[[it]].rename(columns={it: "y"}), how="inner").dropna(subset=["y", "c_imp"])
        j = j[j.y > 0]
        j["age_m"] = (j.vintage.dt.year - j.index.get_level_values(1).year) * 12 + (
            j.vintage.dt.month - j.index.get_level_values(1).month)
        logerr = np.log(j.c_imp) - np.log(j.y)

        # Changes: need both t and t-1 in hold-out set with the same vintage-independent truth.
        jj = j.reset_index().sort_values(KEY)
        jj["prev_date"] = jj.price_date - pd.DateOffset(months=1)
        prev = jj[["geo_id", "price_date", "c_imp", "y", "vintage"]].rename(
            columns={"price_date": "prev_date", "c_imp": "c_imp_prev", "y": "y_prev", "vintage": "vintage_prev"})
        ch = jj.merge(prev, on=["geo_id", "prev_date"], how="inner")
        # Both months must come from the same vintage, otherwise the change mixes two model runs.
        ch = ch[ch.vintage == ch.vintage_prev]
        ch["d_imp"] = np.log(ch.c_imp) - np.log(ch.c_imp_prev)
        ch["d_obs"] = np.log(ch.y) - np.log(ch.y_prev)
        ch["mu"] = ch.price_date.map(mu[it])
        ch = ch.dropna(subset=["d_imp", "d_obs", "mu"])
        ch["e_imp"], ch["e_obs"] = ch.d_imp - ch.mu, ch.d_obs - ch.mu

        def slope(y, x):
            if len(x) < 10:
                return np.nan
            return sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1]

        out_rows.append(dict(
            item=it,
            n_cells=len(j),
            MAPE_pct=round(100 * np.mean(np.abs(j.c_imp - j.y) / j.y), 1),
            medianAPE_pct=round(100 * np.median(np.abs(j.c_imp - j.y) / j.y), 1),
            bias_pct=round(100 * logerr.mean(), 1),
            share_err_gt10pct=round(np.mean(np.abs(logerr) > np.log(1.10)), 2),
            n_changes=len(ch),
            MAE_change_pp=round(100 * np.mean(np.abs(ch.d_imp - ch.d_obs)), 2),
            corr_change=round(ch[["d_imp", "d_obs"]].corr().iloc[0, 1], 2),
            MZ_slope_obs_on_imp=round(slope(ch.e_obs, ch.e_imp), 2),
            reverse_slope_imp_on_obs=round(slope(ch.e_imp, ch.e_obs), 2),
            var_ratio_e=round(ch.e_imp.var() / ch.e_obs.var(), 2) if len(ch) > 10 else np.nan,
            corr_common_mu_vs_obs=round(ch[["mu", "d_obs"]].corr().iloc[0, 1], 2),
            spike_gt10_obs=round(np.mean(np.abs(ch.d_obs) > 0.10), 3),
            spike_gt10_imp=round(np.mean(np.abs(ch.d_imp) > 0.10), 3),
            spikes_caught=round(np.mean(np.abs(ch.d_imp[np.abs(ch.d_obs) > 0.10]) > 0.05), 2),
        ))

        # Range check: does the imputed [l, h] band contain the later-observed value?
        band = j.dropna(subset=["h_imp", "l_imp"])
        inside = (band.y >= band.l_imp) & (band.y <= band.h_imp)
        width_imp = ((band.h_imp - band.l_imp) / band.c_imp).median()
        obs_lat = latest[latest[it].notna() & latest[f"h_{it}"].notna()]
        width_obs = ((obs_lat[f"h_{it}"] - obs_lat[f"l_{it}"]) / obs_lat[f"c_{it}"]).median()
        inside_obs = ((obs_lat[it] >= obs_lat[f"l_{it}"]) & (obs_lat[it] <= obs_lat[f"h_{it}"])).mean()
        range_rows.append(dict(
            item=it,
            band_width_imputed_pct=round(100 * width_imp, 1),
            band_width_observed_pct=round(100 * width_obs, 1),
            truth_inside_imputed_band=round(inside.mean(), 2),
            observed_inside_own_band=round(inside_obs, 2),
            n=len(band),
        ))

    pd.set_option("display.width", 250)
    print("=== Hold-out accuracy (imputed in an earlier vintage vs later observed) ===")
    print(pd.DataFrame(out_rows).to_string(index=False))
    print("\n=== Published high/low band ===")
    print(pd.DataFrame(range_rows).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
