"""Quick look: are RTFP imputed month-on-month price changes smoother than observed ones?
Observed cells = non-missing raw commodity column; imputed = raw missing, c_<item> present."""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1], parse_dates=["price_date"])
df = df.sort_values(["mkt_name", "price_date"])
items = ["rice", "wheat_flour", "oil", "lentils"]
rows = []
for it in items:
    d = df[["mkt_name", "price_date", it, f"c_{it}", "spatially_interpolated", f"trust_{it}"]].copy()
    d["obs"] = d[it].notna()
    # sanity: c_ equals raw where observed?
    m = d["obs"] & d[f"c_{it}"].notna()
    match = np.mean(np.isclose(d.loc[m, it], d.loc[m, f"c_{it}"], rtol=1e-3)) if m.any() else np.nan
    g = d.groupby("mkt_name")
    d["dlog"] = g[f"c_{it}"].transform(lambda s: np.log(s).diff())
    d["obs_prev"] = g["obs"].shift(1)
    d["pair"] = np.select([d.obs & (d.obs_prev == True), ~d.obs & (d.obs_prev == False)], ["OO", "II"], "mixed")
    d = d.dropna(subset=["dlog"])
    # within-month comparison: only months having both OO and II cells
    mm = d[d.pair.isin(["OO", "II"])]
    per_month = mm.groupby(["price_date", "pair"])["dlog"].agg(["std", "count"]).unstack()
    per_month = per_month[(per_month["count"]["OO"] >= 5) & (per_month["count"]["II"] >= 5)]
    ratio = (per_month["std"]["II"] / per_month["std"]["OO"])
    sd = mm.groupby("pair")["dlog"].std()
    spike = mm.assign(sp=mm.dlog.abs() > 0.10).groupby("pair")["sp"].mean()
    rows.append(dict(item=it, raw_eq_close=round(match, 3),
                     n_OO=int((mm.pair == "OO").sum()), n_II=int((mm.pair == "II").sum()),
                     sd_OO=round(sd.get("OO", np.nan), 4), sd_II=round(sd.get("II", np.nan), 4),
                     spike10_OO=round(spike.get("OO", np.nan), 3), spike10_II=round(spike.get("II", np.nan), 3),
                     months_compared=len(ratio), median_within_month_sd_ratio_II_over_OO=round(ratio.median(), 3),
                     share_months_II_smoother=round((ratio < 1).mean(), 3)))
print(pd.DataFrame(rows).to_string(index=False))
