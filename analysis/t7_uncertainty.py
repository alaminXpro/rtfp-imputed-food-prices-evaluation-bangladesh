"""Table 7: calibrated uncertainty for imputed RTFP cells (Task 5.1, methodology D5).

Target r = log(y_eval / c_final) on clean hold-out cells (forecast + cross_sectional).
Features (all at the imputing release): gap, distance to nearest observed market (NaN for forecast cells),
trust, market's observed share of the previous 24 months, item, calendar month, vintage age, stratum.
Split by the release in which the truth arrived (v_truth): train 60% | model selection 10% | conformal\ncalibration 10% | test 20%. Candidates (unconditional, group-empirical by stratum x gap, quantile GB) are\nchosen on the selection block only; every candidate's test result is reported.
Quantile gradient boosting (scikit-learn HistGradientBoosting, quantile loss) + conformalized quantile
regression, Mondrian by stratum x gap bin. Baselines: published [l_, h_] band, constant-width split conformal
on |r|, Gaussian, unconditional empirical quantiles.
Online adaptive conformal (ACI, Gibbs & Candes 2021; gamma = 0.05) processes test releases in order, as the
release flow reveals truth; static calibration under-covers because the error distribution drifts.
Deliverable: intervals for imputed cells of the latest release that lie inside the hold-out's covariate
support, from the selected candidate with the ACI state after the last release; all other cells are flagged
'outside calibrated support'.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

from rtfp.uncertainty import aci_next_alpha, conformal_quantile, cqr_intervals, cqr_scores, group_quantiles, mondrian_thresholds, obs_share

OUT = Path("outputs/tables")
LEVELS = [0.05, 0.10, 0.50, 0.90, 0.95]
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
FEATS = ["gap", "dist_obs", "trust", "obs_share24", "item_code", "cal_month", "vintage_age", "is_forecast"]
SEED = 7


GAMMA = 0.05  # ACI step size


def gap_bin(g):
    return np.where(g.isna(), "na", np.where(g <= 1, "1", np.where(g <= 2, "2", "3+")))


def features(df, gap, dist, trust, share, vint):
    out = pd.DataFrame({"gap": df[gap], "dist_obs": df[dist], "trust": df[trust], "obs_share24": df[share],
                        "item_code": df["item"].map({k: i for i, k in enumerate(ITEMS)}),
                        "cal_month": df.month.dt.month,
                        "vintage_age": (df[vint].dt.year - df.month.dt.year) * 12 + (df[vint].dt.month - df.month.dt.month),
                        "is_forecast": (df.stratum == "forecast").astype(int)}, index=df.index)
    return out


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def main():
    h = pd.read_parquet("data/interim/holdout.parquet")
    h["item"] = h["item"].astype(str)
    h = h[h.stratum.isin(["forecast", "cross_sectional"]) & h.y_eval.notna()].copy()
    stack = pd.read_parquet("data/interim/vintages.parquet",
                            columns=["vintage", "mkt_uid", "item", "month", "obs", "c", "h", "l", "trust", "gap",
                                     "dist_obs_km", "n_obs_month"])
    stack["item"] = stack["item"].astype(str)
    latest_v = stack.vintage.max()
    need = set(h.v_final) | {latest_v}
    sh = obs_share(stack[stack.vintage.isin(need)])
    h = h.merge(sh.rename(columns={"vintage": "v_final", "obs_share": "share_final"})[
        ["v_final", "mkt_uid", "item", "month", "share_final"]], on=["v_final", "mkt_uid", "item", "month"], how="left")
    h["r"] = np.log(h.y_eval / h.c_final)
    X = features(h, "gap_final", "dist_obs_final", "trust_final", "share_final", "v_final")

    # four-way time split by the release in which the truth arrived (whole releases per block):
    # train (first 60%) | select (next 10%, model choice) | calib (next 10%, conformal) | test (last 20%)
    qs_ = h.v_truth.quantile([0.6, 0.7, 0.8]).tolist()
    v = pd.to_datetime(h.v_truth)
    h["block"] = np.select([v <= qs_[0], v <= qs_[1], v <= qs_[2]], ["train", "select", "calib"], "test")
    tr, se, ca, te = (h.block == k for k in ("train", "select", "calib", "test"))
    grp = (h.stratum + "|" + gap_bin(h.gap_final)).to_numpy()

    models = {}
    for a in LEVELS:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=40, random_state=SEED,
                                          categorical_features=[FEATS.index("item_code")])
        models[a] = m.fit(X[tr], h.r[tr])
    Qgb = pd.DataFrame(np.sort(np.column_stack([models[a].predict(X) for a in LEVELS]), axis=1), index=h.index, columns=LEVELS)
    unc = {a: float(h.r[tr].quantile(a)) for a in LEVELS}
    Qun = pd.DataFrame({a: np.full(len(h), unc[a]) for a in LEVELS}, index=h.index)
    gq = group_quantiles(h.r[tr], grp[tr.to_numpy()], LEVELS, min_group=30)
    Qgr = pd.DataFrame({a: [gq.get(g, gq["__pooled__"])[a] for g in grp] for a in LEVELS}, index=h.index)
    cands = {"quantile_gb": Qgb, "group_empirical": Qgr, "unconditional": Qun}
    sel_score = {k: np.mean([pinball(h.r[se], Q[a][se], a) for a in LEVELS]) for k, Q in cands.items()}
    chosen = min(sel_score, key=sel_score.get)
    test_score = {k: np.mean([pinball(h.r[te], Q[a][te], a) for a in LEVELS]) for k, Q in cands.items()}
    mu, sd = h.r[tr].mean(), h.r[tr].std()
    test_score["gaussian"] = np.mean([pinball(h.r[te], mu + norm.ppf(a) * sd, a) for a in LEVELS])

    res, permonth, thresholds = [], [], {}
    for cov, (alo, ahi) in {0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}.items():
        for name, Q in cands.items():
            sc = cqr_scores(h.r[ca], Q[alo][ca], Q[ahi][ca])
            th = mondrian_thresholds(sc, grp[ca.to_numpy()], alpha=1 - cov, min_group=30)
            thresholds[(name, cov)] = th
            thr = np.array([th.get(g, th["__pooled__"]) for g in grp])
            lo, hi = cqr_intervals(Q[alo], Q[ahi], thr)
            inside = (h.r >= lo) & (h.r <= hi)
            for sub, mask in {"all": te, "forecast": te & (h.stratum == "forecast"),
                              "cross_sectional": te & (h.stratum == "cross_sectional")}.items():
                res.append({"nominal": cov, "method": f"{name}+conformal", "selected": name == chosen, "subset": sub,
                            "n_test": int(mask.sum()), "coverage": inside[mask].mean(),
                            "mean_width_logpts": 100 * np.asarray(hi - lo)[mask.to_numpy()].mean()})
            if name == chosen:
                pm = inside[te].groupby(h.month[te].dt.to_period("M")).agg(["mean", "size"])
                permonth.append(pm.assign(nominal=cov).reset_index())
        abs_q = conformal_quantile(np.abs(h.r[ca]), alpha=1 - cov)
        z = norm.ppf(0.5 + cov / 2)
        for name, (l, u) in {"constant_width_conformal": (-abs_q, abs_q), "gaussian": (mu - z * sd, mu + z * sd)}.items():
            inside = (h.r >= l) & (h.r <= u)
            res.append({"nominal": cov, "method": name, "selected": False, "subset": "all", "n_test": int(te.sum()),
                        "coverage": inside[te].mean(), "mean_width_logpts": 100 * (u - l)})
    # online adaptive conformal (ACI): process test releases in order; each release reveals truth for its cells
    aci_state = {}
    rel_order = sorted(h.v_truth[te].unique())
    for cov, (alo, ahi) in {0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}.items():
        for name, Q in cands.items():
            sc_all = cqr_scores(h.r, Q[alo], Q[ahi])
            # decomposition rows: static pooled threshold (no Mondrian groups) and expanding pool at fixed alpha
            th_static = conformal_quantile(sc_all[ca.to_numpy()], 1 - cov)
            ins = sc_all[te.to_numpy()] <= th_static
            res.append({"nominal": cov, "method": f"{name}+static_pooled", "selected": name == chosen, "subset": "all",
                        "n_test": int(te.sum()), "coverage": float(ins.mean()),
                        "mean_width_logpts": float(np.mean(100 * (Q[ahi].to_numpy()[te.to_numpy()] - Q[alo].to_numpy()[te.to_numpy()] + 2 * th_static)))})
            pool_fx, cov_fx, wid_fx = list(sc_all[ca.to_numpy()]), [], []
            for vt in rel_order:
                mk = (te & (h.v_truth == vt)).to_numpy()
                thf = conformal_quantile(np.asarray(pool_fx), 1 - cov)
                cov_fx.extend((sc_all[mk] <= thf).tolist())
                wid_fx.extend((100 * (Q[ahi].to_numpy()[mk] - Q[alo].to_numpy()[mk] + 2 * thf)).tolist())
                pool_fx.extend(sc_all[mk].tolist())
            res.append({"nominal": cov, "method": f"{name}+expanding_fixed_alpha", "selected": name == chosen,
                        "subset": "all", "n_test": len(cov_fx), "coverage": float(np.mean(cov_fx)),
                        "mean_width_logpts": float(np.mean(wid_fx))})
            pool, alpha = list(sc_all[ca.to_numpy()]), 1 - cov
            covered, widths, months_cov = [], [], []
            for vt in rel_order:
                mk = (te & (h.v_truth == vt)).to_numpy()
                th = conformal_quantile(np.asarray(pool), alpha)
                inside = sc_all[mk] <= th
                covered.extend(inside.tolist())
                widths.extend((100 * (Q[ahi].to_numpy()[mk] - Q[alo].to_numpy()[mk] + 2 * th)).tolist())
                alpha = aci_next_alpha(alpha, 1 - cov, 1 - inside.mean(), gamma=GAMMA)
                pool.extend(sc_all[mk].tolist())
            aci_state[(name, cov)] = (pool, alpha)
            res.append({"nominal": cov, "method": f"{name}+ACI", "selected": name == chosen, "subset": "all",
                        "n_test": len(covered), "coverage": float(np.mean(covered)),
                        "mean_width_logpts": float(np.mean(widths))})
    band = te & h.l_final.notna() & h.h_final.notna()
    bl, bu = np.log(h.l_final / h.c_final), np.log(h.h_final / h.c_final)
    res.append({"nominal": np.nan, "method": "published_band", "selected": False, "subset": "all",
                "n_test": int(band.sum()), "coverage": ((h.r >= bl) & (h.r <= bu))[band].mean(),
                "mean_width_logpts": 100 * (bu - bl)[band].mean()})
    qs = {"selection_block": sel_score, "test_block": test_score, "chosen": chosen}
    t = pd.DataFrame(res)
    pm = pd.concat(permonth)
    t1, t2 = qs_[0], qs_[2]

    # deliverable: imputed cells in the latest release, inside the hold-out covariate support
    L = stack[(stack.vintage == latest_v) & ~stack.obs & stack.c.notna()].copy()
    L = L.merge(sh[sh.vintage == latest_v][["mkt_uid", "item", "month", "obs_share"]], on=["mkt_uid", "item", "month"], how="left")
    L["stratum"] = np.where(L.n_obs_month == 0, "forecast", "cross_sectional")
    XL = features(L.assign(vint=latest_v), "gap", "dist_obs_km", "trust", "obs_share", "vint")
    ever = stack[(stack.vintage == latest_v) & stack.obs].groupby(["mkt_uid", "item"]).size().rename("n_obs").reset_index()
    L = L.merge(ever, on=["mkt_uid", "item"], how="left")
    sup = ((XL.gap >= X.gap.min()) & (XL.gap <= X.gap.max()) & (XL.vintage_age <= X.vintage_age.max())
           & (XL.vintage_age >= X.vintage_age.min()) & (XL.dist_obs.isna() | (XL.dist_obs <= X.dist_obs.max()))
           & (L.n_obs.fillna(0) > 0))
    gl = (L.stratum + "|" + gap_bin(L.gap)).to_numpy()
    if chosen == "quantile_gb":
        QL = pd.DataFrame(np.sort(np.column_stack([models[a].predict(XL) for a in LEVELS]), axis=1), index=L.index, columns=LEVELS)
    elif chosen == "group_empirical":
        QL = pd.DataFrame({a: [gq.get(g, gq["__pooled__"])[a] for g in gl] for a in LEVELS}, index=L.index)
    else:
        QL = pd.DataFrame({a: np.full(len(L), unc[a]) for a in LEVELS}, index=L.index)
    deliver = L[["mkt_uid", "item", "month", "c", "stratum", "gap"]].copy()
    deliver["inside_support"] = sup.to_numpy()
    for cov, (alo, ahi) in {0.80: (0.10, 0.90), 0.90: (0.05, 0.95)}.items():
        pool, alpha_final = aci_state[(chosen, cov)]  # ACI state after the last test release
        thr = np.full(len(L), conformal_quantile(np.asarray(pool), alpha_final))
        lo, hi = cqr_intervals(QL[alo], QL[ahi], thr)
        k = int(cov * 100)
        deliver[f"lo{k}"] = np.where(sup, deliver.c * np.exp(lo), np.nan)
        deliver[f"hi{k}"] = np.where(sup, deliver.c * np.exp(hi), np.nan)

    OUT.mkdir(parents=True, exist_ok=True)
    t["n_test_releases"] = len(rel_order)  # ACI updates alpha once per test release
    t.to_csv(OUT / "T7_uncertainty.csv", index=False)
    pm.to_csv(OUT / "T7b_coverage_by_month.csv", index=False)
    pd.DataFrame({"selection_block": qs["selection_block"], "test_block": qs["test_block"]}).assign(
        chosen=lambda d: d.index == chosen).to_csv(OUT / "T7c_quantile_scores.csv")
    deliver.to_csv(OUT / f"rtfp_bgd_calibrated_intervals_{latest_v.date()}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"split: train <= {qs_[0].date()} < select <= {qs_[1].date()} < calib <= {qs_[2].date()} < test; "
          f"n = {tr.sum()}/{se.sum()}/{ca.sum()}/{te.sum()}; chosen on selection block: {chosen}")
    print(t.round(3).to_string(index=False))
    print("avg quantile score, selection block:", {k: round(v, 5) for k, v in sel_score.items()})
    print("avg quantile score, test block:", {k: round(v, 5) for k, v in test_score.items()})
    print("coverage by test month (cqr, 80%):", pm[pm.nominal == 0.8]["mean"].describe()[["count", "mean", "min", "max", "std"]].round(3).to_dict())
    print(f"deliverable: {len(deliver)} imputed cells in {latest_v.date()}; inside support {deliver.inside_support.sum()}")


if __name__ == "__main__":
    main()
