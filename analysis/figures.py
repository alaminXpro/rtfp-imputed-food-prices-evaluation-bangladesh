"""Publication figures 1-6 for the co-author, built from EXISTING outputs only (read-only inputs).

Inputs : data/interim/{panel_latest,vintages,holdout,change_pairs}.parquet
         outputs/tables/{T2_levels,T3_local_changes,T4b_error_by_trust,T7_uncertainty,T7b_coverage_by_month}.csv
Outputs: <fig-dir>/Fig{1..6}_*.png (300 dpi), .pdf, .csv (the plotted numbers: the figure's table view)
         <notes>  (default outputs/results_notes/FIGURES.md): one caption per figure with exact source and N.

Every number printed in a caption is computed here. Values that also appear in a table are recomputed from the
interim data and cross-checked against that table; the script stops on any mismatch.

    Fig1  example series: survey values vs published series 2023-2026, hold-out estimates highlighted
    Fig2  observed vs imputed local monthly change (clean strata, both-imputed pairs, own-mu), MZ slope lines
    Fig3  revision paths: imputed value relative to the later survey value, by releases before the truth arrived
    Fig4  absolute level error binned by gap and by RTFP trust score (clean strata)
    Fig5  level correlation (toggle vs clean cells) next to MAPE and local-change correlation
    Fig6  coverage of the published band vs calibrated 80% intervals

Run from the project root:  .venv/bin/python -I analysis/figures.py [--fig-dir DIR] [--notes PATH] [--boot B]
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["rice", "wheat_flour", "oil", "lentils"]
ITEM_LABEL = {"rice": "Rice", "wheat_flour": "Wheat flour", "oil": "Palm oil", "lentils": "Lentils"}
UNIT = {"rice": "Tk per kg", "wheat_flour": "Tk per kg", "oil": "Tk per litre", "lentils": "Tk per kg"}
CLEAN = ["cross_sectional", "forecast"]
KEY = ["mkt_uid", "month", "item"]
FIGURE_STEMS = ["Fig1_example_series", "Fig2_local_change_scatter", "Fig3_revision_paths",
                "Fig4_error_vs_gap_trust", "Fig5_level_vs_local_metrics", "Fig6_interval_coverage"]

# Colour roles (dataviz reference palette, light mode). Categorical slots 1-3 validated all-pairs on white with
# scripts/validate_palette.js: worst CVD dE 9.2, normal-vision dE 24.0; aqua is < 3:1 contrast on white, so every
# chart that uses it carries a legend and a second encoding (marker shape).
STRATUM_COLOR = {"cross_sectional": "#2a78d6", "forecast": "#eb6834", "toggle": "#1baf7a"}
STRATUM_MARKER = {"cross_sectional": "o", "forecast": "s", "toggle": "D"}
STRATUM_LABEL = {"cross_sectional": "Cross-sectional", "forecast": "Forecast", "toggle": "Toggle"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

GAP_EDGES, GAP_LABELS = [0, 1, 2, 3, 6, 12, np.inf], ["1", "2", "3", "4-6", "7-12", "13+"]
TRUST_EDGES, TRUST_LABELS = [0, 5.999, 7.999, 9.999, 10], ["<6", "6.0-7.9", "8.0-9.9", "10"]  # = T4b bins
PATH_MIN_N = 30  # Fig3: a (item, stratum, k) point is drawn only with at least this many cells


# ---------------------------------------------------------------- pure helpers (tested on toy data)

def pick_example_series(clean: pd.DataFrame) -> list:
    """Fig1 selection rule, applied to clean hold-out cells (one row per cell).

    (a) the market-item series with the most clean hold-out cells; ties: more cross-sectional cells, then
        alphabetical by mkt_uid and item.
    (b) among the remaining series, the one with the most cross-sectional cells (the headline stratum);
        ties: more clean cells, then alphabetical.
    """
    g = clean.groupby(["mkt_uid", "item"]).agg(n_clean=("month", "size"),
                                                n_cs=("stratum", lambda s: int((s == "cross_sectional").sum())))
    g = g.reset_index()
    a = g.sort_values(["n_clean", "n_cs", "mkt_uid", "item"], ascending=[False, False, True, True]).iloc[0]
    rest = g[~((g.mkt_uid == a.mkt_uid) & (g["item"] == a["item"]))]
    b = rest.sort_values(["n_cs", "n_clean", "mkt_uid", "item"], ascending=[False, False, True, True]).iloc[0]
    rules = ["most clean hold-out cells", "most cross-sectional hold-out cells"]
    return [{"mkt_uid": r.mkt_uid, "item": r["item"], "n_clean": int(r.n_clean), "n_cs": int(r.n_cs), "rule": rule}
            for r, rule in zip([a, b], rules)]


def revision_paths(stack: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Every imputed value a hold-out cell received before its survey value arrived.

    k = 1 is the last imputing release (v_final), k = 2 the one before, ... counted over releases that published
    an imputed value for the cell (releases lacking the item have no row and are not counted).
    err_logpct = 100 * log(c_v / y_eval).
    """
    m = stack[KEY + ["vintage", "c", "obs"]].merge(holdout[KEY + ["stratum", "v_final", "v_truth", "y_eval"]], on=KEY)
    m = m[(m.vintage < m.v_truth) & ~m.obs.astype(bool) & m.c.notna()].copy()
    m["k"] = m.groupby(KEY).vintage.rank(ascending=False, method="first").astype(int)
    k1 = m[m.k == 1]
    if not (k1.vintage == k1.v_final).all():
        raise ValueError("k = 1 release differs from v_final for some cells")
    m["err_logpct"] = 100 * np.log(m.c / m.y_eval)
    return m[KEY + ["stratum", "vintage", "k", "c", "y_eval", "err_logpct"]].reset_index(drop=True)


def path_summary(paths: pd.DataFrame, min_n: int = PATH_MIN_N) -> pd.DataFrame:
    s = paths.groupby(["item", "stratum", "k"]).err_logpct.agg(
        n="size", median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).reset_index()
    return s[s.n >= min_n].reset_index(drop=True)


def gap_bin(gap: pd.Series) -> pd.Series:
    return pd.cut(gap, GAP_EDGES, labels=GAP_LABELS)


def trust_bin(trust: pd.Series) -> pd.Series:
    return pd.cut(trust, TRUST_EDGES, labels=TRUST_LABELS)


def wilson(p: float, n: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval for a binomial share p observed on n trials."""
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


# ---------------------------------------------------------------- formatting and checks

def fint(n) -> str:
    return f"{int(n):,}"


def f2(x) -> str:
    return f"{x:.2f}"


def f1(x) -> str:
    return f"{x:.1f}"


def num(x, d=2) -> str:
    """Number for in-figure text, with a typographic minus sign."""
    return f"{x:.{d}f}".replace("-", "\u2212")


def ym(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m")


def dash(label: str) -> str:
    return label.replace("-", "–")


def check_close(a, b, what: str, tol: float = 1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.allclose(a, b, rtol=tol, atol=tol, equal_nan=True):
        raise AssertionError(f"cross-check failed for {what}: {a} vs {b}")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- plotting setup

def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.titleweight": "regular", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False, "lines.linewidth": 1.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.default": "regular",
    })
    return plt


def panel_title(ax, letter: str, text: str):
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc="left")


def save(fig, fig_dir: Path, stem: str, data: pd.DataFrame):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, metadata={"Software": None})
    fig.savefig(fig_dir / f"{stem}.pdf", metadata={"Creator": None, "Producer": None, "CreationDate": None})
    data.to_csv(fig_dir / f"{stem}.csv", index=False)
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------- Figure 1

def fig1(plt, panel, clean, fig_dir, start="2023-01-01"):
    import matplotlib.dates as mdates

    picks = pick_example_series(clean)
    c = clean.assign(ape=100 * np.abs(clean.c_final / clean.y_eval - 1))
    series_mape = c.groupby(["mkt_uid", "item"]).ape.mean()
    names = panel.drop_duplicates("mkt_uid").set_index("mkt_uid").mkt_name
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    rows, facts = [], []
    for ax, letter, pk in zip(axes, "ab", picks):
        s = panel[(panel.mkt_uid == pk["mkt_uid"]) & (panel["item"] == pk["item"]) & (panel.month >= start)]
        s = s.sort_values("month")
        hc = c[(c.mkt_uid == pk["mkt_uid"]) & (c["item"] == pk["item"])].sort_values("month")
        ax.plot(s.month, s.c, color=MUTED, lw=1.5, zorder=2, label="Published series, latest release")
        o = s[s.obs]
        ax.scatter(o.month, o.y, s=14, color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=4,
                   label="Survey value, latest release")
        for st in CLEAN:
            x = hc[hc.stratum == st]
            if x.empty:
                continue
            ax.vlines(x.month, np.minimum(x.c_final, x.y_eval), np.maximum(x.c_final, x.y_eval),
                      color=STRATUM_COLOR[st], lw=0.8, alpha=0.8, zorder=3)
            ax.scatter(x.month, x.c_final, s=22, marker=STRATUM_MARKER[st], color=STRATUM_COLOR[st],
                       edgecolor=SURFACE, linewidth=0.6, zorder=5,
                       label=f"Hold-out cell: estimate before the survey value arrived ({STRATUM_LABEL[st].lower()})")
        name = names[pk["mkt_uid"]]
        panel_title(ax, letter, f"{name}, {ITEM_LABEL[pk['item']].lower()}: {pk['rule']}")
        ax.set_ylabel(f"Price ({UNIT[pk['item']]})")
        mape = series_mape[(pk["mkt_uid"], pk["item"])]
        pct = 100 * (series_mape <= mape).mean()
        facts.append({**pk, "name": name, "n_survey": int(s.obs.sum()), "n_imputed_latest": int((~s.obs).sum()),
                      "n_months": len(s), "n_hold_cs": int((hc.stratum == "cross_sectional").sum()),
                      "n_hold_fc": int((hc.stratum == "forecast").sum()), "mape": mape, "mape_pct": pct,
                      "hold_min": hc.month.min(), "hold_max": hc.month.max(),
                      "n_exact_zero": int((np.abs(hc.c_final / hc.y_eval - 1) < 1e-12).sum()),
                      "n_c_ne_y": int((np.abs(o.c / o.y - 1) > 0.01).sum())})
        out = s[["mkt_uid", "item", "month", "y", "c", "obs"]].rename(
            columns={"y": "y_latest", "c": "c_latest", "obs": "observed_latest"})
        out = out.merge(hc[["month", "stratum", "v_final", "c_final", "y_eval"]], on="month", how="left")
        rows.append(out.assign(panel=letter))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    h, lab = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh), lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, fig_dir, FIGURE_STEMS[0], pd.concat(rows, ignore_index=True))
    return {"facts": facts, "n_series": len(series_mape), "median_series_mape": series_mape.median(),
            "start": start, "end": panel.month.max()}


# ---------------------------------------------------------------- Figure 2

def fig2(plt, pairs, t3, fig_dir):
    from rtfp.metrics import local_components, local_stats

    p = pairs[pairs.stratum.isin(CLEAN) & (pairs.pair_type == "both_imputed")]
    e = local_components(p, mode="own_mu")
    e = e.assign(x=100 * e.e_pub, y=100 * e.e_obs)
    q99 = np.quantile(np.abs(np.r_[e.x, e.y]), 0.99)
    lim = 5 * np.ceil(q99 / 5)
    stats, clipped = [], []
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 8.0), sharex=True, sharey=True)
    for ax, letter, it in zip(axes.flat, "abcd", ITEMS):
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax.axvline(0, color=AXIS, lw=0.6, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, zorder=1, label="45° line")
        for st in ["forecast", "cross_sectional"]:
            g = e[(e["item"] == it) & (e.stratum == st)]
            s = local_stats(g.e_pub, g.e_obs)
            ref = t3[(t3.stratum == st) & (t3.pair_type == "both_imputed") & (t3["mode"] == "own_mu") & (t3["item"] == it)]
            check_close([s["n"], s["beta_mz"], s["corr"]], [ref.n.iloc[0], ref.beta_mz.iloc[0], ref["corr"].iloc[0]],
                        f"Fig2 {it} {st} vs T3")
            a = g.y.mean() - s["beta_mz"] * g.x.mean()
            check_close(g.x.var(ddof=0) / g.y.var(ddof=0), ref.lam.iloc[0], f"Fig2 {it} {st} variance ratio vs T3 lam")
            out_mask = (g.x.abs() > lim) | (g.y.abs() > lim)
            outside = int(out_mask.sum())
            clipped.append(g[out_mask][["item", "stratum", "mkt_uid", "month", "x", "y"]])
            stats.append({"item": it, "stratum": st, "n": s["n"], "beta_mz": s["beta_mz"], "corr": s["corr"],
                          "beta_lo": ref.beta_mz_lo.iloc[0], "beta_hi": ref.beta_mz_hi.iloc[0],
                          "sd_pub": g.x.std(ddof=0), "sd_obs": g.y.std(ddof=0), "outside": outside})
            ax.scatter(g.x, g.y, s=10, marker=STRATUM_MARKER[st], color=STRATUM_COLOR[st], alpha=0.6,
                       edgecolor=SURFACE, linewidth=0.3, zorder=2 if st == "forecast" else 3)
            xs = np.array([g.x.min(), g.x.max()])
            ax.plot(xs, a + s["beta_mz"] * xs, color=STRATUM_COLOR[st], lw=1.5, zorder=4,
                    label=f"{STRATUM_LABEL[st]}: $\\beta_{{MZ}}$ {num(s['beta_mz'])}, r {num(s['corr'])}, n {s['n']}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        panel_title(ax, letter, ITEM_LABEL[it])
        hh, ll = ax.get_legend_handles_labels()
        # legend below the axes so that no point is hidden; bottom row clears the tick labels
        ax.legend(hh[1:] + hh[:1], ll[1:] + ll[:1], loc="upper left", handlelength=1.2, borderaxespad=0,
                  bbox_to_anchor=(0, -0.03 if letter in "ab" else -0.1))
    fig.supxlabel("Imputed local change, published in the imputing release (log points)", fontsize=8)
    fig.supylabel("Survey local change, latest release (log points)", fontsize=8)
    fig.tight_layout()
    data = e[["item", "stratum", "mkt_uid", "month", "v_final", "x", "y"]].rename(
        columns={"x": "e_pub_logpts", "y": "e_obs_logpts"})
    save(fig, fig_dir, FIGURE_STEMS[1], data)
    return {"stats": pd.DataFrame(stats), "lim": lim, "q99": q99, "n": len(e), "n_input": len(p),
            "clipped": pd.concat(clipped, ignore_index=True).sort_values(["item", "stratum", "month"])}


# ---------------------------------------------------------------- Figure 3

def fig3(plt, stack, clean, t2, fig_dir):
    paths = revision_paths(stack, clean)
    k1 = paths[paths.k == 1].groupby(["item", "stratum"]).size()
    for (it, st), n in k1.items():
        ref = t2[(t2["item"] == it) & (t2.stratum == st) & (t2.estimate == "c_final")].n.iloc[0]
        check_close(n, ref, f"Fig3 cells {it} {st} vs T2")
    summ = path_summary(paths)
    kmax = int(summ.k.max())
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharex=True, sharey=True)
    for ax, letter, it in zip(axes.flat, "abcd", ITEMS):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for st in ["forecast", "cross_sectional"]:
            g = summ[(summ["item"] == it) & (summ.stratum == st)].sort_values("k")
            if g.empty:
                continue
            ax.fill_between(g.k, g.q25, g.q75, color=STRATUM_COLOR[st], alpha=0.12, linewidth=0, zorder=2)
            ax.plot(g.k, g["median"], color=STRATUM_COLOR[st], marker=STRATUM_MARKER[st], ms=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3,
                    label=f"{STRATUM_LABEL[st]}: median (line) and interquartile range (band)")
        ax.set_xlim(kmax + 0.5, 0.5)
        ax.set_xticks(range(1, kmax + 1))
        panel_title(ax, letter, f"{ITEM_LABEL[it]}: {fint(k1.get((it, 'cross_sectional'), 0))} cross-sectional, "
                                f"{fint(k1.get((it, 'forecast'), 0))} forecast cells")
    hh, ll = axes.flat[0].get_legend_handles_labels()
    fig.legend(hh[::-1], ll[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=1.6)
    fig.supxlabel("Releases before the survey value arrived (1 = the last imputing release, judged in Tables 2-3)",
                  fontsize=8)
    fig.supylabel("Imputed value vs later survey value, 100·log(ĉ/y)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, fig_dir, FIGURE_STEMS[2], summ)
    return {"paths": paths, "summ": summ, "k1": k1, "kmax": kmax}


# ---------------------------------------------------------------- Figure 4

def binned_error(d, bins, boot, seed, markets, months):
    from rtfp.metrics import pigeonhole_weights

    codes = bins.cat.codes.to_numpy()
    ok = codes >= 0
    d, codes = d[ok], codes[ok]
    k = len(bins.cat.categories)
    err = d.abs_err_pct.to_numpy()
    draws = []
    for b in range(boot):
        w = pigeonhole_weights(d, np.random.default_rng(seed + b), markets, months)
        sw = np.bincount(codes, weights=w, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            draws.append(np.bincount(codes, weights=w * err, minlength=k) / sw)
    draws = np.array(draws)
    out = pd.DataFrame({"bin": bins.cat.categories, "n": np.bincount(codes, minlength=k)})
    out["mean"] = [err[codes == i].mean() if (codes == i).any() else np.nan for i in range(k)]
    out["median"] = [np.median(err[codes == i]) if (codes == i).any() else np.nan for i in range(k)]
    out["n_markets"] = [d.mkt_uid[codes == i].nunique() for i in range(k)]
    out["mean_lo"] = np.nanquantile(draws, 0.025, axis=0)
    out["mean_hi"] = np.nanquantile(draws, 0.975, axis=0)
    return out


def fig4(plt, clean, t4b, fig_dir, boot, seed=20261012):
    d = clean.assign(abs_err_pct=100 * np.abs(np.log(clean.c_final / clean.y_eval)))
    markets, months = sorted(set(d.mkt_uid)), sorted(set(d.month))
    gb = binned_error(d, gap_bin(d.gap_final), boot, seed, markets, months)
    tb = binned_error(d, trust_bin(d.trust_final), boot, seed, markets, months)
    check_close(tb[["n", "mean", "median"]].to_numpy(),
                t4b[["n", "mean_abs_err_pct", "median_abs_err_pct"]].to_numpy(), "Fig4 trust bins vs T4b")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True, gridspec_kw={"width_ratios": [6, 4]})
    for ax, letter, b, xl in [(axes[0], "a", gb, "Gap: months since the market's last survey value (imputing release)"),
                              (axes[1], "b", tb, "RTFP trust score of the cell (imputing release)")]:
        x = np.arange(len(b))
        ax.errorbar(x - 0.1, b["mean"], yerr=[b["mean"] - b.mean_lo, b.mean_hi - b["mean"]], fmt="o", ms=5, color=INK,
                    ecolor=INK, elinewidth=1.0, capsize=0, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    label="Mean, 95% CI (two-way cluster bootstrap)", zorder=3)
        ax.scatter(x + 0.1, b["median"], s=24, marker="D", facecolor=SURFACE, edgecolor=INK2, linewidth=1.0,
                   label="Median", zorder=4)
        ax.set_xticks(x, [f"{dash(str(l))}\nn {fint(n)}" for l, n in zip(b.bin, b.n)])
        ax.set_xlabel(xl)
        ax.set_xlim(-0.5, len(b) - 0.5)
        ax.grid(axis="x", visible=False)
        panel_title(ax, letter, "Absolute level error by gap" if letter == "a" else "Absolute level error by trust score")
    axes[0].set_ylabel("Absolute level error, 100·|log(ĉ/y)|")
    axes[0].set_ylim(bottom=0)
    hh, ll = axes[0].get_legend_handles_labels()
    order = [ll.index(x) for x in sorted(ll, key=lambda t: not t.startswith("Mean"))]
    axes[0].legend([hh[i] for i in order], [ll[i] for i in order], loc="upper left")
    fig.tight_layout()
    data = pd.concat([gb.assign(panel="a_gap"), tb.assign(panel="b_trust")], ignore_index=True)
    data["bin"] = data["bin"].astype(str)
    save(fig, fig_dir, FIGURE_STEMS[3], data)
    return {"gap": gb, "trust": tb, "n": len(d), "n_gap_na": int(d.gap_final.isna().sum()), "boot": boot, "seed": seed}


# ---------------------------------------------------------------- Figure 5

def fig5(plt, holdout, t2, t3, fig_dir):
    from rtfp.metrics import level_stats

    strata = ["toggle", "forecast", "cross_sectional"]
    h = holdout[holdout.y_eval.notna() & holdout.c_final.notna() & holdout.stratum.isin(strata)]
    rows = []
    for it in ITEMS:
        for st in strata:
            g = h[(h["item"] == it) & (h.stratum == st)]
            ref = t2[(t2["item"] == it) & (t2.stratum == st) & (t2.estimate == "c_final")].iloc[0]
            corr = np.corrcoef(np.log(g.c_final), np.log(g.y_eval))[0, 1]
            mape = level_stats(g.c_final, g.y_eval)["mape"]
            check_close([len(g), corr, mape], [ref.n, ref.corr_log_levels, ref.mape], f"Fig5 {it} {st} vs T2")
            r = {"item": it, "stratum": st, "n_cells": len(g), "corr_log_levels": corr, "mape": mape,
                 "mape_lo": ref.mape_lo, "mape_hi": ref.mape_hi, "sd_log_level": np.log(g.y_eval).std(),
                 "month_min": ym(g.month.min()), "month_max": ym(g.month.max()),
                 "n_markets": g.mkt_uid.nunique(), "n_months": g.month.nunique()}
            lc = t3[(t3.stratum == st) & (t3.pair_type == "both_imputed") & (t3["mode"] == "own_mu") & (t3["item"] == it)]
            if len(lc):
                r.update({"n_change_pairs": int(lc.n.iloc[0]), "corr_change": lc["corr"].iloc[0],
                          "corr_change_lo": lc.corr_lo.iloc[0], "corr_change_hi": lc.corr_hi.iloc[0]})
            rows.append(r)
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
    off = {"toggle": -0.22, "forecast": 0.0, "cross_sectional": 0.22}
    ypos = {it: len(ITEMS) - 1 - i for i, it in enumerate(ITEMS)}
    specs = [("a", "Log-level correlation", "corr_log_levels", None, None, (0, 1.02)),
             ("b", "Level MAPE (%)", "mape", "mape_lo", "mape_hi", None),
             ("c", "Local-change correlation", "corr_change", "corr_change_lo", "corr_change_hi", (-0.6, 1.02))]
    for ax, (letter, title, col, lo, hi, xlim) in zip(axes, specs):
        for st in strata:
            g = t[t.stratum == st]
            if col not in g or g[col].isna().all():
                continue
            y = g["item"].map(ypos) + off[st]
            if lo:
                ok = g[lo].notna()
                ax.hlines(y[ok], g[lo][ok], g[hi][ok], color=STRATUM_COLOR[st], lw=1.0, zorder=2)
            ax.scatter(g[col], y, s=26, marker=STRATUM_MARKER[st], color=STRATUM_COLOR[st], edgecolor=SURFACE,
                       linewidth=0.6, zorder=3, label=STRATUM_LABEL[st])
        if xlim:
            ax.set_xlim(*xlim)
        if letter == "c":
            ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        if letter == "b":
            ax.set_xlim(left=0)
        ax.set_yticks(list(ypos.values()), [ITEM_LABEL[i] for i in ypos])
        ax.grid(axis="y", visible=False)
        panel_title(ax, letter, title)
    h_, l_ = axes[0].get_legend_handles_labels()
    span = {STRATUM_LABEL[st]: f"{t[t.stratum == st].month_min.min()[:4]}\u2013{t[t.stratum == st].month_max.max()[:4]}"
            for st in strata}
    lab = {"Toggle": f"Toggle (re-released values, {span['Toggle']})",
           "Forecast": f"Forecast (clean, {span['Forecast']})",
           "Cross-sectional": f"Cross-sectional (clean, {span['Cross-sectional']})"}
    order = [l_.index(x) for x in ["Cross-sectional", "Forecast", "Toggle"]]
    fig.legend([h_[i] for i in order], [lab[l_[i]] for i in order], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, fig_dir, FIGURE_STEMS[4], t)
    return {"t": t}


# ---------------------------------------------------------------- Figure 6

CAND_LABEL = {"unconditional": "Unconditional quantiles", "group_empirical": "Group quantiles",
              "quantile_gb": "Quantile GB"}


def method_label(method: str) -> str:
    """Short in-figure label; 'static' = conformal thresholds set once, '+ ACI' = recalibrated online."""
    if method == "published_band":
        return "Published band [l_, h_]"
    if method == "constant_width_conformal":
        return "Constant width, static"
    if method == "gaussian":
        return "Gaussian"
    base, _, cal = method.partition("+")
    cal = {"ACI": " + ACI", "conformal": ", static"}.get(cal, f" + {cal}" if cal else "")
    return f"{CAND_LABEL.get(base, base)}{cal}"


def fig6(plt, t7, t7b, fig_dir):
    """Published band vs 80% intervals on the time-blocked test set (T7), and per-month coverage (T7b)."""
    m = t7[((t7.nominal == 0.8) & (t7.subset == "all")) | (t7.method == "published_band")].copy()
    sel = m[m.selected.astype(bool)]
    chosen = sorted({x.split("+")[0] for x in sel.method})
    if len(chosen) != 1:
        raise AssertionError(f"expected one selected candidate in T7, got {chosen}")
    chosen = chosen[0]
    deliverable, static = f"{chosen}+ACI", f"{chosen}+conformal"
    if not {deliverable, static, "published_band"} <= set(m.method):
        raise AssertionError(f"T7 lacks one of {deliverable}, {static}, published_band")
    pm = t7b[t7b.nominal == 0.8].copy()
    st_row = m[m.method == static].iloc[0]
    by_stratum = t7[(t7.nominal == 0.8) & (t7.method == static)].set_index("subset").n_test
    check_close(by_stratum[CLEAN].sum(), st_row.n_test, "Fig6 T7 stratum subsets vs all")
    check_close(pm["size"].sum(), st_row.n_test, "Fig6 T7b month sizes vs T7 n_test")
    check_close((pm["mean"] * pm["size"]).sum() / pm["size"].sum(), st_row.coverage,
                "Fig6 T7b pooled coverage vs T7 selected static-conformal coverage")
    m[["cov_lo", "cov_hi"]] = [wilson(p, n) for p, n in zip(m.coverage, m.n_test)]
    pm[["cov_lo", "cov_hi"]] = [wilson(p, n) for p, n in zip(pm["mean"], pm["size"])]
    first = ["published_band", deliverable, static]
    others = m[~m.method.isin(first)].sort_values("coverage", ascending=False).method.tolist()
    m = m.set_index("method").loc[first + others].reset_index()
    m["label"] = [method_label(x) + (" (released)" if x == deliverable else "") for x in m.method]
    focus = {"published_band", deliverable}
    y = np.arange(len(m))[::-1]

    fig = plt.figure(figsize=(7.0, 5.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1], hspace=0.04)
    axa, axb = top.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    axc = bottom.subplots()
    for ax, col, lo, hi in [(axa, "coverage", "cov_lo", "cov_hi"), (axb, "mean_width_logpts", None, None)]:
        for yi, (_, r) in zip(y, m.iterrows()):
            f = r.method in focus
            if lo:
                ax.hlines(yi, r[lo], r[hi], color=INK if f else MUTED, lw=1.0, zorder=2)
            ax.scatter(r[col], yi, s=30 if f else 24, color=INK if f else SURFACE, edgecolor=INK if f else MUTED,
                       linewidth=1.0, zorder=3)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(m) - 0.4)
    axa.set_yticks(y, m.label)
    for tick, meth in zip(axa.get_yticklabels(), m.method):
        tick.set_color(INK if meth in focus else INK2)
    axa.axvline(0.8, color=MUTED, lw=0.8, zorder=1)
    axa.annotate("nominal 80%", (0.8, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                 textcoords="offset points", ha="left", va="bottom", color=INK2, fontsize=7)
    axa.set_xlim(min(0.5, 0.05 * np.floor(m.cov_lo.min() / 0.05)), 1.0)
    axa.set_xlabel("Share of later survey values inside")
    panel_title(axa, "a", "Coverage (95% Wilson interval)")
    axb.set_xlim(0, 5 * np.ceil(m.mean_width_logpts.max() / 5))
    axb.set_xlabel("Mean width (log points)")
    panel_title(axb, "b", "Interval width")
    x = np.arange(len(pm))
    axc.axhline(0.8, color=MUTED, lw=0.8, zorder=1)
    axc.vlines(x, pm.cov_lo, pm.cov_hi, color=INK2, lw=1.0, zorder=2)
    axc.scatter(x, pm["mean"], s=26, color=SURFACE, edgecolor=INK2, linewidth=1.0, zorder=3)
    axc.set_xticks(x, [f"{pd.Period(mm).strftime('%b %Y')}\nn {fint(n)}" for mm, n in zip(pm.month, pm["size"])])
    axc.set_ylim(min(0.4, 0.1 * np.floor(pm.cov_lo.min() / 0.1)), 1.0)
    axc.set_xlim(-0.5, len(pm) - 0.5)
    axc.grid(axis="x", visible=False)
    axc.set_ylabel("Coverage (95% Wilson interval)")
    axc.set_xlabel("Price month of the test cells")
    panel_title(axc, "c", f"{method_label(static)} (calibrated once), by price month")
    data = pd.concat([m.assign(panel="a_b")[["panel", "method", "label", "selected", "nominal", "subset", "n_test",
                                             "coverage", "cov_lo", "cov_hi", "mean_width_logpts"]],
                      pm.assign(panel="c", method=static, label=method_label(static)).rename(
                          columns={"mean": "coverage", "size": "n_test"})[
                          ["panel", "method", "label", "nominal", "month", "n_test", "coverage", "cov_lo", "cov_hi"]]],
                     ignore_index=True)
    save(fig, fig_dir, FIGURE_STEMS[5], data)
    return {"m": m.set_index("method"), "pm": pm, "chosen": chosen, "deliverable": deliverable, "static": static,
            "n_test_stratum": by_stratum[CLEAN].astype(int).to_dict()}


# ---------------------------------------------------------------- captions (FIGURES.md)

def rel(p: Path, root: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(p)


def captions(r, root, fig_dir, inputs, latest_vintage, n_clean, clean_range):
    fd = rel(fig_dir, root)
    files = lambda stem: f"`{fd}/{stem}.png` (300 dpi), `{fd}/{stem}.pdf`; plotted values `{fd}/{stem}.csv`"
    L = ["# Figures 1-6: captions, sources and N", "",
         "*Generated by `analysis/figures.py`; do not edit by hand. Every number below is computed by that script; "
         "values that also appear in Tables 2, 3 and 4b are recomputed from the interim data and cross-checked "
         "against them, and Table 7b is cross-checked against Table 7 (the script stops on a mismatch). Statements "
         "marked 'Inference' are interpretations, not measurements.*", "",
         f"Latest RTFP release in the stack: {latest_vintage}. Clean hold-out = forecast + cross-sectional strata: "
         f"{fint(n_clean)} cells, price months {clean_range[0]} to {clean_range[1]}.", "",
         "Inputs (sha256, first 12 hex):", ""]
    L += [f"- `{rel(p, root)}` {sha12(p)}" for p in inputs]
    L += ["", "Colour key used throughout: cross-sectional = blue circles, forecast = orange squares, toggle = aqua "
          "diamonds (shape repeats the colour, so the figures read in greyscale and under colour-vision deficiency).",
          ""]

    # ---- Figure 1
    f = r[1]
    L += [f"## Figure 1 - Example series: survey values vs the published series ({FIGURE_STEMS[0]})", "",
          f"**Files.** {files(FIGURE_STEMS[0])}.", "",
          f"**Shows.** For two market-item series, {ym(f['start'])} to {ym(f['end'])}: black dots = survey values in "
          f"the latest release ({latest_vintage}); grey line = the published estimate c_ in the latest release; "
          "coloured markers = clean hold-out cells, placed at the estimate the last imputing release (v_final) "
          "published before the survey value arrived, with a thin vertical segment to that survey value.", "",
          "**Selection rule (applied by code, not by eye).** (a) The market-item series with the most clean hold-out "
          "cells; ties broken by more cross-sectional cells, then alphabetically. (b) Among the remaining series, the "
          "one with the most cross-sectional hold-out cells; ties broken by more clean cells, then alphabetically.", "",
          "**Source.** `data/interim/panel_latest.parquet` (line and dots; months >= "
          f"{ym(f['start'])}); `data/interim/holdout.parquet`, strata forecast + cross_sectional (coloured markers).",
          ""]
    nl = []
    for letter, x in zip("ab", f["facts"]):
        nl.append(f"({letter}) {x['name']}, {ITEM_LABEL[x['item']].lower()}: {x['n_months']} months shown, "
                  f"{x['n_survey']} with a survey value and {x['n_imputed_latest']} still imputed in the latest "
                  f"release; {x['n_clean']} clean hold-out cells ({x['n_hold_cs']} cross-sectional, "
                  f"{x['n_hold_fc']} forecast), price months {ym(x['hold_min'])} to {ym(x['hold_max'])}.")
    L += ["**N.** " + " ".join(nl), ""]
    kl = []
    for letter, x in zip("ab", f["facts"]):
        kl.append(f"({letter}) hold-out MAPE {f1(x['mape'])}%, at the {f1(x['mape_pct'])}th percentile of the "
                  f"{fint(f['n_series'])} market-item series with clean hold-out cells (median series MAPE "
                  f"{f1(f['median_series_mape'])}%); {x['n_exact_zero']} of its {x['n_clean']} hold-out estimates "
                  "equal the later survey value exactly; in "
                  f"{x['n_c_ne_y']} of the {x['n_survey']} months with a survey value, the latest published value "
                  "differs from it by more than 1% (black dot off the grey line).")
    L += ["**Key numbers.** " + " ".join(kl), "",
          "**Caveats.** Illustration only; the selection rule ranks series by the number of hold-out cells, not by "
          "error. In months where the latest release has a survey value the grey line mostly coincides with it "
          "(c_ equals y in most observed cells, Table A1b), which is why the published series looks validated; "
          "the coloured markers are what the release showed before the survey value was available. Where a black "
          "dot sits off the grey line, the latest published value differs from the survey value; RTFP's metadata "
          "describes outlier adjustment, which we have not verified as the cause (Table A1b).", ""]

    # ---- Figure 2
    f = r[2]
    s = f["stats"]
    tab = s.assign(item=s["item"].map(ITEM_LABEL), stratum=s.stratum.map(STRATUM_LABEL),
                   n=s.n.map(fint), beta_mz=[f"{f2(b)} [{f2(lo)}, {f2(hi)}]" for b, lo, hi in zip(s.beta_mz, s.beta_lo, s.beta_hi)],
                   corr=s["corr"].map(f2), sd_pub=s.sd_pub.map(f2), sd_obs=s.sd_obs.map(f2), outside=s.outside.map(fint))
    tab = tab[["item", "stratum", "n", "beta_mz", "corr", "sd_pub", "sd_obs", "outside"]].rename(columns={
        "beta_mz": "beta_MZ [95% CI, T3]", "corr": "r", "sd_pub": "SD imputed (log pts)",
        "sd_obs": "SD survey (log pts)", "outside": "points outside view"})
    L += [f"## Figure 2 - Observed vs imputed local monthly change ({FIGURE_STEMS[1]})", "",
          f"**Files.** {files(FIGURE_STEMS[1])}.", "",
          "**Shows.** Each point is a hold-out change pair (t-1, t) with both months imputed in the same release. "
          "x = imputed local change (published log change minus that release's observed-market mean change); "
          "y = survey local change in the latest release (log change minus the latest release's observed-market "
          "mean); log points = 100 x log change. Coloured lines = Mincer-Zarnowitz fit of y on x (slope beta_MZ, "
          "the attenuation factor when imputed changes are used as a regressor); grey line = 45 degrees. "
          f"Axes are clipped to +/-{f['lim']:.0f} log points (rule: the 99th percentile of all |values|, "
          f"{f1(f['q99'])}, rounded up to a multiple of 5); fits use all points.", "",
          "**Source.** `data/interim/change_pairs.parquet`, strata forecast + cross_sectional, pair_type "
          "both_imputed, local components from `rtfp.metrics.local_components(mode='own_mu')`; n, beta_MZ and r "
          "cross-checked against `outputs/tables/T3_local_changes.csv` (CIs are T3's two-way cluster bootstrap).", "",
          f"**N.** {fint(f['n'])} change pairs ({fint(f['n_input'])} input pairs; none dropped).", "",
          md_table(tab), "",
          f"SD imputed^2 / SD survey^2 equals T3's variance ratio lambda in every row (cross-checked). SD imputed is "
          f"below SD survey in {int((s.sd_pub < s.sd_obs).sum())} of {len(s)} rows.", "",
          f"Points outside the plotted range ({len(f['clipped'])}; included in every fit):", "",
          md_table(f["clipped"].assign(item=f["clipped"]["item"].map(ITEM_LABEL),
                                       stratum=f["clipped"].stratum.map(STRATUM_LABEL),
                                       month=f["clipped"].month.map(ym), x=f["clipped"].x.map(f1),
                                       y=f["clipped"].y.map(f1)).rename(
              columns={"x": "imputed (log pts)", "y": "survey (log pts)"})), "",
          "**Caveats.** beta_MZ and r here are not corrected for survey measurement noise (see T1 for the rho* "
          "band). The forecast stratum mixes national nowcast error into the local component. OLS slopes are "
          "sensitive to the extreme survey changes listed above (compare the wide T3 interval for wheat flour, "
          "forecast). Inference: the narrow horizontal spread of points is the variance ratio lambda < 1 of Table 3 "
          "made visible; the flat fitted lines are beta_MZ < 1.", ""]

    # ---- Figure 3
    f = r[3]
    sm = f["summ"]
    rows = []
    for it in ITEMS:
        for st in CLEAN:
            g = sm[(sm["item"] == it) & (sm.stratum == st)].sort_values("k")
            if g.empty:
                continue
            a, b = g.iloc[0], g.iloc[-1]
            rows.append({"item": ITEM_LABEL[it], "stratum": STRATUM_LABEL[st], "cells (k=1)": fint(a.n),
                         "k shown": f"1-{int(b.k)}", "median [IQR] at k=1": f"{f1(a['median'])} [{f1(a.q25)}, {f1(a.q75)}]",
                         f"median [IQR] at largest k": f"{f1(b['median'])} [{f1(b.q25)}, {f1(b.q75)}] (n {fint(b.n)})"})
    paths = f["paths"]
    L += [f"## Figure 3 - Revision paths of imputed values before the survey value arrived ({FIGURE_STEMS[2]})", "",
          f"**Files.** {files(FIGURE_STEMS[2])} (the CSV holds the plotted summary).", "",
          "**Shows.** For clean hold-out cells, every imputed value the cell received in earlier releases, relative "
          "to the survey value that arrived later: 100 x log(c_v / y), where y is the latest-release survey value. "
          "x = number of releases before the survey value arrived, counted back from the last imputing release "
          "(k = 1, the estimate judged in Tables 2-3), over releases that published the cell (releases without the "
          "item are skipped). Line = median; band = interquartile range.", "",
          "**Source.** `data/interim/vintages.parquet` (vintage, mkt_uid, month, item, c, obs) joined to "
          "`data/interim/holdout.parquet`, strata forecast + cross_sectional; k = 1 checked to equal v_final for "
          "every cell; cells at k = 1 cross-checked against n in `outputs/tables/T2_levels.csv`.", "",
          f"**N.** {fint(paths[KEY].drop_duplicates().shape[0])} cells, {fint(len(paths))} cell-release values. "
          f"A point is drawn only when at least {PATH_MIN_N} cells contribute.", "",
          md_table(pd.DataFrame(rows)), "",
          "**Caveats.** The set of cells changes with k: only cells imputed in many releases reach large k, so the "
          "curve mixes revision with composition. Not a balanced panel.", ""]

    # ---- Figure 4
    f = r[4]

    def btab(b, label):
        return md_table(pd.DataFrame({label: [dash(str(x)) for x in b.bin], "n": b.n.map(fint),
                                      "markets": b.n_markets.map(fint),
                                      "mean [95% CI]": [f"{f2(m)} [{f2(lo)}, {f2(hi)}]" for m, lo, hi in
                                                        zip(b["mean"], b.mean_lo, b.mean_hi)],
                                      "median": b["median"].map(f2)}))
    L += [f"## Figure 4 - Absolute level error by gap and by trust score ({FIGURE_STEMS[3]})", "",
          f"**Files.** {files(FIGURE_STEMS[3])}.", "",
          "**Shows.** Absolute level error 100 x |log(c_final / y_eval)| of clean hold-out cells, binned (a) by gap "
          "(months since the market's last survey value, in the imputing release) and (b) by RTFP's trust score in "
          "the imputing release. Filled dots = mean with 95% CI from a two-way (market x month) pigeonhole "
          f"bootstrap (B = {f['boot']}, seed {f['seed']}); hollow diamonds = median. Same y-axis in both panels.", "",
          "**Source.** `data/interim/holdout.parquet`, strata forecast + cross_sectional (columns c_final, y_eval, "
          "gap_final, trust_final). Trust bins are those of `outputs/tables/T4b_error_by_trust.csv`; n, mean and "
          "median per trust bin cross-checked against it.", "",
          f"**N.** {fint(f['n'])} cells; panel (a) excludes {fint(f['n_gap_na'])} cells with no earlier survey value "
          "for the market-item (gap undefined).", "",
          btab(f["gap"], "gap (months)"), "", btab(f["trust"], "trust score"), "",
          "**Caveats.** Unadjusted bins; Table 4 gives the regression with item x month fixed effects and two-way "
          "clustered SEs. Most cells have gap 1 and trust 10, so the other bins are small and some come from few "
          "markets (n and markets above).", ""]

    # ---- Figure 5
    f = r[5]
    t = f["t"]
    tab = pd.DataFrame({"item": t["item"].map(ITEM_LABEL), "stratum": t.stratum.map(STRATUM_LABEL),
                        "cells": t.n_cells.map(fint), "markets": t.n_markets, "price months": t.n_months,
                        "span": t.month_min + " to " + t.month_max, "SD log level": t.sd_log_level.map(f2),
                        "corr log levels": t.corr_log_levels.map(f2),
                        "MAPE % [95% CI]": [f"{f2(a)}" + ("" if pd.isna(lo) else f" [{f2(lo)}, {f2(hi)}]")
                                            for a, lo, hi in zip(t.mape, t.mape_lo, t.mape_hi)],
                        "corr local change [95% CI] (pairs)": [
                            "not computed" if pd.isna(c) else f"{f2(c)} [{f2(lo)}, {f2(hi)}] ({fint(n)})"
                            for c, lo, hi, n in zip(t.corr_change, t.corr_change_lo, t.corr_change_hi,
                                                    t.n_change_pairs.fillna(0))]})
    tg, cl = t[t.stratum == "toggle"], t[t.stratum.isin(CLEAN)]
    sd_order = all(tg.set_index("item").sd_log_level[it] > cl[cl["item"] == it].sd_log_level.max() for it in ITEMS)
    corr_order = all(tg.set_index("item").corr_log_levels[it] > cl[cl["item"] == it].corr_log_levels.max()
                     for it in ITEMS)
    n_zero = int(((cl.corr_change_lo < 0) & (cl.corr_change_hi > 0)).sum())
    rng = lambda x, d=2: f"{x.min():.{d}f}-{x.max():.{d}f}"
    L += [f"## Figure 5 - Level validation metrics vs local-change agreement ({FIGURE_STEMS[4]})", "",
          f"**Files.** {files(FIGURE_STEMS[4])}.", "",
          "**Shows.** By item and stratum: (a) correlation of log published vs log survey levels over hold-out cells "
          "(the Adewopo-style validation metric); (b) level MAPE with 95% CI; (c) correlation of local monthly "
          "changes (own-mu, both-imputed pairs) with 95% CI. Toggle cells (values that appeared in an earlier "
          "release, dropped and returned) span many years; clean cells are recent.", "",
          "**Source.** (a, b) `outputs/tables/T2_levels.csv`, estimate c_final, recomputed from "
          "`data/interim/holdout.parquet` and cross-checked (n, corr, MAPE); MAPE CIs from T2 (two-way cluster "
          "bootstrap; computed for clean strata only). (c) `outputs/tables/T3_local_changes.csv`, pair_type "
          "both_imputed, mode own_mu (computed for clean strata only). SD of log level and spans from "
          "`data/interim/holdout.parquet`.", "",
          f"**N.** {fint(t.n_cells.sum())} hold-out cells with a confirmed survey value in these three strata; "
          f"{fint(t.n_change_pairs.fillna(0).sum())} change pairs in panel (c).", "",
          md_table(tab), "",
          f"**Key numbers.** Log-level correlation: toggle {rng(tg.corr_log_levels)}, clean strata "
          f"{rng(cl.corr_log_levels)}. SD of log level: toggle {rng(tg.sd_log_level)}, clean {rng(cl.sd_log_level)}. "
          f"MAPE (%): toggle {rng(tg.mape)}, clean {rng(cl.mape)}. In every item the toggle stratum has "
          f"{'both the largest SD of log levels and the highest level correlation' if sd_order and corr_order else 'NOT uniformly the largest SD and highest level correlation (check the table)'}. "
          f"{n_zero} of {len(cl)} clean item-stratum local-change correlations have a 95% CI that includes zero.", "",
          "**Caveats.** Toggle cells are not a test of imputation (their survey value was already in an earlier "
          "release) and are shown only to show what the level correlation does on a long-span sample. "
          "Inference: the level correlation tracks the spread of price levels in the sample (SD column) more than the "
          "size of the errors (MAPE column), so a high level correlation can coexist with local monthly changes that "
          "the imputation does not track (panel c).", ""]

    # ---- Figure 6
    f = r[6]
    m, pm = f["m"], f["pm"]
    tab = pd.DataFrame({"method": m.label, "nominal": [("-" if pd.isna(x) else f"{x:.0%}") for x in m.nominal],
                        "test cells": m.n_test.map(fint),
                        "coverage [95% Wilson CI]": [f"{f2(c)} [{f2(lo)}, {f2(hi)}]" for c, lo, hi in
                                                     zip(m.coverage, m.cov_lo, m.cov_hi)],
                        "mean width (log pts)": m.mean_width_logpts.map(f1)})
    dv, sc, pb = m.loc[f["deliverable"]], m.loc[f["static"]], m.loc["published_band"]
    L += [f"## Figure 6 - Published band vs calibrated 80% intervals ({FIGURE_STEMS[5]})", "",
          f"**Files.** {files(FIGURE_STEMS[5])}.", "",
          "**Shows.** Time-blocked test set of clean hold-out cells (cells whose survey value arrived in the last "
          "releases). (a) Coverage = share of later survey values inside each interval, with 95% Wilson binomial "
          "intervals computed here from coverage and n (they ignore clustering by market and month, so they are too "
          "narrow); grey line = nominal 80%. (b) Mean interval width. Labels: ', static' = conformal thresholds set "
          "once on the calibration block; '+ ACI' = adaptive conformal inference, thresholds updated after each test "
          "release. Filled black = RTFP's published band [l_, h_] "
          "at the imputing release (no nominal level is assumed for it) and the released calibrated interval "
          f"(`{f['deliverable']}`: the candidate chosen on the selection block, `{f['chosen']}`, recalibrated online "
          "as each test release reveals survey values); hollow grey = the other 80% methods in T7. (c) Coverage of "
          f"the same candidate calibrated once (`{f['static']}`) by price month of the test cells.", "",
          "**Source.** `outputs/tables/T7_uncertainty.csv` (nominal 0.8, subset all, plus the published_band row; "
          "candidate from the `selected` column) and `outputs/tables/T7b_coverage_by_month.csv` (nominal 0.8; its "
          f"month sizes and pooled coverage cross-checked against the `{f['static']}` row of T7). Model and split: "
          "`analysis/t7_uncertainty.py`.", "",
          f"**N.** {fint(dv.n_test)} test cells for the calibrated interval, {fint(pb.n_test)} for the published "
          f"band; panel (c) {len(pm)} price months, {pm.month.min()} to {pm.month.max()}.", "",
          f"**Key numbers.** Published band coverage {f2(pb.coverage)} (mean width {f1(pb.mean_width_logpts)} log "
          f"points); released calibrated 80% interval {f2(dv.coverage)} (width {f1(dv.mean_width_logpts)}); same "
          f"candidate calibrated once {f2(sc.coverage)} (width {f1(sc.mean_width_logpts)}).", "",
          md_table(tab), "",
          "**By month (panel c):** " + "; ".join(
              f"{mm}: {f2(c)} (n {fint(n)})" for mm, c, n in zip(pm.month, pm["mean"], pm["size"])) + ".", "",
          "**Caveats.** Per-month coverage is available in T7b only for the once-calibrated candidate, not for the "
          "online-recalibrated interval. Pooled coverage hides month-to-month dispersion; months are not independent "
          "test sets. The test block is the last part of the release history only; it holds "
          f"{fint(f['n_test_stratum']['forecast'])} forecast and {fint(f['n_test_stratum']['cross_sectional'])} "
          "cross-sectional cells (T7 subsets), so coverage for cross-sectional cells is barely tested.", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args(argv)
    root = a.root
    fig_dir = a.fig_dir or root / "outputs" / "figures"
    notes = a.notes or root / "outputs" / "results_notes" / "FIGURES.md"
    D, T = root / "data" / "interim", root / "outputs" / "tables"
    inputs = [D / "panel_latest.parquet", D / "vintages.parquet", D / "holdout.parquet", D / "change_pairs.parquet",
              T / "T2_levels.csv", T / "T3_local_changes.csv", T / "T4b_error_by_trust.csv",
              T / "T7_uncertainty.csv", T / "T7b_coverage_by_month.csv"]

    panel = pd.read_parquet(inputs[0], columns=["mkt_uid", "mkt_name", "item", "month", "y", "c", "obs", "vintage"])
    stack = pd.read_parquet(inputs[1], columns=["vintage", "mkt_uid", "month", "item", "c", "obs"])
    holdout = pd.read_parquet(inputs[2])
    pairs = pd.read_parquet(inputs[3])
    for d in (panel, stack, holdout, pairs):
        d["item"] = d["item"].astype(str)
    t2, t3, t4b, t7, t7b = (pd.read_csv(p) for p in inputs[4:])
    clean = holdout[holdout.stratum.isin(CLEAN) & holdout.y_eval.notna() & holdout.c_final.notna()].copy()
    if len(clean) != holdout.stratum.isin(CLEAN).sum():
        raise AssertionError("clean hold-out cells without a confirmed survey value or estimate")

    plt = setup_matplotlib()
    r = {1: fig1(plt, panel, clean, fig_dir), 2: fig2(plt, pairs, t3, fig_dir),
         3: fig3(plt, stack, clean, t2, fig_dir), 4: fig4(plt, clean, t4b, fig_dir, a.boot),
         5: fig5(plt, holdout, t2, t3, fig_dir), 6: fig6(plt, t7, t7b, fig_dir)}
    latest = str(pd.Timestamp(stack.vintage.max()).date())
    if str(panel.vintage.iloc[0]) != latest:
        raise AssertionError(f"panel_latest vintage {panel.vintage.iloc[0]} != latest stack vintage {latest}")
    text = captions(r, root, fig_dir, inputs, latest, len(clean), (ym(clean.month.min()), ym(clean.month.max())))
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(text)
    print(f"wrote {len(FIGURE_STEMS)} figures to {fig_dir} and captions to {notes}")


if __name__ == "__main__":
    main()
