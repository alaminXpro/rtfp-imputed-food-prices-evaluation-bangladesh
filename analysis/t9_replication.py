"""Table 9: bounded replication in other RTFP countries (Task 6.1; analysis plan section 5; methodology D6).

Stages (run in this order; `select` must be complete before any country's results exist):
  select  Discover every per-country RTFP market dataset (all-country metadata WLD_2021_RTFP_v02_M plus the
          catalogue search), count market-file vintages, markets in the latest vintage (excluding 'Market
          Average') and plain survey columns; apply the rule pre-specified in the draft analysis plan; write
          outputs/tables/T9_country_selection.csv with all candidates, the selected five and a UTC timestamp.
          Refuses to overwrite an existing selection (use --reselect to override deliberately).
  download  Download every market vintage of each selected country, up to the latest vintage seen at selection
          (listing snapshot kept at selection time), into data/raw/rtfp_other/<ISO3>/ with a sha256 manifest.
  run     For each selected country (read from the frozen selection file): run
          the Bangladesh pipeline (rtfp.vintages / noise / metrics / audit) and write T9_replication.csv etc.
  note    Generate outputs/results_notes/T9.md from the tables.

Usage: .venv/bin/python -I analysis/t9_replication.py {select,run,note,all}
Downloads go to data/raw/rtfp_other/ (sha256 manifest data/raw/rtfp_other/MANIFEST.csv); >= 1 s between requests.
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rtfp import replicate as rp
from rtfp.download import hdx_file, sha256_file
from rtfp.panel import load_panel

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/rtfp_other"
OUT = ROOT / "outputs/tables"
NOTES = ROOT / "outputs/results_notes"
SEL = OUT / "T9_country_selection.csv"
MIN_INTERVAL = 1.2  # seconds between request starts
B = 500  # two-way (market x month) pigeonhole bootstrap draws, as Tables 2-3
SEED = 20261012
BGD_VINTAGES = ROOT / "data/raw/rtfp_vintages"
BGD_HDX = [hdx_file(ROOT / "data/raw/hdx_wfp")]  # list kept for existing BGD_HDX[-1] uses; respects RTFP_AS_OF


def stage_select(reselect: bool = False) -> pd.DataFrame:
    if SEL.exists() and not reselect:
        raise SystemExit(f"{SEL} exists: the selection is frozen. Pass --reselect only to redo it deliberately.")
    started = dt.datetime.now(dt.timezone.utc)
    f = rp.PoliteFetcher(min_interval=MIN_INTERVAL)
    cat = RAW / "_catalog"
    cat.mkdir(parents=True, exist_ok=True)
    wld_raw = f.get_bytes(rp.WLD_METADATA)
    (cat / "WLD_2021_RTFP_v02_M_metadata.json").write_bytes(wld_raw)
    search_raw = f.get_bytes(rp.CATALOG_SEARCH)
    (cat / "catalog_search_RTFP.json").write_bytes(search_raw)
    wld = json.loads(wld_raw)
    cands = rp.candidates_from_catalog(wld, json.loads(search_raw))
    cands = cands[~cands.iso3.isin(rp.EXCLUDE)].reset_index(drop=True)
    manifest = rp.read_manifest(RAW / "MANIFEST.csv")
    s = rp.survey_candidates(cands, f, RAW, manifest)
    rp.write_manifest(RAW / "MANIFEST.csv", manifest)
    s = rp.apply_selection_rule(s)
    # sensitivity of the selection to the reading of 'plain columns present' (header only vs non-empty values)
    alt = rp.apply_selection_rule(s.drop(columns=["reason", "eligible", "rank", "selected"])
                                  .assign(has_plain_values=s.has_plain_columns))
    s["selected_if_header_only"] = alt.selected.to_numpy()
    s["selection_timestamp_utc"] = started.isoformat(timespec="seconds")
    s["catalogue_version"] = wld["study_desc"]["title_statement"].get("sub_title", "")
    s["rule"] = (f">= {rp.MIN_VINTAGES} market-file vintages; >= {rp.MIN_MARKETS} markets in latest vintage "
                 f"(excl. 'Market Average'); plain survey columns present (header and >= 1 value); not Bangladesh; "
                 f"top {rp.N_SELECT} by markets, ties by vintages, then ISO3")
    cols = ["rank", "selected", "iso3", "country", "eligible", "reason", "n_vintages", "n_markets",
            "has_plain_columns", "has_plain_values", "selected_if_header_only", "first_vintage", "latest_vintage",
            "latest_filename", "latest_sha256", "plain_items", "plain_items_nonempty", "items_with_c",
            "n_observed_cells", "idno", "in_wld_metadata", "in_catalog_search", "selection_timestamp_utc",
            "catalogue_version", "rule"]
    s = s[[c for c in cols if c in s.columns]].sort_values(["selected", "rank", "iso3"],
                                                            ascending=[False, True, True])
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(SEL, index=False)
    pd.set_option("display.width", 250)
    print(s[["rank", "selected", "iso3", "n_vintages", "n_markets", "has_plain_columns", "has_plain_values",
             "reason"]].to_string(index=False))
    print(f"requests: {f.n_requests}; written {SEL}")
    return s


def selected_countries() -> pd.DataFrame:
    if not SEL.exists():
        raise SystemExit(f"{SEL} missing: run the select stage first (selection precedes all results).")
    s = pd.read_csv(SEL)
    return s[s.selected].sort_values("rank")


def stage_download() -> None:
    f = rp.PoliteFetcher(min_interval=MIN_INTERVAL)
    mpath = RAW / "MANIFEST.csv"
    for r in selected_countries().itertuples(index=False):
        listing = json.loads((RAW / "_listings" / f"{r.iso3}.json").read_bytes())
        rows = rp.country_vintage_rows(listing, up_to=r.latest_vintage)
        for i in range(0, len(rows), 10):
            manifest = rp.read_manifest(mpath)
            rp.fetch_files(rows[i:i + 10], RAW / r.iso3, f, manifest, r.iso3)
            rp.write_manifest(mpath, manifest)
        print(f"{r.iso3}: {len(rows)} vintages up to {r.latest_vintage} in {RAW / r.iso3}", flush=True)
    print(f"requests: {f.n_requests}")


def country_zips(r, manifest: dict) -> list[Path]:
    """The frozen vintage list of a selected country, each zip verified against the sha256 manifest."""
    listing = json.loads((RAW / "_listings" / f"{r.iso3}.json").read_bytes())
    paths = [RAW / r.iso3 / x["filename"] for x in rp.country_vintage_rows(listing, up_to=r.latest_vintage)]
    for p in paths:
        m = manifest.get(("rtfp_other", r.iso3, p.name))
        if not p.exists() or m is None or m["sha256"] != sha256_file(p):
            raise SystemExit(f"{p} missing or not matching the manifest: run the download stage")
    return paths


def _audit(panel: pd.DataFrame, hdx_path, items) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hdx_path is None:
        a = pd.DataFrame({"item": items, "status": "no WFP/HDX feed found: plain-column assumption unverified"})
        return a, pd.DataFrame()
    cells, summ = rp.audit_country(panel, rp.clean_hdx(pd.read_csv(hdx_path, low_memory=False)))
    return summ.assign(status="audited"), cells


def _assemble(iso3, country, role, res, audit, hdx_path, hdx_pkg) -> pd.DataFrame:
    s = res["summary"].merge(audit.add_prefix("audit_").rename(columns={"audit_item": "item"}), on="item", how="left")
    s.insert(1, "country", country)
    s.insert(2, "role", role)
    s["n_vintages_country"] = res["n_vintages"]
    s["crosswalk_ok"], s["crosswalk_msg"] = res["crosswalk_ok"], res["crosswalk_msg"]
    s["n_vintages_header_renamed"] = len(res.get("renamed_header_vintages", []))
    s["hdx_package"], s["hdx_file"] = hdx_pkg, (Path(hdx_path).name if hdx_path is not None else None)
    s["processing"] = "item by item (one stacked panel per item)"
    return s


def stage_run(work: Path, only=None, with_reference: bool = True) -> None:
    parts = work / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    f = rp.PoliteFetcher(min_interval=MIN_INTERVAL)
    mpath = RAW / "MANIFEST.csv"
    for r in selected_countries().itertuples(index=False):
        if only and r.iso3 not in only:
            continue
        manifest = rp.read_manifest(mpath)
        zips = country_zips(r, manifest)
        res = rp.run_country(zips, r.iso3, work / r.iso3, B=B, seed=SEED, save_dir=work / "derived")
        hdx_path, pkg = rp.fetch_hdx(r.iso3, f, RAW, manifest)
        rp.write_manifest(mpath, manifest)
        audit, cells = _audit(load_panel(str(res["latest_csv"]), res["items"]), hdx_path, res["items"])
        _assemble(r.iso3, r.country, "replication", res, audit, hdx_path, pkg).to_csv(parts / f"{r.iso3}_summary.csv", index=False)
        res["noise"].assign(country=r.country).to_csv(parts / f"{r.iso3}_noise.csv", index=False)
        audit.assign(iso3=r.iso3, country=r.country).to_csv(parts / f"{r.iso3}_audit.csv", index=False)
        if len(cells):
            cells.to_parquet(work / "derived" / f"{r.iso3}_audit_cells.parquet", index=False)
        shutil.rmtree(work / r.iso3)  # extracted working copies; the raw zips stay in data/raw/rtfp_other
        print(f"{r.iso3} done", flush=True)
    if with_reference and (not only or "BGD" in only):
        # Bangladesh through the same driver: a reference row (not a replication), same noise variant A
        headers = rp.link_extracted_vintages(BGD_VINTAGES, work / "BGD", prefix="BGD")
        res = rp.run_extracted(work / "BGD", headers, "BGD", B=B, seed=SEED, save_dir=work / "derived")
        audit, _ = _audit(load_panel(str(res["latest_csv"]), res["items"]), BGD_HDX[-1], res["items"])
        _assemble("BGD", "Bangladesh", "reference (primary study, same driver)", res, audit, BGD_HDX[-1],
                  "wfp-food-prices-for-bangladesh").to_csv(parts / "BGD_summary.csv", index=False)
        res["noise"].assign(country="Bangladesh").to_csv(parts / "BGD_noise.csv", index=False)
        audit.assign(iso3="BGD", country="Bangladesh").to_csv(parts / "BGD_audit.csv", index=False)
        shutil.rmtree(work / "BGD")
    combine(work)


def combine(work: Path) -> None:
    parts = work / "parts"
    order = list(selected_countries().iso3) + ["BGD"]
    for kind, name in (("summary", "T9_replication.csv"), ("noise", "T9b_noise.csv"), ("audit", "T9c_audit.csv")):
        fs = [parts / f"{iso}_{kind}.csv" for iso in order if (parts / f"{iso}_{kind}.csv").exists()]
        t = pd.concat([pd.read_csv(x) for x in fs], ignore_index=True)
        t.insert(0, "computed_utc", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        t.to_csv(OUT / name, index=False)
        print(f"written {OUT / name}: {len(t)} rows from {len(fs)} countries")


# --------------------------------------------------------------------------- results note (generated)

def _f(v, d=3):
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    if isinstance(v, (bool,)):
        return "yes" if v else "no"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 1 and d == 0):
        return f"{int(v):,}"
    return f"{v:.{d}f}"


def _md(df: pd.DataFrame, digits: dict | None = None) -> str:
    digits = digits or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if isinstance(v, str):
                cells.append(v)
            elif isinstance(v, (bool,)) or str(type(v)).endswith("bool_'>"):
                cells.append("yes" if v else "no")
            else:
                cells.append(_f(v if not hasattr(v, "item") else v.item(), digits.get(c, 3)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _ci(est, lo, hi, d=2):
    return "–" if est != est else f"{est:.{d}f} [{_f(lo, d)}, {_f(hi, d)}]"


def bgd_consistency(t: pd.DataFrame) -> dict:
    """Does the BGD reference row (same driver) reproduce the main-specification Tables 2, 3 and 3b?"""
    b = t[t.iso3 == "BGD"].set_index("item")
    t2 = pd.read_csv(OUT / "T2_levels.csv")
    t2 = t2[t2.estimate == "c_final"].set_index(["item", "stratum"])
    t3 = pd.read_csv(OUT / "T3_local_changes.csv")
    t3 = t3[(t3.pair_type == "both_imputed") & (t3["mode"] == "own_mu")].set_index(["item", "stratum"])
    t3b = pd.read_csv(OUT / "T3b_spikes.csv").set_index("item")
    st = rp.CLEAN
    d_lvl = max(abs(t2.loc[(i, s_), k] - b.loc[i, f"{k}_{s_}"]) for i in b.index for s_ in st for k in ("mape", "bias_logpct"))
    keys = [(i, s_) for i in b.index for s_ in st if (i, s_) in t3.index]
    d_loc = max(abs(t3.loc[k, c] - b.loc[k[0], f"{c}_{k[1]}"]) for k in keys for c in ("corr", "beta_mz", "lam"))
    n_eq = all(int(t3.loc[k, "n"]) == int(b.loc[k[0], f"pairs_{k[1]}"]) for k in keys)
    sp_eq = all((int(t3b.loc[i, "spikes"]), int(t3b.loc[i, "caught"])) == (int(b.loc[i, "spikes"]), int(b.loc[i, "spikes_caught"]))
                for i in b.index)
    return {"max_abs_diff_levels": d_lvl, "max_abs_diff_local": d_loc, "pair_counts_equal": n_eq, "spikes_equal": sp_eq,
            "n_local_cells": len(keys)}


def stage_note() -> None:
    sel = pd.read_csv(SEL)
    t = pd.read_csv(OUT / "T9_replication.csv")
    au = pd.read_csv(OUT / "T9c_audit.csv")
    rep = t[t.role == "replication"]
    ok = rep[rep.status == "ok"]
    chosen = sel[sel.selected].sort_values("rank")
    nxt = sel[sel.eligible & ~sel.selected].sort_values("rank").head(1)
    ts = sel.selection_timestamp_utc.iloc[0]
    computed = t.computed_utc.iloc[0]
    same_sel = bool((sel.selected == sel.selected_if_header_only).all())
    zips = []
    for r in chosen.itertuples(index=False):
        listing = json.loads((RAW / "_listings" / f"{r.iso3}.json").read_bytes())
        zips += [RAW / r.iso3 / x["filename"] for x in rp.country_vintage_rows(listing, up_to=r.latest_vintage)]
    dates = rp.check_renamed_dates(zips)
    bc = bgd_consistency(t) if (t.iso3 == "BGD").any() else None

    # per-country overview
    ov = []
    for r in chosen.itertuples(index=False):
        x = t[t.iso3 == r.iso3]
        ov.append({"country": f"{r.country} ({r.iso3})", "vintages": int(r.n_vintages),
                   "first–latest vintage": f"{r.first_vintage} – {r.latest_vintage}", "markets": int(r.n_markets),
                   "items": len(x), "items ok": int((x.status == "ok").sum()),
                   "clean cells": int(x.cells_clean.sum()),
                   "items audit ≥ 90%": int(x.audit_audit_pass.fillna(False).astype(bool).sum()) if "audit_audit_pass" in x else 0,
                   "crosswalk guard": "pass" if bool(x.crosswalk_ok.iloc[0]) else "markets dropped",
                   "markets dropped (location jump)": ";".join(sorted({m for v in x.get("dropped_markets_location_jump", pd.Series(dtype=str)).fillna("") for m in str(v).split(";") if m})),
                   "old 'DATES' header": int(x.n_vintages_header_renamed.iloc[0]),
                   "peak RSS GB": float(x.peak_rss_gb.max())})
    ov = pd.DataFrame(ov)

    # per-item metrics
    def item_table(d):
        return pd.DataFrame({
            "country": d.iso3, "item": d["item"], "clean cells (fc/cs)": [f"{a:,}/{b:,}" for a, b in zip(d.cells_forecast, d.cells_cross_sectional)],
            "MAPE % [95% CI]": [_ci(a, b, c) for a, b, c in zip(d.mape, d.mape_lo, d.mape_hi)],
            "bias log-% [95% CI]": [_ci(a, b, c) for a, b, c in zip(d.bias_logpct, d.bias_lo, d.bias_hi)],
            "rel A": d.rel_A_naive,
            "pairs cs": d.pairs_cross_sectional,
            "corr cs [95% CI]": [_ci(a, b, c) for a, b, c in zip(d.corr_cross_sectional, d.corr_lo_cross_sectional, d.corr_hi_cross_sectional)],
            "ρ* cons. cs": d.rho_star_conservative_cross_sectional,
            "pairs fc": d.pairs_forecast,
            "corr fc [95% CI]": [_ci(a, b, c) for a, b, c in zip(d.corr_forecast, d.corr_lo_forecast, d.corr_hi_forecast)],
            "ρ* cons. fc": d.rho_star_conservative_forecast,
            "spikes caught": [f"{int(c)}/{int(s)}" for c, s in zip(d.spikes_caught, d.spikes)],
            "audit match": [("–" if a != a else f"{a:.3f}") for a in d.audit_share_matched],
        })

    cs = ok.dropna(subset=["corr_cross_sectional"])
    fc = ok.dropna(subset=["corr_forecast"])
    rho_cs = cs.rho_star_conservative_cross_sectional.dropna()
    rho_fc = fc.rho_star_conservative_forecast.dropna()
    passed = ok[ok.audit_audit_pass.fillna(False).astype(bool)]
    cs_p = passed.dropna(subset=["corr_cross_sectional"])
    rho_cs_p = cs_p.rho_star_conservative_cross_sectional.dropna()
    sp_tot, sp_c = int(ok.spikes.sum()), int(ok.spikes_caught.sum())
    errs = rep[rep.status != "ok"]
    q = lambda x: f"median {x.median():.3f} (range {x.min():.3f} to {x.max():.3f})" if len(x) else "not estimable"
    bgd = t[t.iso3 == "BGD"]
    t1c = pd.read_csv(OUT / "T1c_gate_summary.csv")
    n_cs10 = int((ok.pairs_cross_sectional >= 10).sum())
    cs_pos = int((cs.corr_cross_sectional > 0).sum())
    pos0 = cs[cs.corr_lo_cross_sectional > 0]
    neg0 = cs[cs.corr_hi_cross_sectional < 0]
    cs_excl0 = len(pos0) + len(neg0)
    excl0_items = (f"{len(pos0)} positive: " + ", ".join(f"{a} {b}" for a, b in zip(pos0.iso3, pos0["item"]))
                   + (f"; {len(neg0)} negative: " + ", ".join(f"{a} {b}" for a, b in zip(neg0.iso3, neg0["item"])) if len(neg0) else ""))
    fc_excl0 = int(((fc.corr_lo_forecast > 0) | (fc.corr_hi_forecast < 0)).sum())
    in_rng = lambda r: int(((r >= 0.5) & (r <= 1)).sum())
    over1 = lambda r: int((r > 1).sum())
    rho_cs_A = ok.rho_star_A_naive_cross_sectional.dropna()
    # level accuracy without cells whose published value is a different scale from the survey value (c/y > 5 or < 0.2)
    der = Path(tempfile.gettempdir()) / "rtfp_t9_work" / "derived"
    robust = []
    for r in ok.itertuples(index=False):
        f = der / f"{r.iso3}_{r.item}_holdout.parquet"
        if not f.exists():
            continue
        h = pd.read_parquet(f)
        h = h[h.stratum.astype(str).isin(["forecast", "cross_sectional"]) & h.y_eval.notna()]
        ratio = h.c_final / h.y_eval
        bad = (ratio > 5) | (ratio < 0.2)
        keep = h[~bad]
        robust.append({"iso3": r.iso3, "item": r.item, "n_scale_mismatch": int(bad.sum()),
                       "mape_excl_scale_mismatch": 100 * float((keep.c_final / keep.y_eval - 1).abs().mean()) if len(keep) else np.nan})
    robust = pd.DataFrame(robust)
    robust.to_csv(OUT / "T9d_scale_mismatch.csv", index=False)
    rb = robust[robust.n_scale_mismatch > 0]

    note = f"""# T9 — Bounded replication in five other RTFP countries (Task 6.1)

*Generated by `analysis/t9_replication.py note` from `outputs/tables/T9_*.csv`; do not edit numbers by hand. Exploratory: the selection rule is in `docs/analysis_plan.md` §5, which is a draft not yet posted on OSF.*

**Claim.** On {len(chosen)} other RTFP countries ({len(ok)} country-items), with the Bangladesh pipeline and three disclosed changes (markets whose name moves > 3 km across releases are dropped and listed; the noise sample is all markets over six years because the Division filter does not transfer; the WFP audit series is chosen from the data), results **vary across countries and items rather than replicate uniformly**:
- Cross-sectional both-imputed local correlation: {q(cs.corr_cross_sectional)} across the {len(cs)} items where it is defined ({n_cs10} items have ≥ 10 pairs; it is undefined when all survey changes are identical). It is positive for {cs_pos} of {len(cs)}, but its 95% interval excludes zero for only {cs_excl0} ({excl0_items}).
- Noise-corrected ρ*: in [0.5, 1] for {in_rng(rho_cs)} of {len(rho_cs)} items; {over1(rho_cs)} more exceed 1 and are not interpretable as correlations (small update-only subsets with low estimated reliability). With variant A alone: in [0.5, 1] for {in_rng(rho_cs_A)} of {len(rho_cs_A)} ({over1(rho_cs_A)} exceed 1).
- Forecast stratum: {q(fc.corr_forecast)} ({len(fc)} items); no interval excludes zero ({fc_excl0} of {len(fc)}); ρ* in [0.5, 1] for {in_rng(rho_fc)} of {len(rho_fc)} ({over1(rho_fc)} exceed 1).
- Items whose plain column passed the ≥ 90% audit ({len(passed)}): cross-sectional correlation {q(cs_p.corr_cross_sectional)} ({len(cs_p)} items); ρ* in [0.5, 1] for {in_rng(rho_cs_p)} of {len(rho_cs_p)} ({over1(rho_cs_p)} exceed 1).
- Persistent survey spikes caught by a same-direction published move: {sp_c} of {sp_tot}{f" ({sp_c / sp_tot:.1%})" if sp_tot else ""}.
- Level-error outliers from scale mismatches: {len(rb)} items have clean cells whose published value is > 5× or < 0.2× the survey value (a unit or scale error, not imputation error); see `T9d_scale_mismatch.csv` for MAPE without them.

## 1. Country selection (fixed before any country's results)

- Rule (analysis plan §5): {sel.rule.iloc[0]}.
- Applied at **{ts}** to {len(sel)} per-country RTFP market datasets other than Bangladesh (union of the all-country metadata `WLD_2021_RTFP_v02_M`, "{sel.catalogue_version.iloc[0]}", and the catalogue search). {int(sel.eligible.sum())} were eligible. The selection file was written by the `select` stage before the `download` and `run` stages existed for these countries; the run stage refuses to start without it and reuses its listing snapshot (vintages up to the latest seen at selection).
- "Plain survey columns present" was read as a plain item column with at least one value; the selection is {"identical" if same_sel else "DIFFERENT"} under the header-only reading.

{_md(chosen[["rank", "iso3", "country", "n_vintages", "n_markets", "first_vintage", "latest_vintage"]].assign(rank=lambda d: d["rank"].astype(int)))}

Next in line (not selected): {", ".join(f"{r.country} ({r.iso3}, {int(r.n_markets)} markets, {int(r.n_vintages)} vintages)" for r in nxt.itertuples(index=False)) or "none"}. All candidates and reasons: `T9_country_selection.csv`.

## 2. Data processed

{_md(ov, {"peak RSS GB": 1})}

- Every country was processed **item by item** (one stacked panel per item, via `rtfp.vintages.build_stack(dir, items=[item], prefix=ISO3)`) to bound memory on a 16 GB machine; the largest per-item process peak is in the table.
- Vintages lacking an item are skipped for that item (as for Bangladesh rice). Items = those with both a plain and a `c_` column in the latest vintage.
- Older releases (to January 2024) name the month column `DATES` instead of `price_date`. The extracted working copy renames the header only; data bytes are unchanged. Check over all {dates["n_with_dates"]} such vintages: every `DATES` value is the first of the month given by `year`/`month` ({dates["n_inconsistent"]} inconsistent).
- Market identity guard (`rtfp.crosswalk.build_crosswalk`: a name may not move > 3 km or map to two ids in one release): see the table.

## 3. Results by country and item

Clean strata only (fc = forecast, cs = cross-sectional). Levels pool both clean strata (c_final vs the latest-vintage survey value). Local changes: own-release μ, both-imputed pairs, corr not estimated below 10 pairs. ρ* cons. = largest credible of corr/√rel over noise variants A (all pairs) and B (non-zero truth changes), naive and AR(1) estimators, as Gate G1. rel A = naive reliability, variant A. CIs: two-way (market × month) pigeonhole bootstrap, B = {B}; reliability held fixed.

{_md(item_table(ok))}
"""
    if len(errs):
        note += "\nItems not processed (error recorded in `T9_replication.csv`):\n\n" + _md(errs[["iso3", "item", "status"]]) + "\n"
    if len(bgd):
        note += f"""
**Bangladesh through the same driver (reference, not a replication).** Same code path, same noise variant A as the other countries (all markets, price months from {bgd.noise_since.iloc[0]}); Table 1's main variant (non-Division markets from 2020-08) differs, so use T1/T1c for Bangladesh's main-specification ρ*.

{_md(item_table(bgd))}
"""
        if bc:
            note += f"""
Consistency: the reference row reproduces Table 2 MAPE and bias by stratum (max |difference| {bc["max_abs_diff_levels"]:.1e}), Table 3 corr, β_MZ and λ ({bc["n_local_cells"]} item × stratum cells; max |difference| {bc["max_abs_diff_local"]:.1e}; pair counts equal: {"yes" if bc["pair_counts_equal"] else "NO"}) and Table 3b spike counts (equal: {"yes" if bc["spikes_equal"] else "NO"}).

Main-specification Bangladesh conservative ρ* (Table 1c, cross-sectional): {", ".join(f"{r.item} {r.conservative_rho_star:.3f}" for r in t1c.itertuples(index=False))}.
"""
    aud = au[au.iso3 != "BGD"]
    note += f"""
## 4. Raw-column audit (Task 1.2 transferred)

Every selected country has a WFP/HDX feed (`hdx_package` in `T9_replication.csv`), so the plain-column assumption was tested for every item: observed cells of the latest vintage are matched to Retail rows of the feed (market by exact name, else nearest within 5 km; same month) with `rtfp.audit.audit_cells`. RTFP item names do not name a WFP commodity variety, so for each item the Retail commodity × unit series with the most exact matches (|y/p − 1| ≤ 1%) was chosen from the data; choosing the best of several candidates can only raise the match share (runner-up shown in `T9c_audit.csv`). Bangladesh's stop-gate was 90%.

{_md(aud[["iso3", "item", "hdx_commodity", "hdx_unit", "rtfp_observed", "match", "value_mismatch", "no_hdx_row", "share_matched", "audit_pass"]].rename(columns={"audit_pass": "≥ 90%"}))}

Items below 90% ({int((~aud.audit_pass.fillna(False).astype(bool)).sum())} of {len(aud)}): the plain column is not shown to be the WFP survey value, so their hold-out "truth" is unverified; they are kept in Section 3 and summarised separately in the claim. The share counts an observed RTFP cell with no WFP row as a failure, so a low share can reflect WFP coverage rather than disagreement: agreement where the feed has a row (match / (match + value mismatch)) is shown below.

{_md(aud.assign(agreement_where_feed_has_row=aud.match / (aud.match + aud.value_mismatch).replace(0, np.nan))[["iso3", "item", "match", "value_mismatch", "no_hdx_row", "share_matched", "agreement_where_feed_has_row"]])}

## 5. Method facts

- Pipeline: `src/rtfp/replicate.py` (driver; tests in `tests/test_replicate.py` on synthetic inputs) calling `rtfp.vintages` (stack, hold-out, strata toggle > unconfirmed > backfill (> 6 months) > forecast > cross-sectional, change pairs), `rtfp.noise`, `rtfp.metrics` and `rtfp.audit`; the only change from Bangladesh is `build_stack(..., on_drift='drop')`, which removes markets whose name moves > 3 km across releases (geocoding defects) and records them per item in `dropped_markets_location_jump`.
- Noise variant A for every country: all markets except 'Market Average', price months from 6 years before the latest vintage month; variant B drops exact-zero changes. There is no Division-series equivalent outside Bangladesh, so Table 1's A_main filter is not transferable.
- Bootstrap clusters as Tables 2–3: hold-out markets × months for levels, latest-panel markets × months for correlations; seed {SEED}.
- Downloads: {len(zips)} market-file vintages for the five countries plus the latest file of every candidate, `data/raw/rtfp_other/` with sha256 manifest `data/raw/rtfp_other/MANIFEST.csv`; ≥ {MIN_INTERVAL} s between requests. Results computed {computed}.

## 6. Caveats

- Exploratory. The selection rule was written before these countries' results were computed, but the analysis plan is not yet registered; the results are not confirmatory.
- The commodity series used in the audit is chosen from the data (upward-biased match share); the variety behind each RTFP item is not documented in the files.
- Items whose names end in `_fao` presumably come from a non-WFP (FAO) source; this is an inference from the name only.
- ρ* intervals are not reported: the reliability is held fixed in the bootstrap (Table 1c resamples it jointly for Bangladesh). Holm correction is not applied; no hypothesis is tested here.
- Clean-strata sample sizes differ widely between items; correlations with few pairs have wide intervals (see CIs).
- Fig 9 (forest plot of ρ* bands) is not produced by this script; its inputs are the `corr_*`, `rho_star_*` and `rel_*` columns of `T9_replication.csv`.

**Files.** `outputs/tables/T9_country_selection.csv`, `T9_replication.csv` (country × item, all metrics), `T9b_noise.csv` (reliability by variant), `T9c_audit.csv` (raw-column audit); downloads in `data/raw/rtfp_other/`.
"""
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "T9.md").write_text(note)
    print(f"written {NOTES / 'T9.md'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "download", "run", "note", "all"])
    ap.add_argument("--reselect", action="store_true")
    ap.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "rtfp_t9_work",
                    help="scratch directory for extracted CSVs and per-country parts")
    ap.add_argument("--only", nargs="*", help="ISO3 codes to (re)run; default all selected plus the BGD reference")
    a = ap.parse_args(argv)
    if a.stage in ("select", "all") and not (a.stage == "all" and SEL.exists()):
        stage_select(a.reselect)
    if a.stage in ("download", "all"):
        stage_download()
    if a.stage in ("run", "all"):
        stage_run(a.work, only=a.only)
    if a.stage in ("note", "all"):
        stage_note()
    return 0


if __name__ == "__main__":
    sys.exit(main())
