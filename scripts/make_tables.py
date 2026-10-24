#!/usr/bin/env python
"""Generate LaTeX table fragments from the frozen benchmark results.

Single numeric source of truth for the manuscript. Reads the frozen
`results/benchmark_raw.csv`, uses the full benchmark roster and the main method
set (external published baselines + the proposed method), recomputes all ranks
honestly within each (dataset, seed), and emits:

  * the main table (proposed method last) with held-out MMD^2 as the
    conventional single primary endpoint, separation--fidelity rank (SFR) as a
    secondary two-axis summary, Rank_avg over six law-matching families
    (MMD, W1, cov/third/fourth, d_min; mean_err dropped 2026-10-08),
    and Rank_W1; SERA ranks are appendix-only;
  * per-dataset PFR and MMD^2 tables;
  * a lineage ablation table (MomentAnchor vs Simplex-LOWEM);
  * certificate statistics (U*, r_marg, r_mom), the realized average pool size,
    and the U*-vs-test-MMD^2 association with its sample size and CI.

Numbers are never hand-typed into the manuscript; this script writes them all
to results/latex_snippets.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bench_config as cfg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RANK_LOWER = cfg.RANK_METRICS_LOWER
RANK_APPENDIX = getattr(cfg, "RANK_APPENDIX_LOWER", [])


def parse_audit(meta_json: str) -> dict:
    """Read either the live audit_record or the older certificate alias."""
    if not isinstance(meta_json, str) or not meta_json:
        return {}
    try:
        meta = json.loads(meta_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    cert = meta.get("certificate") or {}
    audit = meta.get("audit_record") or {}
    return {
        "U_star": cert.get("U_star", audit.get("U_train", np.nan)),
        "r_marg": cert.get("r_marg", audit.get("r_marg", np.nan)),
        "r_mom": cert.get("r_mom", audit.get("r_mom", np.nan)),
        "d_priv": cert.get("d_priv", audit.get("d_sep", np.nan)),
        "pool_size": meta.get("pool_size", np.nan),
        "active_support": meta.get("active_support", np.nan),
    }


def sci(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 2:
        return f"${x:.{max(digits, 2)}g}$"
    mant = x / 10 ** exp
    return f"${mant:.{digits}f}\\times10^{{{exp}}}$"


def sci_pm(mean: float, std: float) -> str:
    """Compact mean±std sharing the mean's scientific scale."""
    if not np.isfinite(mean):
        return "--"
    if abs(mean) < 1e-12:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(mean))))
    if exp >= -1:
        return rf"${mean:.2f}\pm{std:.2f}$"
    if exp == -2:
        return rf"${mean:.3f}\pm{std:.3f}$"
    m = mean / 10 ** exp
    s = std / 10 ** exp
    return rf"${m:.1f}{{\scriptstyle\pm{s:.1f}}}\times10^{{{exp}}}$"


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "--"


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute per-(dataset,seed) ranks over the methods present in df."""
    df = df.copy()
    g = ["dataset", "seed"]
    for m in list(RANK_LOWER) + list(RANK_APPENDIX):
        df[f"rank_{m}"] = df.groupby(g)[m].rank(method="average", ascending=True)
    df["rank_min_train_distance"] = df.groupby(g)["min_train_distance"].rank(
        method="average", ascending=False)
    fid_cols = [f"rank_{m}" for m in RANK_LOWER] + ["rank_min_train_distance"]
    df["avg_rank"] = df[fid_cols].mean(axis=1)
    # Secondary separation--fidelity summary (SFR; column key kept as pfr).
    df["pfr"] = 0.5 * (df["rank_mmd2_test"] + df["rank_min_train_distance"])
    return df


def per_method_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std (over seeds, after averaging datasets) for rank-type
    metrics, and plain means for level metrics."""
    per_seed = df.groupby(["method", "seed"]).agg(
        pfr=("pfr", "mean"),
        avg_rank=("avg_rank", "mean"),
        rank_w1=("rank_sliced_w1_test", "mean"),
        rank_sera=("rank_utility_sera", "mean"),
    ).reset_index()
    agg = per_seed.groupby("method").agg(
        pfr_mean=("pfr", "mean"), pfr_std=("pfr", "std"),
        rank_mean=("avg_rank", "mean"), rank_std=("avg_rank", "std"),
        w1_rank=("rank_w1", "mean"), w1_rank_std=("rank_w1", "std"),
        sera_rank=("rank_sera", "mean"), sera_rank_std=("rank_sera", "std"),
    )
    flat = df.groupby("method").agg(
        mmd2=("mmd2_test", "mean"),
        dmin=("min_train_distance", "mean"),
        fourth=("fourth_err", "mean"),
        sw1=("sliced_w1_test", "mean"),
    )
    return agg.join(flat)


def main_table(df_main: pd.DataFrame, out: list[str]) -> None:
    exclude = set(getattr(cfg, "POOLED_MEAN_EXCLUDE", []))
    df_pool = df_main[~df_main.dataset.isin(exclude)].copy()
    agg = per_method_stats(df_pool)
    agg_all = per_method_stats(df_main)
    best = {
        "pfr_mean": agg["pfr_mean"].idxmin(),
        "mmd2": agg["mmd2"].idxmin(),
        "dmin": agg["dmin"].idxmax(),
        "rank_mean": agg["rank_mean"].idxmin(),
        "w1_rank": agg["w1_rank"].idxmin(),
    }

    def cell(method: str, col: str, text: str) -> str:
        if best[col] != method:
            return text
        return rf"{{\boldmath{text}}}" if text.startswith("$") else rf"\textbf{{{text}}}"

    # External baselines sorted by the primary endpoint; proposed method last.
    ext = [m for m in cfg.EXTERNAL_MAIN if m in agg.index]
    ext_sorted = sorted(ext, key=lambda m: agg.loc[m, "mmd2"])
    order = ext_sorted + [cfg.OURS]

    n_pool = df_pool.dataset.nunique()
    n_all = df_main.dataset.nunique()
    out.append("%% ---- MAIN TABLE ROWS (between \\midrule and \\bottomrule) ----")
    out.append(
        f"%% pooled means exclude {sorted(exclude)} "
        f"(n={n_pool}); per-dataset table and win counts still use n={n_all}"
    )
    out.append("%% columns: Method & MMD2 & dmin & SFR & Rank_fid & Rank_W1")
    out.append("%% full-roster means (Melbourne included; not the headline):")
    for m in [x for x in cfg.EXTERNAL_MAIN if x in agg_all.index] + [cfg.OURS]:
        if m not in agg_all.index:
            continue
        r = agg_all.loc[m]
        out.append(
            f"%% ALL12 {m}: mmd2={r.mmd2:.5g} dmin={r.dmin:.4g} "
            f"pfr={r.pfr_mean:.3f}"
        )
    for m in order:
        r = agg.loc[m]
        label = rf"\textbf{{{m} (ours)}}" if m == cfg.OURS else cfg.CITES.get(m, m)
        row = " & ".join([
            label,
            cell(m, "mmd2", sci(r.mmd2)),
            cell(m, "dmin", sci(r.dmin)),
            cell(m, "pfr_mean", f"{fmt(r.pfr_mean)} $\\pm$ {fmt(r.pfr_std)}"),
            cell(m, "rank_mean", f"{fmt(r.rank_mean)} $\\pm$ {fmt(r.rank_std)}"),
            cell(m, "w1_rank", f"{fmt(r.w1_rank)} $\\pm$ {fmt(r.w1_rank_std)}"),
        ])
        out.append(row + r" \\")

    # Wins on the primary endpoint and on the secondary SFR summary.
    for prim, lab, asc in (("mmd2_test", "MMD2-primary", True), ("pfr", "SFR-secondary", True)):
        per_ds = df_main.groupby(["dataset", "method"])[prim].mean().reset_index()
        idx = per_ds.groupby("dataset")[prim].idxmin() if asc else per_ds.groupby("dataset")[prim].idxmax()
        wins = per_ds.loc[idx, "method"].value_counts()
        out.append(f"%% wins on {lab}: " + json.dumps(wins.to_dict()))

    out.append("%% ---- APPENDIX SERA RANK ROWS (not Table 1) ----")
    out.append("%% columns: Method & Rank_SERA")
    sera_best = agg["sera_rank"].idxmin()
    for m in order:
        r = agg.loc[m]
        label = rf"\textbf{{{m} (ours)}}" if m == cfg.OURS else cfg.CITES.get(m, m)
        txt = f"{fmt(r.sera_rank)} $\\pm$ {fmt(r.sera_rank_std)}"
        if m == sera_best:
            txt = rf"\textbf{{{txt}}}"
        out.append(f"{label} & {txt} \\\\")
    per_ds = df_main.groupby(["dataset", "method"])["utility_sera"].mean().reset_index()
    idx = per_ds.groupby("dataset")["utility_sera"].idxmin()
    wins = per_ds.loc[idx, "method"].value_counts()
    out.append("%% wins on SERA-dataset-means: " + json.dumps(wins.to_dict()))


def perdataset_tables(df_main: pd.DataFrame, out: list[str]) -> None:
    order = [m for m in cfg.EXTERNAL_MAIN if m in df_main["method"].unique()]
    order = sorted(order, key=lambda m: df_main[df_main.method == m]["mmd2_test"].mean()) + [cfg.OURS]
    for col, lab, nd in (("mmd2_test", "MMD2", None), ("pfr", "SFR", 2)):
        out.append(f"%% ---- PER-DATASET {lab}: seed mean±std; rows=dataset, cols=method ----")
        piv = df_main.pivot_table(index="dataset", columns="method", values=col, aggfunc="mean")
        piv_std = df_main.pivot_table(index="dataset", columns="method", values=col, aggfunc="std")
        piv = piv.reindex(index=cfg.ROSTER)
        piv_std = piv_std.reindex(index=cfg.ROSTER)
        out.append("%% header: " + " & ".join(["Dataset"] + order) + r" \\")
        for ds_name in piv.index:
            best_m = piv.loc[ds_name, order].idxmin()
            cells = []
            for m in order:
                v = float(piv.loc[ds_name, m])
                s = float(piv_std.loc[ds_name, m])
                txt = sci_pm(v, s) if nd is None else fmt(v, nd)
                if m == best_m:
                    txt = rf"{{\boldmath{txt}}}" if txt.startswith("$") else rf"\textbf{{{txt}}}"
                cells.append(txt)
            out.append(" & ".join([ds_name] + cells) + r" \\")


def ablation_lineage(df_all: pd.DataFrame, out: list[str]) -> None:
    """MomentAnchor vs the Simplex-LOWEM lineage on the retained roster."""
    methods = [cfg.OURS] + cfg.LINEAGE_METHODS
    sub = df_all[df_all.method.isin(methods)].copy()
    if sub.method.nunique() < 2:
        out.append("%% ---- LINEAGE ABLATION: lineage rows missing ----")
        return
    cols = [("mmd2_test", sci, "MMD2"), ("min_train_distance", sci, "dmin"),
            ("cov_err", lambda v: fmt(v, 3), "cov_err"),
            ("fourth_err", lambda v: fmt(v, 3), "fourth_err")]
    means = sub.groupby("method")[[c for c, _, _ in cols]].mean()
    out.append("%% ---- LINEAGE ABLATION TABLE ROWS ----")
    out.append("%% columns: Method & MMD2 & dmin & cov_err & fourth_err")
    # mean_err omitted from lineage table (H18): uninformative on this roster.
    for m in methods:
        if m not in means.index:
            continue
        label = rf"\textbf{{{m} (ours)}}" if m == cfg.OURS else m
        row = " & ".join([label] + [f(means.loc[m, c]) for c, f, _ in cols])
        out.append(row + r" \\")


def certificate_stats(df_main: pd.DataFrame, out: list[str]) -> None:
    ma = df_main[df_main.method == cfg.OURS].copy()
    cert = ma["meta"].apply(parse_audit)
    ma["U_star"] = cert.apply(lambda c: c.get("U_star", np.nan))
    ma["r_marg"] = cert.apply(lambda c: c.get("r_marg", np.nan))
    ma["r_mom"] = cert.apply(lambda c: c.get("r_mom", np.nan))
    ma["d_priv"] = cert.apply(lambda c: c.get("d_priv", np.nan))
    ma["pool_size"] = cert.apply(lambda c: c.get("pool_size", np.nan))
    ma["active_support"] = cert.apply(lambda c: c.get("active_support", np.nan))

    out.append("%% ---- CERTIFICATE STATISTICS (MomentAnchor, roster) ----")
    out.append(f"%% U_star: min={ma.U_star.min():.2g} max={ma.U_star.max():.2g} "
               f"median={ma.U_star.median():.2g}")
    out.append(f"%% r_mom: min={ma.r_mom.min():.2g} max={ma.r_mom.max():.2g} "
               f"median={ma.r_mom.median():.2g}")
    out.append(f"%% r_marg: min={ma.r_marg.min():.2g} max={ma.r_marg.max():.2g} "
               f"median={ma.r_marg.median():.2g}")
    if ma["d_priv"].notna().any():
        out.append(f"%% d_priv: min={ma.d_priv.min():.2g} max={ma.d_priv.max():.2g} "
                   f"median={ma.d_priv.median():.2g}")
    out.append(f"%% realized pool size: mean={ma.pool_size.mean():.0f} "
               f"min={ma.pool_size.min():.0f} max={ma.pool_size.max():.0f}")
    out.append(f"%% active support: mean={ma.active_support.mean():.0f} "
               f"of pool mean {ma.pool_size.mean():.0f}")

    # Association between the certificate U* and the measured test MMD^2 over
    # dataset means, with sample size and a dataset-bootstrap 95% CI.
    ds_mean = ma.groupby("dataset")[["U_star", "mmd2_test"]].mean().dropna()
    n = len(ds_mean)
    if n >= 3:
        r = float(np.corrcoef(ds_mean["U_star"], ds_mean["mmd2_test"])[0, 1])
        rng = np.random.default_rng(0)
        boots = []
        idx = np.arange(n)
        for _ in range(5000):
            b = rng.choice(idx, n, replace=True)
            u = ds_mean["U_star"].to_numpy()[b]
            v = ds_mean["mmd2_test"].to_numpy()[b]
            if np.std(u) > 0 and np.std(v) > 0:
                boots.append(np.corrcoef(u, v)[0, 1])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(f"%% U*-vs-testMMD2 Pearson r={r:.2f} (n={n} datasets; "
                   f"95%% dataset-bootstrap CI [{lo:.2f}, {hi:.2f}])")


def dataset_table(df_all: pd.DataFrame, out: list[str]) -> None:
    out.append("%% ---- DATASET TABLE (retained roster; train/test after caps) ----")
    info = df_all.groupby("dataset").agg(
        ntr=("n_train_eval", "max"), nte=("n_test_eval", "max"),
        src=("source", "first"),
    )
    for name in cfg.ROSTER:
        if name in info.index:
            r = info.loc[name]
            out.append(f"%% {name}: {int(r.ntr)}/{int(r.nte)}  source={r.src}")


def fourth_moment_note(df_main: pd.DataFrame, out: list[str]) -> None:
    fourth = df_main.groupby("method")["fourth_err"].mean().sort_values()
    out.append("%% ---- FOURTH-MOMENT (heavy-tail) error means on roster ----")
    for m, v in fourth.items():
        tag = " (ours)" if m == cfg.OURS else ""
        out.append(f"%% {m}{tag}: fourth_err={v:.3f}")


def ablation_fw_summary(out: list[str]) -> None:
    p = RESULTS / "ablation_fw_raw.csv"
    if not p.exists():
        return
    ab = pd.read_csv(p)
    ab = ab[ab.dataset.isin(cfg.ROSTER)]
    out.append("%% ---- ABLATION FW/moment-order (roster, method-level means) ----")
    ab["U_star"] = ab["meta"].apply(lambda s: parse_audit(s).get("U_star", np.nan))
    summ = ab.groupby("method").agg(mmd2=("mmd2_test", "mean"),
                                    fourth=("fourth_err", "mean"),
                                    ustar=("U_star", "mean"))
    for m, r in summ.iterrows():
        out.append(f"%% {m}: mmd2={r.mmd2:.4g} fourth={r.fourth:.3f} U*={r.ustar:.4g}")


def main() -> int:
    df = pd.read_csv(RESULTS / "benchmark_raw.csv")
    df = df[df.dataset.isin(cfg.ROSTER)].copy()

    df_main = add_ranks(df[df.method.isin(cfg.MAIN_METHODS)].copy())

    out: list[str] = []
    main_table(df_main, out)
    perdataset_tables(df_main, out)
    ablation_lineage(df, out)
    fourth_moment_note(df_main, out)
    certificate_stats(df_main, out)
    dataset_table(df, out)
    ablation_fw_summary(out)

    path = RESULTS / "latex_snippets.txt"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] wrote {path} ({len(out)} lines)")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
