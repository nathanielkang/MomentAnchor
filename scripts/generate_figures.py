#!/usr/bin/env python
"""Regenerate all [PYTHON] figures for the MomentAnchor manuscript from frozen results.

Reads:  results/benchmark_raw.csv, results/ablation_fw_raw.csv
Writes: ../1_Manuscript/figures/fig_ranking_bar.pdf
        ../1_Manuscript/figures/fig_mmd_scatter.pdf
        ../1_Manuscript/figures/fig_certificate.pdf
        ../1_Manuscript/figures/fig_ablation.pdf

Policy: PDF is the manuscript include; render PDFs outside figures/ for the
mandatory visual-inspection pass. No caption text inside images; axis labels
>= 11 pt and ticks >= 10 pt at final print width. All panels are restricted to
the locked twelve-dataset roster in bench_config.ROSTER; the Frank--Wolfe
ablation grid predates the roster, so only its six surviving datasets appear.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bench_config as cfg
from make_tables import add_ranks, parse_audit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT.parent / "1_Manuscript" / "figures"

OURS = cfg.OURS
TEAL = "#1B9E77"
STEEL = "#4C72B0"
ORANGE = "#D95F02"
# Baseline palette deliberately excludes TEAL so ours stays visually unique.
BASELINE_COLORS = ["#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666", "#4C72B0"]

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
})

ABBREV = {
    "Abalone": "Abal", "CPUActivity": "CPU", "CaliforniaHousing": "CalH",
    "House16H": "H16H", "Insurance": "Insur", "AmesHousing": "Ames",
    "IBMEmployee": "IBM", "MelbourneHousing": "Melb", "BikeSharing": "Bike",
    "WineQualityRed": "WineR", "Concrete": "Conc", "EnergyHeat": "EnH",
    "WineQualityWhite": "WineW", "Yacht": "Yacht", "PowerPlant": "Pwr",
}


def parse_cert(meta_json: str) -> dict:
    return parse_audit(meta_json)


def fig_ranking_bar(df: pd.DataFrame) -> None:
    per_seed = df.groupby(["method", "seed"])["pfr"].mean().reset_index()
    agg = per_seed.groupby("method")["pfr"].agg(["mean", "std"]).sort_values("mean")
    # §10g: wide and short (native height ~1.7 in before bbox).
    fig, ax = plt.subplots(figsize=(8.2, 1.7))
    colors = [TEAL if m == OURS else STEEL for m in agg.index]
    y = np.arange(len(agg))[::-1]
    ax.barh(y, agg["mean"], xerr=agg["std"], color=colors, edgecolor="black",
            linewidth=0.6, height=0.62, error_kw={"lw": 1.2, "capsize": 3})
    ax.set_yticks(y)
    ax.set_yticklabels(agg.index)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlabel("Separation\u2013fidelity balanced rank (lower is better)", fontsize=12)
    ax.grid(axis="x", color="0.85", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "fig_ranking_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_mmd_scatter(df: pd.DataFrame) -> None:
    seed_mean = df.groupby(["dataset", "method"])["mmd2_test"].mean().reset_index()
    datasets = sorted(seed_mean["dataset"].unique())
    methods = list(df.groupby("method")["mmd2_test"].mean().sort_values().index)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    n_methods = len(methods)
    jitter = 0.075
    for mi, m in enumerate(methods):
        sub = seed_mean[seed_mean["method"] == m].set_index("dataset").reindex(datasets)
        x = np.arange(len(datasets)) + (mi - n_methods / 2) * jitter
        filled = m == OURS
        edge = TEAL if filled else BASELINE_COLORS[mi % len(BASELINE_COLORS)]
        ax.scatter(x, sub["mmd2_test"],
                   marker=markers[mi % len(markers)],
                   s=40 if filled else 28,
                   facecolors=TEAL if filled else "none",
                   edgecolors=edge,
                   linewidths=1.4 if filled else 1.1,
                   label=m, zorder=4 if filled else 2)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels([ABBREV.get(d, d) for d in datasets], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(r"Held-out MMD$^2$", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="y", color="0.88", lw=0.8)
    ax.set_axisbelow(True)
    # Legend outside the axes, above the plot, so the canvas stays narrow.
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4,
                    frameon=False, fontsize=9, columnspacing=1.0, handletextpad=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.78)
    for ext, kw in (("pdf", {}),):
        fig.savefig(FIGURES / f"fig_mmd_scatter.{ext}", bbox_inches="tight", pad_inches=0.1,
                    bbox_extra_artists=[leg], **kw)
    plt.close(fig)


def fig_certificate(df: pd.DataFrame) -> None:
    """Audit record (U_train, r_mom, d_sep) per dataset; matches eq:audit."""
    ma = df[df["method"] == OURS].copy()
    certs = ma["meta"].apply(parse_cert)
    ma["U_train"] = certs.apply(lambda c: c.get("U_star", np.nan))
    ma["r_mom"] = certs.apply(lambda c: c.get("r_mom", np.nan))
    ma["d_sep"] = certs.apply(lambda c: c.get("d_priv", np.nan))
    agg = ma.groupby("dataset")[["U_train", "r_mom", "d_sep", "mmd2_test"]].mean()
    datasets = list(agg.index)
    labels = [ABBREV.get(d, d) for d in datasets]
    # Horizontal bars: dataset names stay upright and legible after the float is
    # scaled down, and the panel row stays short enough to sit near its text.
    y = np.arange(len(datasets))[::-1]
    # Use the canvas for the twelve horizontal rows; the external legend and
    # axis labels are included by the tight bounding box (about 46 mm in TeX).
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 1.44), sharey=True)
    (ax1, ax2, ax3) = axes
    ax1.barh(y, agg["U_train"], color=TEAL, edgecolor="black", lw=0.5, height=0.62,
             label=r"$U_{\mathrm{train}}$")
    ax1.scatter(agg["mmd2_test"], y, marker="D", facecolors="none",
                edgecolors="black", s=36, label=r"held-out MMD$^2$", zorder=3)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$U_{\mathrm{train}}$", fontsize=11)
    ax2.barh(y, agg["r_mom"], color=ORANGE, edgecolor="black", lw=0.5, height=0.62)
    ax2.set_xlabel(r"$r_{\mathrm{mom}}$", fontsize=11)
    ax3.barh(y, agg["d_sep"], color=STEEL, edgecolor="black", lw=0.5, height=0.62)
    ax3.set_xscale("log")
    ax3.set_xlabel(r"$d_{\mathrm{sep}}$", fontsize=11)
    for ax, panel in zip(axes, "abc"):
        ax.tick_params(axis="both", labelsize=9.5)
        # Keep panel tags above the data, at right, without a separate title row.
        ax.text(0.99, 1.01, f"({panel})", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="bottom", ha="right")
        ax.grid(axis="x", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.set_xlim(right=agg["mmd2_test"].max() * 4)
    handles, legend_labels = ax1.get_legend_handles_labels()
    # Legend outside the axes, above the panel row, clear of every axis label.
    leg = fig.legend(handles, legend_labels, loc="lower center",
                     bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.18, top=0.94, bottom=0.04)
    for ext, kw in (("pdf", {}),):
        fig.savefig(FIGURES / f"fig_certificate.{ext}", bbox_inches="tight", pad_inches=0.1,
                    bbox_extra_artists=[leg], **kw)
    plt.close(fig)


def fig_ablation_fw(ab: pd.DataFrame) -> None:
    certs = ab["meta"].apply(parse_cert)
    ab = ab.copy()
    ab["U_star"] = certs.apply(lambda c: c.get("U_star", np.nan))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 2.6))
    # Panel (a): vs T at r=3
    Ts, mmds, ustars = [], [], []
    for T in (50, 200, 800, 1200):
        sub = ab[ab["method"] == f"MomentAnchor (T={T})"]
        if len(sub):
            Ts.append(T)
            mmds.append(sub["mmd2_test"].mean())
            ustars.append(sub["U_star"].mean())
    ax1.plot(Ts, mmds, "o-", color=TEAL, lw=1.6, ms=6, label=r"held-out MMD$^2$")
    ax1.plot(Ts, ustars, "s--", color=STEEL, lw=1.6, ms=6,
             label=r"$U_{\mathrm{train}}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    # Headroom so the T=50 markers do not collide with the (a) panel label.
    ax1.set_ylim(top=max(mmds + ustars) * 6)
    ax1.set_xlim(left=32)
    ax1.set_xlabel("Frank\u2013Wolfe iterations $T$")
    ax1.set_ylabel("Discrepancy")
    # Panel (b): vs moment order r at T=1200
    rs, mmds_r, fourth_r = [], [], []
    for r in (0, 2, 3):
        sub = ab[ab["method"] == f"MomentAnchor (r={r})"]
        if len(sub):
            rs.append(r)
            mmds_r.append(sub["mmd2_test"].mean())
            fourth_r.append(sub["fourth_err"].mean())
    ax2.plot(rs, mmds_r, "o-", color=TEAL, lw=1.6, ms=6, label=r"held-out MMD$^2$")
    ax2b = ax2.twinx()
    ax2b.plot(rs, fourth_r, "^--", color=ORANGE, lw=1.6, ms=6,
              label="fourth-moment error")
    ax2.set_xticks(rs)
    ax2.set_xlabel(r"Moment order $r$")
    ax2.set_ylabel(r"Held-out MMD$^2$", color=TEAL)
    ax2b.set_ylabel("Fourth-moment err.", color=ORANGE)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    # Legend outside both axes, above the panel row.
    leg = fig.legend(lines1 + lines2, labels1 + labels2, loc="lower center",
                     bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, fontsize=12)
    for ax, panel in ((ax1, "a"), (ax2, "b")):
        ax.text(0.0, 1.03, f"({panel})", transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="left")
        ax.grid(color="0.9", lw=0.7)
        ax.set_axisbelow(True)
    for ax in (ax1, ax2, ax2b):
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
    fig.subplots_adjust(left=0.09, right=0.90, wspace=0.55, top=0.82, bottom=0.2)
    for ext, kw in (("pdf", {}),):
        fig.savefig(FIGURES / f"fig_ablation.{ext}", bbox_inches="tight", pad_inches=0.12,
                    bbox_extra_artists=[leg], **kw)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS / "benchmark_raw.csv")
    df = df[df.dataset.isin(cfg.ROSTER) & df.method.isin(cfg.MAIN_METHODS)].copy()
    df = add_ranks(df)
    fig_ranking_bar(df)
    fig_mmd_scatter(df)
    fig_certificate(df)
    print("[OK] fig_ranking_bar.pdf, fig_mmd_scatter.pdf, fig_certificate.pdf")
    ab_path = RESULTS / "ablation_fw_raw.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        ab = ab[ab.dataset.isin(cfg.ROSTER)].copy()
        fig_ablation_fw(ab)
        print("[OK] fig_ablation.pdf")
    else:
        print("[SKIP] ablation_fw_raw.csv not found; fig_ablation.pdf not regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
