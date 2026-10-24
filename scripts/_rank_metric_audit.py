"""One-off: locked-grid ranks for every available family (not a generator)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
raw = pd.read_csv(ROOT / "results" / "benchmark_raw.csv")
methods = [
    "Bootstrap",
    "Gaussian Copula",
    "KDE Sampler",
    "Global Mixup",
    "SMOTER",
    "CTGAN",
    "TVAE",
    "MomentAnchor",
]
roster = [
    "Abalone",
    "CPUActivity",
    "CaliforniaHousing",
    "House16H",
    "AmesHousing",
    "IBMEmployee",
    "WineQualityRed",
    "Concrete",
    "EnergyHeat",
    "WineQualityWhite",
    "Yacht",
    "PowerPlant",
]
df = raw[raw.method.isin(methods) & raw.dataset.isin(roster)].copy()
print("rows", len(df), "methods", df.method.nunique(), "ds", df.dataset.nunique())

lower = [
    "mmd2_test",
    "sliced_w1_test",
    "mean_err",
    "cov_err",
    "third_err",
    "fourth_err",
    "utility_sera",
    "utility_rare_rmse",
    "utility_rmse",
]
g = ["dataset", "seed"]
for m in lower:
    df[f"rank_{m}"] = df.groupby(g)[m].rank(method="average", ascending=True)
df["rank_dmin"] = df.groupby(g)["min_train_distance"].rank(
    method="average", ascending=False
)
df["pfr"] = 0.5 * (df["rank_mmd2_test"] + df["rank_dmin"])

fid7 = [
    "rank_mmd2_test",
    "rank_sliced_w1_test",
    "rank_mean_err",
    "rank_cov_err",
    "rank_third_err",
    "rank_fourth_err",
    "rank_dmin",
]
fid6 = [c for c in fid7 if c != "rank_fourth_err"]
cur9 = [
    "rank_mmd2_test",
    "rank_sliced_w1_test",
    "rank_mean_err",
    "rank_cov_err",
    "rank_third_err",
    "rank_fourth_err",
    "rank_utility_sera",
    "rank_utility_rare_rmse",
    "rank_dmin",
]
law6 = fid7[:6]
df["rank_fid7"] = df[fid7].mean(axis=1)
df["rank_fid6"] = df[fid6].mean(axis=1)
df["rank_cur9"] = df[cur9].mean(axis=1)
df["rank_law6"] = df[law6].mean(axis=1)
df["rank_mmdw1"] = df[["rank_mmd2_test", "rank_sliced_w1_test"]].mean(axis=1)
df["rank_mmdw1d"] = df[
    ["rank_mmd2_test", "rank_sliced_w1_test", "rank_dmin"]
].mean(axis=1)
df["rank_mmd_rmse"] = df[["rank_mmd2_test", "rank_utility_rmse"]].mean(axis=1)

cols = {
    "MMD": "rank_mmd2_test",
    "W1": "rank_sliced_w1_test",
    "mean": "rank_mean_err",
    "cov": "rank_cov_err",
    "third": "rank_third_err",
    "fourth": "rank_fourth_err",
    "dmin": "rank_dmin",
    "PFR": "pfr",
    "SERA": "rank_utility_sera",
    "rareRMSE": "rank_utility_rare_rmse",
    "RMSE": "rank_utility_rmse",
    "cur9": "rank_cur9",
    "fid7": "rank_fid7",
    "fid6_nofourth": "rank_fid6",
    "law6": "rank_law6",
    "mmd+w1": "rank_mmdw1",
    "mmd+w1+dmin": "rank_mmdw1d",
    "mmd+RMSE": "rank_mmd_rmse",
}


def seed_then_mean(col: str) -> pd.DataFrame:
    per = df.groupby(["method", "seed"])[col].mean().reset_index()
    return per.groupby("method")[col].agg(["mean", "std"])


print()
for name, col in cols.items():
    s = seed_then_mean(col).sort_values("mean")
    winner = s.index[0]
    ma = s.loc["MomentAnchor"]
    order = " < ".join(f"{m}:{s.loc[m, 'mean']:.2f}" for m in s.index)
    print(
        f"{name:16s} WIN={winner:18s} {s.loc[winner, 'mean']:.3f}  "
        f"MA={ma['mean']:.3f}+/-{ma['std']:.3f}  {order}"
    )
