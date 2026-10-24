"""Read-only full-roster diagnostic for the two-axis PFR summary."""

from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAIN = [
    "Bootstrap",
    "Gaussian Copula",
    "KDE Sampler",
    "Global Mixup",
    "SMOTER",
    "CTGAN",
    "TVAE",
    "MomentAnchor",
]

df = pd.read_csv(RESULTS / "benchmark_raw.csv")
d = df[df.method.isin(MAIN)].copy()
d["r_mmd"] = d.groupby(["dataset", "seed"])["mmd2_test"].rank(ascending=True)
d["r_dmin"] = d.groupby(["dataset", "seed"])["min_train_distance"].rank(ascending=False)
d["pfr"] = (d["r_mmd"] + d["r_dmin"]) / 2

print(f"=== fixed full roster: {d.dataset.nunique()} datasets ===")
print(
    d.groupby("method")[["mmd2_test", "min_train_distance", "pfr"]]
    .mean()
    .sort_values("mmd2_test")
    .round(4)
    .to_string()
)
