#!/usr/bin/env python
"""Shared reporting configuration for the MomentAnchor manuscript.

The frozen benchmark uses the locked twelve-dataset roster after the
2026-08-23 screens (card add3_swap_weak). The conventional single primary
endpoint is held-out MMD^2; the privacy--fidelity rank is a secondary
two-axis summary.
"""

from __future__ import annotations

OURS = "MomentAnchor"

# External published baselines shown in the main table (row order; proposed
# method appended last by the table builder). Every entry carries a citation.
EXTERNAL_MAIN = [
    "Bootstrap",
    "Gaussian Copula",
    "KDE Sampler",
    "Global Mixup",
    "SMOTER",
    "CTGAN",
    "TVAE",
]

MAIN_METHODS = EXTERNAL_MAIN + [OURS]

# Structure-preserving-mixup lineage variant: ablation table only, never the
# main table (it is not an external published baseline and not the proposed
# method).
LINEAGE_METHODS = ["Simplex-LOWEM"]

# Reported dataset roster: all datasets in the benchmark, fixed independently
# of method performance after the 2026-08-23 extra-6 pick.
ROSTER = [
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

# Melbourne left the roster (card add3_swap_weak). Keep the hook empty.
POOLED_MEAN_EXCLUDE = []

PRIMARY_METRIC = "mmd2_test"

# Citations for table cells (main table). Proposed method + lineage carry none.
CITES = {
    "Bootstrap": r"Bootstrap~\cite{efron1979bootstrap}",
    "Global Mixup": r"Global Mixup~\cite{zhang2018mixup}",
    "SMOTER": r"SMOTER~\cite{torgo2013smote}",
    "Gaussian Copula": r"Gaussian Copula~\cite{patki2016sdv}",
    "KDE Sampler": r"KDE Sampler~\cite{silverman1986density}",
    "CTGAN": r"CTGAN~\cite{xu2019modeling}",
    "TVAE": r"TVAE~\cite{xu2019modeling}",
}

# Law-matching families in Table 1 Rank_avg (2026-10-08 MR fix H12).
# Dropped mean_err: near-zero training means make relative mean error
# uninformative on this roster (appendix note). SERA / rare-RMSE stay appendix-only.
RANK_METRICS_LOWER = [
    "mmd2_test",
    "sliced_w1_test",
    "cov_err",
    "third_err",
    "fourth_err",
]

RANK_APPENDIX_LOWER = [
    "utility_sera",
    "utility_rare_rmse",
]
