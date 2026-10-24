#!/usr/bin/env python
"""Shared benchmark core for training-free tabular synthesizers built on LOWEM.

Methods
-------
- MomentAnchor   : convex MMD^2 minimization over a candidate-pool selection
                   distribution by Frank-Wolfe, with a moment-residual penalty,
                   an encoded-space separation filter, and a per-run audit
                   record. The proposed method.
- Simplex-LOWEM  : local K-vertex simplex draws + full affine mean/covariance
                   calibration (structure-preserving-mixup lineage; ablation).
- Bootstrap      : resampling of real training rows.
- Global Mixup   : Beta(0.4, 0.4) convex combinations of random pairs.
- SMOTER         : relevance-weighted nearest-neighbor interpolation.
- Gaussian Copula: Gaussian copula with empirical marginals.
- KDE Sampler    : Gaussian kernel density estimate (Scott bandwidth).
- CTGAN          : conditional tabular GAN (Xu et al., 2019); deep generator.
- TVAE           : tabular variational autoencoder (Xu et al., 2019); deep
                   generator.
- TabDDPM        : Gaussian diffusion baseline (Kotelnikov et al., ICML 2023).
- CTAB-GAN+      : WGAN-GP + regression-head GAN (Zhao et al., 2023).

Datasets: the locked twelve-dataset roster after the 2026-08-23 screens
(card add3_swap_weak), loaded from OpenML / scikit-learn. Real loaders only;
failures are reported, never silently replaced with synthetic data.
"""

from __future__ import annotations

import json
import hashlib
import math
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from tab_generators import _as_frame, _torch_seed_from_rng, gen_ctab_gan_plus, gen_tabddpm

UAI_DATASETS = [
    "abalone",
    "cpu_activity",
    "california_housing",
    "house_16h",
    "insurance",
    "ames_housing",
    "ibm_employee",
    "wine_quality_red",
    "concrete",
    "energy_heat",
    "wine_quality_white",
    "yacht",
    "power_plant",
]

# Screened 2026-08-23 (6-candidate 5-seed run). Wine / Concrete / EnergyHeat
# are now in UAI_DATASETS. The leftover three stay here for extra6 replay.
EXTRA_CANDIDATE_DATASETS = [
    "wine_quality_red",
    "concrete",
    "elevators",
    "airfoil",
    "kin8nm",
    "energy_heat",
]

# Second numeric screen (card roster_metric, 2026-08-23). Not in the locked
# roster until a full 5-seed run and a pick. Non-housing on purpose.
SCREEN2_CANDIDATE_DATASETS = [
    "wine_quality_white",
    "yacht",
    "power_plant",
    "parkinsons_tele",
    "diabetes",
    "auto_mpg",
]


@dataclass
class Dataset:
    key: str
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    cat_indices: list[int]
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_method_seed(seed: int, dataset_key: str, method: str) -> int:
    """Return a process-independent 32-bit seed for one benchmark cell."""
    payload = f"{seed}|{dataset_key}|{method}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2 ** 32)


# ----------------------------------------------------------------------
# Dataset loading (real data only; raises on failure)
# ----------------------------------------------------------------------

def _prepare(key, name, X_df, y, *, source, random_state, test_size=0.30) -> Dataset:
    X_df = X_df.copy()
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y)
    X_df = X_df.loc[valid].reset_index(drop=True)
    y = y[valid]

    drop_cols = [c for c in X_df.columns if c.lower() in {"id", "employee_number", "employeenumber"}]
    if drop_cols:
        X_df = X_df.drop(columns=drop_cols)

    cat_cols = X_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_indices = [list(X_df.columns).index(c) for c in cat_cols]
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()
    for col in cat_cols:
        train_values = X_train_df[col].astype("string").fillna("__MISSING__")
        test_values = X_test_df[col].astype("string").fillna("__MISSING__")
        levels = sorted(train_values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(levels)}
        X_train_df[col] = train_values.map(mapping).astype(float)
        X_test_df[col] = test_values.map(mapping).fillna(-1).astype(float)

    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_test = X_test_df.to_numpy(dtype=np.float64)
    num_indices = [i for i in range(X_train.shape[1]) if i not in cat_indices]
    if num_indices:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train[:, num_indices] = scaler.fit_transform(imp.fit_transform(X_train[:, num_indices]))
        X_test[:, num_indices] = scaler.transform(imp.transform(X_test[:, num_indices]))
    if cat_indices:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X_train[:, cat_indices] = imp_cat.fit_transform(X_train[:, cat_indices])
        X_test[:, cat_indices] = imp_cat.transform(X_test[:, cat_indices])

    return Dataset(
        key=key,
        name=name,
        X_train=X_train.astype(np.float64),
        y_train=y_train.astype(np.float64),
        X_test=X_test.astype(np.float64),
        y_test=y_test.astype(np.float64),
        cat_indices=cat_indices,
        source=source,
    )


def _fetch_openml_by_id(data_id: int, target: str | None = None):
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = data.data.copy()
    y = data.target if target is None else X.pop(target)
    return X, np.asarray(y, dtype=float)


def _fetch_openml_named(name: str, *, version: int | None = None, target: str | None = None):
    kwargs = {"name": name, "as_frame": True, "parser": "auto"}
    if version is not None:
        kwargs["version"] = version
    data = fetch_openml(**kwargs)
    X = data.data.copy()
    y = data.target if target is None else X.pop(target)
    return X, pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)


def _first_openml_load(attempts: list[dict], target_names: tuple[str, ...] = ()):
    """Try OpenML name/id attempts; optionally pop a named target column."""
    last_exc: Exception | None = None
    for spec in attempts:
        try:
            if "data_id" in spec:
                X, y = _fetch_openml_by_id(spec["data_id"], target=spec.get("target"))
            else:
                X, y = _fetch_openml_named(
                    spec["name"],
                    version=spec.get("version"),
                    target=spec.get("target"),
                )
            if target_names:
                for col in target_names:
                    if col in X.columns:
                        y = pd.to_numeric(X.pop(col), errors="coerce").to_numpy(dtype=float)
                        break
            return X, y
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"OpenML load failed after {len(attempts)} attempts: {last_exc}") from last_exc


def load_uai_dataset(key: str, seed: int) -> Dataset:
    if key == "abalone":
        data = fetch_openml("abalone", version=1, as_frame=True, parser="auto")
        return _prepare(key, "Abalone", data.data, data.target.astype(float), source="openml:abalone", random_state=seed)
    if key == "cpu_activity":
        data = fetch_openml("cpu_act", version=1, as_frame=True, parser="auto")
        return _prepare(key, "CPUActivity", data.data, data.target.astype(float), source="openml:cpu_act", random_state=seed)
    if key == "california_housing":
        data = fetch_california_housing(as_frame=True)
        return _prepare(key, "CaliforniaHousing", data.data, data.target.to_numpy(), source="sklearn:california_housing", random_state=seed)
    if key == "house_16h":
        data = fetch_openml("house_16H", version=1, as_frame=True, parser="auto")
        return _prepare(key, "House16H", data.data, data.target.astype(float), source="openml:house_16H", random_state=seed)
    if key == "insurance":
        data = fetch_openml("insurance", version=1, as_frame=True, parser="auto")
        return _prepare(key, "Insurance", data.data, data.target.astype(float), source="openml:insurance", random_state=seed)
    if key == "ames_housing":
        X, y = _fetch_openml_by_id(41211)
        return _prepare(key, "AmesHousing", X, y, source="openml:41211", random_state=seed)
    if key == "ibm_employee":
        for data_id in (43893, 43894, 43696):
            try:
                data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
                df = pd.concat([data.data, data.target.rename(data.target.name or "target")], axis=1)
                if "MonthlyIncome" in df.columns:
                    y = df.pop("MonthlyIncome")
                    return _prepare(key, "IBMEmployee", df, y, source=f"openml:{data_id}", random_state=seed)
            except Exception:
                pass
        url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
        try:
            df = pd.read_csv(url)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"IBM Employee real-data load failed: {exc}") from exc
        y = df.pop("MonthlyIncome")
        return _prepare(key, "IBMEmployee", df, y, source="github:IBM/employee-attrition-aif360", random_state=seed)
    if key == "melbourne_housing":
        X, y = _fetch_openml_by_id(43447)
        if "Price" in X.columns:
            y = X.pop("Price")
        return _prepare(key, "MelbourneHousing", X, y, source="openml:43447", random_state=seed)
    if key == "bike_sharing":
        data = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="auto")
        return _prepare(key, "BikeSharing", data.data, data.target.astype(float), source="openml:Bike_Sharing_Demand_v2", random_state=seed)
    if key == "wine_quality_red":
        X, y = _first_openml_load(
            [
                {"name": "wine-quality-red"},
                {"data_id": 40691},
                {"data_id": 287},
                {"name": "winequality-red"},
            ],
            target_names=("quality", "Quality"),
        )
        return _prepare(key, "WineQualityRed", X, y, source="openml:wine-quality-red", random_state=seed)
    if key == "concrete":
        # OpenML 4353 stores the target as a data column and leaves .target empty.
        # Do not fall back to id 188 (unrelated ecology table).
        X, _unused = _fetch_openml_by_id(4353)
        strength = [c for c in X.columns if "compressive" in str(c).lower() or "strength" in str(c).lower()]
        if not strength:
            raise RuntimeError(f"Concrete strength column missing; columns={list(X.columns)}")
        y = pd.to_numeric(X.pop(strength[-1]), errors="coerce").to_numpy(dtype=float)
        return _prepare(key, "Concrete", X, y, source="openml:4353", random_state=seed)
    if key == "elevators":
        X, y = _first_openml_load(
            [{"data_id": 216}, {"name": "elevators", "version": 1}],
            target_names=("Goal", "goal"),
        )
        return _prepare(key, "Elevators", X, y, source="openml:216", random_state=seed)
    if key == "airfoil":
        X, y = _first_openml_load(
            [
                {"data_id": 43919},
                {"data_id": 294},
                {"name": "airfoil_self_noise", "version": 1},
            ],
            target_names=("sound_pressure", "Scaled_sound_pressure", "y"),
        )
        return _prepare(key, "Airfoil", X, y, source="openml:43919", random_state=seed)
    if key == "kin8nm":
        X, y = _first_openml_load(
            [{"data_id": 189}, {"name": "kin8nm", "version": 1}],
            target_names=("y", "Y"),
        )
        return _prepare(key, "Kin8nm", X, y, source="openml:189", random_state=seed)
    if key == "energy_heat":
        X, y = _first_openml_load(
            [
                {"name": "energy-efficiency"},
                {"data_id": 242},
                {"data_id": 147},
                {"name": "ENB2012"},
            ],
            target_names=("Y1", "Heating_Load", "heating_load", "HL"),
        )
        return _prepare(key, "EnergyHeat", X, y, source="openml:energy-efficiency", random_state=seed)
    if key == "wine_quality_white":
        X, y = _first_openml_load(
            [
                {"name": "wine-quality-white"},
                {"data_id": 40498},
                {"data_id": 40692},
                {"name": "winequality-white"},
            ],
            target_names=("quality", "Quality"),
        )
        return _prepare(key, "WineQualityWhite", X, y, source="openml:wine-quality-white", random_state=seed)
    if key == "yacht":
        X, y = _first_openml_load(
            [
                {"data_id": 42370},
                {"data_id": 22960},
                {"name": "yacht_hydrodynamics"},
                {"name": "yacht"},
            ],
            target_names=("Residuary_resistance", "residuary_resistance", "y"),
        )
        return _prepare(key, "Yacht", X, y, source="openml:yacht", random_state=seed)
    if key == "power_plant":
        X, y = _first_openml_load(
            [
                {"name": "combined-cycle-power-plant"},
                {"data_id": 23515},
                {"name": "CCPP"},
                {"data_id": 41474},
            ],
            target_names=("PE", "Net_hourly_electrical_energy", "energy_output", "y"),
        )
        return _prepare(key, "PowerPlant", X, y, source="openml:combined-cycle-power-plant", random_state=seed)
    if key == "parkinsons_tele":
        X, y = _first_openml_load(
            [
                {"name": "parkinsons_telemonitoring"},
                {"data_id": 201},
                {"name": "Parkinsons-Telemonitoring"},
            ],
            target_names=("total_UPDRS", "motor_UPDRS", "total-UPDRS"),
        )
        return _prepare(key, "ParkinsonsTele", X, y, source="openml:parkinsons_telemonitoring", random_state=seed)
    if key == "diabetes":
        data = load_diabetes(as_frame=True)
        return _prepare(
            key,
            "Diabetes",
            data.data,
            data.target.to_numpy(),
            source="sklearn:diabetes",
            random_state=seed,
        )
    if key == "auto_mpg":
        X, y = _first_openml_load(
            [
                {"data_id": 196},
                {"name": "autoMpg"},
                {"name": "auto_mpg"},
            ],
            target_names=("class", "mpg", "MPG"),
        )
        return _prepare(key, "AutoMPG", X, y, source="openml:196", random_state=seed)
    raise ValueError(f"Unknown dataset key: {key}")


def make_joint(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-12)
    z = np.column_stack([X, (y - y_mean) / y_std])
    return z.astype(np.float64), y_mean, y_std


def make_joint_with_scale(X, y, y_mean, y_std) -> np.ndarray:
    return np.column_stack([X, (y - y_mean) / max(y_std, 1e-12)]).astype(np.float64)


def split_joint(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return z[:, :-1], z[:, -1]


def relevance(y: np.ndarray) -> np.ndarray:
    q1, med, q3 = np.quantile(y, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-8)
    phi = np.maximum(0.0, np.abs(y - med) - 0.5 * iqr) / (1.5 * iqr)
    return np.clip(phi, 0.0, 1.0)


def relevance_from_ref(y: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    """Relevance of ``y`` under the IQR law of ``y_ref`` (train-only)."""
    q1, med, q3 = np.quantile(y_ref, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-8)
    phi = np.maximum(0.0, np.abs(y - med) - 0.5 * iqr) / (1.5 * iqr)
    return np.clip(phi, 0.0, 1.0)


def smoter_implied_rare_mass(y: np.ndarray) -> float:
    """Rare mass of the published SMOTER proposal ``p ∝ 0.05 + phi``."""
    phi = relevance(y)
    rare = phi > 0.5
    if not np.any(rare):
        return 0.0
    p = 0.05 + phi
    p = p / p.sum()
    return float(p[rare].sum())


def _constrained_fw_vertex(
    grad: np.ndarray, rare_atom: np.ndarray, pi: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linear oracle on the simplex slice ``1_R^T q >= pi``."""
    s = int(np.argmin(grad))
    if pi <= 0.0 or rare_atom[s] or not rare_atom.any():
        return np.array([s], dtype=int), np.array([1.0])
    rare_idx = np.flatnonzero(rare_atom)
    s_r = int(rare_idx[np.argmin(grad[rare_idx])])
    return np.array([s, s_r], dtype=int), np.array([1.0 - pi, pi])


# ----------------------------------------------------------------------
# Shared local-selection machinery
# ----------------------------------------------------------------------

def pairwise_affinity(z: np.ndarray, k: int) -> np.ndarray:
    """Row-normalized Gaussian affinity restricted to the k nearest neighbors."""
    n = len(z)
    dist = cdist(z, z, metric="euclidean")
    kth = np.partition(dist, min(k, n - 1), axis=1)[:, min(k, n - 1)]
    sigma = float(np.median(kth[kth > 0])) if np.any(kth > 0) else 1.0
    sigma = max(sigma, 1e-6)
    aff = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(aff, 0.0)
    if k < n - 1:
        keep = np.zeros_like(aff, dtype=bool)
        idx = np.argpartition(dist + (aff <= 0) * 1e12, kth=min(k, n - 1), axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        keep[rows, idx] = True
        aff = np.where(keep, aff, 0.0)
    row_sum = aff.sum(axis=1, keepdims=True)
    dead = row_sum[:, 0] <= 0
    if np.any(dead):
        aff[dead] = 1.0 / max(n - 1, 1)
        aff[dead, np.where(dead)[0]] = 0.0
        row_sum = aff.sum(axis=1, keepdims=True)
    return aff / np.maximum(row_sum, 1e-12)


EXPAND_EPS = (math.sqrt(3.0) - 1.0) / 2.0  # Uniform[-eps, 1+eps] gives E[W] = E[W^2] = 1/2.


def nested_weights(rng: np.random.Generator, m: int) -> np.ndarray:
    """Nested structure-preserving weights on K = 4 vertices.

    W = (S T, S (1-T), (1-S) T', (1-S) (1-T')) with S, T, T' i.i.d.
    Uniform[-EXPAND_EPS, 1 + EXPAND_EPS], so E[.] = E[.^2] = 1/2 and W sums
    to one. Every off-diagonal product W_k W_l carries a factor S(1-S),
    T(1-T), or T'(1-T') with mean zero, so E[W_k W_l] = 0 for k != l. The
    signed local affine mix therefore reproduces the empirical mean and second
    moment for every index coupling with uniform marginals. With an exchangeable
    index tuple, it also preserves third-order moments by applying the
    two-record identity within each block and then to the outer pair. These
    are proposal-law identities, not finite-pool or selected-law equalities.
    This is a nested application of
    the two-record condition E[W] = E[W^2]; it is not a barycenter.
    """
    S = rng.uniform(-EXPAND_EPS, 1.0 + EXPAND_EPS, size=m)
    T = rng.uniform(-EXPAND_EPS, 1.0 + EXPAND_EPS, size=m)
    Tp = rng.uniform(-EXPAND_EPS, 1.0 + EXPAND_EPS, size=m)
    return np.stack([S * T, S * (1.0 - T), (1.0 - S) * Tp, (1.0 - S) * (1.0 - Tp)], axis=1)


def multiscale_simplex_draws(
    z: np.ndarray,
    n_draw: int,
    rng: np.random.Generator,
    *,
    k_values: tuple[int, ...] = (5, 10, 20, 40),
    K: int = 4,
    weight_mode: str = "nested",
    alpha: float = 0.8,
    min_weight: float = 0.03,
    return_composition: bool = False,
):
    """Multi-scale kernel-local affine mixes (K conditionally i.i.d. vertices).

    A scale k is chosen uniformly from ``k_values``. With P the row-normalized
    k-NN Gaussian affinity, the anchor j is drawn with probability
    c_j / n, c_j = sum_i P_ij, and the K vertices are i.i.d. given j with
    probability P_ij / c_j. Each vertex index is then marginally uniform on
    the training rows.

    weight_mode
    -----------
    "nested"    : signed nested weights (K = 4, see ``nested_weights``); the
                  raw law reproduces the empirical mean and second moment
                  for every coupling with uniform marginals, and also order 3
                  for the exchangeable conditional-iid index law used here.
    "dirichlet" : nonnegative Dirichlet(alpha) weights with a floor, i.e. a
                  local barycenter (Simplex-LOWEM ablation only); no
                  second-moment guarantee.
    """
    n = len(z)
    if weight_mode == "nested" and K != 4:
        raise ValueError("nested weight mode is defined for K = 4")
    k_values = tuple(min(k, n - 1) for k in k_values)
    affinities = {k: pairwise_affinity(z, k=k) for k in sorted(set(k_values))}
    scale_choice = rng.integers(0, len(k_values), size=n_draw)
    out = np.empty((n_draw, z.shape[1]), dtype=float)
    comp_idx = np.empty((n_draw, K), dtype=np.int64)
    comp_w = np.empty((n_draw, K), dtype=float)

    for s_id, k in enumerate(k_values):
        rows = np.where(scale_choice == s_id)[0]
        if len(rows) == 0:
            continue
        row_norm = affinities[k]
        col_mass = row_norm.sum(axis=0)
        anchor_prob = col_mass / max(col_mass.sum(), 1e-12)
        anchors = rng.choice(n, size=len(rows), replace=True, p=anchor_prob)
        # Group draws by anchor so vertex sampling is vectorized per anchor.
        order = np.argsort(anchors)
        rows_sorted = rows[order]
        anchors_sorted = anchors[order]
        uniq, starts = np.unique(anchors_sorted, return_index=True)
        boundaries = list(starts) + [len(anchors_sorted)]
        for u_i, j in enumerate(uniq):
            block = rows_sorted[boundaries[u_i]:boundaries[u_i + 1]]
            p = row_norm[:, j]
            p = p / max(p.sum(), 1e-12)
            verts = rng.choice(n, size=(len(block), K), replace=True, p=p)
            if weight_mode == "nested":
                weights = nested_weights(rng, len(block))
            else:
                weights = rng.dirichlet(np.full(K, alpha), size=len(block))
                weights = np.maximum(weights, min_weight)
                weights = weights / weights.sum(axis=1, keepdims=True)
                # Reject fully degenerate simplices (all vertices identical).
                for _ in range(4):
                    degen = (verts == verts[:, [0]]).all(axis=1)
                    if not degen.any():
                        break
                    verts[degen] = rng.choice(n, size=(int(degen.sum()), K), replace=True, p=p)
            out[block] = np.einsum("bk,bkd->bd", weights, z[verts])
            comp_idx[block] = verts
            comp_w[block] = weights
    if return_composition:
        return out, comp_idx, comp_w
    return out


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------

def _spd_sqrt(mat: np.ndarray, inverse: bool = False) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.maximum(vals, 1e-10)
    power = -0.5 if inverse else 0.5
    return vecs @ np.diag(vals ** power) @ vecs.T


def covariance_matching_map(z_syn: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    """Unique symmetric positive-definite A with A Sigma_syn A = Sigma_ref
    (the optimal-transport map between the Gaussian moment surrogates):
    A = Sigma_s^{-1/2} (Sigma_s^{1/2} Sigma_r Sigma_s^{1/2})^{1/2} Sigma_s^{-1/2}.
    """
    d = z_syn.shape[1]
    cs = np.cov(z_syn, rowvar=False) + 1e-6 * np.eye(d)
    cr = np.cov(z_ref, rowvar=False) + 1e-6 * np.eye(d)
    cs_half = _spd_sqrt(cs)
    cs_inv_half = _spd_sqrt(cs, inverse=True)
    middle = _spd_sqrt(cs_half @ cr @ cs_half)
    return cs_inv_half @ middle @ cs_inv_half


def affine_match_mean_cov(z_syn: np.ndarray, z_ref: np.ndarray, *, clip: bool = False) -> np.ndarray:
    A = covariance_matching_map(z_syn, z_ref)
    out = (z_syn - z_syn.mean(axis=0)) @ A + z_ref.mean(axis=0)
    if clip:
        lo = np.quantile(z_ref, 0.001, axis=0)
        hi = np.quantile(z_ref, 0.999, axis=0)
        out = np.clip(out, lo, hi)
    return out


def shrinkage_moment_calibration(z_syn: np.ndarray, z_ref: np.ndarray, rho: float) -> np.ndarray:
    """Z_cal = (1-rho) Z + rho [mu_ref + (Z - mu_syn) A] with A the unique
    symmetric covariance-matching map (Eq. shrinkcal in the manuscript)."""
    repaired = affine_match_mean_cov(z_syn, z_ref, clip=False)
    return (1.0 - rho) * z_syn + rho * repaired


RHO_GRID = (0.0, 0.35, 0.7, 1.0)


def nearest_train_dist(
    z_syn: np.ndarray,
    z_ref: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Exact nearest-training distance for every synthetic row."""
    if len(z_syn) == 0 or len(z_ref) == 0:
        raise ValueError("nearest_train_dist requires nonempty arrays")
    out = np.empty(len(z_syn), dtype=float)
    for start in range(0, len(z_syn), batch_size):
        stop = min(start + batch_size, len(z_syn))
        out[start:stop] = cdist(z_syn[start:stop], z_ref).min(axis=1)
    return out


def adaptive_separation_radius(z_ref: np.ndarray, *, q: float = 0.02) -> float:
    """Quantile of exact leave-one-out nearest-neighbor distances."""
    if len(z_ref) < 2:
        return 0.0
    nn_model = NearestNeighbors(n_neighbors=2).fit(z_ref)
    nn = nn_model.kneighbors(z_ref, return_distance=True)[0][:, 1]
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return 0.0
    return float(np.quantile(nn, q))


SEPARATION_RELAX_FACTOR = 0.75
SEPARATION_MAX_RELAX = 12


class SeparationFailure(RuntimeError):
    """The filter cannot retain 2m candidates, or an active candidate is a copy
    of a training row (d_b = 0); nothing is released."""


def quantile_radius_filter(
    d_train: np.ndarray,
    radius: float,
    need: int,
) -> tuple[np.ndarray, float, int] | None:
    """Keep candidates with d_b >= radius_used, relaxing geometrically.

    The applied radius starts at ``radius`` (the 2% leave-one-out quantile, no
    positive floor; it is 0 when the training sample has duplicate rows) and is
    multiplied by SEPARATION_RELAX_FACTOR at most SEPARATION_MAX_RELAX times
    while fewer than ``need`` candidates survive. Returns None when fewer than
    ``need`` survive. A zero radius removes nothing; copies are then excluded
    by the release check on the active support in ``gen_momentanchor``.
    """
    used = max(float(radius), 0.0)
    keep = d_train >= used
    n_relax = 0
    while keep.sum() < need and n_relax < SEPARATION_MAX_RELAX and used > 0.0:
        used *= SEPARATION_RELAX_FACTOR
        n_relax += 1
        keep = d_train >= used
    if keep.sum() < need:
        return None
    return keep, used, n_relax


def reference_rbf_gamma(reference: np.ndarray) -> float:
    """Median-heuristic RBF precision fitted to one fixed reference sample."""
    d2 = cdist(reference, reference, "sqeuclidean")
    positive = d2[d2 > 0]
    med = np.median(positive) if len(positive) else 1.0
    return 1.0 / max(float(med), 1e-8)


def composite_score(
    z_syn: np.ndarray,
    z_val: np.ndarray,
    z_ref: np.ndarray,
    *,
    mmd_gamma: float,
) -> float:
    """Held-out composite with one validation-fitted kernel for every rho."""
    mom = moment_errors(z_syn, z_ref)
    mmd = rbf_mmd2(z_syn, z_val, gamma=mmd_gamma)
    radius = adaptive_separation_radius(z_ref, q=0.01)
    dmin = float(np.min(nearest_train_dist(z_syn, z_ref)))
    penalty = max(radius - dmin, 0.0)
    return (
        0.20 * mmd
        + mom["mean_err"]
        + mom["cov_err"]
        + 0.35 * mom["third_err"]
        + 0.25 * mom["fourth_err"]
        + 2.0 * penalty
    )


def select_rho_heldout(
    z: np.ndarray,
    n_syn: int,
    rng: np.random.Generator,
    *,
    weight_mode: str = "nested",
) -> tuple[float, list[dict]]:
    """Choose the shrinkage strength on an internal 70/30 split of the
    training sample (fit/validation), disjoint from the test split."""
    idx = rng.permutation(len(z))
    split = max(64, int(0.7 * len(z)))
    fit, val = z[idx[:split]], z[idx[split:]]
    if len(val) < 32:
        val = fit
    raw = multiscale_simplex_draws(fit, n_syn, rng, weight_mode=weight_mode)
    mmd_gamma = reference_rbf_gamma(val)
    grid = []
    for rho in RHO_GRID:
        cand = shrinkage_moment_calibration(raw, fit, rho=rho)
        score = composite_score(cand, val, fit, mmd_gamma=mmd_gamma)
        grid.append({"rho": rho, "score": float(score)})
    best = min(grid, key=lambda g: g["score"])
    return float(best["rho"]), grid


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def rbf_mmd2(x: np.ndarray, y: np.ndarray, *, gamma: float | None = None) -> float:
    """Biased MMD^2 using all supplied rows and a fixed reference kernel."""
    if gamma is None:
        gamma = reference_rbf_gamma(y)
    kxx = np.exp(-gamma * cdist(x, x, "sqeuclidean"))
    kyy = np.exp(-gamma * cdist(y, y, "sqeuclidean"))
    kxy = np.exp(-gamma * cdist(x, y, "sqeuclidean"))
    return max(0.0, float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()))


def make_projections(d: int, rng: np.random.Generator, n_proj: int = 64) -> np.ndarray:
    projections = rng.normal(size=(n_proj, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True) + 1e-12
    return projections


def sliced_w1(x: np.ndarray, y: np.ndarray, projections: np.ndarray) -> float:
    vals = []
    for v in projections:
        vals.append(wasserstein_distance(x @ v, y @ v))
    return float(np.mean(vals))


def moment_errors(x: np.ndarray, y: np.ndarray) -> dict:
    def rel(a, b):
        return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - mu_x, y - mu_y
    return {
        "mean_err": rel(mu_x, mu_y),
        "cov_err": rel(np.cov(xc, rowvar=False), np.cov(yc, rowvar=False)),
        "third_err": rel(np.mean(xc ** 3, axis=0), np.mean(yc ** 3, axis=0)),
        "fourth_err": rel(np.mean(xc ** 4, axis=0), np.mean(yc ** 4, axis=0)),
    }


def utility_metrics(z_syn: np.ndarray, y_mean: float, y_std: float, ds: Dataset) -> dict:
    X_syn, yz_syn = split_joint(z_syn)
    y_syn = yz_syn * y_std + y_mean
    model = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, random_state=0)
    model.fit(X_syn, y_syn)
    pred = model.predict(ds.X_test)
    rmse = math.sqrt(mean_squared_error(ds.y_test, pred))
    phi = relevance(ds.y_test)
    rare = phi > 0.5
    rare_rmse = math.sqrt(mean_squared_error(ds.y_test[rare], pred[rare])) if rare.any() else float("nan")
    # SERA: integrating the relevance-threshold indicator gives phi-weighted SE.
    sera = float(np.mean(phi * (ds.y_test - pred) ** 2))
    return {"rmse": float(rmse), "rare_rmse": float(rare_rmse), "sera": sera, "n_rare": int(rare.sum())}


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------


def _moment_features(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, pair_idx: np.ndarray, order: int) -> np.ndarray:
    xs = (x - mu) / sd
    feats = [xs]
    if order >= 2:
        feats.append(xs ** 2)
        if len(pair_idx):
            feats.append(xs[:, pair_idx[:, 0]] * xs[:, pair_idx[:, 1]])
    if order >= 3:
        feats.append(xs ** 3)
    return np.concatenate(feats, axis=1)


def _gen_momentanchor_two_stage(
    z: np.ndarray,
    n_syn: int,
    rng: np.random.Generator,
    **fw_kw,
) -> tuple[np.ndarray, dict]:
    """Bulk FW on the full table plus rare-neighbor FW on the rare slice.

    Release count for the rare block is the SMOTER implied rare mass, not a
    free mix weight. Each block is the locked unconstrained MomentAnchor
    selector on its own target sample.
    """
    y = z[:, -1]
    rare = relevance(y) > 0.5
    pi = smoter_implied_rare_mass(y)
    if int(rare.sum()) < 4 or pi <= 0.0:
        return gen_momentanchor(z, n_syn, rng, **fw_kw)
    m_r = int(round(pi * n_syn))
    m_r = min(max(m_r, 1), n_syn - 1)
    m_b = n_syn - m_r
    seed_r, seed_b, seed_sh = rng.integers(0, 2 ** 32, size=3, dtype=np.uint64)
    out_r, meta_r = gen_momentanchor(
        z[rare], m_r, np.random.default_rng(seed_r), **fw_kw
    )
    out_b, meta_b = gen_momentanchor(
        z, m_b, np.random.default_rng(seed_b), **fw_kw
    )
    out = np.vstack([out_r, out_b])
    np.random.default_rng(seed_sh).shuffle(out)
    audit = dict(meta_b.get("audit_record") or {})
    audit.update({
        "pi_smoter": float(pi),
        "n_rare_release": int(m_r),
        "n_bulk_release": int(m_b),
    })
    return out, {
        "backend": "momentanchor_two_stage",
        "pi_smoter": float(pi),
        "n_rare_release": int(m_r),
        "n_bulk_release": int(m_b),
        "n_rare_train": int(rare.sum()),
        "rare_meta": meta_r,
        "bulk_meta": meta_b,
        "audit_record": audit,
        "certificate": meta_b.get("certificate", {}),
    }


def gen_momentanchor(
    z: np.ndarray,
    n_syn: int,
    rng: np.random.Generator,
    *,
    fw_iters: int = 2000,
    moment_order: int = 3,
    lambda_mom: float = 1.0,
    pool_factor: int = 6,
    pool_cap: int = 4096,
    use_rare_kernel: bool = False,
    use_rare_mass_constraint: bool = False,
    use_two_stage_rare: bool = False,
) -> tuple[np.ndarray, dict]:
    """Frank-Wolfe candidate selection with an auditable objective.

    A fixed pool of signed local affine mixes is calibrated, filtered by
    encoded-space output separation at the 2% leave-one-out quantile radius
    (failure, not copying, when the filter cannot retain 2m candidates or an
    active candidate sits at distance 0), and reweighted on the simplex to minimize exact-kernel
    MMD^2 plus a quadratic moment-feature residual. Optional
    ``use_rare_kernel`` is off by default (2026-08-24 smoke failed). The
    audit record reports the global training MMD; only the hard separation
    statement transfers without Monte Carlo error to the finite multinomial
    resample.
    """
    n, d = z.shape
    if use_two_stage_rare:
        return _gen_momentanchor_two_stage(
            z,
            n_syn,
            rng,
            fw_iters=fw_iters,
            moment_order=moment_order,
            lambda_mom=lambda_mom,
            pool_factor=pool_factor,
            pool_cap=pool_cap,
        )
    proposal_seed, calibration_seed, feature_seed, release_seed = rng.integers(
        0, 2 ** 32, size=4, dtype=np.uint64
    )
    proposal_rng = np.random.default_rng(proposal_seed)
    calibration_rng = np.random.default_rng(calibration_seed)
    feature_rng = np.random.default_rng(feature_seed)
    release_rng = np.random.default_rng(release_seed)
    B = int(min(max(pool_factor * n_syn, n_syn), pool_cap))
    need = 2 * n_syn
    if B < need:
        raise ValueError(f"initial pool size B0={B} must be at least 2m={need}")
    pool, comp_idx, comp_w = multiscale_simplex_draws(
        z,
        B,
        proposal_rng,
        return_composition=True,
    )

    # Pool-level shrinkage calibration; strength chosen on an internal
    # 70/30 split of the training sample (disjoint from the test split).
    rho, rho_grid = select_rho_heldout(
        z,
        min(B, 2 * n_syn),
        calibration_rng,
        weight_mode="nested",
    )
    pool = shrinkage_moment_calibration(pool, z, rho=rho)

    # Output-level encoded-space separation filter at the 2% leave-one-out
    # quantile (the protocol of the reported campaign; no positive floor).
    radius = adaptive_separation_radius(z, q=0.02)
    d_train = nearest_train_dist(pool, z)
    filt = quantile_radius_filter(d_train, radius, need)
    n_regen = 0
    if filt is None:
        # Redraw and recalibrate the pool once with the same strength.
        n_regen = 1
        pool, comp_idx, comp_w = multiscale_simplex_draws(
            z, B, proposal_rng, return_composition=True
        )
        pool = shrinkage_moment_calibration(pool, z, rho=rho)
        d_train = nearest_train_dist(pool, z)
        filt = quantile_radius_filter(d_train, radius, need)
        if filt is None:
            raise SeparationFailure(
                f"fewer than 2m={need} candidates survive the relaxed radius after one regeneration"
            )
    keep, used_radius, n_relax = filt
    pool, comp_idx, comp_w, d_train = (
        pool[keep],
        comp_idx[keep],
        comp_w[keep],
        d_train[keep],
    )
    B = len(pool)
    y_train = z[:, -1]
    pi_smoter = (
        smoter_implied_rare_mass(y_train) if use_rare_mass_constraint else 0.0
    )
    if use_rare_mass_constraint and pi_smoter > 0.0:
        train_rare = np.flatnonzero(relevance(y_train) > 0.5)
        if len(train_rare) >= 2:
            n_extra = max(128, int(0.25 * B))
            extra, extra_idx, extra_w = multiscale_simplex_draws(
                z[train_rare],
                n_extra,
                proposal_rng,
                return_composition=True,
            )
            extra_idx = train_rare[extra_idx]
            extra_d = nearest_train_dist(extra, z)
            extra_keep = extra_d >= used_radius
            if extra_keep.sum() < 16:
                extra_keep = np.zeros(len(extra), dtype=bool)
            extra, extra_idx, extra_w, extra_d = (
                extra[extra_keep], extra_idx[extra_keep], extra_w[extra_keep], extra_d[extra_keep]
            )
            pool = np.vstack([pool, extra])
            comp_idx = np.vstack([comp_idx, extra_idx])
            comp_w = np.vstack([comp_w, extra_w])
            d_train = np.concatenate([d_train, extra_d])
            B = len(pool)

    # Exact RBF kernel quantities with one training-reference bandwidth.
    gamma = reference_rbf_gamma(z)
    K_pp = np.exp(-gamma * cdist(pool, pool, "sqeuclidean"))   # (B, B)
    k_pt = np.exp(-gamma * cdist(pool, z, "sqeuclidean")).mean(axis=1)  # (B,)
    k_tt = float(np.exp(-gamma * cdist(z, z, "sqeuclidean")).mean())

    # Rare-region kernel. Same bandwidth; beta is the rare-row fraction.
    # Threshold 1/2 matches the downstream rare mask in utility_metrics.
    y_train = z[:, -1]
    rare_mask = relevance(y_train) > 0.5
    n_rare = int(rare_mask.sum())
    beta_rare = (
        float(n_rare) / float(n)
        if use_rare_kernel and n_rare >= 5
        else 0.0
    )
    if beta_rare > 0.0:
        z_rare = z[rare_mask]
        k_pr = np.exp(-gamma * cdist(pool, z_rare, "sqeuclidean")).mean(axis=1)
        k_rr = float(np.exp(-gamma * cdist(z_rare, z_rare, "sqeuclidean")).mean())
    else:
        k_pr = np.zeros(B)
        k_rr = 0.0

    # Moment features (coordinatewise orders 1..r + pairwise cross second moments).
    mu, sd = z.mean(axis=0), z.std(axis=0) + 1e-8
    pair_idx = np.array([(i, j) for i in range(d) for j in range(i + 1, d)], dtype=int)
    if len(pair_idx) > 800:
        sel = feature_rng.choice(len(pair_idx), 800, replace=False)
        pair_idx = pair_idx[sel]
    pair_idx = pair_idx.reshape(-1, 2)
    if moment_order >= 2:
        Psi_raw = _moment_features(pool, mu, sd, pair_idx, moment_order)
        F_train = _moment_features(z, mu, sd, pair_idx, moment_order)
        f_sd = F_train.std(axis=0)
        # Standardize each moment feature by its training-sample dispersion so
        # heavy-tailed coordinates cannot dominate the penalty; features with
        # (near-)zero dispersion carry no moment information and are dropped
        # (they arise from constant-encoded columns and would otherwise blow
        # up the standardized residual).
        keep = f_sd > 1e-6
        if keep.any():
            Psi = Psi_raw[:, keep] / f_sd[keep]
            m_hat = (F_train[:, keep] / f_sd[keep]).mean(axis=0)
            lam = lambda_mom / Psi.shape[1]
        else:
            Psi = np.zeros((B, 1))
            m_hat = np.zeros(1)
            lam = 0.0
    else:
        Psi = np.zeros((B, 1))
        m_hat = np.zeros(1)
        lam = 0.0

    # Frank-Wolfe on the simplex with gamma_t = 2/(t+2).
    # Optional linear slice: rare-atom mass >= SMOTER implied rare mass.
    phi_pool = relevance_from_ref(pool[:, -1], y_train)
    rare_atom = phi_pool > 0.5
    if use_rare_mass_constraint and pi_smoter > 0.0 and rare_atom.any() and (~rare_atom).any():
        q = np.zeros(B)
        q[rare_atom] = pi_smoter / float(rare_atom.sum())
        q[~rare_atom] = (1.0 - pi_smoter) / float((~rare_atom).sum())
    else:
        q = np.full(B, 1.0 / B)
    kq = K_pp @ q          # (B,)
    mm = Psi.T @ q         # (P,)
    for t in range(fw_iters):
        grad = (
            2.0 * (kq - k_pt)
            + 2.0 * beta_rare * (kq - k_pr)
            + 2.0 * lam * (Psi @ (mm - m_hat))
        )
        gamma_t = 2.0 / (t + 2.0)
        if use_rare_mass_constraint and pi_smoter > 0.0:
            idxs, wts = _constrained_fw_vertex(grad, rare_atom, pi_smoter)
        else:
            idxs, wts = np.array([int(np.argmin(grad))], dtype=int), np.array([1.0])
        q *= (1.0 - gamma_t)
        kq *= (1.0 - gamma_t)
        mm *= (1.0 - gamma_t)
        for i, w in zip(idxs, wts):
            q[i] += gamma_t * w
            kq += gamma_t * w * K_pp[i]
            mm += gamma_t * w * Psi[i]

    # Audit record at the solution (exact kernel and exact separation).
    # U_train remains the global training MMD, not the rare-region term.
    u_train = max(0.0, float(q @ kq - 2.0 * (q @ k_pt) + k_tt))
    u_rare = (
        max(0.0, float(q @ kq - 2.0 * (q @ k_pr) + k_rr))
        if beta_rare > 0.0
        else 0.0
    )
    r_mom = float(np.linalg.norm(mm - m_hat)) if moment_order >= 2 else 0.0
    grad = (
        2.0 * (kq - k_pt)
        + 2.0 * beta_rare * (kq - k_pr)
        + 2.0 * lam * (Psi @ (mm - m_hat))
    )
    fw_gap = max(0.0, float(q @ grad - np.min(grad)))
    objective = float(u_train + beta_rare * u_rare + lam * r_mom ** 2)
    usage = np.zeros(n)
    np.add.at(usage, comp_idx.ravel(), (q[:, None] * comp_w).ravel())
    # This is a proposal-lineage diagnostic only: calibration changes candidate
    # coordinates, so it is not a marginal constraint or a mean-error bound.
    r_marg = float(np.abs(usage - 1.0 / n).sum())
    support = np.flatnonzero(q > 0.0)
    d_sep = float(d_train[support].min()) if len(support) else float("nan")
    if not d_sep > 0.0:
        # Release check: a copy of a training row in the active support
        # (possible only when the applied radius is 0) means no release.
        raise SeparationFailure("an active candidate coincides with a training row (d_sep = 0)")

    draw = release_rng.choice(B, size=n_syn, replace=True, p=q)
    out = pool[draw]
    audit = {
        "U_train": u_train,
        "r_marg": r_marg,
        "r_mom": r_mom,
        "d_sep": d_sep,
        "objective": objective,
        "fw_gap": fw_gap,
        "U_rare": u_rare,
        "beta_rare": beta_rare,
        "n_rare": n_rare,
        "pi_smoter": float(pi_smoter) if use_rare_mass_constraint else 0.0,
        "rare_mass_attained": float(q[rare_atom].sum()) if rare_atom.any() else 0.0,
    }
    return out, {
        "backend": "momentanchor_fw",
        "fw_iters": fw_iters,
        "moment_order": moment_order,
        "use_rare_kernel": bool(use_rare_kernel),
        "use_rare_mass_constraint": bool(use_rare_mass_constraint),
        "beta_rare": beta_rare,
        "n_rare": n_rare,
        "pi_smoter": float(pi_smoter) if use_rare_mass_constraint else 0.0,
        "pool_size": int(B),
        "rho_pool": rho,
        "rho_grid": rho_grid,
        "separation_radius_requested": radius,
        "separation_radius_used": used_radius,
        "separation_relaxations": int(n_relax),
        "pool_regenerations": int(n_regen),
        "privacy_radius_requested": radius,
        "privacy_radius_used": used_radius,
        "audit_record": audit,
        "certificate": {
            "U_star": u_train,
            "r_marg": r_marg,
            "r_mom": r_mom,
            "d_priv": d_sep,
        },
        "active_support": int(len(support)),
    }


def gen_simplex_lowem(z: np.ndarray, n_syn: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    selection_seed, draw_seed = rng.integers(0, 2 ** 32, size=2, dtype=np.uint64)
    rho, grid = select_rho_heldout(
        z,
        n_syn,
        np.random.default_rng(selection_seed),
        weight_mode="dirichlet",
    )
    raw = multiscale_simplex_draws(
        z,
        n_syn,
        np.random.default_rng(draw_seed),
        weight_mode="dirichlet",
    )
    out = shrinkage_moment_calibration(raw, z, rho=rho)
    return out, {
        "backend": "simplex_lowem",
        "rho_selected": rho,
        "rho_grid": grid,
        "weight_mode": "dirichlet",
    }


def gen_bootstrap(z: np.ndarray, n_syn: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    idx = rng.choice(len(z), size=n_syn, replace=True)
    return z[idx].copy(), {"backend": "bootstrap_rows"}


def gen_global_mixup(z: np.ndarray, n_syn: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = len(z)
    i = rng.choice(n, size=n_syn, replace=True)
    ip = rng.choice(n, size=n_syn, replace=True)
    for _ in range(5):
        same = i == ip
        if not np.any(same):
            break
        ip[same] = rng.choice(n, size=int(same.sum()), replace=True)
    w = rng.beta(0.4, 0.4, size=n_syn)[:, None]
    return w * z[i] + (1.0 - w) * z[ip], {"backend": "global_beta_mixup"}


def gen_smoter(z: np.ndarray, n_syn: int, rng: np.random.Generator, *, k: int = 8) -> tuple[np.ndarray, dict]:
    y = z[:, -1]
    phi = relevance(y)
    prob = 0.05 + phi
    prob = prob / prob.sum()
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(z))).fit(z)
    neigh = nn.kneighbors(z, return_distance=False)
    anchors = rng.choice(len(z), size=n_syn, replace=True, p=prob)
    out = np.empty((n_syn, z.shape[1]), dtype=float)
    for t, i in enumerate(anchors):
        choices = neigh[i][1:] if len(neigh[i]) > 1 else neigh[i]
        j = int(rng.choice(choices))
        w = rng.uniform(0.0, 1.0)
        out[t] = w * z[i] + (1.0 - w) * z[j]
    return out, {"backend": "smoter_relevance_knn", "k": k}


def gen_gaussian_copula(z: np.ndarray, n_syn: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n, d = z.shape
    u = np.empty_like(z)
    for j in range(d):
        u[:, j] = sps.rankdata(z[:, j], method="average") / (n + 1)
    g = sps.norm.ppf(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.atleast_2d(np.corrcoef(g, rowvar=False))
    # Constant columns yield NaN correlations; treat them as independent.
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = corr + 1e-6 * np.eye(d)
    vals, vecs = np.linalg.eigh(corr)
    L = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-10)))
    gs = rng.standard_normal((n_syn, d)) @ L.T
    us = sps.norm.cdf(gs)
    out = np.empty((n_syn, d))
    for j in range(d):
        out[:, j] = np.quantile(z[:, j], us[:, j])
    return out, {"backend": "gaussian_copula_empirical_marginals"}


def gen_kde(z: np.ndarray, n_syn: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    scale = z.std(axis=0)
    pos = scale[scale > 0]
    fill = float(np.median(pos)) if len(pos) else 1.0
    scale = np.where(scale > 0, scale, fill)
    kde = None
    backend = "gaussian_kde_scott"
    for jitter in (0.0, 1e-4, 1e-3, 1e-2):
        zj = z if jitter == 0.0 else z + jitter * scale * rng.standard_normal(z.shape)
        try:
            kde = sps.gaussian_kde(zj.T)
            backend = "gaussian_kde_scott" if jitter == 0.0 else f"gaussian_kde_scott_jitter_{jitter:g}"
            break
        except Exception:
            kde = None
    lo = np.quantile(z, 0.001, axis=0)
    hi = np.quantile(z, 0.999, axis=0)
    if kde is None:
        # Last resort: independent coordinate-wise Gaussians (still a published
        # KDE-family fallback, not a silent row copy).
        out = rng.normal(loc=z.mean(axis=0), scale=scale, size=(n_syn, z.shape[1]))
        return np.clip(out, lo, hi), {"backend": "independent_gaussian_kde_fallback"}
    seed = int(rng.integers(0, 2 ** 31 - 1))
    out = kde.resample(n_syn, seed=seed).T
    return np.clip(out, lo, hi), {"backend": backend}


def gen_ctgan(z: np.ndarray, n_syn: int, rng: np.random.Generator, *, epochs: int = 100) -> tuple[np.ndarray, dict]:
    """Conditional tabular GAN (Xu et al., 2019). All (label-encoded, standardized)
    columns are treated as continuous. Epoch budget is fixed for CPU runs."""
    from ctgan import CTGAN

    _torch_seed_from_rng(rng)
    df = _as_frame(z)
    model = CTGAN(epochs=epochs, verbose=False)
    model.fit(df, discrete_columns=[])
    out = model.sample(n_syn)[df.columns].to_numpy(dtype=np.float64)
    return out, {"backend": "ctgan", "epochs": epochs}


def gen_tvae(z: np.ndarray, n_syn: int, rng: np.random.Generator, *, epochs: int = 100) -> tuple[np.ndarray, dict]:
    """Tabular variational autoencoder (Xu et al., 2019); deep generator."""
    from ctgan import TVAE

    _torch_seed_from_rng(rng)
    df = _as_frame(z)
    model = TVAE(epochs=epochs)
    model.fit(df, discrete_columns=[])
    out = model.sample(n_syn)[df.columns].to_numpy(dtype=np.float64)
    return out, {"backend": "tvae", "epochs": epochs}


GENERATORS: dict[str, Callable] = {
    "MomentAnchor": gen_momentanchor,
    "Simplex-LOWEM": gen_simplex_lowem,
    "Bootstrap": gen_bootstrap,
    "Global Mixup": gen_global_mixup,
    "SMOTER": gen_smoter,
    "Gaussian Copula": gen_gaussian_copula,
    "KDE Sampler": gen_kde,
    "CTGAN": gen_ctgan,
    "TVAE": gen_tvae,
}

# Experimental approximations used only for explicit smoke studies. These are
# not official implementations and never enter submission tables.
EXPERIMENTAL_GENERATORS: dict[str, Callable] = {
    "TabDDPM-style": gen_tabddpm,
    "CTAB-GAN+-style": gen_ctab_gan_plus,
}

OURS = {"MomentAnchor"}


# ----------------------------------------------------------------------
# Evaluation and ranking
# ----------------------------------------------------------------------

RANK_METRICS_LOWER = [
    "mmd2_test",
    "sliced_w1_test",
    "mean_err",
    "cov_err",
    "third_err",
    "fourth_err",
]


def project_categorical_codes(
    z_syn: np.ndarray,
    z_train: np.ndarray,
    cat_indices: list[int],
) -> np.ndarray:
    """Project encoded categorical features to levels observed in training."""
    if not cat_indices:
        return z_syn
    out = np.asarray(z_syn, dtype=float).copy()
    for col in cat_indices:
        levels = np.unique(z_train[:, col])
        if len(levels):
            nearest = np.abs(out[:, col, None] - levels[None, :]).argmin(axis=1)
            out[:, col] = levels[nearest]
    return out


def evaluate_generator(
    method,
    z_syn,
    z_train,
    z_test,
    ds,
    y_mean,
    y_std,
    *,
    mmd_gamma,
    projections,
) -> dict:
    min_dist = float(np.min(nearest_train_dist(z_syn, z_train)))
    out = {
        "method": method,
        "mmd2_test": rbf_mmd2(z_syn, z_test, gamma=mmd_gamma),
        "sliced_w1_test": sliced_w1(z_syn, z_test, projections),
        "min_train_distance": min_dist,
        "eval_rbf_gamma": float(mmd_gamma),
    }
    out.update(moment_errors(z_syn, z_train))
    out.update({f"utility_{k}": v for k, v in utility_metrics(z_syn, y_mean, y_std, ds).items()})
    return out


def rank_results(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    group = ["dataset", "seed"]
    for m in RANK_METRICS_LOWER:
        df[f"rank_{m}"] = df.groupby(group)[m].rank(method="average", ascending=True)
    df["rank_min_train_distance"] = df.groupby(group)["min_train_distance"].rank(method="average", ascending=False)
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df["avg_rank"] = df[rank_cols].mean(axis=1)
    return df


def run_benchmark(
    *,
    datasets: list[str],
    seeds: list[int],
    methods: dict[str, Callable],
    n_synthetic: int,
    max_train: int,
    max_test: int,
    results_dir: Path,
    run_name: str,
    primary_metric: str,
    log=print,
) -> pd.DataFrame:
    rows: list[dict] = []
    load_errors: list[dict] = []
    t0 = time.time()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / f"{run_name}_checkpoint.jsonl"
    done: set[tuple] = set()
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key_t = (rec["dataset_key"], int(rec["seed"]), rec["method"])
            if key_t in done:
                continue
            done.add(key_t)
            rows.append(rec)
        log(f"[RESUME] loaded {len(done)} completed cells from {checkpoint_path.name}")

    def _append_checkpoint(row: dict) -> None:
        with checkpoint_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()

    for key in datasets:
        for seed in seeds:
            remaining = [
                m for m in methods
                if (key, seed, m) not in done
            ]
            if not remaining:
                log(f"[DATASET] {key} seed={seed} (all methods cached)")
                continue
            log(f"[DATASET] {key} seed={seed}")
            try:
                ds = load_uai_dataset(key, seed)
            except Exception as exc:
                log(f"  [LOAD-ERROR] {exc}")
                load_errors.append({"dataset": key, "seed": seed, "error": str(exc)})
                continue
            rng = np.random.default_rng(seed)
            z_train, y_mean, y_std = make_joint(ds.X_train, ds.y_train)
            z_test = make_joint_with_scale(ds.X_test, ds.y_test, y_mean, y_std)
            if len(z_train) > max_train:
                z_train = z_train[rng.choice(len(z_train), max_train, replace=False)]
            if len(z_test) > max_test:
                z_test = z_test[rng.choice(len(z_test), max_test, replace=False)]
            eval_rng = np.random.default_rng(stable_method_seed(seed, key, "__evaluation__"))
            mmd_gamma = reference_rbf_gamma(z_test)
            projections = make_projections(z_train.shape[1], eval_rng, n_proj=64)
            for method, gen in methods.items():
                if (key, seed, method) in done:
                    log(f"  [SKIP] {method}")
                    continue
                seed_key = method.split(" (", 1)[0]
                m_rng = np.random.default_rng(stable_method_seed(seed, key, seed_key))
                t_m = time.time()
                try:
                    result = gen(z_train, n_synthetic, m_rng)
                    z_syn, meta = result if isinstance(result, tuple) else (result, {})
                    z_syn = project_categorical_codes(z_syn, z_train, ds.cat_indices)
                    meta = dict(meta)
                    meta["categorical_projection"] = bool(ds.cat_indices)
                    row = evaluate_generator(
                        method,
                        z_syn,
                        z_train,
                        z_test,
                        ds,
                        y_mean,
                        y_std,
                        mmd_gamma=mmd_gamma,
                        projections=projections,
                    )
                    row.update({
                        "dataset": ds.name,
                        "dataset_key": key,
                        "seed": seed,
                        "source": ds.source,
                        "n_train_eval": int(len(z_train)),
                        "n_test_eval": int(len(z_test)),
                        "n_synthetic": int(len(z_syn)),
                        "gen_seconds": round(time.time() - t_m, 2),
                        "meta": json.dumps(meta, default=float),
                    })
                    rows.append(row)
                    done.add((key, seed, method))
                    _append_checkpoint(row)
                    log(f"  [METHOD] {method}: mmd2={row['mmd2_test']:.4g} ({row['gen_seconds']}s)")
                except Exception as exc:
                    log(f"  [METHOD-ERROR] {method}: {exc}")
                    rows.append({"dataset": ds.name, "dataset_key": key, "seed": seed, "method": method, "error": str(exc)})
    failed_rows = [r for r in rows if "error" in r]
    expected_rows = len(datasets) * len(seeds) * len(methods)
    ok_rows = [r for r in rows if "error" not in r]
    if load_errors or failed_rows or len(ok_rows) != expected_rows:
        results_dir.mkdir(parents=True, exist_ok=True)
        failure_path = results_dir / f"{run_name}_failures.json"
        failure_path.write_text(
            json.dumps(
                {
                    "created_utc": utc_now(),
                    "expected_rows": expected_rows,
                    "completed_rows": len(ok_rows),
                    "load_errors": load_errors,
                    "method_errors": failed_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Incomplete benchmark grid: {len(ok_rows)}/{expected_rows} cells; "
            f"see {failure_path}"
        )
    df = rank_results(ok_rows)
    write_outputs(df, load_errors, results_dir, run_name, primary_metric,
                  config={"datasets": datasets, "seeds": seeds, "n_synthetic": n_synthetic,
                          "max_train": max_train, "max_test": max_test,
                          "methods": list(methods),
                          "evaluation": {
                              "mmd": "all rows; one test-reference median bandwidth per dataset-seed",
                              "sliced_w1": "64 shared Gaussian projections per dataset-seed",
                              "distance": "exact all synthetic-to-training rows",
                              "categorical": "train-only encoding; projection to observed training levels",
                          }})
    print(f"[DONE] elapsed={time.time() - t0:.0f}s rows={len(df)}")
    return df


def write_outputs(df, load_errors, results_dir: Path, run_name: str, primary_metric: str, config: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = results_dir / f"{run_name}_raw.csv"
    df.to_csv(raw_csv, index=False)

    agg_metrics = RANK_METRICS_LOWER + ["min_train_distance", "avg_rank"]
    table = df.groupby(["dataset", "method"])[agg_metrics].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.reset_index()
    table_csv = results_dir / f"{run_name}_table.csv"
    table.to_csv(table_csv, index=False)

    overall = df.groupby("method")[agg_metrics].mean().sort_values("avg_rank")
    overall_std = df.groupby(["method", "dataset"])["avg_rank"].mean().groupby("method").std()

    # Wins on the primary metric: per dataset, method with best seed-mean.
    per_ds = df.groupby(["dataset", "method"])[primary_metric].agg(["mean", "std"]).reset_index()
    wins: dict[str, int] = {m: 0 for m in df["method"].unique()}
    ties: dict[str, int] = {m: 0 for m in df["method"].unique()}
    for ds_name, grp in per_ds.groupby("dataset"):
        best_idx = grp["mean"].idxmin()
        best = grp.loc[best_idx]
        wins[best["method"]] += 1
        for _, r in grp.iterrows():
            if r["method"] != best["method"]:
                s = 0.0 if not np.isfinite(r["std"]) else r["std"]
                sb = 0.0 if not np.isfinite(best["std"]) else best["std"]
                if r["mean"] - s <= best["mean"] + sb:
                    ties[r["method"]] += 1

    payload = {
        "schema": "lowem_benchmark/v2",
        "created_at": utc_now(),
        "config": config,
        "primary_metric": primary_metric,
        "load_errors": load_errors,
        "overall_by_method": [
            {
                "method": m,
                **{k: float(overall.loc[m, k]) for k in agg_metrics},
                "avg_rank_std_over_datasets": float(overall_std.get(m, float("nan"))),
                "primary_wins": wins.get(m, 0),
                "primary_ties_with_best": ties.get(m, 0),
            }
            for m in overall.index
        ],
        "outputs": {"raw_csv": str(raw_csv), "table_csv": str(table_csv)},
    }
    summary_path = results_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {table_csv}")
    print(f"[OK] wrote {summary_path}")
    print(overall[["avg_rank", "mmd2_test", "fourth_err", "min_train_distance"]].to_string())
