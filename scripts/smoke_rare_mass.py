"""Smoke: SMOTER-mass constrained MomentAnchor (does not touch GENERATORS)."""
from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lowem_bench as lb

ROOT = Path(__file__).resolve().parents[1]

# Pre-declared bar (seed 42, n=1024, 3 datasets):
# SERA: beat SMOTER on >=2/3, OR within 20% of SMOTER on all 3.
# MMD: still beat SMOTER; stay below Copula if possible. Bootstrap may win MMD.


def main() -> int:
    lb.run_benchmark(
        datasets=["abalone", "energy_heat", "yacht"],
        seeds=[42],
        methods={
            "MomentAnchor": partial(
                lb.gen_momentanchor, use_rare_mass_constraint=True
            ),
            "Bootstrap": lb.gen_bootstrap,
            "SMOTER": lb.gen_smoter,
            "Global Mixup": lb.gen_global_mixup,
            "Gaussian Copula": lb.gen_gaussian_copula,
        },
        n_synthetic=1024,
        max_train=2000,
        max_test=1000,
        results_dir=ROOT / "results",
        run_name="smoke_rare_mass",
        primary_metric="mmd2_test",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
