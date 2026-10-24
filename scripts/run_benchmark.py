#!/usr/bin/env python
"""Run the multi-seed benchmark (or an ablation) for this paper pack.

Profiles
--------
smoke : 1 dataset (abalone), 1 seed, 256 synthetic samples (~2 min sanity run)
full  : 12 datasets, 5 seeds, 1024 synthetic samples per method per dataset
new_baselines_smoke : abalone, seed 42, n_synthetic=256, TabDDPM + CTAB-GAN+ only
extra6 : screened six-candidate replay (not the locked roster)

Ablations
---------
fw  : MomentAnchor Frank-Wolfe iteration grid and moment-order grid

Examples
--------
python scripts/run_benchmark.py --profile smoke
python scripts/run_benchmark.py --profile full
python scripts/run_benchmark.py --profile new_baselines_smoke
python scripts/run_benchmark.py --methods TabDDPM "CTAB-GAN+" --datasets abalone
python scripts/run_benchmark.py --ablation fw --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lowem_bench as lb
from tab_generators import (
    SMOKE_EPOCHS_CTAB,
    SMOKE_EPOCHS_TABDDPM,
    SMOKE_STEPS_TABDDPM,
    gen_ctab_gan_plus,
    gen_tabddpm,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PRIMARY_METRIC_DEFAULT = "mmd2_test"  # MomentAnchor pack: MMD^2 (pre-declared)


def _new_baseline_generators() -> dict:
    return {
        "TabDDPM-style": partial(
            gen_tabddpm,
            epochs=SMOKE_EPOCHS_TABDDPM,
            n_timesteps=SMOKE_STEPS_TABDDPM,
        ),
        "CTAB-GAN+-style": partial(gen_ctab_gan_plus, epochs=SMOKE_EPOCHS_CTAB),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["smoke", "full", "new_baselines_smoke", "extra6", "screen6"],
        default=None,
    )
    parser.add_argument("--ablation", choices=["fw"], default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help='Optional subset of GENERATORS to run (e.g. TabDDPM "CTAB-GAN+").',
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--primary", default=PRIMARY_METRIC_DEFAULT)
    args = parser.parse_args()

    if args.ablation is None and args.profile is None:
        args.profile = "smoke"

    if args.ablation == "fw":
        methods = {}
        for iters in (50, 200, 800, 1200):
            methods[f"MomentAnchor (T={iters})"] = partial(lb.gen_momentanchor, fw_iters=iters, moment_order=3)
        for order in (0, 2, 3):
            methods[f"MomentAnchor (r={order})"] = partial(lb.gen_momentanchor, fw_iters=1200, moment_order=order)
        datasets = args.datasets or lb.UAI_DATASETS
        seeds = args.seeds or [42, 43, 44]
        n_syn = args.n_synthetic or 1024
        run_name = "ablation_fw"
    elif args.profile == "extra6":
        methods = dict(lb.GENERATORS)
        datasets = args.datasets or lb.EXTRA_CANDIDATE_DATASETS
        seeds = args.seeds or [42, 43, 44, 45, 46]
        n_syn = args.n_synthetic or 1024
        run_name = "benchmark_extra6"
    elif args.profile == "screen6":
        methods = dict(lb.GENERATORS)
        datasets = args.datasets or lb.SCREEN2_CANDIDATE_DATASETS
        seeds = args.seeds or [42, 43, 44, 45, 46]
        n_syn = args.n_synthetic or 1024
        run_name = "benchmark_screen6"
    elif args.profile == "new_baselines_smoke":
        methods = _new_baseline_generators()
        datasets = args.datasets or ["abalone"]
        seeds = args.seeds or [42]
        n_syn = args.n_synthetic or 256
        run_name = "new_baselines_smoke"
    elif args.profile == "smoke":
        methods = dict(lb.GENERATORS)
        methods.update(_new_baseline_generators())
        datasets = args.datasets or ["abalone"]
        seeds = args.seeds or [42]
        n_syn = args.n_synthetic or 256
        run_name = "smoke"
    else:
        methods = dict(lb.GENERATORS)
        datasets = args.datasets or lb.UAI_DATASETS
        seeds = args.seeds or [42, 43, 44, 45, 46]
        n_syn = args.n_synthetic or 1024
        run_name = "benchmark"

    if args.methods:
        pool = methods if args.profile == "new_baselines_smoke" else lb.GENERATORS
        if args.profile != "new_baselines_smoke":
            methods = dict(lb.GENERATORS)
        methods = {m: methods[m] for m in args.methods if m in methods}
        if not methods:
            known = sorted(pool.keys())
            raise SystemExit(f"No known methods in {args.methods}; known: {known}")
    if args.run_name:
        run_name = args.run_name

    lb.run_benchmark(
        datasets=datasets,
        seeds=seeds,
        methods=methods,
        n_synthetic=n_syn,
        max_train=args.max_train,
        max_test=args.max_test,
        results_dir=RESULTS,
        run_name=run_name,
        primary_metric=args.primary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
