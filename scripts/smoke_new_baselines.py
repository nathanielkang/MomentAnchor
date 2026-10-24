#!/usr/bin/env python
"""One-liner smoke entry: TabDDPM + CTAB-GAN+ on abalone (seed 42, n_syn=256)."""

from run_benchmark import main

if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "--profile", "new_baselines_smoke"]
    raise SystemExit(main())
