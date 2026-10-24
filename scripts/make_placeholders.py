#!/usr/bin/env python
"""Gray-box placeholder PDFs for the two conceptual [VISUAL EDITOR] figures.

These are diagnostic placeholders only; full drawing specifications live in
1_Manuscript/figure_prompts.txt. No TikZ, no Python "art" — just a labelled
gray box so the manuscript compiles and reviewers see the intended slot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1].parent / "1_Manuscript" / "figures"

PLACEHOLDERS = {
    "fig_overview.pdf": (10.0, 3.0, "Conceptual figure (VISUAL EDITOR placeholder)\n"
                                     "MomentAnchor pipeline: pool -> privacy filter ->\n"
                                     "Frank--Wolfe selection -> certified sample\n"
                                     "See figure_prompts.txt [fig:overview]"),
    "fig_moment_factorization.pdf": (9.0, 3.2, "Conceptual figure (VISUAL EDITOR placeholder)\n"
                                                "Moment geometry: orders 1--3 exact;\n"
                                                "order-4 defect linear in Q\n"
                                                "See figure_prompts.txt [fig:moment_factorization]"),
}


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, (w, h, text) in PLACEHOLDERS.items():
        fig, ax = plt.subplots(figsize=(w, h))
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   facecolor="0.88", edgecolor="0.55", lw=1.2))
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="0.25")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.savefig(FIGURES / name, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
