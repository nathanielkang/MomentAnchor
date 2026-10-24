#!/usr/bin/env python
"""Generate [VISUAL EDITOR] conceptual figures for the MomentAnchor manuscript.

Writes PDF + PNG to 1_Manuscript/figures/:
  - fig_overview.pdf / .png
  - fig_moment_factorization.pdf / .png

Spec: 1_Manuscript/figure_prompts.txt (fig:overview, fig:moment_factorization).
Full caption text is rendered inside each image (8.5 pt serif, bottom).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import patches

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT.parent / "1_Manuscript" / "figures"
DPI = 300

# Palette (figure_prompts.txt)
GRAY = "#8C8C8C"
LIGHT_BLUE = "#7FB3E0"
RED = "#D95F02"
DARK_BLUE = "#2C6FBB"
TEAL = "#1B9E77"
LIGHT_GREEN = "#CCEBC5"
LIGHT_RED = "#FCAE91"
LIGHT_ORANGE = "#FDD0A2"
WHITE = "#FFFFFF"

CAPTION_OVERVIEW = (
    "The MomentAnchor pipeline. Local barycenter candidates are drawn from the training sample with uniform marginal usage,\n"
    "calibrated, and filtered by an output-level privacy rule; Frank-Wolfe then optimizes the selection distribution on the "
    "candidate simplex against the empirical measure in MMD$^2$ with a moment-residual penalty; the emitted sample carries "
    "the certificate $(U^\\star, r_{\\mathrm{marg}}, r_{\\mathrm{mom}})$."
)

CAPTION_MOMENT = (
    "Moment geometry of structure-preserving mixup. Every exchangeable coupling with uniform marginals preserves moments "
    "through order three (Theorem~1); the order-four deviation factorizes into a weight scalar $\\kappa$ times a coupling "
    "defect that is linear in the selection law $Q$ (Theorem~2), so higher-order exactness is a linear constraint on $Q$."
)


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0,
        }
    )


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(pdf, dpi=DPI, facecolor=WHITE, edgecolor="none", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=DPI, facecolor=WHITE, edgecolor="none", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return pdf, png


def _draw_caption(ax: plt.Axes, text: str, y: float = 0.045) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontfamily="serif",
        color="black",
        linespacing=1.18,
    )


def _straight_arrow(ax: plt.Axes, x0: float, x1: float, y: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 1.6,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 12,
        },
    )


def _cloud_points(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    t = rng.uniform(-1.4, 1.4, n)
    x = 0.55 * np.sin(t) + 0.12 * rng.normal(size=n)
    y = 0.35 * np.cos(1.3 * t) + 0.08 * rng.normal(size=n)
    return np.column_stack([x, y]) * scale


def _panel_box(ax: plt.Axes, x0: float, x1: float, y0: float, y1: float) -> None:
    rect = patches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        transform=ax.transAxes,
        fill=False,
        edgecolor="0.75",
        linewidth=0.8,
        zorder=0,
    )
    ax.add_patch(rect)


def _draw_scatter_block(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    title: str,
    *,
    base_pts: np.ndarray | None = None,
    overlay_open: np.ndarray | None = None,
    overlay_color: str = LIGHT_BLUE,
    output_teal: np.ndarray | None = None,
    cross_indices: np.ndarray | None = None,
    privacy_center: tuple[float, float] | None = None,
    privacy_radius: float = 0.12,
    cert_tag: bool = False,
    show_base: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    _panel_box(ax, x0, x1, y0, y1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w = x1 - x0
    h = y1 - y0

    ax.text(cx, y1 - 0.028, title, transform=ax.transAxes, ha="center", va="top", fontsize=11, fontweight="medium")

    if base_pts is None:
        assert rng is not None
        base_pts = _cloud_points(25, rng)

    def to_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = cx + 0.38 * w * xy[:, 0]
        ys = cy + 0.04 * h + 0.28 * h * xy[:, 1]
        return xs, ys

    bx, by = to_axes(base_pts)
    if show_base:
        ax.scatter(bx, by, s=16, c=GRAY, edgecolors="none", zorder=2)

    if overlay_open is not None:
        ox, oy = to_axes(overlay_open)
        ax.scatter(
            ox,
            oy,
            s=14,
            facecolors="none",
            edgecolors=overlay_color,
            linewidths=0.9,
            zorder=3,
        )

    if cross_indices is not None and overlay_open is not None:
        ox, oy = to_axes(overlay_open)
        for idx in cross_indices:
            xi, yi = ox[idx], oy[idx]
            d = 0.008 * min(w, h)
            ax.plot([xi - d, xi + d], [yi - d, yi + d], color=RED, lw=1.0, zorder=5, solid_capstyle="butt")
            ax.plot([xi - d, xi + d], [yi + d, yi - d], color=RED, lw=1.0, zorder=5, solid_capstyle="butt")

    if privacy_center is not None:
        pc = np.array([privacy_center])
        px, py = to_axes(pc)
        circle = patches.Circle(
            (px[0], py[0]),
            privacy_radius * min(w, h) * 0.55,
            transform=ax.transAxes,
            fill=False,
            edgecolor=RED,
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(px[0] + 0.04 * w, py[0] + 0.05 * h, r"$\varepsilon$", transform=ax.transAxes, fontsize=11)

    if output_teal is not None:
        tx, ty = to_axes(output_teal)
        ax.scatter(tx, ty, s=16, c=TEAL, edgecolors="none", zorder=2)
        if cert_tag:
            tag_x = x1 - 0.04 * w
            tag_y = y0 + 0.06 * h
            box = patches.FancyBboxPatch(
                (tag_x - 0.11 * w, tag_y - 0.025 * h),
                0.105 * w,
                0.05 * h,
                boxstyle="round,pad=0.004",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor="0.45",
                linewidth=0.8,
                zorder=6,
            )
            ax.add_patch(box)
            ax.text(
                tag_x - 0.055 * w,
                tag_y,
                r"$(U^\star, r_{\mathrm{marg}}, r_{\mathrm{mom}})$",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

    return base_pts


def fig_overview() -> tuple[Path, Path]:
    """Five-block MomentAnchor pipeline (fig:overview)."""
    _setup_rc()
    w_in, h_in = 2100 / DPI, 620 / DPI
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rng = np.random.default_rng(42)
    base = _cloud_points(25, rng)
    pool = np.vstack([base + rng.normal(scale=0.06, size=2) for _ in range(2)])
    pool = np.vstack([pool, _cloud_points(25, rng, scale=0.95)])

    # indices closest to training points for privacy crosses
    dists = np.min(np.linalg.norm(pool[:, None, :] - base[None, :, :], axis=2), axis=1)
    cross_idx = np.argsort(dists)[:8]

    y0, y1 = 0.17, 0.86
    cols = [(0.02, 0.17), (0.19, 0.34), (0.36, 0.51), (0.53, 0.70), (0.72, 0.98)]
    arrow_y = 0.52

    def _panel_note(x0: float, x1: float, text: str) -> None:
        ax.text(
            0.5 * (x0 + x1),
            y0 + 0.038,
            text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.25",
            linespacing=1.05,
        )

    _draw_scatter_block(ax, *cols[0], y0, y1, "Training sample", base_pts=base, rng=rng)
    _draw_scatter_block(
        ax,
        *cols[1],
        y0,
        y1,
        "Candidate pool",
        base_pts=base,
        overlay_open=pool,
        rng=rng,
    )
    _panel_note(cols[1][0], cols[1][1], "uniform marginal\nvertex usage")

    _draw_scatter_block(
        ax,
        *cols[2],
        y0,
        y1,
        "Privacy filter",
        base_pts=base,
        overlay_open=pool,
        cross_indices=cross_idx,
        privacy_center=base[4],
        rng=rng,
    )
    _panel_note(cols[2][0], cols[2][1], r"min distance $\geq \varepsilon$")

    # Block 4: Frank-Wolfe simplex
    x0, x1 = cols[3]
    _panel_box(ax, x0, x1, y0, y1)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    ax.text(cx, y1 - 0.028, "Convex selection (Frank-Wolfe)", transform=ax.transAxes, ha="center", va="top", fontsize=11)

    tri_h = 0.30 * (y1 - y0)
    tri_w = 0.26 * (x1 - x0)
    v0 = np.array([cx, cy + 0.04 * (y1 - y0)])
    v1 = np.array([cx - tri_w, cy - 0.10 * (y1 - y0)])
    v2 = np.array([cx + tri_w, cy - 0.10 * (y1 - y0)])
    tri = plt.Polygon([v0, v1, v2], fill=False, edgecolor="0.35", lw=1.5, transform=ax.transAxes, zorder=2)
    ax.add_patch(tri)

    centroid = (v0 + v1 + v2) / 3.0
    edge_mid = 0.5 * (v1 + v2)
    path_pts = [
        centroid,
        0.75 * centroid + 0.25 * edge_mid,
        0.55 * centroid + 0.45 * edge_mid,
        0.35 * centroid + 0.65 * edge_mid,
        0.15 * centroid + 0.85 * edge_mid,
        edge_mid,
    ]
    px = [p[0] for p in path_pts]
    py = [p[1] for p in path_pts]
    ax.plot(px, py, color=DARK_BLUE, lw=1.6, zorder=3, solid_capstyle="butt")
    ax.scatter([edge_mid[0]], [edge_mid[1]], s=38, c=DARK_BLUE, edgecolors="white", linewidths=0.6, zorder=4, transform=ax.transAxes)
    ax.text(edge_mid[0] + 0.018, edge_mid[1] - 0.02, r"$q^\star$", transform=ax.transAxes, fontsize=11, color=DARK_BLUE)
    ax.text(
        x0 + 0.02 * (x1 - x0),
        cy + 0.14 * (y1 - y0),
        r"$\gamma_t = 2/(t{+}2)$",
        transform=ax.transAxes,
        fontsize=9,
        color="0.3",
    )
    ax.text(
        cx,
        y0 + 0.055,
        r"$\min_q\ \mathrm{MMD}^2 + \lambda\,\|\Psi^{\top} q - \hat m\|^2$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
    )

    _draw_scatter_block(
        ax,
        *cols[4],
        y0,
        y1,
        "Certified sample",
        base_pts=base,
        output_teal=base,
        show_base=False,
        cert_tag=True,
        rng=rng,
    )

    for i in range(4):
        _straight_arrow(ax, cols[i][1] + 0.005, cols[i + 1][0] - 0.005, arrow_y)

    _draw_caption(ax, CAPTION_OVERVIEW)
    return _save(fig, "fig_overview")


def fig_moment_factorization() -> tuple[Path, Path]:
    """Two-panel moment geometry schematic (fig:moment_factorization)."""
    _setup_rc()
    w_in, h_in = 1900 / DPI, 640 / DPI
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel (a)
    pa_x0, pa_x1 = 0.03, 0.49
    py0, py1 = 0.21, 0.88
    _panel_box(ax, pa_x0, pa_x1, py0, py1)
    ax.text(pa_x0 + 0.018, py1 - 0.028, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(
        pa_x0 + 0.075,
        py1 - 0.028,
        "Orders 1–3: exact for any\nadmissible coupling",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="medium",
        va="top",
        ha="left",
        linespacing=1.05,
    )

    line_y = 0.58
    x_start, x_end = pa_x0 + 0.06, pa_x1 - 0.04
    ax.plot([x_start, x_end], [line_y, line_y], color="0.35", lw=1.2, transform=ax.transAxes, solid_capstyle="butt")
    orders = np.linspace(x_start, x_end, 5)
    for i, xo in enumerate(orders):
        s = i + 1
        ax.text(xo, line_y + 0.04, f"$s={s}$", transform=ax.transAxes, ha="center", fontsize=9.5)
        ax.plot([xo], [line_y], "o", color="0.35", ms=5, transform=ax.transAxes)

    # shaded regions
    ax.add_patch(
        patches.Rectangle(
            (orders[0] - 0.04, line_y - 0.08),
            orders[2] - orders[0] + 0.08,
            0.16,
            transform=ax.transAxes,
            facecolor=LIGHT_GREEN,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (orders[3] - 0.04, line_y - 0.08),
            orders[3] - orders[3] + 0.08,
            0.16,
            transform=ax.transAxes,
            facecolor=LIGHT_RED,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
    )

    for xo in orders[:3]:
        ax.text(xo, line_y - 0.055, "\u2713", transform=ax.transAxes, ha="center", va="center", fontsize=16, color="#1B7837", fontweight="bold")
    ax.text(orders[3], line_y - 0.055, r"$\delta$", transform=ax.transAxes, ha="center", va="center", fontsize=15, color=RED, fontweight="bold")

    ax.text(
        0.5 * (pa_x0 + pa_x1),
        py0 + 0.085,
        "preserved for every $Q$ with uniform marginals",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.28",
        clip_on=False,
    )

    # Panel (b)
    pb_x0, pb_x1 = 0.51, 0.97
    _panel_box(ax, pb_x0, pb_x1, py0, py1)
    ax.text(pb_x0 + 0.018, py1 - 0.028, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(
        pb_x0 + 0.075,
        py1 - 0.028,
        "Order 4 defect is LINEAR\nin the selection law $Q$",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="medium",
        va="top",
        ha="left",
        linespacing=1.05,
    )

    eq_y = 0.56
    ax.text(
        0.5 * (pb_x0 + pb_x1),
        eq_y,
        r"$\mathbb{E}\left[(a^{\top}\tilde{X})^4\right] - m_4"
        r"= 2\kappa\,(m_4 - 4C_{3,1} + 3C_{2,2})$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
    )

    box_x0 = pb_x0 + 0.05
    box_w = 0.52 * (pb_x1 - pb_x0)
    box_h = 0.18
    box_y = py0 + 0.09
    orange_box = patches.FancyBboxPatch(
        (box_x0, box_y),
        box_w,
        box_h,
        boxstyle="round,pad=0.008",
        transform=ax.transAxes,
        facecolor=LIGHT_ORANGE,
        edgecolor="#E6550D",
        linewidth=0.9,
        zorder=1,
    )
    ax.add_patch(orange_box)
    ax.text(
        box_x0 + 0.02,
        box_y + box_h / 2,
        r"$C_{s,t}$ are linear functionals of $Q$ $\Rightarrow$"
        "\nmoment matching to any order $r$ is a linear constraint on $Q$",
        transform=ax.transAxes,
        va="center",
        fontsize=9,
        linespacing=1.15,
    )

    # simplex + hyperplane icon (right of box)
    sx = pb_x0 + 0.72 * (pb_x1 - pb_x0)
    sy = box_y + box_h / 2
    sw = 0.14 * (pb_x1 - pb_x0)
    sh = 0.15
    tv0 = (sx, sy + sh * 0.45)
    tv1 = (sx - sw * 0.55, sy - sh * 0.35)
    tv2 = (sx + sw * 0.55, sy - sh * 0.35)
    ax.add_patch(plt.Polygon([tv0, tv1, tv2], fill=False, edgecolor="0.4", lw=1.2, transform=ax.transAxes))
    # hyperplane segment through simplex
    hx0, hx1 = sx - sw * 0.35, sx + sw * 0.35
    hy = sy + sh * 0.05
    ax.plot([hx0, hx1], [hy, hy - sh * 0.25], color=DARK_BLUE, lw=1.4, transform=ax.transAxes, solid_capstyle="butt")

    # connector line from equation to box
    eq_bottom = eq_y - 0.06
    box_top = box_y + box_h
    mid_x = box_x0 + box_w * 0.35
    ax.plot([mid_x, mid_x], [eq_bottom, box_top + 0.01], color="0.45", lw=1.0, transform=ax.transAxes, solid_capstyle="butt")

    _draw_caption(ax, CAPTION_MOMENT)
    return _save(fig, "fig_moment_factorization")


def self_check() -> dict[str, object]:
    """Verify outputs exist and are not gray placeholders."""
    checks: dict[str, object] = {"ok": True, "files": {}, "errors": []}
    stems = ("fig_overview", "fig_moment_factorization")
    for stem in stems:
        info: dict[str, object] = {}
        for ext in ("pdf", "png"):
            path = FIGURES / f"{stem}.{ext}"
            info[ext] = str(path)
            if not path.exists():
                checks["ok"] = False
                checks["errors"].append(f"missing: {path}")
                continue
            size = path.stat().st_size
            info[f"{ext}_bytes"] = size
            if size < 8000:
                checks["ok"] = False
                checks["errors"].append(f"suspiciously small ({size} B): {path}")
        pdf_path = FIGURES / f"{stem}.pdf"
        if pdf_path.exists():
            raw = pdf_path.read_bytes()
            if b"PLACEHOLDER" in raw or b"VISUAL EDITOR placeholder" in raw:
                checks["ok"] = False
                checks["errors"].append(f"placeholder marker in {pdf_path}")
        checks["files"][stem] = info
    return checks


def main() -> int:
    paths = []
    paths.append(fig_overview())
    paths.append(fig_moment_factorization())
    for pdf, png in paths:
        print(f"[OK] {pdf.name} ({pdf.stat().st_size} B)")
        print(f"[OK] {png.name} ({png.stat().st_size} B)")
    report = self_check()
    print(f"[SELF-CHECK] ok={report['ok']}")
    if report["errors"]:
        for err in report["errors"]:
            print(f"  ERROR: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
