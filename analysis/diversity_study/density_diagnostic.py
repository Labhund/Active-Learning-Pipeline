#!/usr/bin/env python3
"""
Density diagnostic for diversity sampling quality.

Loads cached UMAP embeddings and produces two diagnostic plots on the
reverse UMAP space (fit on library background — the ground-truth library topology):

  1. Log2 density-ratio heatmap  — log2(picks_density / bg_density)
       zero  = proportional coverage
       blue  = undersampled relative to library density
       red   = oversampled relative to library density

  2. Hexbin coverage-fraction    — picks / (picks + background) per hex cell
       Intuitive complementary view; same interpretation.

Both are computed via 2D histogram binning (fast, no KDE fitting needed).
Bins with insufficient background support are masked out (white).

Usage:
    python analysis/diversity_study/density_diagnostic.py --target trpv1_8gfa
    python analysis/diversity_study/density_diagnostic.py --target trpv1_8gfa --bins 150
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
CACHE_DIR   = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR / "figures"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=str, default="trpv1_8gfa")
    p.add_argument("--bins",   type=int, default=120,
                   help="Number of 2D histogram bins per axis (default 120)")
    p.add_argument("--min-bg-count", type=int, default=3,
                   help="Min background counts per bin to include in ratio (default 3)")
    return p.parse_args()


def load_cache(target: str) -> tuple:
    keys = ["emb_bg_rev", "emb_24k_on_bg"]
    paths = {k: CACHE_DIR / f"{k}_{target}.npy" for k in keys}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        print(f"ERROR: missing cache files: {missing}")
        print("Run umap_diversity_study.py first to generate embeddings.")
        sys.exit(1)
    emb_bg  = np.load(paths["emb_bg_rev"])
    emb_24k = np.load(paths["emb_24k_on_bg"])
    return emb_bg, emb_24k


def compute_ratio(emb_bg, emb_24k, bins, min_bg_count):
    """
    2D histogram log2-ratio: log2(picks_density / bg_density).

    Both histograms are normalised to density (sum = 1 over non-masked bins)
    before taking the ratio, so the comparison is scale-independent.
    """
    # Shared bin edges covering both datasets
    all_x = np.concatenate([emb_bg[:, 0], emb_24k[:, 0]])
    all_y = np.concatenate([emb_bg[:, 1], emb_24k[:, 1]])
    x_margin = (all_x.max() - all_x.min()) * 0.02
    y_margin = (all_y.max() - all_y.min()) * 0.02
    xedges = np.linspace(all_x.min() - x_margin, all_x.max() + x_margin, bins + 1)
    yedges = np.linspace(all_y.min() - y_margin, all_y.max() + y_margin, bins + 1)

    h_bg,  _, _ = np.histogram2d(emb_bg[:,0],  emb_bg[:,1],  bins=(xedges, yedges))
    h_24k, _, _ = np.histogram2d(emb_24k[:,0], emb_24k[:,1], bins=(xedges, yedges))

    # Mask bins with insufficient background support
    mask = h_bg < min_bg_count

    # Normalise to fraction of total points (density, sum≈1)
    bg_frac   = h_bg   / h_bg.sum()
    pick_frac = h_24k  / max(h_24k.sum(), 1)

    eps = 1e-9
    ratio = np.log2((pick_frac + eps) / (bg_frac + eps))
    ratio[mask] = np.nan   # white = no library support here

    # Hexbin coverage fraction: picks / (picks + bg) per bin, raw counts
    total = h_bg + h_24k
    cover = np.where(total > 0, h_24k / total, np.nan)
    cover[mask] = np.nan

    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])
    return ratio, cover, xcen, ycen, h_bg, h_24k


def make_density_figure(emb_bg, emb_24k, h_bg, h_24k, xcen, ycen, args):
    """
    Two-panel raw density map (log10 counts) for background and picks.
    Shared colour scale so visual comparison is direct.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, Y = np.meshgrid(xcen, ycen, indexing="ij")

    # log10(count + 1) — handles zeros cleanly, compresses dynamic range
    d_bg  = np.log10(h_bg  + 1).astype(float)
    d_24k = np.log10(h_24k + 1).astype(float)

    # Mask bins outside the library support (no bg counts at all)
    outside = h_bg == 0
    d_bg[outside]  = np.nan
    d_24k[outside] = np.nan

    # Shared colour scale across both panels
    vmax = max(np.nanmax(d_bg), np.nanmax(d_24k))

    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], d_bg,  f"100K library background\nraw point density (reverse UMAP space)",
         f"n = {len(emb_bg):,}"),
        (axes[1], d_24k, f"24K MaxMin diversity picks\nraw point density (same space, projected)",
         f"n = {len(emb_24k):,}"),
    ]

    for ax, data, title, subtitle in panels:
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=vmax,
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("log₁₀(counts + 1) per bin", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        # Annotate a few count levels on the colorbar
        ticks = [t for t in [0, 0.5, 1, 1.5, 2, 2.5, 3] if t <= vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}\n({int(10**t - 1)})" for t in ticks])
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP-1 (library space)", fontsize=9)
        ax.set_ylabel("UMAP-2 (library space)", fontsize=9)
        ax.tick_params(labelsize=7)

    footer = (
        f"target={args.target}  bins={args.bins}×{args.bins}  "
        f"colour scale: shared log₁₀(counts+1), vmax={vmax:.2f} ({int(10**vmax-1)} counts)"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555")
    plt.suptitle(
        "Raw Density Maps — Library vs Diversity Picks (Reverse UMAP space)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = FIGURES_DIR / f"density_maps_{args.target}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


def make_figure(emb_bg, emb_24k, ratio, cover, xcen, ycen, h_bg, args):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    X, Y = np.meshgrid(xcen, ycen, indexing="ij")

    # ------------------------------------------------------------------ #
    # Panel 1 — Log2 density ratio
    # ------------------------------------------------------------------ #
    ax = axes[0]

    # Symmetric color range centred on 0; clamp at ±4 (16× over/under)
    vmax = 4.0
    cmap_ratio = plt.cm.RdBu_r
    cmap_ratio.set_bad("white")

    im = ax.pcolormesh(
        X, Y, ratio,
        cmap=cmap_ratio, vmin=-vmax, vmax=vmax,
        shading="nearest", rasterized=True,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log₂(picks density / library density)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.set_ticks([-4, -2, 0, 2, 4])
    cb.set_ticklabels(["−4\n(16× under)", "−2", "0\n(equal)", "+2", "+4\n(16× over)"])

    ax.set_title(
        f"Log₂ density ratio — {args.target}\n"
        "Blue = undersampled relative to library; Red = oversampled",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.set_xlabel("UMAP-1 (library space)", fontsize=9)
    ax.set_ylabel("UMAP-2 (library space)", fontsize=9)
    ax.tick_params(labelsize=7)

    # Overlay background contour to show library structure
    bg_smooth = h_bg.astype(float)
    levels = np.percentile(bg_smooth[bg_smooth > 0], [50, 80, 95])
    ax.contour(X, Y, bg_smooth, levels=levels, colors="k", linewidths=0.4, alpha=0.35)

    # ------------------------------------------------------------------ #
    # Panel 2 — Coverage fraction hexbin
    # ------------------------------------------------------------------ #
    ax = axes[1]

    cmap_cov = plt.cm.RdYlGn   # red=0 (no picks), green=1 (picks dominate)
    cmap_cov.set_bad("white")

    # Expected fraction if picks were drawn proportionally to library density:
    expected = len(emb_24k) / (len(emb_24k) + len(emb_bg))

    im2 = ax.pcolormesh(
        X, Y, cover,
        cmap=cmap_cov, vmin=0, vmax=min(1.0, 4 * expected),
        shading="nearest", rasterized=True,
    )
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.04, pad=0.02)
    cb2.set_label("picks / (picks + background) per bin", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    # Mark the expected proportional fraction
    cb2.ax.axhline(expected / min(1.0, 4 * expected), color="k", lw=1.5, ls="--")

    ax.set_title(
        f"Coverage fraction per bin — {args.target}\n"
        f"Dashed line = proportional expectation ({expected:.4f})",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.set_xlabel("UMAP-1 (library space)", fontsize=9)
    ax.set_ylabel("UMAP-2 (library space)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.contour(X, Y, h_bg.astype(float), levels=levels, colors="k", linewidths=0.4, alpha=0.35)

    # ------------------------------------------------------------------ #
    # Shared footer
    # ------------------------------------------------------------------ #
    n_bg   = len(emb_bg)
    n_24k  = len(emb_24k)
    n_masked = np.sum(np.isnan(ratio))
    n_total  = ratio.size
    footer = (
        f"n_diversity={n_24k:,}  n_background={n_bg:,}  "
        f"bins={args.bins}×{args.bins}  min_bg_count={args.min_bg_count}  "
        f"masked={n_masked}/{n_total} bins ({100*n_masked/n_total:.0f}% outside library support)"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555")
    plt.suptitle(
        "Diversity Sampling Quality — Density Diagnostics (Reverse UMAP space)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = FIGURES_DIR / f"density_diagnostic_{args.target}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Print summary statistics
    # ------------------------------------------------------------------ #
    valid = ratio[~np.isnan(ratio)]
    print(f"\n--- Log2 ratio summary (valid bins: {len(valid)}) ---")
    print(f"  mean  : {valid.mean():.3f}")
    print(f"  median: {np.median(valid):.3f}")
    print(f"  std   : {valid.std():.3f}")
    print(f"  <−2 (≥4× under) : {(valid < -2).sum()} bins ({100*(valid<-2).mean():.1f}%)")
    print(f"  −2..0 (under)   : {((valid >= -2) & (valid < 0)).sum()} bins")
    print(f"   0..+2 (over)   : {((valid >= 0) & (valid < 2)).sum()} bins")
    print(f"  >+2 (≥4× over)  : {(valid >= 2).sum()} bins ({100*(valid>=2).mean():.1f}%)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    print(f"Loading cached embeddings for target '{args.target}'…")
    emb_bg, emb_24k = load_cache(args.target)
    print(f"  background: {emb_bg.shape}   picks: {emb_24k.shape}")

    print(f"Computing 2D histograms ({args.bins}×{args.bins} bins, min_bg={args.min_bg_count})…")
    ratio, cover, xcen, ycen, h_bg, h_24k = compute_ratio(
        emb_bg, emb_24k, args.bins, args.min_bg_count
    )

    print("Generating ratio/coverage figure…")
    make_figure(emb_bg, emb_24k, ratio, cover, xcen, ycen, h_bg, args)

    print("Generating raw density figure…")
    make_density_figure(emb_bg, emb_24k, h_bg, h_24k, xcen, ycen, args)


if __name__ == "__main__":
    main()
