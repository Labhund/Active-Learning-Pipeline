#!/usr/bin/env python3
"""
umap_compare_experiments.py — Static density-ratio comparison of two AL experiments.

3-panel figure (1×3, figsize≈18×6):

  Panel 1 — MaxMin AL coverage: log₂ density ratio vs library background
  Panel 2 — Random AL coverage: log₂ density ratio vs library background
  Panel 3 — Difference map: MaxMin − Random (blue = random explored more,
             red = maxmin explored more)

All panels share the same 2D bin edges (from bg + both experiments combined).
Grey contour lines show library density structure in all panels.

Requires:
    cache/emb_al_maxmin_init_{target}.npy   (from umap_al_rounds.py)
    cache/emb_al_random_init_{target}.npy   (from umap_al_rounds.py --experiment-id random_init)
    cache/emb_bg_rev_{target}.npy           (from umap_diversity_study.py)

Usage:
    python analysis/diversity_study/umap_compare_experiments.py \\
        --target trpv1_8gfa \\
        [--exp-a maxmin_init] [--exp-b random_init] \\
        [--bins 120] [--min-bg-count 3] \\
        [--out analysis/figures/umap_compare_experiments_trpv1_8gfa.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed.", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR  = Path(__file__).resolve().parent
CACHE_DIR   = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"


# ---------------------------------------------------------------------------
# Density computation (same logic as compare_random_baseline.py)
# ---------------------------------------------------------------------------

def shared_edges(emb_bg, emb_a, emb_b, bins):
    all_x = np.concatenate([emb_bg[:, 0], emb_a[:, 0], emb_b[:, 0]])
    all_y = np.concatenate([emb_bg[:, 1], emb_a[:, 1], emb_b[:, 1]])
    mx = (all_x.max() - all_x.min()) * 0.02
    my = (all_y.max() - all_y.min()) * 0.02
    xedges = np.linspace(all_x.min() - mx, all_x.max() + mx, bins + 1)
    yedges = np.linspace(all_y.min() - my, all_y.max() + my, bins + 1)
    return xedges, yedges


def density_ratio(emb_picks, emb_bg, xedges, yedges, min_bg_count):
    """
    Compute log₂ density ratio: log₂(picks_density / bg_density).
    Returns (ratio, h_bg, h_pk) where h_bg and h_pk are raw histograms.
    Bins with h_bg < min_bg_count are masked (NaN).
    """
    h_bg, _, _ = np.histogram2d(emb_bg[:, 0], emb_bg[:, 1], bins=(xedges, yedges))
    h_pk, _, _ = np.histogram2d(emb_picks[:, 0], emb_picks[:, 1], bins=(xedges, yedges))
    mask = h_bg < min_bg_count
    eps = 1e-9
    ratio = np.log2(
        (h_pk / max(h_pk.sum(), 1) + eps) /
        (h_bg / max(h_bg.sum(), 1) + eps)
    )
    ratio[mask] = np.nan
    return ratio, h_bg, h_pk


def summarise(label, ratio, n, verbose=True):
    v = ratio[~np.isnan(ratio)]
    std = v.std()
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    pct_under = 100 * (v < -2).mean()
    pct_over  = 100 * (v >  2).mean()
    if verbose:
        print(f"\n  {label}  (n={n:,}, valid bins={len(v)})")
        print(f"    std  log₂-ratio    : {std:.3f}   (lower = more uniform)")
        print(f"    IQR                : {iqr:.3f}")
        print(f"    bins <−2 (≥4× under): {(v<-2).sum():4d}  ({pct_under:.1f}%)")
        print(f"    bins >+2 (≥4× over) : {(v>+2).sum():4d}  ({pct_over:.1f}%)")
    return v, std, iqr, pct_under, pct_over


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(emb_bg, emb_a, emb_b, exp_a, exp_b,
                ratio_a, ratio_b, h_bg, xedges, yedges,
                target, args):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcen, ycen, indexing="ij")

    # Difference map: exp_a − exp_b (both masked where either is NaN)
    diff = ratio_a - ratio_b
    both_valid = ~np.isnan(ratio_a) & ~np.isnan(ratio_b)
    diff[~both_valid] = np.nan

    vmax_ratio = 4.0

    # Contour levels from library density (background)
    bg_f = h_bg.astype(float)
    levels = np.percentile(bg_f[bg_f > 0], [50, 80, 95])

    cmap_ratio = plt.cm.RdBu_r.copy()
    cmap_ratio.set_bad("white")

    cmap_diff = plt.cm.RdBu_r.copy()
    cmap_diff.set_bad("white")

    diff_abs = np.nanmax(np.abs(diff))
    vmax_diff = min(diff_abs, 3.0) if diff_abs > 0 else 1.0

    exp_a_label = exp_a.replace("_", " ")
    exp_b_label = exp_b.replace("_", " ")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], ratio_a, cmap_ratio, -vmax_ratio, vmax_ratio,
         f"{exp_a_label}\nlog₂ density ratio vs library",
         "log₂(AL / library density)"),
        (axes[1], ratio_b, cmap_ratio, -vmax_ratio, vmax_ratio,
         f"{exp_b_label}\nlog₂ density ratio vs library",
         "log₂(AL / library density)"),
        (axes[2], diff, cmap_diff, -vmax_diff, vmax_diff,
         f"Difference: {exp_a_label} − {exp_b_label}\n(red = maxmin more; blue = random more)",
         "log₂ density ratio difference"),
    ]

    for ax, data, cmap, vmin, vmax, title, cblabel in panels:
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(cblabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Tick marks for ratio panels
        if vmax == vmax_ratio:
            cb.set_ticks([-4, -2, 0, 2, 4])
            cb.set_ticklabels(["−4\n(16×\nunder)", "−2", "0\n(equal)", "+2", "+4\n(16×\nover)"])

        # Library density contours
        ax.contour(X, Y, bg_f, levels=levels, colors="k",
                   linewidths=0.4, alpha=0.30)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP-1 (library space)", fontsize=8)
        ax.set_ylabel("UMAP-2 (library space)", fontsize=8)
        ax.tick_params(labelsize=7)

    footer = (
        f"target={target}  n_maxmin={len(emb_a):,}  n_random={len(emb_b):,}  "
        f"n_bg={len(emb_bg):,}  bins={args.bins}×{args.bins}  "
        f"min_bg_count={args.min_bg_count}"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom",
             fontsize=7.5, color="#555")

    plt.suptitle(
        f"Chemical space coverage comparison — {target}\n"
        f"{exp_a_label} vs {exp_b_label} (reverse UMAP, library frame)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_path = Path(args.out) if args.out else (
        FIGURES_DIR / f"umap_compare_experiments_{target}.png"
    )
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target",        type=str,  default="trpv1_8gfa")
    p.add_argument("--exp-a",         type=str,  default="maxmin_init",
                   help="First experiment ID (default: maxmin_init)")
    p.add_argument("--exp-b",         type=str,  default="random_init",
                   help="Second experiment ID (default: random_init)")
    p.add_argument("--bins",          type=int,  default=120)
    p.add_argument("--min-bg-count",  type=int,  default=3)
    p.add_argument("--out",           type=str,  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    target = args.target
    exp_a  = args.exp_a
    exp_b  = args.exp_b

    # ------------------------------------------------------------------
    # Load cached embeddings
    # ------------------------------------------------------------------
    path_a  = CACHE_DIR / f"emb_al_{exp_a}_{target}.npy"
    path_b  = CACHE_DIR / f"emb_al_{exp_b}_{target}.npy"
    path_bg = CACHE_DIR / f"emb_bg_rev_{target}.npy"

    missing = []
    for p in [path_a, path_b, path_bg]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print("ERROR: missing required cache files:")
        for m in missing:
            print(f"  {m}")
        if str(path_b) in missing:
            print(f"\nRun first:  python analysis/diversity_study/umap_al_rounds.py "
                  f"--target {target} --experiment-id {exp_b}")
        sys.exit(1)

    print(f"Loading embeddings for {exp_a} ...")
    emb_a = np.load(path_a)
    print(f"  shape: {emb_a.shape}")

    print(f"Loading embeddings for {exp_b} ...")
    emb_b = np.load(path_b)
    print(f"  shape: {emb_b.shape}")

    print("Loading background embeddings ...")
    emb_bg = np.load(path_bg)
    print(f"  shape: {emb_bg.shape}")

    # ------------------------------------------------------------------
    # Density computation
    # ------------------------------------------------------------------
    print(f"\nComputing shared bin edges ({args.bins}×{args.bins}) ...")
    xedges, yedges = shared_edges(emb_bg, emb_a, emb_b, args.bins)

    print(f"Computing density ratio for {exp_a} ...")
    ratio_a, h_bg, h_a = density_ratio(emb_a, emb_bg, xedges, yedges, args.min_bg_count)

    print(f"Computing density ratio for {exp_b} ...")
    ratio_b, _,    h_b = density_ratio(emb_b, emb_bg, xedges, yedges, args.min_bg_count)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n=== Coverage summary ===")
    va, std_a, iqr_a, under_a, over_a = summarise(
        exp_a.replace("_", " "), ratio_a, len(emb_a))
    vb, std_b, iqr_b, under_b, over_b = summarise(
        exp_b.replace("_", " "), ratio_b, len(emb_b))

    print("\n=== Relative comparison ===")
    delta_std = std_b - std_a
    delta_iqr = iqr_b - iqr_a
    print(f"  std reduction (b−a) : {delta_std:+.3f}  "
          f"({'MaxMin more uniform ✓' if delta_std > 0 else 'Random more uniform ✓'})")
    print(f"  IQR reduction (b−a) : {delta_iqr:+.3f}")
    print(f"  ≥4× undersampled bins: {exp_a}={under_a:.1f}%  {exp_b}={under_b:.1f}%  "
          f"Δ={under_b - under_a:+.1f}pp")
    print(f"  ≥4× oversampled  bins: {exp_a}={over_a:.1f}%  {exp_b}={over_b:.1f}%  "
          f"Δ={over_b - over_a:+.1f}pp")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    print("\nGenerating comparison figure ...")
    make_figure(
        emb_bg, emb_a, emb_b, exp_a, exp_b,
        ratio_a, ratio_b, h_bg, xedges, yedges,
        target, args,
    )


if __name__ == "__main__":
    main()
