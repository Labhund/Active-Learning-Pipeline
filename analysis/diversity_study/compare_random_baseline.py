#!/usr/bin/env python3
"""
MaxMin diversity vs random baseline — fresh-sample comparison.

Loads the saved PCA + reverse-UMAP models from cache, then:
  1. Samples a FRESH 24K random compounds from the DB — entirely distinct
     from both the 100K background and the MaxMin 24K diversity picks.
  2. Projects them through PCA → reverse UMAP (seconds, no re-fitting).
  3. Produces a side-by-side density diagnostic:

       Left  — MaxMin diversity picks in library UMAP space
       Right — Fresh random 24K in the same library UMAP space

Both evaluated against the same 100K background → directly comparable.

Prerequisites:
    Run umap_diversity_study.py first (saves PCA + UMAP models to cache/).

Usage:
    python analysis/diversity_study/compare_random_baseline.py --target trpv1_8gfa
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg2

SCRIPT_DIR  = Path(__file__).resolve().parent
CACHE_DIR   = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR / "figures"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target",       type=str,  default="trpv1_8gfa")
    p.add_argument("--fp-file",      type=Path, default=Path("data/fingerprints/compounds.h5"))
    p.add_argument("--bins",         type=int,  default=120)
    p.add_argument("--min-bg-count", type=int,  default=3)
    p.add_argument("--seed",         type=int,  default=99)
    return p.parse_args()


# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def db_connect():
    return psycopg2.connect(
        dbname="analgesics",
        user="labhund",
        host=os.environ.get("PGHOST", "/tmp"),
    )


def sample_fresh_random_ids(target: str, n: int, exclude_ids: np.ndarray) -> np.ndarray:
    """
    Sample n random compound IDs from DB via TABLESAMPLE, excluding any IDs
    already used as background or diversity picks.
    """
    exclude_set = set(exclude_ids.tolist())
    conn = db_connect()
    cur  = conn.cursor()

    # Overshoot to account for exclusions (~0.04% of library excluded, negligible)
    frac = (n / 300_000_000) * 100 * 2.0
    frac = max(frac, 0.001)

    logging.info("TABLESAMPLE BERNOULLI(%.4f%%) for fresh random sample…", frac)
    cur.execute(
        "SELECT id FROM compounds TABLESAMPLE BERNOULLI(%s)",
        (frac,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    # Filter out excluded IDs
    mask = np.array([i not in exclude_set for i in ids], dtype=bool)
    ids  = ids[mask]

    if len(ids) < n:
        logging.warning("Only %d fresh IDs after exclusion (wanted %d) — using all", len(ids), n)
    else:
        # Random subsample to exactly n (without replacement)
        rng  = np.random.default_rng(99)
        ids  = rng.choice(ids, size=n, replace=False)

    logging.info("Fresh random sample: %d compound IDs", len(ids))
    return ids


# ---------------------------------------------------------------------------
# FP loading (same as main script)
# ---------------------------------------------------------------------------

def load_fps(fp_file: Path, compound_ids: np.ndarray) -> np.ndarray:
    rows     = compound_ids - 1
    sort_idx = np.argsort(rows)
    rows_sorted = rows[sort_idx]

    fps_list = []
    with h5py.File(fp_file, "r") as f:
        dset = f["fps"]
        runs = []
        if len(rows_sorted) > 0:
            s = e = rows_sorted[0]
            for r in rows_sorted[1:]:
                if r == e + 1:
                    e = r
                else:
                    runs.append((s, e + 1)); s = e = r
            runs.append((s, e + 1))
        for (s, e) in runs:
            fps_list.append(dset[s:e])

    if not fps_list:
        return np.zeros((0, 4096), dtype=np.float32)

    fp_raw = np.concatenate(fps_list, axis=0)
    fps    = np.unpackbits(fp_raw, axis=1).astype(np.float32)
    return fps[np.argsort(sort_idx)]


# ---------------------------------------------------------------------------
# Density diagnostic
# ---------------------------------------------------------------------------

def shared_edges(emb_bg, emb_a, emb_b, bins):
    all_x = np.concatenate([emb_bg[:,0], emb_a[:,0], emb_b[:,0]])
    all_y = np.concatenate([emb_bg[:,1], emb_a[:,1], emb_b[:,1]])
    mx = (all_x.max() - all_x.min()) * 0.02
    my = (all_y.max() - all_y.min()) * 0.02
    xedges = np.linspace(all_x.min() - mx, all_x.max() + mx, bins + 1)
    yedges = np.linspace(all_y.min() - my, all_y.max() + my, bins + 1)
    return xedges, yedges


def density_ratio(emb_picks, emb_bg, xedges, yedges, min_bg_count):
    h_bg, _, _ = np.histogram2d(emb_bg[:,0],    emb_bg[:,1],    bins=(xedges, yedges))
    h_pk, _, _ = np.histogram2d(emb_picks[:,0], emb_picks[:,1], bins=(xedges, yedges))
    mask  = h_bg < min_bg_count
    eps   = 1e-9
    ratio = np.log2((h_pk / max(h_pk.sum(), 1) + eps) / (h_bg / h_bg.sum() + eps))
    ratio[mask] = np.nan
    total = h_bg + h_pk
    cover = np.where(total > 0, h_pk / total, np.nan)
    cover[mask] = np.nan
    return ratio, cover, h_bg, h_pk


def summarise(label, ratio, n):
    v = ratio[~np.isnan(ratio)]
    print(f"\n  {label}  (n={n:,}, valid bins={len(v)})")
    print(f"    std  log2-ratio    : {v.std():.3f}   (lower = more uniform)")
    print(f"    IQR                : {np.percentile(v,75)-np.percentile(v,25):.3f}")
    print(f"    bins <−2 (≥4× under): {(v<-2).sum():4d}  ({100*(v<-2).mean():.1f}%)")
    print(f"    bins >+2 (≥4× over) : {(v>+2).sum():4d}  ({100*(v>+2).mean():.1f}%)")
    return v


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_density_figure(emb_bg, emb_rand, h_bg, h_rand, xedges, yedges, args):
    """
    Raw log-density maps: 100K background (left) vs fresh random 24K (right).
    Shared colour scale across both panels for direct visual comparison.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcen, ycen, indexing="ij")

    d_bg   = np.log10(h_bg   + 1).astype(float)
    d_rand = np.log10(h_rand + 1).astype(float)

    # Mask bins with no background support
    outside = h_bg == 0
    d_bg[outside]   = np.nan
    d_rand[outside] = np.nan

    vmax = max(np.nanmax(d_bg), np.nanmax(d_rand))

    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], d_bg,   f"100K library background\nraw point density (reverse UMAP space)",   f"n = {len(emb_bg):,}"),
        (axes[1], d_rand, f"Fresh random 24K (never seen by UMAP)\nraw point density (projected)", f"n = {len(emb_rand):,}"),
    ]

    for ax, data, title, subtitle in panels:
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=vmax,
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("log₁₀(counts + 1) per bin", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ticks = [t for t in [0, 0.5, 1, 1.5, 2, 2.5, 3] if t <= vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:.1f}\n({int(10**t - 1)})" for t in ticks])
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP-1 (library space)", fontsize=9)
        ax.set_ylabel("UMAP-2 (library space)", fontsize=9)
        ax.tick_params(labelsize=7)

    footer = (
        f"target={args.target}  bins={args.bins}×{args.bins}  seed={args.seed}  "
        f"colour scale: shared log₁₀(counts+1), vmax={vmax:.2f} ({int(10**vmax - 1)} counts)"
    )
    fig.text(0.5, 0.005, footer, ha="center", va="bottom", fontsize=7.5, color="#555")
    plt.suptitle(
        "Raw Density Maps — Library Background vs Fresh Random 24K (Reverse UMAP space)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = FIGURES_DIR / f"density_maps_random_{args.target}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

def make_figure(emb_bg, emb_maxmin, ratio_mm, cover_mm,
                emb_rand,   ratio_rd, cover_rd,
                h_bg, xedges, yedges, args):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])
    X, Y = np.meshgrid(xcen, ycen, indexing="ij")

    vmax_r   = 4.0
    expected = len(emb_maxmin) / (len(emb_maxmin) + len(emb_bg))
    vmax_c   = min(1.0, 4 * expected)
    bg_h     = h_bg.astype(float)
    levels   = np.percentile(bg_h[bg_h > 0], [50, 80, 95])

    cmap_r = plt.cm.RdBu_r;  cmap_r.set_bad("white")
    cmap_c = plt.cm.RdYlGn;  cmap_c.set_bad("white")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0,0], ratio_mm, cmap_r, -vmax_r, vmax_r,
         "MaxMin diversity picks\nLog₂ density ratio vs library",
         "log₂(picks / library density)"),
        (axes[0,1], ratio_rd, cmap_r, -vmax_r, vmax_r,
         "Fresh random 24K (never seen by UMAP)\nLog₂ density ratio vs library",
         "log₂(picks / library density)"),
        (axes[1,0], cover_mm, cmap_c, 0, vmax_c,
         "MaxMin — coverage fraction per bin",
         "picks / (picks + background)"),
        (axes[1,1], cover_rd, cmap_c, 0, vmax_c,
         "Random — coverage fraction per bin",
         "picks / (picks + background)"),
    ]

    for ax, data, cmap, vmin, vmax, title, cblabel in panels:
        im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(cblabel, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        if cmap is cmap_r:
            cb.set_ticks([-4, -2, 0, 2, 4])
            cb.set_ticklabels(["−4\n(16×\nunder)", "−2", "0\n(equal)", "+2", "+4\n(16×\nover)"])
        else:
            cb.ax.axhline(expected / vmax_c, color="k", lw=1.5, ls="--")
        ax.contour(X, Y, bg_h, levels=levels, colors="k", linewidths=0.4, alpha=0.35)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("UMAP-1 (library space)", fontsize=8)
        ax.set_ylabel("UMAP-2 (library space)", fontsize=8)
        ax.tick_params(labelsize=7)

    # Column headers
    for col, label in enumerate(["MaxMin diversity (diversity_init)", "Fresh random baseline"]):
        axes[0, col].annotate(
            label, xy=(0.5, 1.07), xycoords="axes fraction",
            ha="center", fontsize=12, fontweight="bold", color="#222",
        )

    footer = (
        f"target={args.target}  n=24,000 each  n_background={len(emb_bg):,}  "
        f"bins={args.bins}×{args.bins}  fresh_random_seed={args.seed}"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=8, color="#555")
    plt.suptitle(
        "MaxMin Diversity vs Random Baseline — Density Diagnostic (Reverse UMAP space)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out = FIGURES_DIR / f"compare_random_{args.target}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    # ------------------------------------------------------------------
    # Load cached embeddings + models
    # ------------------------------------------------------------------
    required = {
        "emb_bg_rev":    CACHE_DIR / f"emb_bg_rev_{args.target}.npy",
        "emb_24k_on_bg": CACHE_DIR / f"emb_24k_on_bg_{args.target}.npy",
        "bg_ids":        CACHE_DIR / f"bg_ids_{args.target}.npy",
        "pca_model":     CACHE_DIR / f"pca_{args.target}.joblib",
        "umap_rev":      CACHE_DIR / f"umap_rev_{args.target}.joblib",
    }
    missing = [str(v) for v in required.values() if not v.exists()]
    if missing:
        print("ERROR: missing cache/model files:")
        for m in missing: print(" ", m)
        print("\nRe-run:  python analysis/diversity_study/umap_diversity_study.py --force")
        sys.exit(1)

    print("Loading cached embeddings and models…")
    emb_bg        = np.load(required["emb_bg_rev"])
    emb_maxmin    = np.load(required["emb_24k_on_bg"])
    bg_ids        = np.load(required["bg_ids"])
    pca_bundle    = joblib.load(required["pca_model"])   # {"pca": ..., "n95": ...}
    umap_rev      = joblib.load(required["umap_rev"])
    pca_model     = pca_bundle["pca"]
    n95           = pca_bundle["n95"]
    print(f"  background: {emb_bg.shape}   maxmin: {emb_maxmin.shape}   n95={n95}")

    # ------------------------------------------------------------------
    # Sample fresh 24K — never seen by UMAP/PCA during fitting
    # ------------------------------------------------------------------
    n_picks = len(emb_maxmin)
    print(f"\nSampling {n_picks:,} fresh random compounds from DB (excluding {len(bg_ids):,} background IDs)…")
    ids_fresh = sample_fresh_random_ids(args.target, n_picks, bg_ids)

    # ------------------------------------------------------------------
    # Load FPs and project into library UMAP space
    # ------------------------------------------------------------------
    print(f"Loading FPs for {len(ids_fresh):,} fresh compounds…")
    t0 = time.time()
    fps_fresh = load_fps(args.fp_file, ids_fresh)
    print(f"  FP load: {time.time()-t0:.1f}s  shape={fps_fresh.shape}")

    print("Projecting through PCA…")
    t0 = time.time()
    pca_fresh = pca_model.transform(fps_fresh)[:, :n95]
    print(f"  PCA transform: {time.time()-t0:.1f}s  shape={pca_fresh.shape}")
    del fps_fresh

    print("Projecting through reverse UMAP…")
    t0 = time.time()
    emb_rand = umap_rev.transform(pca_fresh)
    print(f"  UMAP transform: {time.time()-t0:.1f}s  shape={emb_rand.shape}")
    del pca_fresh

    # ------------------------------------------------------------------
    # Density diagnostics
    # ------------------------------------------------------------------
    print(f"\nComputing density histograms ({args.bins}×{args.bins} bins)…")
    xedges, yedges = shared_edges(emb_bg, emb_maxmin, emb_rand, args.bins)
    ratio_mm, cover_mm, h_bg, _      = density_ratio(emb_maxmin, emb_bg, xedges, yedges, args.min_bg_count)
    ratio_rd, cover_rd, _,   h_rand  = density_ratio(emb_rand,   emb_bg, xedges, yedges, args.min_bg_count)

    print("\n=== Summary statistics ===")
    v_mm = summarise("MaxMin diversity", ratio_mm, len(emb_maxmin))
    v_rd = summarise("Fresh random    ", ratio_rd, len(emb_rand))

    print("\n=== Improvement (MaxMin vs random) ===")
    delta_std = v_rd.std() - v_mm.std()
    delta_iqr = (np.percentile(v_rd,75)-np.percentile(v_rd,25)) - (np.percentile(v_mm,75)-np.percentile(v_mm,25))
    print(f"  std reduction : {delta_std:+.3f}  ({'MaxMin more uniform ✓' if delta_std > 0 else 'Random more uniform'})")
    print(f"  IQR reduction : {delta_iqr:+.3f}")
    under_mm = (v_mm < -2).mean() * 100
    under_rd = (v_rd < -2).mean() * 100
    print(f"  ≥4× undersampled bins: MaxMin={under_mm:.1f}%  Random={under_rd:.1f}%  Δ={under_rd-under_mm:+.1f}pp")

    print("\nGenerating ratio/coverage figure…")
    make_figure(emb_bg, emb_maxmin, ratio_mm, cover_mm,
                emb_rand,   ratio_rd, cover_rd,
                h_bg, xedges, yedges, args)

    print("Generating raw density figure…")
    make_density_figure(emb_bg, emb_rand, h_bg, h_rand, xedges, yedges, args)


if __name__ == "__main__":
    main()
