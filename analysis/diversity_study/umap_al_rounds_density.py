#!/usr/bin/env python3
"""
Interactive UMAP density ratio — AL rounds vs library chemical space.

For each AL round, shows the log₂ density ratio of docked compounds vs the
2.4M library background in UMAP space. A slider steps through rounds one at
a time.

  Red   = oversampled vs library (AL focused here)
  Blue  = undersampled vs library (AL avoided here)
  White = same density as library

Requires cache files from umap_al_rounds.py:
    cache/emb_al_{exp_id}_{target}.npy
    cache/al_meta_{exp_id}_{target}.csv
    cache/emb_bg_rev_{target}.npy

Usage:
    python analysis/diversity_study/umap_al_rounds_density.py \\
        --target trpv1_8gfa \\
        --experiment-id maxmin_init \\
        [--bins 120] \\
        [--min-bg-count 3] \\
        [--out analysis/figures/umap_al_rounds_density_trpv1_8gfa_maxmin_init.html]
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
CACHE_DIR   = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target",        type=str,  default="trpv1_8gfa")
    p.add_argument("--experiment-id", type=str,  default="maxmin_init")
    p.add_argument("--bins",          type=int,  default=120,
                   help="Number of histogram bins along each UMAP axis (default 120)")
    p.add_argument("--min-bg-count",  type=int,  default=3,
                   help="Mask bins with fewer than N background points (default 3)")
    p.add_argument("--bg-subsample",  type=int,  default=30_000,
                   help="Library background scatter points shown (default 30000)")
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--out",           type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_al_cache(target: str, exp_id: str):
    """
    Load cached AL embeddings + metadata written by umap_al_rounds.py.

    Returns
    -------
    al_emb    : (N, 2) float64
    scores    : (N,) float64
    al_rounds : (N,) int
    zinc_ids  : list of str
    """
    emb_path  = CACHE_DIR / f"emb_al_{exp_id}_{target}.npy"
    meta_path = CACHE_DIR / f"al_meta_{exp_id}_{target}.csv"

    if not emb_path.exists() or not meta_path.exists():
        LOG.error("Cache not found — run umap_al_rounds.py first:")
        LOG.error("  %s", emb_path)
        LOG.error("  %s", meta_path)
        sys.exit(1)

    al_emb = np.load(emb_path)

    with open(meta_path, newline="") as f:
        meta = list(csv.DictReader(f))

    scores    = np.array([float(r["score"])    for r in meta], dtype=np.float64)
    al_rounds = np.array([int(r["al_round"])   for r in meta], dtype=np.int32)
    zinc_ids  =         [r["zinc_id"]          for r in meta]

    return al_emb, scores, al_rounds, zinc_ids


def load_bg(target: str) -> np.ndarray:
    bg_path = CACHE_DIR / f"emb_bg_rev_{target}.npy"
    if not bg_path.exists():
        LOG.error("Background embeddings not found: %s", bg_path)
        LOG.error("Re-run umap_diversity_study.py")
        sys.exit(1)
    return np.load(bg_path)


# ---------------------------------------------------------------------------
# Density computation (matches compare_random_baseline.py exactly)
# ---------------------------------------------------------------------------

def shared_edges(emb_bg: np.ndarray, al_emb: np.ndarray, bins: int):
    """Shared bin edges spanning background + all AL rounds."""
    all_x = np.concatenate([emb_bg[:, 0], al_emb[:, 0]])
    all_y = np.concatenate([emb_bg[:, 1], al_emb[:, 1]])
    mx = (all_x.max() - all_x.min()) * 0.02
    my = (all_y.max() - all_y.min()) * 0.02
    xedges = np.linspace(all_x.min() - mx, all_x.max() + mx, bins + 1)
    yedges = np.linspace(all_y.min() - my, all_y.max() + my, bins + 1)
    return xedges, yedges


def density_ratio(emb_round: np.ndarray, h_bg: np.ndarray,
                  xedges: np.ndarray, yedges: np.ndarray,
                  min_bg_count: int):
    """
    Log₂ density ratio of round vs library background.

    h_bg : pre-computed background histogram (n_x_bins, n_y_bins)

    Returns ratio array (n_x_bins, n_y_bins), same shape as h_bg.
    Bins with h_bg < min_bg_count are set to NaN.
    """
    h_rnd, _, _ = np.histogram2d(
        emb_round[:, 0], emb_round[:, 1], bins=(xedges, yedges)
    )
    eps   = 1e-9
    ratio = np.log2(
        (h_rnd / max(h_rnd.sum(), 1) + eps) /
        (h_bg  / max(h_bg.sum(),  1) + eps)
    )
    ratio[h_bg < min_bg_count] = np.nan
    return ratio, h_rnd


def compute_all_ratios(emb_bg, al_emb, al_rounds_arr, xedges, yedges, min_bg_count):
    """Compute log₂ density ratio for each AL round. Returns dict round→ratio."""
    h_bg, _, _ = np.histogram2d(
        emb_bg[:, 0], emb_bg[:, 1], bins=(xedges, yedges)
    )
    rounds = sorted(set(al_rounds_arr.tolist()))
    ratios = {}
    h_rnds = {}
    for r in rounds:
        mask = al_rounds_arr == r
        ratio, h_rnd = density_ratio(al_emb[mask], h_bg, xedges, yedges, min_bg_count)
        ratios[r] = ratio
        h_rnds[r] = h_rnd
        n_valid = (~np.isnan(ratio)).sum()
        LOG.info("  Round %d: %d compounds, %d valid bins, "
                 "std log2-ratio=%.3f, oversampled(>+2)=%.1f%%, undersampled(<-2)=%.1f%%",
                 r, mask.sum(), n_valid,
                 ratio[~np.isnan(ratio)].std(),
                 100 * (ratio[~np.isnan(ratio)] > 2).mean(),
                 100 * (ratio[~np.isnan(ratio)] < -2).mean())
    return ratios, h_rnds, h_bg


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def make_figure(emb_bg, al_emb, scores, al_rounds_arr, zinc_ids,
                ratios, xedges, yedges,
                target, exp_id, bg_subsample, seed):
    import plotly.graph_objects as go

    rounds  = sorted(ratios.keys())
    xcen    = 0.5 * (xedges[:-1] + xedges[1:])
    ycen    = 0.5 * (yedges[:-1] + yedges[1:])

    # Symmetric colorscale limit: cap at 4.0 (16×), driven by 99th percentile
    all_finite = np.concatenate([ratios[r][~np.isnan(ratios[r])] for r in rounds])
    vmax = min(4.0, float(np.percentile(np.abs(all_finite), 99.5)))

    # Per-round stats for slider labels
    stats = {}
    for r in rounds:
        mask = al_rounds_arr == r
        sc   = scores[mask]
        stats[r] = dict(n=int(mask.sum()), best=float(sc.min()), mean=float(sc.mean()))

    traces = []

    # --- Background scatter (always visible, very faint) ---
    rng   = np.random.default_rng(seed)
    n_bg  = min(bg_subsample, len(emb_bg))
    idx   = rng.choice(len(emb_bg), size=n_bg, replace=False)
    bg_xy = emb_bg[idx]
    traces.append(go.Scattergl(
        x=bg_xy[:, 0],
        y=bg_xy[:, 1],
        mode="markers",
        marker=dict(color="rgba(200,200,200,0.06)", size=1.5),
        name=f"Library ({n_bg:,} shown)",
        hoverinfo="skip",
        showlegend=False,
    ))
    N_BG_TRACE = 1  # number of always-visible traces

    # --- One density-ratio heatmap per round ---
    colorbar_cfg = dict(
        title=dict(
            text="log₂(round / library)",
            side="right",
            font=dict(color="#cccccc", size=12),
        ),
        tickvals=[-4, -3, -2, -1, 0, 1, 2, 3, 4],
        ticktext=[
            "−4 (16× under)", "−3", "−2 (4× under)", "−1",
            "0 (equal)",
            "+1", "+2 (4× over)", "+3", "+4 (16× over)",
        ],
        tickfont=dict(color="#cccccc", size=10),
        thickness=16,
        len=0.75,
        bgcolor="rgba(30,30,60,0.6)",
        bordercolor="#555",
        borderwidth=1,
    )

    for i, r in enumerate(rounds):
        # Plotly Heatmap: z[row, col] = z at y=ycen[row], x=xcen[col]
        # np.histogram2d output: (n_x, n_y), so transpose → (n_y, n_x)
        z = ratios[r].T.copy()   # (n_y_bins, n_x_bins), NaN = masked

        traces.append(go.Heatmap(
            x=xcen,
            y=ycen,
            z=z,
            colorscale="RdBu_r",   # blue=undersampled, red=oversampled
            zmin=-vmax,
            zmax=vmax,
            zmid=0.0,
            colorbar=colorbar_cfg,
            visible=(r == rounds[0]),
            name=f"Round {r}",
            hovertemplate=(
                "UMAP-1: %{x:.2f}<br>"
                "UMAP-2: %{y:.2f}<br>"
                "log₂ ratio: %{z:.2f}"
                f"<extra>Round {r}</extra>"
            ),
        ))

    # --- Slider ---
    # Each step: show background + this round's heatmap, hide others
    n_total = N_BG_TRACE + len(rounds)

    def make_visible(round_idx):
        vis = [False] * n_total
        for j in range(N_BG_TRACE):   # always show background
            vis[j] = True
        vis[N_BG_TRACE + round_idx] = True
        return vis

    def round_title(r):
        st = stats[r]
        return (
            f"AL density ratio — {target} [{exp_id}]<br>"
            f"<sup>Round {r}: n={st['n']:,}  "
            f"best={st['best']:.2f}  mean={st['mean']:.2f} kcal/mol  |  "
            f"red = AL oversampled vs library, blue = undersampled</sup>"
        )

    steps = []
    for i, r in enumerate(rounds):
        steps.append({
            "method": "update",
            "args": [
                {"visible": make_visible(i)},
                {"title": {"text": round_title(r),
                           "font": {"size": 15, "color": "#e8e8e8"},
                           "x": 0.5}},
            ],
            "label": f"R{r}",
        })

    slider_cfg = dict(
        active=0,
        currentvalue=dict(
            prefix="Showing: ",
            font=dict(color="#cccccc", size=13),
            visible=True,
            xanchor="left",
        ),
        steps=steps,
        x=0.05,
        len=0.9,
        pad={"t": 10, "b": 10},
        bgcolor="#1e1e3a",
        activebgcolor="#ffb347",
        bordercolor="#555",
        borderwidth=1,
        font=dict(color="#cccccc", size=11),
    )

    layout = go.Layout(
        title=dict(
            text=round_title(rounds[0]),
            font=dict(size=15, color="#e8e8e8"),
            x=0.5,
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#0d1117",
        width=1100,
        height=870,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title="",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title="",
        ),
        hovermode="closest",
        margin=dict(l=20, r=120, t=90, b=100),
        sliders=[slider_cfg],
    )

    return go.Figure(data=traces, layout=layout)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(al_rounds_arr, scores, ratios, target, exp_id, out_html, bins, min_bg_count):
    rounds = sorted(ratios.keys())
    print(f"\n{'='*65}")
    print(f"  Density ratio map — {target} [{exp_id}]")
    print(f"  Bins: {bins}×{bins}  min_bg_count={min_bg_count}")
    print(f"{'='*65}")
    print(f"  {'Round':>5}  {'N':>7}  {'Best':>8}  {'Mean':>8}  "
          f"{'StdLog2R':>9}  {'>4×over':>8}  {'>4×under':>9}")
    print(f"  {'-'*62}")
    for r in rounds:
        mask = al_rounds_arr == r
        sc   = scores[mask]
        v    = ratios[r][~np.isnan(ratios[r])]
        over   = 100 * (v > 2).mean()
        under  = 100 * (v < -2).mean()
        print(f"  {r:>5}  {mask.sum():>7,}  {sc.min():>8.2f}  {sc.mean():>8.2f}  "
              f"{v.std():>9.3f}  {over:>7.1f}%  {under:>8.1f}%")
    print(f"\n  Output → {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    target = args.target
    exp_id = args.experiment_id

    out_html = args.out or (
        FIGURES_DIR / f"umap_al_rounds_density_{target}_{exp_id}.html"
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    LOG.info("Loading cached AL data (target=%s, exp=%s)…", target, exp_id)
    al_emb, scores, al_rounds, zinc_ids = load_al_cache(target, exp_id)
    LOG.info("  %d compounds, rounds %s", len(al_emb), sorted(set(al_rounds.tolist())))

    LOG.info("Loading background embeddings…")
    emb_bg = load_bg(target)
    LOG.info("  Background: %s", emb_bg.shape)

    # Bin edges
    LOG.info("Computing %d×%d density histograms…", args.bins, args.bins)
    xedges, yedges = shared_edges(emb_bg, al_emb, args.bins)
    ratios, h_rnds, h_bg = compute_all_ratios(
        emb_bg, al_emb, al_rounds, xedges, yedges, args.min_bg_count
    )

    # Build figure
    LOG.info("Building Plotly figure with slider…")
    t0  = time.time()
    fig = make_figure(
        emb_bg, al_emb, scores, al_rounds, zinc_ids,
        ratios, xedges, yedges,
        target, exp_id, args.bg_subsample, args.seed,
    )
    LOG.info("  Built in %.1fs", time.time() - t0)

    # Save
    LOG.info("Writing HTML: %s", out_html)
    t0 = time.time()
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    LOG.info("  Saved in %.1fs  (%.1f MB)", time.time() - t0,
             out_html.stat().st_size / 1e6)

    print_summary(al_rounds, scores, ratios, target, exp_id, out_html,
                  args.bins, args.min_bg_count)


if __name__ == "__main__":
    main()
