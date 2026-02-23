#!/usr/bin/env python3
"""
Interactive UMAP — AL rounds projected into library chemical space.

Projects all docked AL compounds through the reverse UMAP fitted on 2.4M
library compounds, producing a self-contained Plotly HTML showing which
regions of chemical space each round explored.

Traces:
  - Grey background: 50K random subsample of the 2.4M library frame
  - One coloured trace per round (brown→orange), toggle-able in legend
  - Hover: ZINC ID + docking score

Prerequisite:
    Run umap_diversity_study.py first (saves PCA + UMAP models to cache/).

Usage:
    python analysis/diversity_study/umap_al_rounds.py \\
        --target trpv1_8gfa \\
        --experiment-id maxmin_init \\
        [--bg-subsample 50000] \\
        [--out analysis/figures/umap_al_rounds_trpv1_8gfa_maxmin_init.html] \\
        [--force]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import psycopg2

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
    p.add_argument("--target",        type=str,  default="trpv1_8gfa",
                   help="Docking target identifier")
    p.add_argument("--experiment-id", type=str,  default="maxmin_init",
                   help="Experiment ID (e.g. maxmin_init, random_init)")
    p.add_argument("--bg-subsample",  type=int,  default=50_000,
                   help="Number of background library points to show (default 50000)")
    p.add_argument("--out",           type=Path, default=None,
                   help="Output HTML path (default: analysis/figures/umap_al_rounds_<target>_<exp>.html)")
    p.add_argument("--seed",          type=int,  default=42,
                   help="RNG seed for background subsample")
    p.add_argument("--force",         action="store_true",
                   help="Recompute UMAP projection even if cache exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def db_connect():
    return psycopg2.connect(
        dbname="analgesics",
        user="labhund",
        host=os.environ.get("PGHOST", "/tmp"),
    )


def fetch_al_compounds(target: str, exp_id: str):
    """
    Return list of (compound_id, zinc_id, smiles_protonated, score, al_round)
    for all docked compounds with valid scores for this target + experiment.
    """
    conn = db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ds.compound_id, c.zinc_id, c.smiles_protonated,
                       ds.score, ds.al_round
                FROM   docking_scores ds
                JOIN   compounds c ON ds.compound_id = c.id
                WHERE  ds.target = %s AND ds.experiment_id = %s
                       AND ds.score IS NOT NULL
                ORDER  BY ds.al_round, ds.compound_id
                """,
                (target, exp_id),
            )
            return cur.fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Morgan FP computation from SMILES
# ---------------------------------------------------------------------------

def compute_fps_from_smiles(rows, radius: int = 4, n_bits: int = 4096):
    """
    Compute Morgan FPs from SMILES strings.

    rows : list of (compound_id, zinc_id, smiles, score, al_round)

    Returns
    -------
    fps_arr   : (N_valid, n_bits) float32
    meta_rows : list of (compound_id, zinc_id, score, al_round) for valid rows
    n_failed  : int — number of SMILES that failed to parse
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fps_list  = []
    meta_rows = []
    n_failed  = 0
    n_total   = len(rows)

    for i, (cid, zinc_id, smiles, score, al_round) in enumerate(rows):
        if i > 0 and i % 20_000 == 0:
            LOG.info("  FP progress: %d / %d  (failed=%d)", i, n_total, n_failed)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_failed += 1
            continue
        bv = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        fps_list.append(bv)
        meta_rows.append((cid, zinc_id, score, al_round))

    fps_arr = np.stack(fps_list, axis=0) if fps_list else np.zeros((0, n_bits), dtype=np.float32)
    return fps_arr, meta_rows, n_failed


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def round_palette(n_rounds: int, exp_id: str = "maxmin_init"):
    """
    Brown→orange for maxmin_init; blue gradient for random_init.
    Returns list of hex colour strings, one per round.
    """
    import matplotlib.colors as mcolors

    if "random" in exp_id:
        start = np.array(mcolors.to_rgb("#08306b"))
        end   = np.array(mcolors.to_rgb("#bdd7e7"))
    else:
        start = np.array(mcolors.to_rgb("#3d0c00"))
        end   = np.array(mcolors.to_rgb("#ffb347"))

    colors = []
    for i in range(n_rounds):
        t   = i / max(n_rounds - 1, 1)
        rgb = start + (end - start) * t
        colors.append(mcolors.to_hex(np.clip(rgb, 0, 1)))
    return colors


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------

def make_figure(emb_bg, al_emb, compound_ids, zinc_ids, scores, al_rounds,
                target, exp_id, bg_subsample, seed):
    """
    Build interactive Plotly Scattergl figure.

    emb_bg       : (N_bg, 2) background embeddings
    al_emb       : (N_al, 2) AL compound embeddings
    compound_ids : (N_al,) int
    zinc_ids     : (N_al,) str
    scores       : (N_al,) float
    al_rounds    : (N_al,) int
    """
    import plotly.graph_objects as go

    rounds  = sorted(set(al_rounds))
    palette = round_palette(len(rounds), exp_id)
    round_color = {r: c for r, c in zip(rounds, palette)}

    traces = []

    # --- Background: random subsample of 2.4M library ---
    rng   = np.random.default_rng(seed)
    n_bg  = min(bg_subsample, len(emb_bg))
    idx   = rng.choice(len(emb_bg), size=n_bg, replace=False)
    bg_xy = emb_bg[idx]

    traces.append(go.Scattergl(
        x=bg_xy[:, 0],
        y=bg_xy[:, 1],
        mode="markers",
        marker=dict(color="rgba(180,180,180,0.07)", size=2),
        name=f"Library (2.4M, n={n_bg:,} shown)",
        hoverinfo="skip",
        showlegend=True,
    ))

    # --- One trace per AL round ---
    al_rounds_arr    = np.array(al_rounds)
    scores_arr       = np.array(scores, dtype=np.float64)
    zinc_ids_arr     = np.array(zinc_ids)

    for r in rounds:
        mask = al_rounds_arr == r
        n_r  = mask.sum()

        # customdata columns: [zinc_id (str), score (float)]
        cdata = np.empty((n_r, 2), dtype=object)
        cdata[:, 0] = zinc_ids_arr[mask]
        cdata[:, 1] = scores_arr[mask]

        traces.append(go.Scattergl(
            x=al_emb[mask, 0],
            y=al_emb[mask, 1],
            mode="markers",
            marker=dict(
                color=round_color[r],
                size=4,
                opacity=0.55,
            ),
            name=f"Round {r} (n={n_r:,})",
            customdata=cdata,
            hovertemplate=(
                "ZINC: %{customdata[0]}<br>"
                "Score: %{customdata[1]:.2f} kcal/mol"
                f"<extra>Round {r}</extra>"
            ),
        ))

    layout = go.Layout(
        title=dict(
            text=f"AL rounds — {target} [{exp_id}] (reverse UMAP, library frame)",
            font=dict(size=16, color="#e8e8e8"),
            x=0.5,
        ),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        width=1200,
        height=800,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title="",
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            title="",
        ),
        legend=dict(
            font=dict(color="#e8e8e8", size=11),
            bgcolor="rgba(30,30,60,0.75)",
            bordercolor="#555",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return go.Figure(data=traces, layout=layout)


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

    # Output paths
    out_html = args.out or (
        FIGURES_DIR / f"umap_al_rounds_{target}_{exp_id}.html"
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_emb  = CACHE_DIR / f"emb_al_{exp_id}_{target}.npy"
    cache_meta = CACHE_DIR / f"al_meta_{exp_id}_{target}.csv"

    # -----------------------------------------------------------------------
    # Step A: Get AL embeddings (from cache or compute fresh)
    # -----------------------------------------------------------------------
    if cache_emb.exists() and cache_meta.exists() and not args.force:
        LOG.info("Cache hit — loading embeddings and metadata:")
        LOG.info("  %s", cache_emb)
        LOG.info("  %s", cache_meta)
        al_emb = np.load(cache_emb)

        # Parse metadata CSV
        import csv
        with open(cache_meta, newline="") as f:
            reader = csv.DictReader(f)
            meta_rows = list(reader)

        compound_ids = [int(r["compound_id"]) for r in meta_rows]
        zinc_ids     = [r["zinc_id"]          for r in meta_rows]
        scores       = [float(r["score"])      for r in meta_rows]
        al_rounds    = [int(r["al_round"])     for r in meta_rows]

        LOG.info("  Loaded %d compounds.", len(al_emb))

    else:
        # -------------------------------------------------------------------
        # Check required model/cache files
        # -------------------------------------------------------------------
        required = {
            "pca_model": CACHE_DIR / f"pca_{target}.joblib",
            "umap_rev":  CACHE_DIR / f"umap_rev_{target}.joblib",
        }
        missing = [str(v) for k, v in required.items() if not v.exists()]
        if missing:
            LOG.error("Missing required model files — re-run umap_diversity_study.py:")
            for m in missing:
                LOG.error("  %s", m)
            sys.exit(1)

        # -------------------------------------------------------------------
        # Step 1: Fetch docked AL compounds from DB
        # -------------------------------------------------------------------
        LOG.info("Fetching AL compounds (target=%s, exp=%s)…", target, exp_id)
        t0   = time.time()
        rows = fetch_al_compounds(target, exp_id)
        LOG.info("  Fetched %d compounds in %.1fs", len(rows), time.time() - t0)

        if not rows:
            LOG.error("No docked compounds found. Check target/experiment-id and DB.")
            sys.exit(1)

        # -------------------------------------------------------------------
        # Step 2: Compute Morgan FPs from SMILES
        # -------------------------------------------------------------------
        LOG.info("Computing Morgan FPs (radius=4, nBits=4096) for %d compounds…", len(rows))
        t0 = time.time()
        fps_arr, meta_rows_valid, n_failed = compute_fps_from_smiles(rows)
        elapsed = time.time() - t0
        LOG.info("  Done in %.1fs — valid=%d  failed=%d", elapsed, len(fps_arr), n_failed)
        if n_failed:
            LOG.warning("  %d SMILES failed to parse and are excluded from the plot.", n_failed)

        if len(fps_arr) == 0:
            LOG.error("No valid fingerprints computed. Aborting.")
            sys.exit(1)

        # Unpack metadata
        compound_ids = [r[0] for r in meta_rows_valid]
        zinc_ids     = [r[1] for r in meta_rows_valid]
        scores       = [r[2] for r in meta_rows_valid]
        al_rounds    = [r[3] for r in meta_rows_valid]

        # -------------------------------------------------------------------
        # Step 3 (part of Step 4 in plan): PCA transform
        # -------------------------------------------------------------------
        LOG.info("Loading PCA model: %s", required["pca_model"])
        pca_bundle = joblib.load(required["pca_model"])
        pca        = pca_bundle["pca"]
        n95        = pca_bundle["n95"]
        LOG.info("  n95=%d components", n95)

        t0     = time.time()
        al_pca = pca.transform(fps_arr)[:, :n95]
        LOG.info("  PCA transform: %.1fs", time.time() - t0)
        del fps_arr

        # -------------------------------------------------------------------
        # Step 4: UMAP project through reverse model
        # -------------------------------------------------------------------
        LOG.info("Loading reverse UMAP model (may take ~30 s for 12 GB joblib)…")
        t0       = time.time()
        umap_rev = joblib.load(required["umap_rev"])
        LOG.info("  Model loaded in %.1fs", time.time() - t0)

        LOG.info("Projecting %d compounds through reverse UMAP (first run ~3–5 min)…",
                 len(al_pca))
        t0     = time.time()
        al_emb = umap_rev.transform(al_pca)
        LOG.info("  UMAP transform done in %.1fs", time.time() - t0)
        del al_pca, umap_rev

        # -------------------------------------------------------------------
        # Step 5: Cache results
        # -------------------------------------------------------------------
        np.save(cache_emb, al_emb)
        LOG.info("Cached embeddings → %s", cache_emb)

        import csv
        with open(cache_meta, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["compound_id", "zinc_id", "score", "al_round"])
            for cid, zid, sc, rnd in zip(compound_ids, zinc_ids, scores, al_rounds):
                writer.writerow([cid, zid, sc, rnd])
        LOG.info("Cached metadata   → %s", cache_meta)

    # -----------------------------------------------------------------------
    # Load background embeddings
    # -----------------------------------------------------------------------
    bg_path = CACHE_DIR / f"emb_bg_rev_{target}.npy"
    if not bg_path.exists():
        LOG.error("Background embeddings not found: %s", bg_path)
        LOG.error("Re-run umap_diversity_study.py to regenerate.")
        sys.exit(1)

    LOG.info("Loading background embeddings: %s", bg_path)
    t0     = time.time()
    emb_bg = np.load(bg_path)
    LOG.info("  Shape %s loaded in %.1fs", emb_bg.shape, time.time() - t0)

    # -----------------------------------------------------------------------
    # Build Plotly figure
    # -----------------------------------------------------------------------
    LOG.info("Building interactive Plotly figure…")
    t0  = time.time()
    fig = make_figure(
        emb_bg, al_emb,
        compound_ids, zinc_ids, scores, al_rounds,
        target, exp_id,
        args.bg_subsample, args.seed,
    )
    LOG.info("  Figure built in %.1fs", time.time() - t0)

    # -----------------------------------------------------------------------
    # Save HTML
    # -----------------------------------------------------------------------
    LOG.info("Writing HTML: %s", out_html)
    t0 = time.time()
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    size_mb = out_html.stat().st_size / 1e6
    LOG.info("  Saved in %.1fs  (%.1f MB)", time.time() - t0, size_mb)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    al_rounds_arr = np.array(al_rounds)
    scores_arr    = np.array(scores, dtype=np.float64)
    rounds        = sorted(set(al_rounds))

    print(f"\n{'='*50}")
    print(f"  UMAP AL rounds — {target} [{exp_id}]")
    print(f"{'='*50}")
    print(f"  {'Round':>5}  {'N':>7}  {'Best':>8}  {'Mean':>8}")
    print(f"  {'-'*35}")
    for r in rounds:
        mask = al_rounds_arr == r
        n_r  = mask.sum()
        best = scores_arr[mask].min()
        mean = scores_arr[mask].mean()
        print(f"  {r:>5}  {n_r:>7,}  {best:>8.2f}  {mean:>8.2f}")
    total = len(al_rounds_arr)
    print(f"  {'Total':>5}  {total:>7,}  {scores_arr.min():>8.2f}  {scores_arr.mean():>8.2f}")
    print(f"\n  Output → {out_html}")


if __name__ == "__main__":
    main()
