#!/usr/bin/env python3
"""
Diversity Study — PCA + UMAP Visualization of Round-0 Initial Batch

Two complementary views of chemical space coverage:
  Forward:  UMAP fit on 24K diversity picks, background projected onto it.
  Reverse:  UMAP fit on background, 24K diversity picks projected onto it.

For large backgrounds (>500K) PCA is fitted on a subsample and the rest
transformed in chunks to stay within RAM limits.

Usage:
    python analysis/diversity_study/umap_diversity_study.py \
        --fp-file data/fingerprints/compounds.h5 \
        --target trpv1_8gfa \
        --n-background 2400000 \
        --pca-subsample 500000 \
        --fp-chunk 200000 \
        --seed 42 \
        --pca-components 300 \
        --umap-neighbors 15 \
        --umap-min-dist 0.1
"""

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from sklearn.decomposition import PCA

SCRIPT_DIR  = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
CACHE_DIR   = SCRIPT_DIR / "cache"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    fp_file:        Path
    target:         str
    n_background:   int
    pca_subsample:  int
    fp_chunk:       int
    seed:           int
    pca_components: int
    umap_neighbors: int
    umap_min_dist:  float
    umap_epochs:    int
    force:          bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp-file",        type=Path, default=Path("data/fingerprints/compounds.h5"))
    p.add_argument("--target",         type=str,  default="trpv1_8gfa")
    p.add_argument("--n-background",   type=int,  default=100_000)
    p.add_argument("--pca-subsample",  type=int,  default=500_000,
                   help="Max background points used for PCA fitting (rest transformed in chunks)")
    p.add_argument("--fp-chunk",       type=int,  default=200_000,
                   help="FP loading chunk size for large backgrounds")
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--pca-components", type=int,  default=300,
                   help="PCA components to fit (trimmed to 95%% variance post-hoc)")
    p.add_argument("--umap-neighbors", type=int,  default=15)
    p.add_argument("--umap-min-dist",  type=float, default=0.1)
    p.add_argument("--umap-epochs",    type=int,  default=None,
                   help="UMAP optimization epochs (default: umap-learn auto, ~200 for small, ~100 for large)")
    p.add_argument("--force",          action="store_true",
                   help="Recompute even if cache files exist")
    a = p.parse_args()
    return Config(
        fp_file=a.fp_file,
        target=a.target,
        n_background=a.n_background,
        pca_subsample=a.pca_subsample,
        fp_chunk=a.fp_chunk,
        seed=a.seed,
        pca_components=a.pca_components,
        umap_neighbors=a.umap_neighbors,
        umap_min_dist=a.umap_min_dist,
        umap_epochs=a.umap_epochs,
        force=a.force,
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def db_connect():
    return psycopg2.connect(
        dbname="analgesics",
        user="labhund",
        host=os.environ.get("PGHOST", "/tmp"),
    )


def load_diversity_ids(cur, target: str) -> np.ndarray:
    cur.execute(
        "SELECT compound_id FROM al_batches WHERE round=0 AND target=%s",
        (target,),
    )
    ids = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
    logging.info("Loaded %d diversity compound IDs from DB", len(ids))
    return ids


def sample_background_ids(cur, n: int) -> np.ndarray:
    """TABLESAMPLE BERNOULLI — ~33ms regardless of table size."""
    frac = (n / 300_000_000) * 100 * 1.5   # 50% overshoot to ensure enough rows
    frac = max(frac, 0.001)
    cur.execute(
        "SELECT id FROM compounds TABLESAMPLE BERNOULLI(%s) LIMIT %s",
        (frac, n),
    )
    ids = np.array([r[0] for r in cur.fetchall()], dtype=np.int64)
    logging.info("Sampled %d background compound IDs", len(ids))
    return ids


# ---------------------------------------------------------------------------
# FP loading
# ---------------------------------------------------------------------------

def load_fps(fp_file: Path, compound_ids: np.ndarray) -> np.ndarray:
    """
    Load Morgan FPs for the given compound IDs.
    Rows sorted for chunk-cache efficiency, restored to original order on return.
    HDF5 contract: compound id k → row k-1.
    """
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
        for s, e in runs:
            fps_list.append(dset[s:e])

    if not fps_list:
        return np.zeros((0, 4096), dtype=np.float32)

    fp_raw = np.concatenate(fps_list, axis=0)
    fps    = np.unpackbits(fp_raw, axis=1).astype(np.float32)
    return fps[np.argsort(sort_idx)]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def cache_path(name: str, target: str) -> Path:
    return CACHE_DIR / f"{name}_{target}.npy"

def model_path(name: str, target: str) -> Path:
    return CACHE_DIR / f"{name}_{target}.joblib"

def cache_exists(names: list, target: str) -> bool:
    array_ok = all(cache_path(n, target).exists() for n in names)
    model_ok  = model_path("pca", target).exists() and model_path("umap_rev", target).exists()
    return array_ok and model_ok

def save_cache(arrays: dict, target: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(cache_path(name, target), arr)
        logging.info("Cached %s → %s", name, cache_path(name, target))

def save_models(pca, n95: int, umap_rev, target: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "n95": n95}, model_path("pca", target))
    logging.info("Saved PCA model → %s", model_path("pca", target))
    joblib.dump(umap_rev, model_path("umap_rev", target))
    logging.info("Saved reverse UMAP reducer → %s", model_path("umap_rev", target))

def load_cache(names: list, target: str) -> dict:
    return {n: np.load(cache_path(n, target)) for n in names}


# ---------------------------------------------------------------------------
# PCA — subsample fit + chunked transform
# ---------------------------------------------------------------------------

def run_pca(fps_24k: np.ndarray, ids_bg: np.ndarray, fp_file: Path, cfg: Config):
    """
    Fit PCA on 24K picks + up to pca_subsample background points.
    Transform remaining background in fp_chunk-sized pieces to stay within RAM.

    Returns: pca_24k, pca_bg (both in original bg_ids order), n95, pca_model
    """
    rng = np.random.default_rng(cfg.seed)
    n_bg = len(ids_bg)
    n_sub = min(cfg.pca_subsample, n_bg)

    # Split bg indices into subsample (for fit) and rest (chunked transform)
    perm     = rng.permutation(n_bg)
    sub_pos  = np.sort(perm[:n_sub])   # positions into ids_bg, sorted for HDF5
    rest_pos = np.sort(perm[n_sub:])

    logging.info("PCA: loading %d background FPs for fitting…", n_sub)
    t0 = time.time()
    fps_bg_sub = load_fps(fp_file, ids_bg[sub_pos])
    logging.info("  loaded in %.1f s", time.time() - t0)

    combined = np.vstack([fps_24k, fps_bg_sub])
    logging.info("PCA: fitting on combined (%d, %d) — randomized SVD, %d components…",
                 *combined.shape, cfg.pca_components)
    t0 = time.time()
    pca = PCA(n_components=cfg.pca_components, svd_solver="randomized", random_state=cfg.seed)
    combined_pca = pca.fit_transform(combined)
    logging.info("PCA fit in %.1f s", time.time() - t0)
    del combined

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n95 = min(int(np.searchsorted(cumvar, 0.95)) + 1, cfg.pca_components)
    logging.info("PCA: %d components → 95%% variance (fitted %d, cumvar[-1]=%.3f)",
                 n95, cfg.pca_components, cumvar[-1])

    pca_24k    = combined_pca[:len(fps_24k), :n95]
    pca_bg_sub = combined_pca[len(fps_24k):, :n95]
    del combined_pca, fps_bg_sub

    # Transform remaining background in chunks
    pca_bg_rest_chunks = []
    n_rest = len(rest_pos)
    if n_rest > 0:
        logging.info("PCA: transforming remaining %d background points in chunks of %d…",
                     n_rest, cfg.fp_chunk)
        t0 = time.time()
        for start in range(0, n_rest, cfg.fp_chunk):
            chunk_pos = rest_pos[start:start + cfg.fp_chunk]
            fps_chunk = load_fps(fp_file, ids_bg[chunk_pos])
            pca_bg_rest_chunks.append(pca.transform(fps_chunk)[:, :n95])
            del fps_chunk
            logging.info("  transformed %d / %d", min(start + cfg.fp_chunk, n_rest), n_rest)
        logging.info("Chunked PCA transform done in %.1f s", time.time() - t0)

    # Reconstruct pca_bg in the original ids_bg order
    pca_bg = np.empty((n_bg, n95), dtype=np.float32)
    pca_bg[sub_pos] = pca_bg_sub
    if pca_bg_rest_chunks:
        pca_bg[rest_pos] = np.vstack(pca_bg_rest_chunks)

    return pca_24k, pca_bg, n95, pca


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def make_umap_reducer(cfg: Config):
    import umap as umap_lib
    kwargs = dict(
        n_components=2,
        n_neighbors=cfg.umap_neighbors,
        min_dist=cfg.umap_min_dist,
        random_state=cfg.seed,
        low_memory=True,
        verbose=True,
    )
    if cfg.umap_epochs is not None:
        kwargs["n_epochs"] = cfg.umap_epochs
    return umap_lib.UMAP(**kwargs)


def run_umap(data_fit: np.ndarray, data_transform: np.ndarray, cfg: Config, label: str):
    """Fit UMAP on data_fit, transform data_transform. Returns (emb_fit, emb_proj, reducer)."""
    reducer = make_umap_reducer(cfg)

    logging.info("UMAP (%s): fitting on %d points (shape %s)…",
                 label, len(data_fit), data_fit.shape)
    t0 = time.time()
    emb_fit = reducer.fit_transform(data_fit)
    logging.info("UMAP (%s) fit in %.1f s", label, time.time() - t0)

    logging.info("UMAP (%s): transforming %d points…", label, len(data_transform))
    t1 = time.time()
    emb_proj = reducer.transform(data_transform)
    logging.info("UMAP (%s) transform in %.1f s", label, time.time() - t1)

    return emb_fit, emb_proj, reducer


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(emb_24k, emb_bg_on_24k, emb_bg, emb_24k_on_bg, cfg: Config) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    color_bg   = "#aaaaaa"
    color_pick = "#e05c2a"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.scatter(emb_bg_on_24k[:, 0], emb_bg_on_24k[:, 1],
               s=1, alpha=0.1, color=color_bg, rasterized=True, label=f"{len(emb_bg_on_24k):,} background (projected)")
    ax.scatter(emb_24k[:, 0], emb_24k[:, 1],
               s=4, alpha=0.6, color=color_pick, rasterized=True, label="24K diversity picks")
    ax.set_title("UMAP fit on 24K diversity picks\nBackground projected onto diversity space",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.scatter(emb_bg[:, 0], emb_bg[:, 1],
               s=1, alpha=0.05, color=color_bg, rasterized=True, label=f"{len(emb_bg):,} background")
    ax.scatter(emb_24k_on_bg[:, 0], emb_24k_on_bg[:, 1],
               s=4, alpha=0.6, color=color_pick, rasterized=True, label="24K diversity picks (projected)")
    ax.set_title("UMAP fit on background sample\n24K diversity picks projected onto library space",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.tick_params(labelsize=7)

    annotation = (
        f"n_diversity={len(emb_24k):,}  n_background={len(emb_bg):,}  "
        f"seed={cfg.seed}  neighbors={cfg.umap_neighbors}  min_dist={cfg.umap_min_dist}"
    )
    fig.text(0.5, 0.01, annotation, ha="center", va="bottom", fontsize=8, color="#555555")
    plt.suptitle(f"Chemical Space Coverage — {cfg.target} Round-0 Diversity Picks",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = FIGURES_DIR / f"umap_diversity_{cfg.target}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    logging.info("Figure saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    cfg = parse_args()

    CACHE_KEYS = ["emb_24k_fwd", "emb_bg_on_24k", "emb_bg_rev", "emb_24k_on_bg"]

    if not cfg.force and cache_exists(CACHE_KEYS, cfg.target):
        logging.info("Cache found — loading embeddings (use --force to recompute)")
        embs          = load_cache(CACHE_KEYS, cfg.target)
        emb_24k       = embs["emb_24k_fwd"]
        emb_bg_on_24k = embs["emb_bg_on_24k"]
        emb_bg        = embs["emb_bg_rev"]
        emb_24k_on_bg = embs["emb_24k_on_bg"]
        make_figure(emb_24k, emb_bg_on_24k, emb_bg, emb_24k_on_bg, cfg)
        return

    # ------------------------------------------------------------------
    # Step 1 — Diversity FPs
    # ------------------------------------------------------------------
    logging.info("=== Step 1: Load diversity compound IDs ===")
    conn = db_connect()
    cur  = conn.cursor()

    ids_24k = load_diversity_ids(cur, cfg.target)
    if len(ids_24k) == 0:
        logging.error("No round-0 diversity compounds for target '%s'.", cfg.target)
        sys.exit(1)

    logging.info("Loading FPs for %d diversity compounds…", len(ids_24k))
    t0 = time.time()
    fps_24k = load_fps(cfg.fp_file, ids_24k)
    logging.info("Loaded diversity FPs in %.1f s — shape %s", time.time() - t0, fps_24k.shape)

    # ------------------------------------------------------------------
    # Step 2 — Background IDs
    # ------------------------------------------------------------------
    logging.info("=== Step 2: Sample %d background compounds ===", cfg.n_background)
    ids_bg = sample_background_ids(cur, cfg.n_background)
    cur.close()
    conn.close()

    # ------------------------------------------------------------------
    # Step 3 — PCA (subsample fit + chunked transform)
    # ------------------------------------------------------------------
    logging.info("=== Step 3: PCA ===")
    pca_24k, pca_bg, n95, pca_model = run_pca(fps_24k, ids_bg, cfg.fp_file, cfg)
    logging.info("PCA output: picks %s  background %s", pca_24k.shape, pca_bg.shape)
    del fps_24k

    # ------------------------------------------------------------------
    # Step 4 — Forward UMAP (fit on 24K, transform background)
    # ------------------------------------------------------------------
    logging.info("=== Step 4: Forward UMAP (fit on 24K) ===")
    emb_24k, emb_bg_on_24k, _ = run_umap(pca_24k, pca_bg, cfg, label="forward")

    # ------------------------------------------------------------------
    # Step 5 — Reverse UMAP (fit on background, transform 24K)
    # ------------------------------------------------------------------
    logging.info("=== Step 5: Reverse UMAP (fit on %d background points) ===", len(pca_bg))
    emb_bg, emb_24k_on_bg, umap_rev = run_umap(pca_bg, pca_24k, cfg, label="reverse")
    del pca_24k, pca_bg

    # ------------------------------------------------------------------
    # Step 6 — Cache
    # ------------------------------------------------------------------
    logging.info("=== Step 6: Save cache ===")
    save_cache(
        {
            "emb_24k_fwd":    emb_24k,
            "emb_bg_on_24k":  emb_bg_on_24k,
            "emb_bg_rev":     emb_bg,
            "emb_24k_on_bg":  emb_24k_on_bg,
            "bg_ids":         ids_bg,
        },
        cfg.target,
    )
    save_models(pca_model, n95, umap_rev, cfg.target)

    # ------------------------------------------------------------------
    # Step 7 — Figure
    # ------------------------------------------------------------------
    logging.info("=== Step 7: Generate figure ===")
    make_figure(emb_24k, emb_bg_on_24k, emb_bg, emb_24k_on_bg, cfg)

    logging.info("Done.")


if __name__ == "__main__":
    main()
