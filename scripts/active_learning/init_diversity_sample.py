"""
init_diversity_sample.py — Round-0 diversity initialization.

Selects 24,000 maximally diverse compounds from the full ~300M library using
parallel MaxMin Tanimoto diversity picking across 24 worker processes.

Usage:
    python scripts/active_learning/init_diversity_sample.py \
        --fp-file data/fingerprints/compounds.h5 \
        --target trpv1_5irz \
        --batch-size 24000 \
        --threads 24 \
        --seed 42
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg2
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# DB config
# ---------------------------------------------------------------------------
DB_NAME = "analgesics"
DB_USER = "labhund"


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


# ---------------------------------------------------------------------------
# Bit-count lookup table (0-255 → popcount)
# ---------------------------------------------------------------------------
_BIT_COUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def tanimoto_maxmin(fp_array: np.ndarray, n_select: int, rng: np.random.Generator,
                    progress_path: Path = None):
    """
    MaxMin Tanimoto diversity picking.

    Parameters
    ----------
    fp_array      : (N, 512) uint8 — packed 4096-bit Morgan FPs
    n_select      : int
    rng           : numpy random Generator
    progress_path : optional Path; if set, writes "iter/total" every 5% for monitoring

    Returns
    -------
    selected_indices : (n_select,) int — row indices into fp_array
    """
    N = len(fp_array)
    n_select = min(n_select, N)

    min_dist = np.full(N, np.inf, dtype=np.float32)
    selected = []

    start_idx = int(rng.integers(0, N))
    selected.append(start_idx)
    min_dist[start_idx] = 0.0

    report_every = max(1, n_select // 20)  # write progress every 5%

    for i in range(n_select - 1):
        q = fp_array[selected[-1]]  # (512,) uint8

        # Tanimoto similarity via packed bit operations
        inter = _BIT_COUNT[fp_array & q].sum(axis=1).astype(np.float32)  # (N,)
        union = _BIT_COUNT[fp_array | q].sum(axis=1).astype(np.float32)  # (N,)
        # guard against all-zero FPs (union == 0 → similarity = 0)
        with np.errstate(invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        dist = 1.0 - sim  # Tanimoto distance

        np.minimum(min_dist, dist, out=min_dist)
        min_dist[selected[-1]] = 0.0  # already selected — exclude from argmax

        best = int(np.argmax(min_dist))
        selected.append(best)
        min_dist[best] = 0.0

        if progress_path and (i + 1) % report_every == 0:
            try:
                progress_path.write_text(f"{i + 1} {n_select}")
            except Exception:
                pass

    return np.array(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------
def worker_fn(args):
    """
    One worker handles a contiguous HDF5 row range.

    Reads a sequential slice of `subsample_size` rows (fast, no scatter I/O),
    filters to valid compound IDs via a shared mmap validity mask, then runs MaxMin.

    Returns (worker_id, selected_compound_ids, elapsed_seconds).
    """
    (
        worker_id,
        start_row,          # first HDF5 row owned by this worker
        n_rows_in_range,    # total rows in this worker's partition
        fp_file,
        subsample_size,
        select_per_worker,
        seed,
        progress_dir,
        valid_mask_path,    # path to bool numpy array; mmap-opened read-only
    ) = args

    rng = np.random.default_rng(seed + worker_id)
    progress_path = Path(progress_dir) / f"worker_{worker_id:02d}.prog"

    # Pick a random contiguous block within this worker's range
    max_offset = max(0, n_rows_in_range - subsample_size)
    block_offset = int(rng.integers(0, max_offset + 1))
    block_start = start_row + block_offset
    block_end   = min(block_start + subsample_size, start_row + n_rows_in_range)

    # Sequential read — one contiguous slice, no chunk scatter
    with h5py.File(fp_file, "r") as f:
        fp_block = f["fps"][block_start : block_end]  # (block_size, 512) uint8

    # Filter to rows with valid compound IDs (mmap — shared OS page cache)
    valid_mask = np.load(valid_mask_path, mmap_mode="r")
    local_valid = valid_mask[block_start : block_end]   # view, no copy
    valid_local_rows = np.where(local_valid)[0]         # local indices into fp_block
    fp_array = fp_block[valid_local_rows]               # in-memory selection

    if len(fp_array) == 0:
        logging.warning("Worker %02d: no valid rows in block — returning empty.", worker_id)
        return worker_id, [], 0.0

    # Safety check: drop any all-zero FP vectors.
    # The validity mask should prevent these, but a compound could have a corrupt FP
    # (all-zero = no bits set = impossible for a real molecule, and distance 1.0 to
    # everything — MaxMin would greedily select them, poisoning diversity picks).
    nonzero_mask = fp_array.any(axis=1)
    n_zero = int((~nonzero_mask).sum())
    if n_zero:
        logging.warning(
            "Worker %02d: dropping %d all-zero FP vectors (corrupt fingerprints in DB)",
            worker_id, n_zero,
        )
        valid_local_rows = valid_local_rows[nonzero_mask]
        fp_array = fp_array[nonzero_mask]
    if len(fp_array) == 0:
        logging.warning("Worker %02d: all FPs were zero — returning empty.", worker_id)
        return worker_id, [], 0.0

    # MaxMin diversity selection on the valid, non-zero subset
    t0 = time.time()
    local_indices = tanimoto_maxmin(fp_array, select_per_worker, rng, progress_path)
    elapsed = time.time() - t0

    progress_path.write_text(f"{select_per_worker} {select_per_worker}")

    # Map back to global HDF5 rows → compound IDs
    global_rows = block_start + valid_local_rows[local_indices]
    compound_ids = (global_rows + 1).tolist()   # compound_id = hdf5_row + 1
    return worker_id, compound_ids, elapsed


# ---------------------------------------------------------------------------
# Progress monitor (runs in main process while workers are active)
# ---------------------------------------------------------------------------
def _log_progress(progress_dir: Path, n_workers: int, n_select: int, t0: float,
                  t_maxmin: list):
    """Read per-worker progress files and log a single summary line.

    t_maxmin is a one-element list used as a mutable box: set to time.time()
    the first time any progress file appears, so ETA excludes FP-load time.
    """
    done = 0
    pcts = []
    for wid in range(n_workers):
        pf = progress_dir / f"worker_{wid:02d}.prog"
        if pf.exists():
            try:
                curr, total = map(int, pf.read_text().strip().split())
                pct = curr / total * 100
                pcts.append(pct)
                if curr >= total:
                    done += 1
            except Exception:
                pass
    elapsed_min = (time.time() - t0) / 60
    if pcts:
        if t_maxmin[0] is None:
            t_maxmin[0] = time.time()  # first checkpoint — MaxMin is running
        maxmin_elapsed = (time.time() - t_maxmin[0]) / 60
        avg_pct = sum(pcts) / len(pcts)
        eta_min = (maxmin_elapsed / avg_pct * 100 - maxmin_elapsed) if avg_pct > 0 else float("inf")
        logging.info(
            "Init progress | elapsed=%.1fmin | done=%d/%d workers | "
            "avg=%.0f%% [%.0f%%–%.0f%%] | ETA≈%.0fmin",
            elapsed_min, done, n_workers,
            avg_pct, min(pcts), max(pcts), eta_min,
        )
    else:
        logging.info("Init progress | elapsed=%.1fmin | workers loading FPs...", elapsed_min)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Round-0 diversity init")
    parser.add_argument("--fp-file", default="data/fingerprints/compounds.h5")
    parser.add_argument("--target", default="trpv1_5irz")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--batch-size", type=int, default=24000)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--subsample-per-worker", type=int, default=1_000_000,
                        help="IDs to subsample per worker before MaxMin")
    parser.add_argument("--select-per-worker", type=int, default=None,
                        help="IDs to select per worker (default: batch_size // threads)")
    parser.add_argument("--seed", type=int, default=42)
    cfg = parser.parse_args(args)

    # Configure logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "init_diversity.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    fp_file = Path(cfg.fp_file)
    if not fp_file.exists():
        logging.error("FP file not found: %s", fp_file)
        sys.exit(1)

    n_threads = cfg.threads
    select_per_worker = cfg.select_per_worker or (cfg.batch_size // n_threads)
    total_select = select_per_worker * n_threads
    logging.info(
        "Diversity init: %d workers × %d selected = %d total",
        n_threads, select_per_worker, total_select,
    )

    # Check if round 0 already exists for this experiment
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM al_batches "
                "WHERE round=0 AND target=%s AND experiment_id=%s",
                (cfg.target, cfg.experiment_id),
            )
            existing = cur.fetchone()[0]
        if existing > 0:
            logging.info(
                "Round 0 already has %d entries for target=%s experiment_id=%s — skipping.",
                existing, cfg.target, cfg.experiment_id,
            )
            return
    finally:
        conn.close()

    # Build validity mask: bool array of shape (N_hdf5_rows,) marking which rows
    # have a valid compound_id in the DB.  Written to /tmp as a numpy file so
    # workers can mmap it read-only (OS shares one copy across all 24 processes).
    with h5py.File(fp_file, "r") as f:
        N_rows = f["fps"].shape[0]
    logging.info("HDF5 rows: %d", N_rows)

    logging.info("Fetching valid compound IDs to build validity mask (~45s)...")
    conn = get_db_conn()
    try:
        with conn.cursor("id_cursor") as cur:
            cur.itersize = 500_000
            cur.execute("SELECT id FROM compounds ORDER BY id")
            all_ids = np.array([row[0] for row in cur], dtype=np.int64)
    finally:
        conn.close()

    valid_mask = np.zeros(N_rows, dtype=np.bool_)
    valid_hdf5_rows = all_ids - 1                    # compound_id → 0-based row
    in_range = valid_hdf5_rows[valid_hdf5_rows < N_rows]
    valid_mask[in_range] = True
    n_valid = int(valid_mask.sum())
    logging.info(
        "Valid HDF5 rows: %d / %d (%.2f%% fill)",
        n_valid, N_rows, n_valid / N_rows * 100,
    )

    valid_mask_path = "/tmp/al_init_valid_mask.npy"
    np.save(valid_mask_path, valid_mask)
    del valid_mask, all_ids, in_range   # free ~2.7 GB before forking

    rows_per_worker = N_rows // n_threads
    logging.info(
        "Partition: %d workers × %d rows/worker, subsample %d → MaxMin %d",
        n_threads, rows_per_worker, cfg.subsample_per_worker, select_per_worker,
    )

    progress_dir = Path("logs/init_progress")
    progress_dir.mkdir(parents=True, exist_ok=True)
    for f in progress_dir.glob("worker_*.prog"):  # clear stale files
        f.unlink()

    worker_args = [
        (
            i,
            i * rows_per_worker,                                    # start_row
            rows_per_worker if i < n_threads - 1                    # n_rows_in_range
                else N_rows - i * rows_per_worker,
            str(fp_file),
            cfg.subsample_per_worker,
            select_per_worker,
            cfg.seed,
            str(progress_dir),
            valid_mask_path,
        )
        for i in range(n_threads)
    ]

    t0 = time.time()
    t_maxmin = [None]  # set when first progress file appears (excludes FP load time)
    with Pool(processes=n_threads) as pool:
        async_result = pool.map_async(worker_fn, worker_args)
        while not async_result.ready():
            _log_progress(progress_dir, n_threads, select_per_worker, t0, t_maxmin)
            async_result.wait(timeout=30)
        raw_results = async_result.get()
    elapsed = time.time() - t0

    _log_progress(progress_dir, n_threads, select_per_worker, t0, t_maxmin)  # final snapshot
    for wid, _, w_elapsed in raw_results:
        logging.info("Worker %02d finished in %.1fs", wid, w_elapsed)
    logging.info("Parallel diversity picking completed in %.1f s", elapsed)

    selected_ids = [cid for _, batch, _ in raw_results for cid in batch]
    logging.info("Total selected: %d compounds", len(selected_ids))

    # Bulk insert into al_batches
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            from psycopg2.extras import execute_values
            rows = [
                (0, int(cid), cfg.target, cfg.experiment_id, "diversity_init")
                for cid in selected_ids
            ]
            execute_values(
                cur,
                "INSERT INTO al_batches (round, compound_id, target, experiment_id, source) "
                "VALUES %s",
                rows,
            )
        conn.commit()
        logging.info(
            "Inserted %d rows into al_batches (round=0, experiment_id=%s)",
            len(rows), cfg.experiment_id,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
