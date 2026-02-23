#!/usr/bin/env python3
"""
Build Morgan fingerprint HDF5 from the compounds table.

Index mapping: compounds.id - 1 → HDF5 row (0-indexed).
Output dataset 'fps': shape (max_id, 512), dtype uint8.
Each row is a 4096-bit radius-4 Morgan FP packed into 512 bytes.

Usage:
    source env_db.sh
    conda activate chem
    python scripts/fingerprints/build_fingerprints.py \\
        --out data/fingerprints/compounds.h5 \\
        --workers 24 \\
        --chunk 10000
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg2
from rdkit import RDLogger
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles

RDLogger.DisableLog("rdApp.*")

DB_NAME = "analgesics"
DB_USER = "labhund"
N_BITS = 4096
RADIUS = 4
PACKED_SIZE = N_BITS // 8  # 512 bytes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("build_fingerprints.log"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker globals (set once per process via initializer)
# ---------------------------------------------------------------------------
_worker_queue = None
_worker_conn = None
_worker_chunk = None


def init_worker(queue, pghost, chunk_size):
    """Runs once per worker process to set up DB connection and shared state."""
    global _worker_queue, _worker_conn, _worker_chunk
    _worker_queue = queue
    _worker_chunk = chunk_size
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=pghost)
    conn.set_session(readonly=True)
    _worker_conn = conn


def process_range(args):
    """
    Worker task: stream (id, smiles) from [lo, hi], compute FPs, enqueue results.
    Returns (n_success, n_null, n_rdkit_fail, lo, hi).
    """
    lo, hi = args
    pid = os.getpid()
    n_success = 0
    n_null = 0
    n_fail = 0

    # Named cursor → server-side, streams rows without loading all into memory
    cur = _worker_conn.cursor(name=f"fp_cur_{pid}_{lo}")
    cur.execute(
        "SELECT id, smiles_protonated FROM compounds"
        " WHERE id BETWEEN %s AND %s ORDER BY id",
        (lo, hi),
    )

    while True:
        rows = cur.fetchmany(_worker_chunk)
        if not rows:
            break

        ids = []
        packed_fps = []

        for compound_id, smiles in rows:
            if smiles is None:
                n_null += 1
                continue

            mol = MolFromSmiles(smiles)
            if mol is None:
                n_fail += 1
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=RADIUS, nBits=N_BITS)
            bit_arr = np.zeros(N_BITS, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, bit_arr)  # fills in-place
            packed = np.packbits(bit_arr)  # 4096 bits → 512 bytes

            ids.append(compound_id)
            packed_fps.append(packed)
            n_success += 1

        if ids:
            id_arr = np.array(ids, dtype=np.int32)
            fp_arr = np.stack(packed_fps, axis=0)  # shape (N, 512)
            _worker_queue.put((id_arr, fp_arr))

    cur.close()
    return n_success, n_null, n_fail, lo, hi


# ---------------------------------------------------------------------------
# Writer process
# ---------------------------------------------------------------------------

def writer_proc(queue, out_path, log_interval=1_000_000):
    """
    Single writer process: drains queue and writes fingerprints to HDF5.
    Receives (id_array, fp_array) tuples; sentinel is None.
    """
    written = 0
    t0 = time.time()
    prev_milestone = 0

    with h5py.File(out_path, "a") as f:
        ds = f["fps"]
        while True:
            item = queue.get()
            if item is None:
                break

            ids, fps = item  # ids: (N,) int32; fps: (N, 512) uint8
            rows = ids.astype(np.intp) - 1  # 0-indexed HDF5 rows (sorted)

            # Fancy-index write (h5py requires sorted indices for efficiency)
            ds[rows] = fps

            written += len(ids)
            milestone = written // log_interval
            if milestone > prev_milestone:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                log.info(
                    f"[writer] {written:>12,} rows written | {rate:>10,.0f} rows/sec"
                )
                prev_milestone = milestone

    elapsed = time.time() - t0
    rate = written / elapsed if elapsed > 0 else 0
    log.info(
        f"[writer] Complete: {written:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/sec)"
    )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def preflight(pghost):
    """Query DB for compound count and max id."""
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=pghost)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), MAX(id) FROM compounds")
        n_compounds, max_id = cur.fetchone()
    conn.close()
    return int(n_compounds), int(max_id)


def preallocate_hdf5(out_path, max_id):
    """Create and pre-allocate the HDF5 dataset (filled with zeros)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_gb = max_id * PACKED_SIZE / 1e9
    log.info(f"Pre-allocating {out_path} — shape=({max_id:,}, {PACKED_SIZE}), ~{size_gb:.1f} GB")

    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "fps",
            shape=(max_id, PACKED_SIZE),
            dtype="uint8",
            chunks=(10_000, PACKED_SIZE),
            fillvalue=0,
        )
        ds.attrs["radius"] = RADIUS
        ds.attrs["nbits"] = N_BITS
        ds.attrs["index_mapping"] = "compound_id - 1"

    log.info("Pre-allocation complete.")


def build_id_ranges(max_id, n_workers):
    """Divide [1, max_id] into n_workers contiguous ranges."""
    chunk = max_id // n_workers
    ranges = [
        (i * chunk + 1, (i + 1) * chunk if i < n_workers - 1 else max_id)
        for i in range(n_workers)
    ]
    return ranges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Morgan FP HDF5 from compounds table"
    )
    parser.add_argument(
        "--out",
        default="data/fingerprints/compounds.h5",
        help="Output HDF5 path (relative to PROJ_ROOT or absolute)",
    )
    parser.add_argument("--workers", type=int, default=24, help="DB reader processes")
    parser.add_argument("--chunk", type=int, default=10_000, help="fetchmany batch size")
    args = parser.parse_args()

    pghost = os.getenv("PGHOST")
    if not pghost:
        sys.exit("PGHOST not set. Run: source env_db.sh")

    # Resolve output path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        proj_root = os.getenv("PROJ_ROOT", "/data2/lipin_lab/markus/analgesics")
        out_path = Path(proj_root) / out_path

    # --- Phase 0: preflight ---
    log.info("Phase 0: preflight query (COUNT, MAX id)...")
    n_compounds, max_id = preflight(pghost)
    log.info(f"  n_compounds = {n_compounds:,}")
    log.info(f"  max_id      = {max_id:,}")
    log.info(f"  id gaps     = {max_id - n_compounds:,}")

    # --- Phase 1: pre-allocate HDF5 ---
    if out_path.exists():
        log.info(f"HDF5 already exists — resuming writes: {out_path}")
    else:
        preallocate_hdf5(out_path, max_id)

    # --- Build ID ranges ---
    ranges = build_id_ranges(max_id, args.workers)
    log.info(
        f"Phase 2-3: {args.workers} workers, ranges {ranges[0]} … {ranges[-1]}"
    )

    # Queue size: 48 × ~5 MB ≈ 240 MB max in-flight
    queue = mp.Queue(maxsize=48)

    # --- Phase 2: start writer process ---
    writer = mp.Process(
        target=writer_proc,
        args=(queue, str(out_path)),
        name="hdf5-writer",
        daemon=False,
    )
    writer.start()

    # --- Phase 3: launch worker pool ---
    t_start = time.time()
    pool_error = None
    try:
        with mp.Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(queue, pghost, args.chunk),
        ) as pool:
            results = pool.map(process_range, ranges)
    except Exception as exc:
        pool_error = exc
        log.error(f"Worker pool error: {exc}")
        results = []
    finally:
        # Always send sentinel so writer can exit cleanly
        queue.put(None)

    writer.join()

    if pool_error:
        sys.exit(f"Aborted due to pool error: {pool_error}")

    # --- Summary ---
    total_success = sum(r[0] for r in results)
    total_null = sum(r[1] for r in results)
    total_fail = sum(r[2] for r in results)
    elapsed = time.time() - t_start

    log.info("=" * 60)
    log.info(f"Finished in {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    log.info(f"  Success FPs :  {total_success:>12,}")
    log.info(f"  Null SMILES :  {total_null:>12,}")
    log.info(f"  RDKit fails :  {total_fail:>12,}")
    log.info(f"  Throughput  :  {total_success / elapsed:>12,.0f} FP/sec")
    log.info(f"  Output      :  {out_path}")
    log.info("=" * 60)

    # Quick shape check
    with h5py.File(out_path, "r") as f:
        shape = f["fps"].shape
    log.info(f"HDF5 dataset shape: {shape}")


if __name__ == "__main__":
    main()
