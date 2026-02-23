"""
dock_batch.py — Ligand PDBQT preparation + AutoDock-GPU docking.

Three-stage producer/consumer pipeline:
  Stage 1: 18 prep PROCESSES (ProcessPoolExecutor, CPU-bound)
  Stage 2: 2 batch_dock_worker THREADS (AutoDock-GPU --filelist, GPU-bound)
  Stage 3: 1 writer THREAD (DB inserts + progress CSV/JSON)

AutoDock-GPU --filelist mode docks N ligands in one process call,
loading grid maps once — eliminates per-compound CUDA setup overhead.

Usage:
    python scripts/active_learning/dock_batch.py \
        --target trpv1_8gfa \
        --experiment-id maxmin_init \
        --round 0 \
        --maps targets/trpv1/grids/trpv1_8gfa.fld \
        --work-dir work/docking \
        --prep-threads 18 \
        --dock-workers 2 \
        --dock-batch-size 100 \
        --fail-threshold 0.0
"""

import argparse
import bisect
import csv
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import psycopg2
from psycopg2.extras import execute_values
from rdkit import Chem
from rdkit.Chem import AllChem

# Meeko imports
try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError:
    print(
        "ERROR: meeko not installed. Run: mamba install -n chem meeko", file=sys.stderr
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# DB config
# ---------------------------------------------------------------------------
DB_NAME = "analgesics"
DB_USER = "labhund"

_SENTINEL = object()


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


# ---------------------------------------------------------------------------
# MPS check
# ---------------------------------------------------------------------------
def check_mps_running():
    """Verify NVIDIA MPS is active. Raises RuntimeError if not."""
    mps_pipe = Path("/tmp/nvidia-mps")
    if not mps_pipe.exists():
        raise RuntimeError(
            "NVIDIA MPS control daemon is not running.\n"
            "Start it with: nvidia-cuda-mps-control -d\n"
            "Then re-run this script."
        )
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found — is the NVIDIA driver installed?")
    logging.info("NVIDIA MPS check passed.")


# ---------------------------------------------------------------------------
# Stage 1: Ligand preparation (meeko)
# ---------------------------------------------------------------------------
def prep_worker(args):
    """
    Prepare a single ligand PDBQT via RDKit + meeko.
    Returns (compound_id, pdbqt_path, failed).
    pdbqt_path is None and failed=True on any failure.
    """
    compound_id, smiles, round_dir = args
    out_pdbqt = round_dir / f"{compound_id}.pdbqt"

    def _fail(reason):
        logging.warning("PREP FAIL cmpd=%d: %s", compound_id, reason)
        return (compound_id, None, True)

    # Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _fail("invalid SMILES")

    mol = Chem.AddHs(mol)

    # Embed 3D coordinates
    ps = AllChem.ETKDGv3()
    rc = AllChem.EmbedMolecule(mol, ps)
    if rc == -1:
        # Retry with random coordinates
        ps.useRandomCoords = True
        rc = AllChem.EmbedMolecule(mol, ps)
    if rc == -1:
        return _fail("3D embedding failed")

    # MMFF geometry optimization (non-fatal)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    # Meeko preparation
    # meeko 0.5.0: prepare() returns a list of RDKitMoleculeSetup objects;
    # write_string takes the setup, not the MoleculePreparation instance.
    try:
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        if not mol_setups:
            return _fail("meeko: prepare() returned no setups")
        pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return _fail(f"meeko write_string: {err_msg}")
    except Exception as e:
        return _fail(f"meeko: {e}")

    out_pdbqt.write_text(pdbqt_str)
    return (compound_id, out_pdbqt, False)


# ---------------------------------------------------------------------------
# Stage 2: AutoDock-GPU docking (--filelist batch mode)
# ---------------------------------------------------------------------------
def parse_autodock_xml(xml_path: Path, fail_threshold: float):
    """
    Parse AutoDock-GPU XML output.
    Returns best_energy (float) or None if parsing fails / score >= fail_threshold.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # AutoDock-GPU XML: <autodock_gpu><runs><run><energies><best_energy>...
        energies = root.findall(".//best_energy")
        if not energies:
            # Try alternate path
            energies = root.findall(".//free_NRG_binding")
        if not energies:
            logging.warning("No energy element in XML: %s", xml_path)
            return None
        best = min(float(e.text) for e in energies)
        if best >= fail_threshold:
            return None  # failed dock by threshold
        return best
    except Exception as e:
        logging.warning("XML parse error %s: %s", xml_path, e)
        return None


def batch_dock_worker(
    batch_queue,
    result_queue,
    grid_fld,
    work_dir,
    exp_id,
    round_num,
    nrun,
    fail_threshold,
    autodock_bin,
):
    """
    Consumes (batch_id, compound_list) from batch_queue where
    compound_list = [(compound_id, pdbqt_path), ...].

    AutoDock-GPU --filelist format (v1.6):
      Line 1:   grid .fld file
      Lines 2+: one .pdbqt file per compound
    Output XMLs are named {resnam}_{1}.xml, {resnam}_{2}.xml, ... (1-indexed).

    Pushes (compound_id, score_or_None) to result_queue for each compound.
    """
    while True:
        item = batch_queue.get()
        if item is _SENTINEL:
            break

        batch_id, compound_list = item
        n = len(compound_list)

        # Batch output prefix: {work_dir}/{exp_id}/round{N}/batch{id}
        # AutoDock-GPU writes: {batch_resnam}_1.xml, {batch_resnam}_2.xml, ...
        batch_resnam = (
            work_dir / exp_id / f"round{round_num}" / f"batch{batch_id}"
        ).resolve()

        # Write filelist: grid fld first, then one pdbqt per line
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix=f"filelist_{batch_id}_", delete=False
        ) as fh:
            filelist_path = Path(fh.name)
            fh.write(f"{grid_fld.resolve()}\n")
            for compound_id, pdbqt_path in compound_list:
                fh.write(f"{pdbqt_path.resolve()}\n")

        # Generous timeout: allow 10s per compound, minimum 300s
        timeout = max(300, n * 10)

        cmd = [
            str(autodock_bin),
            "--filelist",
            str(filelist_path),
            "--resnam",
            str(batch_resnam),
            "--nrun",
            str(nrun),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            batch_ok = proc.returncode == 0
            if not batch_ok:
                logging.warning(
                    "AutoDock-GPU non-zero exit batch=%d (%d cmpds): %s",
                    batch_id,
                    n,
                    proc.stderr[-500:],
                )
        except subprocess.TimeoutExpired:
            logging.warning(
                "AutoDock-GPU timeout batch=%d (%d cmpds, %ds)", batch_id, n, timeout
            )
            batch_ok = False
        except Exception as e:
            logging.warning("AutoDock-GPU error batch=%d: %s", batch_id, e)
            batch_ok = False
        finally:
            # Always clean up filelist
            try:
                filelist_path.unlink()
            except Exception:
                pass

        # Parse each compound's XML (1-indexed: batch0_1.xml, batch0_2.xml, ...)
        for idx, (compound_id, _) in enumerate(compound_list, start=1):
            if not batch_ok:
                result_queue.put((compound_id, None))
                continue
            xml_path = Path(f"{batch_resnam}_{idx}.xml")
            score = parse_autodock_xml(xml_path, fail_threshold)
            result_queue.put((compound_id, score))

        logging.info("Batch %d: %d compounds docked (ok=%s)", batch_id, n, batch_ok)


# ---------------------------------------------------------------------------
# Stage 3: DB writer + progress tracking
# ---------------------------------------------------------------------------
def writer_thread(
    result_queue,
    target,
    experiment_id,
    round_num,
    total_compounds,
    progress_csv_path,
    progress_json_path,
    reporting_interval,
):
    """
    Drains result_queue, bulk-inserts into docking_scores, and writes
    per-round progress CSV/JSON every `reporting_interval` completed docks.

    Progress CSV columns:
        docked, valid, failed, best1, best10, best100, best1000, elapsed_s, timestamp
    """
    conn = get_db_conn()
    buffer = []
    written = 0

    # In-memory sorted list of valid scores for best-N tracking
    valid_scores_sorted = []  # ascending (most negative = best binder = index 0)
    n_valid = 0
    n_failed = 0
    t0 = time.time()

    # Create CSV fresh (truncate any previous run's file)
    progress_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_csv_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            [
                "docked",
                "valid",
                "failed",
                "best1",
                "best10",
                "best100",
                "best1000",
                "elapsed_s",
                "timestamp",
            ]
        )

    def _best_n(n):
        return (
            f"{valid_scores_sorted[n - 1]:.4f}" if len(valid_scores_sorted) >= n else ""
        )

    def _write_progress():
        elapsed = int(time.time() - t0)
        row = [
            written,
            n_valid,
            n_failed,
            _best_n(1),
            _best_n(10),
            _best_n(100),
            _best_n(1000),
            elapsed,
            datetime.now().isoformat(timespec="seconds"),
        ]
        with open(progress_csv_path, "a", newline="") as fh:
            csv.writer(fh).writerow(row)

        snap = {
            "total": total_compounds,
            "docked": written,
            "valid": n_valid,
            "failed": n_failed,
            "best1": valid_scores_sorted[0] if n_valid >= 1 else None,
            "best10": valid_scores_sorted[9] if n_valid >= 10 else None,
            "best100": valid_scores_sorted[99] if n_valid >= 100 else None,
            "best1000": valid_scores_sorted[999] if n_valid >= 1000 else None,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        progress_json_path.write_text(json.dumps(snap, indent=2))

    try:
        while True:
            item = result_queue.get()
            if item is _SENTINEL:
                break

            compound_id, score = item
            buffer.append((compound_id, target, score, round_num, experiment_id))

            written += 1
            if score is not None:
                n_valid += 1
                bisect.insort(valid_scores_sorted, score)
            else:
                n_failed += 1

            if len(buffer) >= 1:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        "INSERT INTO docking_scores "
                        "(compound_id, target, score, al_round, experiment_id) "
                        "VALUES %s",
                        buffer,
                    )
                conn.commit()
                logging.info("Writer: %d / %d docked", written, total_compounds)
                buffer.clear()

            if written % reporting_interval == 0:
                _write_progress()

        # Flush remainder
        if buffer:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO docking_scores "
                    "(compound_id, target, score, al_round, experiment_id) "
                    "VALUES %s",
                    buffer,
                )
            conn.commit()

        # Final progress snapshot
        _write_progress()
        logging.info("Writer finished: %d total records inserted", written)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Dock batch of compounds")
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--maps", required=True, help="AutoDock-GPU .fld grid file")
    parser.add_argument("--work-dir", default="work/docking")
    parser.add_argument("--prep-threads", type=int, default=18)
    parser.add_argument(
        "--dock-workers",
        type=int,
        default=2,
        help="Number of concurrent AutoDock-GPU --filelist workers",
    )
    parser.add_argument(
        "--dock-batch-size",
        type=int,
        default=100,
        help="Ligands per --filelist batch submission",
    )
    parser.add_argument("--fail-threshold", type=float, default=0.0)
    parser.add_argument("--nrun", type=int, default=20)
    parser.add_argument("--autodock-bin", default="bin/autodock_gpu")
    parser.add_argument(
        "--reporting-interval",
        type=int,
        default=500,
        help="Completed docks between progress CSV/JSON writes",
    )
    cfg = parser.parse_args(args)

    al_round = cfg.round
    exp_id = cfg.experiment_id
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"al_round{al_round}_{exp_id}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    # Check prerequisites
    check_mps_running()

    grid_fld = Path(cfg.maps)
    if not grid_fld.exists():
        logging.error("Grid map not found: %s", grid_fld)
        sys.exit(1)

    autodock_bin = Path(cfg.autodock_bin)
    if not autodock_bin.exists():
        logging.error("AutoDock-GPU binary not found: %s", autodock_bin)
        sys.exit(1)

    work_dir = Path(cfg.work_dir)
    round_dir = work_dir / exp_id / f"round{al_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Progress file paths
    progress_csv = log_dir / f"dock_progress_{exp_id}_round{al_round}.csv"
    progress_json = log_dir / f"dock_progress_{exp_id}_round{al_round}.json"

    # Fetch batch from al_batches (experiment-scoped)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ab.compound_id, c.smiles_protonated "
                "FROM al_batches ab "
                "JOIN compounds c ON c.id = ab.compound_id "
                "WHERE ab.round = %s AND ab.target = %s AND ab.experiment_id = %s",
                (al_round, cfg.target, exp_id),
            )
            batch = cur.fetchall()  # [(compound_id, smiles), ...]
    finally:
        conn.close()

    if not batch:
        logging.error(
            "No compounds in al_batches for round=%d target=%s experiment_id=%s",
            al_round,
            cfg.target,
            exp_id,
        )
        sys.exit(1)

    # Resume: find already-docked compound IDs this round + experiment
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT compound_id FROM docking_scores "
                "WHERE al_round=%s AND target=%s AND experiment_id=%s",
                (al_round, cfg.target, exp_id),
            )
            done_ids = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    todo = [(cid, smi) for cid, smi in batch if cid not in done_ids]
    logging.info(
        "Batch size: %d | Already done: %d | To dock: %d",
        len(batch),
        len(done_ids),
        len(todo),
    )

    if not todo:
        logging.info("All compounds already docked for this round.")
        return

    # Queues
    batch_queue = queue.Queue(maxsize=4)  # buffered: up to 4 pending batches
    result_queue = queue.Queue()

    # Stage 3: writer thread
    writer = threading.Thread(
        target=writer_thread,
        args=(
            result_queue,
            cfg.target,
            exp_id,
            al_round,
            len(todo),
            progress_csv,
            progress_json,
            cfg.reporting_interval,
        ),
        daemon=True,
    )
    writer.start()

    # Stage 2: batch dock worker threads
    dock_worker_threads = []
    for _ in range(cfg.dock_workers):
        t = threading.Thread(
            target=batch_dock_worker,
            args=(
                batch_queue,
                result_queue,
                grid_fld,
                work_dir,
                exp_id,
                al_round,
                cfg.nrun,
                cfg.fail_threshold,
                autodock_bin,
            ),
            daemon=True,
        )
        t.start()
        dock_worker_threads.append(t)

    # Stage 1: prep workers (CPU-bound — use processes to bypass GIL)
    # Batch assembler: accumulate successful preps into dock batches;
    # failed preps bypass docking and go directly to result_queue.
    prep_args = [(cid, smi, round_dir) for cid, smi in todo]
    pending_batch = []
    batch_counter = 0

    with ProcessPoolExecutor(max_workers=cfg.prep_threads) as executor:
        for compound_id, pdbqt_path, failed in executor.map(prep_worker, prep_args):
            if failed:
                result_queue.put((compound_id, None))
            else:
                pending_batch.append((compound_id, pdbqt_path))
                if len(pending_batch) >= cfg.dock_batch_size:
                    batch_queue.put((batch_counter, list(pending_batch)))
                    batch_counter += 1
                    pending_batch.clear()

    # Flush any partial batch at the end
    if pending_batch:
        batch_queue.put((batch_counter, pending_batch))

    # Signal each dock worker to stop (one sentinel per worker)
    for _ in range(cfg.dock_workers):
        batch_queue.put(_SENTINEL)
    for t in dock_worker_threads:
        t.join()

    # Signal writer to stop
    result_queue.put(_SENTINEL)
    writer.join()

    logging.info("dock_batch round=%d experiment_id=%s complete.", al_round, exp_id)


if __name__ == "__main__":
    main()
