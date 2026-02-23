import os
import sys
import logging
import multiprocessing as mp
import psycopg2
from psycopg2.extras import execute_values
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog("rdApp.*")

# Configuration
DB_NAME = "analgesics"
DB_USER = "labhund"
NUM_WORKERS = 4
CHUNK_SIZE = 100000
MAX_EXPECTED_ID = 181500000
STATE_FILE = "backfill_state.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("backfill.log"), logging.StreamHandler()],
)

worker_conn = None
worker_cur = None


def init_worker():
    global worker_conn, worker_cur
    worker_conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, host=os.getenv("PGHOST")
    )
    worker_conn.autocommit = False
    worker_cur = worker_conn.cursor()


def process_chunk(chunk_range):
    global worker_conn, worker_cur
    start_id, end_id = chunk_range

    # 1. Log the start of the chunk
    logging.info(f"Starting chunk ID {start_id} to {end_id}...")

    try:
        worker_cur.execute(
            """
            SELECT id, smiles_protonated 
            FROM public.compounds 
            WHERE id >= %s AND id < %s 
              AND (mw IS NULL OR logp IS NULL)
        """,
            (start_id, end_id),
        )

        rows = worker_cur.fetchall()

        if not rows:
            # 2. Log if the chunk was clean (no missing data)
            logging.info(f"Finished chunk ID {start_id} to {end_id}. (0 missing rows)")
            return start_id

        update_batch = []

        for row_id, smiles in rows:
            if not smiles:
                continue

            mol = MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                update_batch.append((row_id, mw, logp))

        if update_batch:
            update_query = """
                UPDATE public.compounds AS c
                SET mw = data.mw, 
                    logp = data.logp
                FROM (VALUES %s) AS data(id, mw, logp)
                WHERE c.id = data.id;
            """
            execute_values(worker_cur, update_query, update_batch, page_size=10000)
            worker_conn.commit()

        # 3. Log a successful update with the row count
        logging.info(
            f"Finished chunk ID {start_id} to {end_id}. Updated {len(update_batch)} rows."
        )
        return start_id

    except Exception as e:
        worker_conn.rollback()
        # 4. Clearly flag failures
        logging.error(f"Failed on chunk ID {start_id} to {end_id}: {e}")
        return None


def get_completed_chunks():
    completed = set()
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            for line in f:
                if line.strip().isdigit():
                    completed.add(int(line.strip()))
    return completed


def main():
    logging.info(f"Targeting backfill up to ID {MAX_EXPECTED_ID}")

    completed_starts = get_completed_chunks()
    logging.info(
        f"Found {len(completed_starts)} previously completed chunks in {STATE_FILE}."
    )

    pending_chunks = []
    for start in range(1, MAX_EXPECTED_ID + 1, CHUNK_SIZE):
        if start not in completed_starts:
            pending_chunks.append((start, start + CHUNK_SIZE))

    if not pending_chunks:
        logging.info("No chunks left to process. Backfill is fully complete!")
        sys.exit(0)

    logging.info(
        f"Created {len(pending_chunks)} pending chunks. Starting pool with {NUM_WORKERS} workers."
    )

    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            with open(STATE_FILE, "a") as state_f:
                for completed_start_id in pool.imap_unordered(
                    process_chunk, pending_chunks
                ):
                    if completed_start_id is not None:
                        state_f.write(f"{completed_start_id}\n")
                        state_f.flush()
        except KeyboardInterrupt:
            logging.warning("Shutdown signaled. Terminating pool. State is saved.")
            pool.terminate()
            pool.join()
            sys.exit(0)

    logging.info("Targeted backfill complete.")


if __name__ == "__main__":
    main()
