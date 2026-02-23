import os
import sys
import logging
import multiprocessing as mp
import psycopg2
from psycopg2.extras import execute_values
from dimorphite_dl import protonate_smiles
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit import RDLogger

# Silence RDKit
RDLogger.DisableLog("rdApp.*")

# Configuration
DB_NAME = "analgesics"
DB_USER = "labhund"
PROJ_ROOT = os.getenv("PROJ_ROOT", "/data2/loo_lab/markus/analgesics")
ZINC_DIR = os.path.join(PROJ_ROOT, "data/zinc20")
NUM_WORKERS = 18

# Bumping batch size to 10k reduces the number of commits and network overhead
BATCH_SIZE = 10000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Worker-%(process)d: %(message)s",
    handlers=[logging.FileHandler("ingestion.log"), logging.StreamHandler()],
)

# Global variables for persistent worker connections
worker_conn = None
worker_cur = None


def init_worker():
    """Initializer runs ONCE per worker process to set up a persistent DB connection."""
    global worker_conn, worker_cur
    worker_conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, host=os.getenv("PGHOST")
    )
    # Disable autocommit to batch transactions manually
    worker_conn.autocommit = False
    worker_cur = worker_conn.cursor()


def process_file_worker(file_path):
    global worker_conn, worker_cur

    tranche_name = "/".join(file_path.split("/")[-2:]).replace(".smi", "")

    try:
        # 1. Check Tranche Status
        worker_cur.execute(
            "SELECT status FROM public.tranche_status WHERE tranche_name = %s",
            (tranche_name,),
        )
        row = worker_cur.fetchone()

        if row and row[0] == "completed":
            return  # Skip already finished files

        # 2. Lock/Start Tranche
        worker_cur.execute(
            """
            INSERT INTO public.tranche_status (tranche_name, status)
            VALUES (%s, 'processing')
            ON CONFLICT (tranche_name) DO UPDATE SET status = 'processing'
            RETURNING id
        """,
            (tranche_name,),
        )
        t_id = worker_cur.fetchone()[0]
        worker_conn.commit()

        logging.info(f"Processing Tranche {t_id}: {tranche_name}")

        batch_results = []
        batch_errors = []

        # 3. Stream the file
        with open(file_path, "r") as f:
            next(f)  # Skip header

            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                smiles, zinc_id = parts[0], parts[1]

                try:
                    prot_list = protonate_smiles(smiles, ph_min=7.4, ph_max=7.4)

                    if not prot_list:
                        batch_errors.append(
                            (zinc_id, smiles, "Protonation Failed", t_id)
                        )
                        continue

                    prot_smiles = prot_list[0]
                    mol = MolFromSmiles(prot_smiles)

                    if not mol:
                        batch_errors.append(
                            (zinc_id, smiles, "RDKit Mol Conversion Failed", t_id)
                        )
                        continue

                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)

                    batch_results.append((zinc_id, prot_smiles, mw, logp, t_id))

                except Exception as e:
                    batch_errors.append((zinc_id, smiles, f"Crash: {str(e)}", t_id))

                # 4. Flush to Database in Batches
                if len(batch_results) >= BATCH_SIZE:
                    execute_values(
                        worker_cur,
                        """INSERT INTO public.compounds (zinc_id, smiles_protonated, mw, logp, tranche_id) 
                           VALUES %s ON CONFLICT (zinc_id) DO NOTHING""",
                        batch_results,
                        page_size=BATCH_SIZE,  # Crucial for speed with execute_values
                    )
                    batch_results.clear()

                if len(batch_errors) >= BATCH_SIZE:
                    execute_values(
                        worker_cur,
                        """INSERT INTO public.ingestion_errors (zinc_id, original_smiles, error_msg, tranche_id) 
                           VALUES %s ON CONFLICT DO NOTHING""",
                        batch_errors,
                        page_size=BATCH_SIZE,
                    )
                    batch_errors.clear()

        # Flush remaining
        if batch_results:
            execute_values(
                worker_cur,
                """INSERT INTO public.compounds (zinc_id, smiles_protonated, mw, logp, tranche_id) 
                   VALUES %s ON CONFLICT (zinc_id) DO NOTHING""",
                batch_results,
                page_size=len(batch_results),
            )
        if batch_errors:
            execute_values(
                worker_cur,
                """INSERT INTO public.ingestion_errors (zinc_id, original_smiles, error_msg, tranche_id) 
                   VALUES %s ON CONFLICT DO NOTHING""",
                batch_errors,
                page_size=len(batch_errors),
            )

        # 5. Mark Completed
        worker_cur.execute(
            "UPDATE public.tranche_status SET status = 'completed' WHERE id = %s",
            (t_id,),
        )
        worker_conn.commit()
        logging.info(f"Finished Tranche {t_id}")

    except Exception as e:
        worker_conn.rollback()
        logging.error(f"Failed to process tranche {tranche_name}: {e}")


def main():
    all_files = []
    for root, _, files in os.walk(ZINC_DIR):
        for f in files:
            if f.endswith(".smi"):
                all_files.append(os.path.join(root, f))

    logging.info(
        f"Found {len(all_files)} files. Resuming ingestion with {NUM_WORKERS} workers."
    )

    # Using imap_unordered + persistent connections
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            for _ in pool.imap_unordered(process_file_worker, sorted(all_files)):
                pass
        except KeyboardInterrupt:
            logging.warning("Shutdown signaled. Terminating pool...")
            pool.terminate()
            pool.join()
            sys.exit(0)


if __name__ == "__main__":
    main()
