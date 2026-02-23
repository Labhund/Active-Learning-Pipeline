import os
import time
import csv
import multiprocessing as mp
from dimorphite_dl import protonate_smiles
from rdkit import RDLogger

# 1. Silence RDKit warnings globally
# RDLogger.DisableLog("rdany.cp")

# Configuration
ZINC_DIR = "/data2/lipin_lab/markus/analgesics/data/zinc20"
NUM_WORKERS = 6
MAX_TASKS_PER_CHILD = 1
SAMPLES_PER_CORE = 1000
ERROR_LOG_PATH = "ingest_errors.csv"


def protonate_worker(chunk):
    results = []
    errors = []
    start_time = time.perf_counter()

    for smiles, zinc_id in chunk:
        try:
            protonated_list = protonate_smiles(smiles, ph_min=5.5, ph_max=7.4)

            if not protonated_list:
                # Capture the "silent" failures
                errors.append((zinc_id, smiles, "RDKit/Dimorphite Sanitization Failed"))
                results.append((zinc_id, smiles, "FAILED"))
            else:
                results.append((zinc_id, smiles, protonated_list[0]))

        except Exception as e:
            errors.append((zinc_id, smiles, f"CRASH: {str(e)}"))
            results.append((zinc_id, smiles, "ERROR"))

    return results, errors, time.perf_counter() - start_time


def main():
    # 1. Gather files
    smi_files = []
    for root, _, files in os.walk(ZINC_DIR):
        for f in files:
            if f.endswith(".smi"):
                smi_files.append(os.path.join(root, f))
        if len(smi_files) >= NUM_WORKERS:
            break

    smi_files = sorted(smi_files)[:NUM_WORKERS]
    print(
        f"Found {len(smi_files)} files. Extracting {SAMPLES_PER_CORE} samples per worker..."
    )

    # 2. Extract data
    test_data = []
    for f_path in smi_files:
        worker_chunk = []
        with open(f_path, "r") as f:
            next(f)  # skip header
            for _ in range(SAMPLES_PER_CORE):
                line = f.readline().strip().split()
                if len(line) >= 2:
                    worker_chunk.append((line[0], line[1]))
        test_data.append(worker_chunk)

    # 3. Execution
    print(f"Spinning up Pool with {NUM_WORKERS} workers...")
    with mp.Pool(processes=NUM_WORKERS, maxtasksperchild=MAX_TASKS_PER_CHILD) as pool:
        raw_output = pool.map(protonate_worker, test_data)

    # 4. Processing Results & Writing Error Log
    all_errors = []
    total_time = 0

    print("\n" + "=" * 95)
    print(
        f"{'CORE':<6} | {'ZINC ID':<12} | {'ORIGINAL SMILES':<30} | {'PROTONATED SMILES'}"
    )
    print("-" * 95)

    for i, (res_list, err_list, duration) in enumerate(raw_output):
        total_time += duration
        all_errors.extend(err_list)

        # Print first 3 samples per core
        for zinc_id, orig, prot in res_list[:3]:
            o_disp = (orig[:27] + "..") if len(orig) > 27 else orig
            p_disp = (prot[:42] + "..") if len(prot) > 42 else prot
            print(f"Core {i} | {zinc_id:<12} | {o_disp:<30} | {p_disp}")
        print(f"Core {i} | ... processed {len(res_list)} total molecules.")
        print("-" * 95)

    # Write errors to CSV
    if all_errors:
        with open(ERROR_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["zinc_id", "smiles", "error_message"])
            writer.writerows(all_errors)
        print(
            f"\n[!] CAUTION: {len(all_errors)} molecules failed. Details saved to {ERROR_LOG_PATH}"
        )

    # 5. Final Analytics
    avg_per_mol = (total_time / (NUM_WORKERS * SAMPLES_PER_CORE)) * 1000
    print("\n### RUNTIME ESTIMATE BREAKDOWN")
    print(f"* **Throughput:** ~{1 / (avg_per_mol / 1000):.1f} mol/sec per core")
    print(
        f"* **Total Expected for 1M molecules:** ~{(avg_per_mol * 1000000 / 1000 / 3600 / NUM_WORKERS):.2f} hours"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
