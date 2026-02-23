# Devlog — AL Loop Bug Fixes, Infrastructure Hardening, and Round-0 Diversity Init
**Date:** 2026-02-20
**Author:** Markus / Claude Code (claude-sonnet-4-6)
**Project:** Active Learning HTVS — analgesics / TRPV1
**Session scope:** Code review of all 6 AL scripts → bug fixes → first successful run of round-0 diversity initialization (24,000 compounds selected, inserted into DB)

---

## 1. Motivation

With receptor prep and grid generation complete (see `devlog_20260220_receptor_prep_8gfa.md`),
this session focused on making the AL loop actually run. A code review prior to the session
identified 7 bugs (3 critical, 2 moderate, 2 minor). Running the smoke test then uncovered
additional infrastructure issues and a root-cause performance/correctness problem in the HDF5
access pattern.

---

## 2. Bug Fixes Applied (Pre-planned)

### Fix 1 — `dock_batch.py`: Invalid AutoDock-GPU v1.5 flags (CRITICAL)

The subprocess command used v1.5-era flags that AutoDock-GPU v1.6 rejects:

```python
# BEFORE (broken)
cmd = [
    str(autodock_bin),
    "-M", str(grid_fld),
    "-L", str(pdbqt_path),
    "-O", str(out_prefix),           # unknown flag in v1.6
    "--nrun", str(nrun),
    "--lsrat", str(lsrat),           # v1.5 integer; v1.6 expects percentage
    "--xml_output", "1",             # unknown flag (v1.6 uses --xmloutput / -x)
]

# AFTER (fixed)
cmd = [
    str(autodock_bin),
    "--ffile",  str(grid_fld),
    "--lfile",  str(pdbqt_path),
    "--resnam", str(out_prefix),     # -N in v1.6
    "--nrun",   str(nrun),
]
```

Also removed `lsrat` from: `dock_worker` signature, `threading.Thread` call, `argparse` parser,
and `run_dock()` in `al_loop.py`.

Without this fix: every compound would have exited with `Error: Unknown argument '-O'` →
all scores NULL → `train_surrogate` raises `ValueError: No valid docking scores found` → loop
dies at end of round 0.

### Fix 2 — `al_loop.py`: `run_score_library` returned wrong file path (CRITICAL)

```python
# BEFORE (off-by-one in round number)
out_path = Path(...) / f"predicted_{cfg['target']}_round{al_round - 1}.h5"

# AFTER
out_path = Path(...) / f"predicted_{cfg['target']}_round{al_round}.h5"
```

Without this: `select_batch` receives a non-existent path → `sys.exit(1)` on round 1.

### Fix 3 — `config/al_loop.yaml`: Remove stale `lsrat` field (MODERATE)

Removed `lsrat: 1500000` from the `autodock:` config block (v1.5-era, meaningless in v1.6).

### Fix 4 — `dock_batch.py`: ThreadPoolExecutor → ProcessPoolExecutor for prep (MODERATE)

Python's GIL prevents true parallelism for CPU-bound RDKit + meeko work.
Switched to `ProcessPoolExecutor` and refactored `prep_worker` to accept a tuple arg and
return `(compound_id, pdbqt_path_or_None, failed_bool)` instead of putting directly to a queue.

### Fix 5 — `train_surrogate.py`: Add `colsample_bytree` to defaults (MINOR)

Default hparams (used when Optuna hasn't run yet) were missing `colsample_bytree`.
XGBoost default is 1.0 (all 4096 features per tree) — slow and overfit-prone on small early
datasets. Added `"colsample_bytree": 0.1` to the fallback dict (consistent with Optuna
search range of [0.05, 0.3]).

### Fix 6 — `select_batch.py`: Guard against all-NaN logits (MINOR)

```python
# BEFORE (crashes with ValueError on empty array)
logits -= logits[~excluded_mask & np.isfinite(logits)].max()

# AFTER
valid_logits = logits[~excluded_mask & np.isfinite(logits)]
if len(valid_logits) == 0:
    logging.error("No valid predicted scores — score file may be corrupt.")
    sys.exit(1)
logits -= valid_logits.max()
```

### Fix 7 — `al_loop.py`: Fix `--init-only` docstring (MINOR)

Module docstring updated to correctly state "diversity init + round-0 docking + **training**,
then stop" (training was already implemented but omitted from the docstring).

---

## 3. Infrastructure Issues Found During Smoke Test

### 3a. `pyyaml` not in chem env

`al_loop.py` imports `yaml` which wasn't installed.

```bash
mamba install -n chem pyyaml
```

### 3b. AL tables didn't exist

The one-time migration hadn't been run.

```bash
psql -h "$PGHOST" -d analgesics -f scripts/utils/create_al_tables.sql
```

Tables created: `docking_scores`, `al_batches`, `surrogate_predictions`.

### 3c. HDF5 dataset key mismatch (`fingerprints` vs `fps`)

`build_fingerprints.py` writes the dataset as `fps`. All AL scripts were reading
`f["fingerprints"]` → `KeyError`. Fixed in:
- `init_diversity_sample.py`
- `train_surrogate.py`
- `score_library.py`

### 3d. `optuna` not in chem env

Added for HPO persistence:

```bash
mamba install -n chem optuna optuna-dashboard
```

### 3e. init used `prep_threads` (18) instead of 24

`run_init` was using `cfg.get("prep_threads", 24)` but `prep_threads` was set to 18 (for meeko
prep, intentionally limited below 24 because prep is CPU-bound with less parallelism benefit).
Added a separate `init_threads: 24` key to the config and updated `run_init` to use it.

---

## 4. Performance Tracking Added

### Optuna SQLite persistence

`train_surrogate.py` now accepts `--optuna-storage` (a SQLite URL). Studies are created with
`load_if_exists=True` so best hyperparameters from round 0's HPO carry forward to round 3, 6, …
`al_loop.py` constructs the URL: `sqlite:///optuna/{target}_surrogate.db`.

### Per-round metrics JSON + CSV

After training, `train_surrogate.py` writes:
```
models/metrics_{target}_round{N}.json
```
containing `val_rmse`, `n_train`, `n_val`, `best_docking_score`.

`al_loop.py`'s `log_round_summary()` reads this JSON and appends a row to:
```
logs/al_metrics.csv
```
Columns: `timestamp, round, target, n_docked, n_failed, best_score_kcal_mol,
mean_score_kcal_mol, val_rmse_kcal_mol, n_train`.

Config additions to `al_loop.yaml`:
```yaml
optuna_dir:  optuna/
metrics_csv: logs/al_metrics.csv
```

---

## 5. Progress Monitoring for Diversity Init

Added per-worker progress files and a monitoring loop:

- `tanimoto_maxmin()` writes `"{curr} {n_select}"` to `logs/init_progress/worker_NN.prog`
  every 5% of iterations (`n_select // 20`)
- `_log_progress()` reads all progress files every 30s and logs:
  ```
  Init progress | elapsed=X.Xmin | done=N/24 workers | avg=YY% [min%–max%] | ETA≈Zmin
  ```
- ETA is calculated from the time the first progress file appears (i.e., when MaxMin
  actually starts), **not** from the overall start — this excludes the FP-load phase which
  would otherwise make the ETA wildly inflated during the first 5–15 minutes.

---

## 6. HDF5 I/O Root Cause Investigation and Fix

### Symptom

Workers were taking 13+ minutes just to load fingerprints. The original code used fancy
(scatter) indexing to pull random compound IDs from the HDF5 file.

### Root cause

The HDF5 file is chunked as `(10000, 512)` = 5 MB per chunk. With 500K randomly-sampled
rows scattered across a 12.4M-row worker partition, each read touched:

```
500,000 / 10,000 × (average chunks touched) ≈ 1,240 chunks × 5 MB = 6.3 GB per worker
```

Across 24 workers: **151 GB** of I/O to read 24 × 256 MB = 6 GB of actual data. 24× overhead.

### Fix: sequential contiguous slice reads

Each worker now reads a single contiguous slice from its HDF5 partition:

```python
# Pick a random contiguous block within this worker's row range
block_start = start_row + block_offset
block_end   = min(block_start + subsample_size, start_row + n_rows_in_range)

with h5py.File(fp_file, "r") as f:
    fp_block = f["fps"][block_start : block_end]  # 50 chunks × 5 MB = 256 MB
```

This reads exactly 50 HDF5 chunks (256 MB) per worker — independent of how many valid rows
are in the block. The validity filtering happens in-memory after the read.

### Subsample reduction

Reduced `subsample_per_worker` from 1,000,000 to 500,000 in `config/al_loop.yaml`.
Also removed the DB ID fetch that was previously used to construct the subsample
(replaced by the simpler validity mask approach).

---

## 7. FK Violation and Root Cause: HDF5 Gap Rows

### The crash (before validity mask fix)

```
psycopg2.errors.ForeignKeyViolation: insert or update on table "al_batches"
violates foreign key constraint "al_batches_compound_id_fkey"
DETAIL: Key (compound_id)=(190686575) is not present in table "compounds".
```

### Root cause

The HDF5 file has **304,996,291 rows** but the `compounds` table has only **297,881,291
rows**. The ~7.1M gap rows (2.3% of the HDF5) correspond to PostgreSQL `SERIAL` sequence
gaps from failed COPY transactions during the original bulk import. These rows exist as
reserved sequence values but were never committed, leaving the HDF5 positions at those
indices with all-zero fingerprint vectors (the HDF5 was pre-allocated).

### Compounding problem: all-zero vectors poison MaxMin

All-zero FP vectors have Tanimoto distance = 1.0 to every real molecule (since
`intersection = 0`, `union = popcount(real)`, `similarity = 0`, `distance = 1`).
MaxMin greedily selects them first as "maximally diverse" — filling the initial batch
with garbage compound IDs that fail the FK constraint on insert.

### Fix: validity mask

In `main()`, before forking workers:

```python
# Fetch all valid IDs from DB
conn = get_db_conn()
with conn.cursor("id_cursor") as cur:
    cur.itersize = 500_000
    cur.execute("SELECT id FROM compounds ORDER BY id")
    all_ids = np.array([row[0] for row in cur], dtype=np.int64)

# Build boolean mask: True = valid compound, HDF5 row index = compound_id - 1
valid_mask = np.zeros(N_rows, dtype=np.bool_)
valid_hdf5_rows = all_ids - 1
in_range = valid_hdf5_rows[valid_hdf5_rows < N_rows]
valid_mask[in_range] = True

# Save for workers to mmap
np.save("/tmp/al_init_valid_mask.npy", valid_mask)
del valid_mask, all_ids, in_range   # free ~2.7 GB before forking
```

Workers load the mask read-only via `mmap_mode="r"` — the OS shares one copy of the
305 MB file across all 24 processes via its page cache:

```python
valid_mask = np.load(valid_mask_path, mmap_mode="r")
local_valid = valid_mask[block_start : block_end]   # view, no copy
valid_local_rows = np.where(local_valid)[0]
fp_array = fp_block[valid_local_rows]
```

### Belt-and-suspenders: all-zero safety check

Even after the validity mask, a corrupt compound in the DB could theoretically have an
all-zero fingerprint. Added a secondary filter:

```python
nonzero_mask = fp_array.any(axis=1)
n_zero = int((~nonzero_mask).sum())
if n_zero:
    logging.warning(
        "Worker %02d: dropping %d all-zero FP vectors (corrupt fingerprints in DB)",
        worker_id, n_zero,
    )
    valid_local_rows = valid_local_rows[nonzero_mask]
    fp_array = fp_array[nonzero_mask]
```

Both `valid_local_rows` and `fp_array` are kept in sync so the compound ID mapping
(`global_row + 1`) remains correct after either filter.

---

## 8. Diversity Init: Successful Completion

**Target:** trpv1_8gfa
**Date/time:** 2026-02-20, ~12:07 → ~13:27 (79 min total)

### Timing breakdown

| Phase | Duration |
|-------|----------|
| DB ID fetch + validity mask build | ~45s |
| Worker FP loading (sequential reads) | ~3–5 min |
| MaxMin diversity picking (24 × 500K → 1000) | ~73 min |
| DB insert (24,000 rows) | ~3s |

### Worker completion times

All 24 workers completed within 4257–4744s (71–79 min). Workers 15–22 were slightly
faster, likely reflecting marginally higher valid-compound density in the upper HDF5 partitions.

### Result

```
al_batches: round=0, target=trpv1_8gfa, source=diversity_init → 24000 rows
```

Confirmed:
```sql
SELECT round, source, count(*)
FROM al_batches WHERE target='trpv1_8gfa'
GROUP BY round, source;
-- round | source          | count
--   0   | diversity_init  | 24000
```

The `source='diversity_init'` flag distinguishes round-0 compounds from all subsequent
rounds which use `source='thompson_sample'`.

---

## 9. Loop Halted: MPS Not Running

After inserting the 24,000 compounds, `al_loop.py` proceeded to `run_dock(cfg, al_round=0)`.
`dock_batch.py` checks for NVIDIA MPS and raised:

```
RuntimeError: NVIDIA MPS control daemon is not running.
Start it with: nvidia-cuda-mps-control -d
```

This is the expected safety guard. The diversity init data is fully saved and idempotent —
`init_diversity_sample.py` will skip if `al_batches` is already populated for round 0.

---

## 10. Next Steps

1. **Start MPS and run round-0 docking + training:**
   ```bash
   nvidia-cuda-mps-control -d
   source env_db.sh && conda activate chem
   python scripts/active_learning/al_loop.py \
       --config config/al_loop.yaml --start-round 0 --rounds 1 --skip-init --init-only
   ```
   `--skip-init` skips diversity init (already done). `--init-only` stops after training.

2. **Monitor docking:**
   ```sql
   SELECT count(*), count(*) FILTER (WHERE score IS NOT NULL) AS valid,
          min(score), avg(score)
   FROM docking_scores WHERE target='trpv1_8gfa' AND al_round=0;
   ```

3. **Verify surrogate training** (after docking completes):
   ```bash
   cat models/metrics_trpv1_8gfa_round0.json
   ```

4. **PyMOL visual verification** (still pending from receptor prep session):
   Load `trpv1_8gfa.pdbqt` and confirm ZEI:D:1203 is inside the grid box and the S4-S5
   linker is at/outside the intracellular edge.

---

## 11. Files Modified This Session

| File | Changes |
|------|---------|
| `scripts/active_learning/dock_batch.py` | Fix AutoDock-GPU flags; remove lsrat; ThreadPool→ProcessPool |
| `scripts/active_learning/al_loop.py` | Fix score path off-by-one; remove lsrat from run_dock; fix docstring; add metrics CSV; add Optuna storage; add init_threads/subsample passthrough |
| `scripts/active_learning/init_diversity_sample.py` | Fix HDF5 key; sequential reads; validity mask; all-zero safety check; progress reporting with corrected ETA |
| `scripts/active_learning/train_surrogate.py` | Fix HDF5 key; add colsample_bytree default; Optuna storage; metrics JSON write |
| `scripts/active_learning/score_library.py` | Fix HDF5 key (2 occurrences) |
| `scripts/active_learning/select_batch.py` | NaN guard before softmax shift |
| `config/al_loop.yaml` | Remove lsrat; add init_threads, subsample_per_worker, optuna_dir, metrics_csv |
