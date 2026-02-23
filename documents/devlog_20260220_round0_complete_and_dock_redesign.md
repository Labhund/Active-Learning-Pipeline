# Devlog — Round-0 Complete, dock_batch --filelist Redesign, Grid Maps Fix
**Date:** 2026-02-20 (evening)
**Author:** Markus / Claude Code (claude-sonnet-4-6)
**Project:** Active Learning HTVS — analgesics / TRPV1
**Session scope:** Grid regeneration (S/I/P atom types), dock_batch.py redesign (--filelist batch mode), round-0 docking + training completion, round-1 library scoring failure (lz4 → lzf fix)

---

## 1. Motivation

Two independent issues were identified with the round-0 docking pipeline before launch:

1. **GPU utilisation was spikey** — the original `dock_batch.py` launched one `autodock_gpu` subprocess per compound (6 concurrent). Each call incurred CUDA setup overhead (~0.08s), the GPU kernel finished in ~0.2–0.25s, then the process exited. AutoDock-GPU v1.6 supports `--filelist` / `-B` batch mode: one process docks N ligands in sequence, loading grid maps once.

2. **`--derivtype` workaround was hiding missing grid maps** — S (aliphatic sulfur), I (iodine), and P (phosphorus) had no grid maps, forcing a `--derivtype S=SA,I=Br,P=C` workaround. With ~18% of ZINC20 drug-like compounds containing these atoms, this was causing silent dock failures. autogrid4 takes ~2s; adding 3 more maps is essentially free.

---

## 2. Grid Regeneration (S, I, P atom types)

**File:** `targets/trpv1/grids/trpv1_8gfa.gpf`

Changed `ligand_types` and added 3 `map` lines:

```
# Before:
ligand_types A C Br Cl F HD N NA OA SA

# After:
ligand_types A C Br Cl F HD N NA OA SA S I P

# Added before elecmap:
map trpv1_8gfa.S.map
map trpv1_8gfa.I.map
map trpv1_8gfa.P.map
```

```bash
cd targets/trpv1/grids/
autogrid4 -p trpv1_8gfa.gpf -l trpv1_8gfa.glg
# → 13 atom-type .map files generated (~2s)
```

Verified: `ls *.map | wc -l` → 15 (13 atom-type + elec + dsol).

Smoke test post-regeneration (100 compounds, `--nrun 20`):
- Prep failures: 3% (embedding only — same as before)
- Dock failures: **0%** (was ~18% with missing S/I/P maps and no `--derivtype`)
- Mean: −8.71 kcal/mol, best: −10.44 kcal/mol

`--derivtype S=SA,I=Br,P=C` removed from `dock_batch.py` and `smoke_test_dock.py`.

---

## 3. dock_batch.py Redesign: --filelist Batch Mode

### Investigation

Discovered the correct AutoDock-GPU v1.6 `--filelist` format via testing:
- 3-line format (fld, pdbqt, out_prefix) → `"No ligands ... specified"` error
- Separate `--ffile` + pdbqt list → `"specify a .fld file before the first ligand"` error
- **Correct format**: `.fld` file first, then one `.pdbqt` per line; `--resnam` sets output prefix; outputs are `{resnam}_{N}.xml` (1-indexed)

### Architecture change

**Old:** `dock_worker()` — 6 threads, one `autodock_gpu --ffile/--lfile/--resnam` subprocess per compound

**New:** `batch_dock_worker()` — 2 threads, 100 compounds per `--filelist` batch:
```
Stage 1: 18 prep PROCESSES (ProcessPoolExecutor, unchanged)
         ↓ (compound_id, pdbqt_path, failed)
[main thread batch assembler]
         failed preps → result_queue directly (score=None)
         good preps → accumulated until dock_batch_size (100) reached → batch_queue
Stage 2: 2 batch_dock_worker THREADS
         write temp filelist: grid.fld \n pdbqt1 \n pdbqt2 \n ...
         run autodock_gpu --filelist {file} --resnam {batch_prefix} --nrun {N}
         parse {batch_prefix}_1.xml, _2.xml, ... → result_queue
Stage 3: 1 writer THREAD (unchanged)
```

### CLI changes (backwards-compatible defaults)

| Old flag | New flag | Default |
|---|---|---|
| `--dock-threads 6` | `--dock-workers 2` | 2 |
| (new) | `--dock-batch-size 100` | 100 |

`al_loop.py` `run_dock()` updated; `config/al_loop_maxmin.yaml` updated.

### Note on first batch run (all-zeros failure)

First attempt: 100% dock failure. The original 3-line filelist format (fld, pdbqt, out_prefix) was silently accepted by AutoDock-GPU (returncode 0) but produced no XML files because the format was wrong — the `.pdbqt` suffix detection was failing. Only after testing the format manually was the correct behaviour identified.

---

## 4. Round-0 Docking Results

**Config:** `config/al_loop_maxmin.yaml`, experiment_id=`maxmin_init`
**Duration:** ~62 min (24,000 compounds, 2 batch workers × 100-compound batches)

| Metric | Value |
|---|---|
| Compounds docked | 24,000 |
| Valid scores | 23,301 (97.1%) |
| Failed | 699 (2.9% — embedding failures) |
| Best score | **−15.49 kcal/mol** |
| Mean score | −10.07 kcal/mol |
| Top 10 mean | −14.48 kcal/mol |
| Top 100 mean | −13.77 kcal/mol |
| Top 1000 mean | −12.65 kcal/mol |

Best score −15.49 kcal/mol is nearly 2× better than capsaicin (−8.77 kcal/mol), consistent with a large diversity-sampled set finding strong binders across chemical space.

---

## 5. Round-0 Surrogate Training

**Duration:** ~13 min (Optuna 30 trials + final training)

| Metric | Value |
|---|---|
| Training samples | 23,301 |
| Optuna CV RMSE (best trial 20) | 1.1875 kcal/mol |
| Final val RMSE | **1.2062 kcal/mol** |
| Trees (early stopped) | 2,714 |

Best hyperparameters:
```json
{
  "max_depth": 10,
  "learning_rate": 0.04158,
  "subsample": 0.8046,
  "colsample_bytree": 0.2238,
  "min_child_weight": 12
}
```

`colsample_bytree=0.22` (22% of 4096 features per tree) is appropriate for sparse binary Morgan FP data — strong regularisation.

Artifacts:
- `models/surrogate_trpv1_8gfa_maxmin_init_round0.json`
- `models/hparams_trpv1_8gfa_maxmin_init_round0.json`
- `models/metrics_trpv1_8gfa_maxmin_init_round0.json`
- `optuna/trpv1_8gfa_maxmin_init_surrogate.db`

---

## 6. Round-1 Library Scoring Failure: lz4 Compression

### What happened

Round-1 library scoring of all 304,996,291 compounds completed successfully in 1903s (160,237 rows/s). At 100%, the script crashed writing the HDF5 output:

```
ValueError: Compression filter "lz4" is unavailable
```

h5py relies on the underlying C HDF5 library for compression. `lz4` requires an external dynamically-loaded plugin not bundled with the conda `h5py`. The chem env doesn't have it.

A partial 800-byte HDF5 file was written (HDF5 header only, no dataset). This caused the idempotency check on restart to skip scoring, then `select_batch.py` to crash reading the non-existent `predicted_scores` dataset.

### Fix

`score_library.py` line 118: `compression="lz4"` → `compression="lzf"`.

`lzf` is natively bundled with h5py (no external dependencies), speed is comparable to lz4, and it is transparently decodable by any h5py installation.

The corrupted 800-byte file was deleted so score_library will re-run on next launch.

### Restart command

```bash
source env_db.sh && conda activate chem
nohup python scripts/active_learning/al_loop.py \
    --config config/al_loop_maxmin.yaml --start-round 1 --rounds 5 \
    >> logs/al_loop_maxmin_init_stdout.log 2>&1 &
```

---

## 7. Files Modified This Session

| File | Change |
|---|---|
| `targets/trpv1/grids/trpv1_8gfa.gpf` | Add S, I, P to `ligand_types` + 3 `map` lines |
| `scripts/active_learning/dock_batch.py` | Replace per-compound `dock_worker` with `--filelist` `batch_dock_worker`; rename `--dock-threads` → `--dock-workers`; add `--dock-batch-size`; remove `--derivtype` |
| `scripts/active_learning/al_loop.py` | `run_dock()`: pass `--dock-workers`, `--dock-batch-size` |
| `scripts/active_learning/score_library.py` | `compression="lz4"` → `compression="lzf"` |
| `scripts/utils/smoke_test_dock.py` | Remove `--derivtype` |
| `config/al_loop_maxmin.yaml` | `dock_workers: 2`, `dock_batch_size: 100` (replaced `dock_threads: 6`) |
| `CLAUDE.md` | Progress update, integrate user note, doc corrections |
