# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Active Learning (AL) HTVS pipeline targeting **TRPV1** (Transient Receptor Potential Vanilloid 1) for analgesic drug discovery. ~300M ZINC20 compounds (pH 7.4 protonated, MW/LogP computed) are stored in a local PostgreSQL DB. An iterative AL loop trains a surrogate model (XGBoost on Morgan FPs) on AutoDock-GPU scores, then uses Thompson sampling to select the next docking batch — focusing compute on the most promising chemical space. Designed to scale to multiple targets and HPC.

## Implementation Status

**Phase 1 COMPLETE (2026-02-20 → 2026-02-22).** Two 6-round experiments completed and compared.
Lab reports: `documents/labarchives_TRPV1_prototype_run1.md`, `documents/labarchives_TRPV1_postscreen_analysis1.md`, `documents/labarchives_TRPV1_comparison_maxmin_vs_random.md`.

**Completed experiments:**

| experiment_id | Init strategy | Rounds | Docked | Best-1 | R5 RMSE |
|---|---|---|---|---|---|
| `maxmin_init` | MaxMin Tanimoto (batch-parallel) | 0–5 | 142,042 | −16.06 | 0.9292 |
| `random_init` | Uniform random | 0–5 | 142,556 | −16.15 | 0.8458 |

Key finding from comparison: Thompson sampling (T=1.0) dominates over initialization choice by round 2.
Random init produced consistently better surrogate RMSE. MaxMin UMAP coverage was less uniform than
random due to the batch-parallel implementation artefact (central ridge oversampling).
See `documents/labarchives_TRPV1_comparison_maxmin_vs_random.md` for full analysis.

**Planned experiments (temperature ablation — Phase 2):**
- `greedy` — T=0 (deterministic top-K, pure exploitation); requires code change in `select_batch.py`
- `explore` — T=5.0 (near-uniform Boltzmann, high exploration)
- Both use `random_init` seed (cleaner surrogate from round 0) for controlled comparison.
- Config files: `config/al_loop_greedy.yaml`, `config/al_loop_explore.yaml` (to be created).
- See "Temperature Ablation" section below for implementation notes.

**Progress:**
1. ~~Run the DB migration~~ **DONE**
2. ~~Prepare receptor PDBQT + AutoDock-GPU grid~~ **DONE (2026-02-20)**
3. ~~Round-0 diversity init (maxmin_init)~~ **DONE (2026-02-20)**
4. ~~Rounds 0–5 maxmin_init~~ **DONE (2026-02-21)**
5. ~~Rounds 0–5 random_init~~ **DONE (2026-02-22)**
6. ~~MaxMin vs random comparison analysis~~ **DONE (2026-02-22)**
7. **Next:** Temperature ablation (greedy T=0 and exploratory T=5)

## Temperature Ablation — Phase 2 Experiments

**Scientific question:** How does the Thompson sampling temperature T affect what gets discovered?
T controls the exploration/exploitation balance in `select_batch.py`:
- `logits = neg_scores / T; probs = softmax(logits)`
- T→0: pure exploitation (all mass on the top-predicted compound)
- T=1.0: current baseline (moderately exploitative)
- T→∞: pure exploration (uniform random, equivalent to random_init at every round)

**Planned experiments (both use random_init seed for apples-to-apples comparison):**

| experiment_id | T | Behaviour |
|---|---|---|
| `random_T0` | 0 (greedy) | Top-K deterministic selection — fastest potential convergence, highest surrogate-error risk |
| `random_T1` | 1.0 | Baseline (already done as `random_init`) |
| `random_T5` | 5.0 | Near-exploratory — mild preference for predicted hits, broad coverage |

**T=0 implementation (greedy mode) — requires change to `select_batch.py`:**
At T=0, `neg_scores / T` is undefined. Add a special case before the softmax block:
```python
if cfg.temperature == 0:
    # Deterministic greedy: rank compounds by predicted score, pick top batch_size
    ranked_idx = np.argsort(-neg_scores)  # descending neg_score = ascending score
    selected = [i for i in ranked_idx
                if valid_mask[i] and not excluded_mask[i]][:cfg.batch_size]
    compound_ids = (np.array(selected) + 1).tolist()
    # ... write to DB as usual
```
Config: set `temperature: 0` in YAML, check for it in `al_loop.py` → `select_batch.py`.

**T=5 — no code changes needed**, just set `temperature: 5.0` in config.

**Expected behaviours:**
- T=0: Should find the best single compounds fastest if the surrogate is accurate.
  Risk: if surrogate has systematic error in a region, all 24K slots go to that region
  (no exploration to correct it). Scaffold diversity will likely be very low.
- T=5: At the score range observed (~8 kcal/mol spread), exp(8/5) ≈ 5× weight on the
  best vs mean compound (vs exp(8/1) ≈ 3000× at T=1). This is much closer to random
  sampling. Will likely show broader scaffold coverage but slower convergence to top hits.

**To run (once configs are created):**
```bash
source env_db.sh && conda activate chem
nvidia-cuda-mps-control -d
# Greedy (requires select_batch.py change first)
nohup python scripts/active_learning/al_loop.py \
    --config config/al_loop_greedy.yaml --start-round 0 --rounds 6 \
    >> logs/al_loop_random_T0_stdout.log 2>&1 &
# Exploratory
nohup python scripts/active_learning/al_loop.py \
    --config config/al_loop_explore.yaml --start-round 0 --rounds 6 \
    >> logs/al_loop_random_T5_stdout.log 2>&1 &
```

## Environment Setup

Always source the environment file first:
```bash
source env_db.sh
```
This sets `PROJ_ROOT`, `PGDATA`, `PGHOST`, `PGLOG` and adds aliases: `db-start`, `db-stop`, `db-status`, `db-sql`, `db-schema`.

All Python work uses the `chem` conda environment:
```bash
conda activate chem   # /home/labhund/mamba_envs/chem
```

Python dependencies in `chem` env: RDKit, psycopg2, h5py, xgboost, meeko, optuna, pyyaml, scikit-learn.
Install any missing with `mamba install -n chem <pkg>`.

## Hardware

Workstation (siloed project — all compute is local):
- **CPU**: AMD Ryzen 9 9900X (12c/24t)
- **GPU**: RTX 5080
- **RAM**: 128 GB DDR5
- **Storage**: 2 TB NVMe

Parallelism target: 24 threads for CPU-bound steps; 1 GPU for docking and surrogate inference.

## Database

Local PostgreSQL instance storing ~300M ZINC20 compounds selected by logP > 3.5 and MW ~350 based on analysis of known binders.

```bash
db-start         # Start PostgreSQL
db-stop          # Stop PostgreSQL
db-status        # Check if running
db-sql           # Open psql shell to "analgesics" DB
./scripts/utils/db_stats.sh   # Show compound count, table size
```

**Compounds table** (read-only at this phase):
```
public.compounds (id SERIAL, zinc_id TEXT, smiles_protonated TEXT, mw FLOAT, logp FLOAT)
```

**AL tables** (created by `scripts/utils/create_al_tables.sql` — run once before first loop):
```sql
-- Docking results per round (score=NULL for failed docks)
docking_scores (id SERIAL PK, compound_id INT FK, target TEXT, score FLOAT, al_round INT, docked_at TIMESTAMP)

-- Compounds selected each round and how
al_batches (id SERIAL PK, round INT, compound_id INT FK, target TEXT, source TEXT)
-- source values: 'diversity_init' (round 0) | 'thompson_sample' (round 1+)

-- Optional: full-library surrogate predictions per round
surrogate_predictions (round INT, compound_id INT FK, target TEXT, predicted_score FLOAT, PK(round,compound_id,target))
```

## Data Artifacts

| Path | Contents |
|------|----------|
| `data/fingerprints/` | 4096-bit radius-4 Morgan FPs, 8-bit packed, HDF5. Row index = `compounds.id - 1`. Shape and runtime recorded after FP generation completes. |
| `data/zinc20/` | ~300M ZINC20 source .smi files (tranche subdirectories) |
| `data/zinc22/H05/` | ZINC22 subset (P350-P600 logP, H20-H40 atoms) |
| `data/postgres/` | Local PostgreSQL data directory |
| `documents/TRPV1_binders.csv` | 11 known TRPV1 binders used as reference compounds |
| `targets/trpv1/structures/pdb/` | CIF source files: 5IRZ, 7MZD, 8U3L, **8GFA** (SB-366791 co-crystal, used for docking). Extracted: `trpv1_8gfa_chains_DA.pdb`, `trpv1_8gfa_receptor.pdb` (protein + lipid shell), `trpv1_8gfa_protein_fixed.pdb` (meeko input). |
| `targets/trpv1/structures/pdbqt/` | `trpv1_8gfa.pdbqt` — chains D+A protein + 8 annular lipids (DU0/POV), prepared with meeko 0.5.0 + obabel (870 KB). |
| `targets/trpv1/grids/` | `trpv1_8gfa.fld` + **13 atom-type maps** (A,C,Br,Cl,F,HD,N,NA,OA,SA,**S,I,P**) + elec + dsol. Grid: 48×44×60 pts, 0.303 Å spacing, center (81.581, 101.331, 86.875). S/I/P maps added 2026-02-20 (eliminates need for `--derivtype` workaround). |
| `memprotmd/TRPV1/8GF9/` | TRPV1 in DPPC membrane (GROMACS CHARMM36, 34 MB PDB) |
| `boltzlab/known_binders/` | BoltzLab scoring results for reference binders |
| `bin/autodock_gpu` | AutoDock-GPU binary (v1.6, CUDA/OpenCL) |
| `models/` | XGBoost surrogate models: `surrogate_{target}_round{N}.json` |
| `models/hparams_{target}_round{N}.json` | Best Optuna hyperparameters (persisted between HPO rounds) |
| `scores/` | Per-round predicted score HDF5s: `predicted_{target}_round{N}.h5` |
| `work/docking/round{N}/` | Per-compound PDBQT inputs + AutoDock-GPU XML outputs |
| `logs/` | Per-round logs (`al_round{N}.log`) + orchestrator log (`al_loop.log`) |
| `config/al_loop.yaml` | Central AL loop configuration (all tunable parameters) |

**Fingerprint HDF5 layout**: one dataset of shape `(N, 512)` uint8, where each row is a 4096-bit Morgan FP packed into 512 bytes. Compound with `compounds.id = k` → HDF5 row `k - 1`.

## Active Learning Loop Architecture

```
INITIALIZATION (round 0, one-time)
  - 300M compounds split into 24 equal batches
  - Per batch (24 threads): subsample 1M → MaxMin Tanimoto diversity picking → 1,000 IDs
  - Combine: 24,000 compound IDs → initial docking batch
  - Record source = 'diversity_init' in al_batches

DOCKING (per round)
  - Fetch smiles_protonated from DB for batch IDs
  - Prepare PDBQT via meeko (installed in chem env)
  - Run bin/autodock_gpu against TRPV1 grid
  - Ingest scores → docking_scores table
  - Exclude failed docks (score ≥ 0 kcal/mol) from training set [threshold configurable]

SURROGATE TRAINING
  - Fetch Morgan FPs from HDF5 for all docked + valid compounds
  - XGBoost regressor: 4096-bit packed Morgan FP → docking score (kcal/mol, raw)
  - No score transformation; model predicts raw values for stability across rounds

LIBRARY SCORING
  - Stream HDF5 in chunks → surrogate.predict(FP chunk) for all ~300M compounds
  - Negate predicted scores (so higher = better predicted binder)
  - Apply softmax with temperature T (controls exploration/exploitation balance)

BATCH SELECTION (Thompson sampling via Boltzmann exploration)
  - Sample 24,000 compound IDs WITHOUT replacement from softmax distribution
  - Low-scoring compounds retain non-zero selection probability → prevents surrogate collapse
  - Temperature T is a tunable hyperparameter; record source = 'thompson_sample' in al_batches

LOOP → back to DOCKING
```

## Key Design Decisions

- **Raw score prediction**: XGBoost predicts raw docking scores (kcal/mol, negative). No min-max normalization — avoids scale drift across rounds.
- **Acquisition function**: softmax(negated predicted scores) → Thompson sampling without replacement. Balances exploration and exploitation.
- **Failed dock exclusion**: compounds with score ≥ 0 kcal/mol are excluded from training. Default threshold is 0; make it a configurable parameter.
- **Batch size**: 24,000 compounds per round — matches 24 CPU threads for parallelism.
- **Multi-target design**: `docking_scores.target` and `al_batches` are target-agnostic; schema supports future expansion beyond TRPV1.

## Docking Setup Notes

- **Vanilloid binding site** is lipid-exposed (S3-S4 transmembrane domain) — critical design consideration.
- Receptor preparation includes **protein + annular lipid shell (15 Å cutoff from ZEI centroid)** — 8GFA's own cryo-EM resolved lipids (DU0 cholesterol derivative + POV POPC) are used directly. MemProtMD (8GF9) is not needed for receptor prep.
- **Active target: `trpv1_8gfa`** — ZEI:D:1203 at the chain D/A interface. Grid restricted to exclude the S4-S5 linker void (prevents surrogate convergence on linker binders across AL rounds).
- **Grid center (81.581, 101.331, 86.875)** set in ADT from ZEI:D:1203 pose. Grid: 48×44×60 pts, 0.303 Å spacing (14.5×13.3×18.2 Å box).
- **Pending visual check**: load `trpv1_8gfa.pdbqt` in PyMOL, confirm ZEI inside box and S4-S5 linker Cα (~res 580-600) at/outside intracellular edge.
- **Capsaicin smoke dock PASSED (2026-02-20):** best `free_NRG_binding` = **−8.77 kcal/mol**, 3 clusters (7/11/2 poses), mean −8.39 kcal/mol across 20 runs. Note: AutoDock-GPU stdout reports `Inter + Intra = −13.22` — this is the raw optimizer value before the torsional correction (+2.98 kcal/mol for 10 rotatable bonds); `free_NRG_binding` from XML is the correct binding energy for DB storage and surrogate training. `dock_batch.py` reads `free_NRG_binding` (falls back from missing `best_energy` tag in v1.6 XML — confirmed correct).
- Ligand PDBQT prep: `meeko` (installed in `chem` env).
- **Receptor prep pipeline** (if re-running): `bash scripts/utils/prep_grid_8gfa.sh`. Requires HIS tautomer fix + CYX disulfide rename — handled by `scripts/utils/preprocess_receptor_pdb.py` (CYS 387↔391 intra-chain disulfide, S-S 2.03 Å; HIS tautomers read from modelled H positions in 8GFA).

## Script Layout

```
scripts/
  active_learning/
    al_loop.py                  # Orchestrator — entry point for full AL loop
    init_diversity_sample.py    # MaxMin initialization (24-process parallel, round 0)
    dock_batch.py               # 3-stage pipeline: meeko prep → MPS docking (--filelist batch) → DB write
    train_surrogate.py          # XGBoost training + Optuna HPO on accumulated scores
    score_library.py            # Full 300M library scoring in 100K-row chunks
    select_batch.py             # Softmax Thompson sampling batch selection
  fingerprints/
    build_fingerprints.py       # Generate and store Morgan FPs to HDF5
  utils/
    create_al_tables.sql        # DB migration — run once before first AL loop
    chem_summary.py             # Print cheminformatics properties for a SMILES string
    db_stats.sh                 # Show compound count and table size
    extract_receptor_8gfa.py    # Extract chains D+A + lipid shell from 8gfa.cif; identify D/A interface ZEI
    preprocess_receptor_pdb.py  # Fix HIS tautomers (from H positions) + CYX disulfide; split protein/lipids
    prep_grid_8gfa.sh           # Orchestrate meeko → obabel → autogrid4 for trpv1_8gfa
```

## Running the AL Loop

### One-time setup
```bash
source env_db.sh && db-start
psql -d analgesics -f scripts/utils/create_al_tables.sql   # create AL tables
nvidia-cuda-mps-control -d                                  # start NVIDIA MPS
```

### Smoke test (diversity init only — no docking yet)
**COMPLETED 2026-02-20** — 24,000 compounds in `al_batches`, `round=0`, `source='diversity_init'`.
```bash
# Verify:
psql -h "$PGHOST" -d analgesics -c "SELECT round, source, count(*) FROM al_batches WHERE target='trpv1_8gfa' GROUP BY round, source;"
# Expected: 0 | diversity_init | 24000
```

### Round-0 docking + training — **COMPLETED 2026-02-20**
Results: 23,301/24,000 valid, best −15.49 kcal/mol, surrogate val_rmse=1.2062.
Use `config/al_loop_maxmin.yaml` (not the generic `al_loop.yaml`) for `maxmin_init` experiment.

### Rounds 1+ (current)
```bash
source env_db.sh && conda activate chem
nohup python scripts/active_learning/al_loop.py \
    --config config/al_loop_maxmin.yaml --start-round 1 --rounds 5 \
    >> logs/al_loop_maxmin_init_stdout.log 2>&1 &
```

### Full loop (from scratch)
```bash
python scripts/active_learning/al_loop.py \
    --config config/al_loop_maxmin.yaml --start-round 0 --rounds 5
```

### Restart after interruption (all scripts are resume-safe)
```bash
# dock_batch: skips already-docked compound IDs
# train_surrogate: skips if model file already exists for that round
# score_library: skips if score HDF5 already exists for that round
# select_batch: skips if al_batches already populated for that round
python scripts/active_learning/al_loop.py \
    --config config/al_loop.yaml --start-round 2 --rounds 3 --skip-init
```

### Individual script CLIs
```bash
# Diversity init (round 0 only)
python scripts/active_learning/init_diversity_sample.py \
    --fp-file data/fingerprints/compounds.h5 --target trpv1_8gfa --batch-size 24000 --threads 24

# Docking
python scripts/active_learning/dock_batch.py \
    --target trpv1_8gfa --round 0 --maps targets/trpv1/grids/trpv1_8gfa.fld

# Train surrogate (add --tune-hparams for Optuna HPO)
python scripts/active_learning/train_surrogate.py \
    --target trpv1_8gfa --round 0 --fp-file data/fingerprints/compounds.h5

# Score full library
python scripts/active_learning/score_library.py \
    --target trpv1_8gfa --round 1 --model models/surrogate_trpv1_8gfa_round0.json

# Select next batch
python scripts/active_learning/select_batch.py \
    --target trpv1_8gfa --round 1 --scores scores/predicted_trpv1_8gfa_round0.h5
```

### Monitoring queries
```bash
# Docking progress per round
psql -d analgesics -c "
SELECT al_round, count(*), count(*) FILTER (WHERE score IS NOT NULL) AS valid,
       min(score), avg(score)
FROM docking_scores WHERE target='trpv1_8gfa' GROUP BY al_round ORDER BY al_round;"

# Batch composition
psql -d analgesics -c "
SELECT round, source, count(*) FROM al_batches GROUP BY round, source ORDER BY round;"
```

## Utility Scripts

```bash
# Print cheminformatics properties for a SMILES string:
python scripts/utils/chem_summary.py "<SMILES>"

# Visualize BoltzLab binding confidence scores:
python boltzlab/known_binders/plot_hist.py
```

## TRPV1 Target Context

Crystal structures:
- **5IRZ** — closed/apo state (reference for capsaicin/RTX binding pose)
- **8U3L** — open state
- **8GFA** — SB-366791 (ZEI) co-crystal, 4-fold tetramer with resolved annular lipids. **Active docking target** (`trpv1_8gfa`). D/A interface binding site used.
- **8GF9** — MemProtMD atomistic membrane simulation (DPPC bilayer, available for MD context if needed)

Known reference binders (in `documents/`): capsaicin, resiniferatoxin, capsazepine, AMG-517, A-425619, GRC-6211, JNJ-17203212, SB-366791.

## Implementation Notes for Developers

- **DB connection pattern** (used in all scripts): `psycopg2.connect(dbname="analgesics", user="labhund", host=os.environ.get("PGHOST", "/tmp"))` — Unix socket via `PGHOST`.
- **HDF5 index contract**: `compounds.id = k` → HDF5 row `k - 1`. This mapping is enforced throughout all scripts.
- **HDF5 dataset name**: the fingerprint dataset is `fps` (not `fingerprints`). Shape: `(304996291, 512)` uint8.
- **HDF5 gap rows**: The HDF5 has 304,996,291 rows but the DB has only 297,881,291 compounds (~2.3% gap). Gap rows are all-zero vectors left by failed PostgreSQL COPY transactions during bulk import. `init_diversity_sample.py` handles this with a validity mask (mmap'd bool array, built from DB IDs before forking workers) plus a secondary all-zero-vector safety check. Do not assume HDF5 row count == compound count.
- **HDF5 chunk layout**: `(10000, 512)` = 5 MB/chunk. Fancy (scatter) indexing over random rows is catastrophically slow — each non-contiguous row touches a separate 5 MB chunk. Always use contiguous slice reads (`f["fps"][start:end]`) and filter in-memory.
- **MPS is required** for `dock_batch.py` — the script checks for `/tmp/nvidia-mps/` and raises a clear error rather than silently degrading.
- **`dock_batch.py` uses `--filelist` batch mode** (redesigned 2026-02-20): 2 `batch_dock_worker` threads each process 100-compound batches via `autodock_gpu --filelist`. Filelist format: grid `.fld` on line 1, then one `.pdbqt` per line. Output XMLs: `{resnam}_{N}.xml` (1-indexed). Old per-compound `dock_worker` (6 threads) removed. CLI: `--dock-workers 2 --dock-batch-size 100` (replaced `--dock-threads 6`).
- **`score_library.py` uses `compression="lzf"`** — `lz4` is not available in the chem conda env (requires external HDF5 plugin). `lzf` is natively bundled with h5py, similar speed.
- **HPO schedule**: `hparam_tune_every_n_rounds: 3` in config → `al_loop.py` passes `--tune-hparams` at rounds 0, 3, 6, … Optuna studies persist in `optuna/{target}_surrogate.db` (SQLite). Best params carry forward between HPO rounds.
- **Fail threshold**: docking scores ≥ 0 kcal/mol excluded from training. Configurable via `fail_threshold` in config or `--fail-threshold` CLI flag. Failed docks stored as NULL in `docking_scores.score`.
- **All scripts are idempotent**: each checks for existing output before running and skips gracefully. Safe to restart the loop after interruption at any stage.
- **`al_loop.py` imports other scripts as Python modules** (not subprocess) — call their `main()` functions directly. Scripts must remain importable.
- **Logging**: all scripts append to `logs/al_round{N}.log` and mirror to stdout. Orchestrator also writes `logs/al_loop.log`.
- **Per-round metrics**: `models/metrics_{target}_round{N}.json` (val_rmse, n_train, best score) written by `train_surrogate.py`; appended to `logs/al_metrics.csv` by `al_loop.py`.

## Python Dependencies

RDKit, psycopg2, h5py, xgboost, meeko, optuna, pyyaml, scikit-learn. No requirements.txt — all assumed installed in the `chem` conda environment.
