# Active Learning HTVS Pipeline

Active learning (AL) pipeline for high-throughput virtual screening of ~300M ZINC20 compounds against **TRPV1** (Transient Receptor Potential Vanilloid 1)

## Method

An iterative loop alternates between docking (AutoDock-GPU) and surrogate model training (XGBoost on 4096-bit Morgan fingerprints). Thompson sampling with a tunable Boltzmann temperature selects 24,000 compounds per round, focusing compute on the most promising chemical space while maintaining exploration.

```
Round 0: Diversity/random initialization → Dock 24K compounds
Round N: Train surrogate on all docked scores → Score full library →
         Thompson sample next 24K batch → Dock → Repeat
```

## Key Results (Phase 1)

Two 6-round experiments completed over the TRPV1 vanilloid binding site (8GFA, SB-366791 co-crystal):

| Experiment | Init strategy | Compounds docked | Best score (kcal/mol) | Final RMSE |
|---|---|---|---|---|
| `maxmin_init` | MaxMin Tanimoto diversity | 142,042 | -16.06 | 0.93 |
| `random_init` | Uniform random | 142,556 | -16.15 | 0.85 |

Thompson sampling dominates over initialization choice by round 2. Full analysis in `documents/`.

## Repository Structure

```
scripts/
  active_learning/          # Core AL pipeline (orchestrator, docking, training, scoring, selection)
  fingerprints/             # Morgan fingerprint generation
  ligand_prep/              # ZINC20 ingestion and preprocessing
  utils/                    # Receptor prep, DB setup, utilities
analysis/                   # Post-screening analysis scripts (UMAP, scaffolds, PAINS, etc.)
config/                     # Experiment YAML configurations
documents/                  # Lab reports and reference data
targets/trpv1/grids/        # AutoDock-GPU grid maps for 8GFA vanilloid site
targets/trpv1/structures/   # Prepared receptor PDBQT
```

Large data artifacts (fingerprint HDF5, docking outputs, trained models, PostgreSQL) are not tracked — see `.gitignore`. They are regenerable from the pipeline scripts and public ZINC20 data.

## Setup

**Hardware:** Designed for a single workstation with an NVIDIA GPU (tested on RTX 5080) and 128 GB RAM.

**Dependencies:** RDKit, psycopg2, h5py, xgboost, meeko, optuna, pyyaml, scikit-learn — all in a conda environment.

```bash
# 1. Create conda env (if not already present)
mamba create -n chem python=3.11 rdkit psycopg2 h5py xgboost meeko optuna pyyaml scikit-learn

# 2. Activate env and load DB config
conda activate chem
source env_db.sh

# 3. Start PostgreSQL and create AL tables (one-time)
db-start
psql -d analgesics -f scripts/utils/create_al_tables.sql

# 4. Start NVIDIA MPS for concurrent docking
nvidia-cuda-mps-control -d
```

## Running Experiments

```bash
# Full AL loop (6 rounds from scratch)
python scripts/active_learning/al_loop.py \
    --config config/al_loop_random.yaml --start-round 0 --rounds 6

# Resume after interruption (all scripts are idempotent)
python scripts/active_learning/al_loop.py \
    --config config/al_loop_random.yaml --start-round 3 --rounds 3 --skip-init

# Stop gracefully after current round
touch STOP_AL_random_init
```

See `CLAUDE.md` for detailed architecture notes, individual script CLIs, and monitoring queries.

## Docking Target

**TRPV1 vanilloid binding site** — chain D/A interface from PDB 8GFA (SB-366791 co-crystal). Receptor includes protein + annular lipid shell (8 cryo-EM resolved lipids). Grid: 48x44x60 points, 0.303 A spacing, 13 atom-type maps.

## License

Internal use — Lipin Lab.
