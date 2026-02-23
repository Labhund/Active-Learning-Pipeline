"""
al_loop.py — Active learning loop orchestrator.

Ties together all AL pipeline stages:
  Round 0: diversity init → dock → train
  Round N: score_library → select_batch → dock → train

Imports individual scripts as modules and calls their main() functions
for clean error handling and shared config.

Usage:
    python scripts/active_learning/al_loop.py \
        --config config/al_loop_maxmin.yaml \
        --start-round 0 \
        --rounds 5 \
        [--init-only]    # run diversity init + round-0 docking + training, then stop
        [--skip-init]    # assume al_batches already populated for start-round
"""

import argparse
import csv
import json
import logging
import sys
import time
import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Stage imports
# ---------------------------------------------------------------------------
# These must be importable — ensure scripts/active_learning/ is on sys.path
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import init_diversity_sample
import dock_batch
import train_surrogate
import score_library
import select_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def verify_prerequisites(cfg: dict):
    """Check that required files exist before starting the loop."""
    fp_file = Path(cfg["fp_file"])
    if not fp_file.exists():
        raise FileNotFoundError(f"FP file not found: {fp_file}")

    receptor = Path(cfg["receptor_pdbqt"])
    if not receptor.exists():
        raise FileNotFoundError(
            f"Receptor PDBQT not found: {receptor}\n"
            "Prepare it with: meeko receptor_preparation ..."
        )

    grid = Path(cfg["grid_maps"])
    if not grid.exists():
        raise FileNotFoundError(
            f"AutoDock-GPU grid map not found: {grid}\n"
            "Prepare the grid before running the AL loop."
        )

    logging.info("Prerequisite check passed.")


def log_round_summary(cfg: dict, al_round: int):
    """
    Query DB for docking stats, read training metrics JSON, log and append to CSV.

    Extended summary includes:
      - best_1, best_10, best_100, best_1000 docking scores for this round
      - cumulative n_screened across all rounds for this experiment
    """
    import psycopg2

    target = cfg["target"]
    exp_id = cfg["experiment_id"]

    conn = psycopg2.connect(
        dbname="analgesics",
        user="labhund",
        host=os.environ.get("PGHOST", "/tmp"),
    )
    try:
        with conn.cursor() as cur:
            # Per-round stats
            cur.execute(
                """
                SELECT
                    count(*) FILTER (WHERE score IS NOT NULL) AS n_valid,
                    count(*) FILTER (WHERE score IS NULL)     AS n_failed,
                    min(score)                                AS best_score,
                    avg(score)                                AS mean_score
                FROM docking_scores
                WHERE target = %s AND experiment_id = %s AND al_round = %s
                """,
                (target, exp_id, al_round),
            )
            row = cur.fetchone()
            n_valid, n_failed, best_score, mean_score = row

            # Best-N scores for this round (top 1000)
            cur.execute(
                """
                SELECT score FROM docking_scores
                WHERE target=%s AND experiment_id=%s AND al_round=%s
                  AND score IS NOT NULL
                ORDER BY score ASC LIMIT 1000
                """,
                (target, exp_id, al_round),
            )
            top_scores = [r[0] for r in cur.fetchall()]

            # Cumulative n_screened (all rounds, valid docks, this experiment)
            cur.execute(
                """
                SELECT count(*) FROM docking_scores
                WHERE target=%s AND experiment_id=%s AND score IS NOT NULL
                """,
                (target, exp_id),
            )
            n_screened = cur.fetchone()[0]
    finally:
        conn.close()

    def _best_n(n):
        return top_scores[n - 1] if len(top_scores) >= n else None

    best_1 = _best_n(1)
    best_10 = _best_n(10)
    best_100 = _best_n(100)
    best_1000 = _best_n(1000)

    # Read val_rmse and n_train written by train_surrogate
    val_rmse = n_train = None
    metrics_path = (
        Path(cfg.get("model_dir", "models/"))
        / f"metrics_{target}_{exp_id}_round{al_round}.json"
    )
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        val_rmse = m.get("val_rmse")
        n_train = m.get("n_train")

    logging.info(
        "=== Round %d summary | target=%s | exp=%s | "
        "valid=%s  failed=%s  best1=%.2f  mean=%.2f kcal/mol  "
        "best10=%s  best100=%s  best1000=%s  "
        "n_screened=%s  val_rmse=%s ===",
        al_round,
        target,
        exp_id,
        n_valid,
        n_failed,
        best_1 if best_1 is not None else float("nan"),
        mean_score if mean_score is not None else float("nan"),
        f"{best_10:.2f}" if best_10 is not None else "n/a",
        f"{best_100:.2f}" if best_100 is not None else "n/a",
        f"{best_1000:.2f}" if best_1000 is not None else "n/a",
        n_screened,
        f"{val_rmse:.4f}" if val_rmse is not None else "n/a",
    )

    # Append row to metrics CSV (experiment-scoped)
    metrics_csv = Path(f"logs/al_metrics_{exp_id}.csv")
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "round",
                    "target",
                    "experiment_id",
                    "n_docked",
                    "n_failed",
                    "best1_kcal_mol",
                    "best10_kcal_mol",
                    "best100_kcal_mol",
                    "best1000_kcal_mol",
                    "mean_score_kcal_mol",
                    "n_screened_cumulative",
                    "val_rmse_kcal_mol",
                    "n_train",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                al_round,
                target,
                exp_id,
                n_valid,
                n_failed,
                f"{best_1:.4f}" if best_1 is not None else "",
                f"{best_10:.4f}" if best_10 is not None else "",
                f"{best_100:.4f}" if best_100 is not None else "",
                f"{best_1000:.4f}" if best_1000 is not None else "",
                f"{mean_score:.4f}" if mean_score is not None else "",
                n_screened,
                f"{val_rmse:.4f}" if val_rmse is not None else "",
                n_train if n_train is not None else "",
            ]
        )


# ---------------------------------------------------------------------------
# Stage callers (build CLI args lists and delegate to module main())
# ---------------------------------------------------------------------------
def run_init(cfg: dict):
    logging.info("--- Stage: diversity init (round 0) ---")
    init_diversity_sample.main(
        [
            "--fp-file",
            cfg["fp_file"],
            "--target",
            cfg["target"],
            "--experiment-id",
            cfg["experiment_id"],
            "--batch-size",
            str(cfg["batch_size"]),
            "--threads",
            str(cfg.get("init_threads", 24)),
            "--subsample-per-worker",
            str(cfg.get("subsample_per_worker", 1_000_000)),
            "--seed",
            str(cfg.get("seed", 42)),
        ]
    )


def run_dock(cfg: dict, al_round: int):
    logging.info("--- Stage: docking (round %d) ---", al_round)
    dock_batch.main(
        [
            "--target",
            cfg["target"],
            "--experiment-id",
            cfg["experiment_id"],
            "--round",
            str(al_round),
            "--maps",
            cfg["grid_maps"],
            "--work-dir",
            cfg.get("work_dir", "work/docking"),
            "--prep-threads",
            str(cfg.get("prep_threads", 18)),
            "--dock-workers",
            str(cfg.get("dock_workers", 2)),
            "--dock-batch-size",
            str(cfg.get("dock_batch_size", 100)),
            "--fail-threshold",
            str(cfg.get("fail_threshold", 0.0)),
            "--nrun",
            str(cfg.get("autodock", {}).get("nrun", 20)),
            "--reporting-interval",
            str(cfg.get("reporting_interval", 500)),
        ]
    )


def run_train(cfg: dict, al_round: int, tune: bool):
    logging.info("--- Stage: train surrogate (round %d) ---", al_round)
    extra = ["--tune-hparams"] if tune else []
    xgb_cfg = cfg.get("xgboost", {})
    exp_id = cfg["experiment_id"]

    # Build Optuna SQLite storage URL from config
    optuna_dir = Path(cfg.get("optuna_dir", "optuna/"))
    optuna_dir.mkdir(parents=True, exist_ok=True)
    optuna_storage = f"sqlite:///{optuna_dir}/{cfg['target']}_{exp_id}_surrogate.db"

    train_surrogate.main(
        [
            "--target",
            cfg["target"],
            "--experiment-id",
            exp_id,
            "--round",
            str(al_round),
            "--fp-file",
            cfg["fp_file"],
            "--out",
            cfg.get("model_dir", "models/"),
            "--fail-threshold",
            str(cfg.get("fail_threshold", 0.0)),
            "--optuna-trials",
            str(cfg.get("optuna_trials", 30)),
            "--optuna-storage",
            optuna_storage,
            "--device",
            str(xgb_cfg.get("device", "cuda")),
            "--seed",
            str(cfg.get("seed", 42)),
        ]
        + extra
    )


def run_score_library(cfg: dict, al_round: int):
    logging.info("--- Stage: score library (round %d) ---", al_round)
    exp_id = cfg["experiment_id"]
    model_path = (
        Path(cfg.get("model_dir", "models/"))
        / f"surrogate_{cfg['target']}_{exp_id}_round{al_round - 1}.json"
    )
    out_path = (
        Path(cfg.get("score_dir", "scores/"))
        / f"predicted_{cfg['target']}_{exp_id}_round{al_round}.h5"
    )
    score_library.main(
        [
            "--target",
            cfg["target"],
            "--experiment-id",
            exp_id,
            "--round",
            str(al_round),
            "--fp-file",
            cfg["fp_file"],
            "--model",
            str(model_path),
            "--out",
            cfg.get("score_dir", "scores/"),
            "--chunk-size",
            str(100_000),
        ]
    )
    return out_path


def run_select_batch(cfg: dict, al_round: int, scores_path: Path):
    logging.info("--- Stage: select batch (round %d) ---", al_round)
    select_batch.main(
        [
            "--target",
            cfg["target"],
            "--experiment-id",
            cfg["experiment_id"],
            "--round",
            str(al_round),
            "--batch-size",
            str(cfg["batch_size"]),
            "--temperature",
            str(cfg.get("temperature", 1.0)),
            "--scores",
            str(scores_path),
            "--seed",
            str(cfg.get("seed", 42)),
        ]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="AL loop orchestrator")
    parser.add_argument("--config", default="config/al_loop.yaml")
    parser.add_argument("--start-round", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to run")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Run diversity init + round-0 docking + training, then stop",
    )
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip diversity init (assume al_batches already populated)",
    )
    cfg_args = parser.parse_args(args)

    cfg = load_config(cfg_args.config)

    exp_id = cfg.get("experiment_id", "maxmin_init")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"al_loop_{exp_id}.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info(
        "Config: target=%s  experiment_id=%s  batch_size=%d",
        cfg["target"],
        exp_id,
        cfg["batch_size"],
    )

    verify_prerequisites(cfg)

    start_round = cfg_args.start_round
    n_rounds = cfg_args.rounds
    tune_interval = cfg.get("hparam_tune_every_n_rounds", 3)

    # Stop-file sentinel (checked at top of each round)
    stop_file = Path(cfg.get("stop_file", f"STOP_AL_{exp_id}"))

    # -------------------------------------------------------------------------
    # Round 0: initialization + dock + train
    # --skip-init skips diversity init only; dock + train still run.
    # -------------------------------------------------------------------------
    if start_round == 0:
        if stop_file.exists():
            logging.info("Stop file %s detected — not starting round 0.", stop_file)
            return

        t0 = time.time()
        if not cfg_args.skip_init:
            run_init(cfg)
        run_dock(cfg, al_round=0)
        tune = 0 % tune_interval == 0
        run_train(cfg, al_round=0, tune=tune)
        log_round_summary(cfg, 0)
        logging.info("Round 0 complete in %.1f s", time.time() - t0)

        if cfg_args.init_only:
            logging.info("--init-only: stopping after round 0.")
            return

    # -------------------------------------------------------------------------
    # Subsequent rounds
    # -------------------------------------------------------------------------
    first_iter_round = max(1, start_round)
    for al_round in range(first_iter_round, start_round + n_rounds):
        # Check stop file at the top of each round
        if stop_file.exists():
            logging.info(
                "Stop file %s detected — halting cleanly after round %d.",
                stop_file,
                al_round - 1,
            )
            break

        t0 = time.time()
        logging.info("========== AL Round %d | exp=%s ==========", al_round, exp_id)

        # Score library using model from previous round
        scores_path = run_score_library(cfg, al_round)

        # Select next batch
        run_select_batch(cfg, al_round, scores_path)

        # Dock selected batch
        run_dock(cfg, al_round)

        # Train surrogate on all data so far
        tune = al_round % tune_interval == 0
        run_train(cfg, al_round, tune)

        log_round_summary(cfg, al_round)
        logging.info("Round %d complete in %.1f s", al_round, time.time() - t0)

    logging.info("AL loop finished: %d rounds complete.", n_rounds)


if __name__ == "__main__":
    main()
