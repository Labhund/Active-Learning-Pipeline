"""
train_surrogate.py — XGBoost surrogate model training with optional Optuna HPO.

Trains on all valid docking scores accumulated so far (all rounds).
Saves model JSON and, if tuning, best hyperparameter JSON.

Usage:
    python scripts/active_learning/train_surrogate.py \
        --target trpv1_5irz \
        --round 0 \
        --fp-file data/fingerprints/compounds.h5 \
        --out models/ \
        --fail-threshold 0.0 \
        [--tune-hparams] \
        [--optuna-trials 30]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import psycopg2
import xgboost as xgb

DB_NAME = "analgesics"
DB_USER = "labhund"


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def load_training_data(target: str, experiment_id: str, fail_threshold: float,
                       fp_file: Path):
    """
    Fetch valid docking scores from DB and corresponding FPs from HDF5.

    Returns
    -------
    fps    : (N, 4096) float32 — unpacked Morgan FPs
    scores : (N,) float32 — raw docking scores (kcal/mol)
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT compound_id, score FROM docking_scores "
                "WHERE target = %s AND experiment_id = %s "
                "AND score IS NOT NULL AND score < %s "
                "ORDER BY compound_id",
                (target, experiment_id, fail_threshold),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No valid docking scores found for target={target} experiment_id={experiment_id}"
        )

    compound_ids = np.array([r[0] for r in rows], dtype=np.int64)
    scores = np.array([r[1] for r in rows], dtype=np.float32)

    hdf5_rows = compound_ids - 1  # 0-based

    with h5py.File(fp_file, "r") as f:
        packed = f["fps"][hdf5_rows]  # (N, 512) uint8

    fps = np.unpackbits(packed, axis=1).astype(np.float32)  # (N, 4096)

    logging.info(
        "Training set: %d compounds | score range [%.2f, %.2f] kcal/mol",
        len(scores), scores.min(), scores.max(),
    )
    return fps, scores


def train_val_split(fps, scores, val_frac=0.1, seed=42):
    """Stratified train/val split by score percentile."""
    rng = np.random.default_rng(seed)
    N = len(scores)
    percentiles = np.argsort(np.argsort(scores)) / N  # rank → [0,1)
    # Stratify: pull val_frac from each decile
    val_mask = np.zeros(N, dtype=bool)
    decile_edges = np.linspace(0, 1, 11)
    for lo, hi in zip(decile_edges[:-1], decile_edges[1:]):
        in_decile = np.where((percentiles >= lo) & (percentiles < hi))[0]
        n_val = max(1, int(len(in_decile) * val_frac))
        chosen = rng.choice(in_decile, size=n_val, replace=False)
        val_mask[chosen] = True
    return fps[~val_mask], scores[~val_mask], fps[val_mask], scores[val_mask]


def find_latest_hparams(model_dir: Path, target: str, experiment_id: str, round_num: int):
    """Load most recent hparams JSON for this target+experiment (any previous round)."""
    candidates = sorted(model_dir.glob(f"hparams_{target}_{experiment_id}_round*.json"))
    if not candidates:
        return None
    # Pick the one with the highest round <= round_num
    best = None
    for path in candidates:
        stem = path.stem  # hparams_{target}_{experiment_id}_round{N}
        try:
            r = int(stem.rsplit("round", 1)[1])
            if r <= round_num:
                best = path
        except (ValueError, IndexError):
            continue
    return best


def run_optuna_hpo(fps, scores, n_trials: int, device: str, seed: int,
                   storage: str = None, study_name: str = None):
    """Run Optuna HPO. Returns best_params dict.

    If storage is provided (e.g. 'sqlite:///optuna/trpv1_8gfa_surrogate.db'),
    trials are persisted and the study resumes across restarts.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logging.error("optuna not installed. Run: mamba install -n chem optuna")
        raise

    from sklearn.model_selection import KFold

    X, y = fps, scores

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.3),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
            "tree_method": "hist",
            "device": device,
            "eval_metric": "rmse",
            "seed": seed,
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        rmses = []
        for train_idx, val_idx in kf.split(X):
            model = xgb.XGBRegressor(**params)
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            preds = model.predict(X[val_idx])
            rmse = float(np.sqrt(np.mean((preds - y[val_idx]) ** 2)))
            rmses.append(rmse)
        return float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        logging.info("Resuming Optuna study '%s' with %d existing trials", study_name, n_existing)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logging.info("Optuna best params: %s  RMSE=%.4f", best, study.best_value)
    return best


def main(args=None):
    parser = argparse.ArgumentParser(description="Train XGBoost surrogate")
    parser.add_argument("--target", default="trpv1_5irz")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--fp-file", default="data/fingerprints/compounds.h5")
    parser.add_argument("--out", default="models/")
    parser.add_argument("--fail-threshold", type=float, default=0.0)
    parser.add_argument("--tune-hparams", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--optuna-storage", default=None,
                        help="SQLite URL for persistent Optuna storage, e.g. "
                             "sqlite:///optuna/trpv1_8gfa_maxmin_init_surrogate.db")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    # Default XGBoost params (used if no saved hparams)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
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

    fp_file = Path(cfg.fp_file)
    model_dir = Path(cfg.out)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"surrogate_{cfg.target}_{exp_id}_round{al_round}.json"
    if model_path.exists():
        logging.info("Model already exists: %s — skipping training.", model_path)
        return

    fps, scores = load_training_data(cfg.target, exp_id, cfg.fail_threshold, fp_file)
    fps_train, scores_train, fps_val, scores_val = train_val_split(
        fps, scores, seed=cfg.seed
    )

    # Determine hyperparameters
    if cfg.tune_hparams:
        logging.info("Running Optuna HPO (%d trials)...", cfg.optuna_trials)
        study_name = f"{cfg.target}_{exp_id}_surrogate"
        best_params = run_optuna_hpo(
            fps, scores, cfg.optuna_trials, cfg.device, cfg.seed,
            storage=cfg.optuna_storage,
            study_name=study_name,
        )
        hparam_path = model_dir / f"hparams_{cfg.target}_{exp_id}_round{al_round}.json"
        hparam_path.write_text(json.dumps(best_params, indent=2))
        logging.info("Saved hparams: %s", hparam_path)
    else:
        latest_hp = find_latest_hparams(model_dir, cfg.target, exp_id, al_round)
        if latest_hp:
            best_params = json.loads(latest_hp.read_text())
            logging.info("Loaded hparams from %s", latest_hp)
        else:
            best_params = {
                "max_depth":        cfg.max_depth,
                "learning_rate":    cfg.learning_rate,
                "subsample":        cfg.subsample,
                "colsample_bytree": 0.1,   # ~410 of 4096 features; consistent with Optuna range
            }
            logging.info("Using default hparams: %s", best_params)

    # Train final model on full data, use val split for early stopping
    train_params = {
        "n_estimators": 5000,  # early stopping will select actual value
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "device": cfg.device,
        "eval_metric": "rmse",
        "seed": cfg.seed,
        **best_params,
    }
    # Remove keys that XGBRegressor doesn't accept directly
    train_params.pop("early_stopping_rounds", None)

    model = xgb.XGBRegressor(
        **train_params,
        early_stopping_rounds=50,
    )

    logging.info("Training final model on %d samples...", len(scores_train))
    model.fit(
        fps_train, scores_train,
        eval_set=[(fps_val, scores_val)],
        verbose=100,
    )

    # Evaluate on val set
    val_preds = model.predict(fps_val)
    val_rmse = float(np.sqrt(np.mean((val_preds - scores_val) ** 2)))
    best_score = float(scores.min())
    logging.info(
        "Round %d | Val RMSE=%.4f kcal/mol | Best docking score=%.2f kcal/mol",
        al_round, val_rmse, best_score,
    )

    model.save_model(str(model_path))
    logging.info("Saved model: %s", model_path)

    # Write metrics JSON for al_loop.py to pick up for the CSV
    metrics = {
        "val_rmse":           val_rmse,
        "n_train":            int(len(scores_train)),
        "n_val":              int(len(scores_val)),
        "best_docking_score": float(scores.min()),
    }
    metrics_path = model_dir / f"metrics_{cfg.target}_{exp_id}_round{al_round}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logging.info("Saved training metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
