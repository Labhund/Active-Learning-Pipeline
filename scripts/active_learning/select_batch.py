"""
select_batch.py — Softmax Thompson sampling without replacement.

Reads predicted scores, excludes already-docked/batched compounds within
this experiment (experiment-scoped exclusion), applies Boltzmann softmax,
and samples the next docking batch. Parallel experiments (e.g. maxmin_init
vs random_init) remain independent and can discover the same top hits.
Inserts into al_batches with the given experiment_id.

Temperature semantics:
  T > 0 : Boltzmann softmax — probs ∝ exp(−score / T).
           T=1.0 is the standard baseline (moderately exploitative).
           T→∞ approaches uniform random sampling.
  T = 0 : Greedy (pure exploitation) — deterministic top-K selection by
           predicted score. No stochasticity; seed is ignored. DB source
           label is 'greedy_select' instead of 'thompson_sample'.

Usage:
    python scripts/active_learning/select_batch.py \
        --target trpv1_8gfa \
        --experiment-id maxmin_init \
        --round 1 \
        --batch-size 24000 \
        --temperature 1.0 \
        --scores scores/predicted_trpv1_8gfa_maxmin_init_round0.h5 \
        --seed 42
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

DB_NAME = "analgesics"
DB_USER = "labhund"


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def main(args=None):
    parser = argparse.ArgumentParser(description="Thompson sampling batch selection")
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=24000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--scores", required=True, help="Predicted scores HDF5 from score_library.py")
    parser.add_argument("--seed", type=int, default=42)
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

    # Idempotency check (experiment-scoped)
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM al_batches "
                "WHERE round=%s AND target=%s AND experiment_id=%s",
                (al_round, cfg.target, exp_id),
            )
            existing = cur.fetchone()[0]
        if existing > 0:
            logging.info(
                "al_batches round=%d target=%s experiment_id=%s already has %d rows — skipping.",
                al_round, cfg.target, exp_id, existing,
            )
            return
    finally:
        conn.close()

    scores_path = Path(cfg.scores)
    if not scores_path.exists():
        logging.error("Scores file not found: %s", scores_path)
        sys.exit(1)

    # Load predicted scores
    with h5py.File(scores_path, "r") as f:
        predicted_scores = f["predicted_scores"][:]  # (N,) float32
    N = len(predicted_scores)
    logging.info("Loaded %d predicted scores", N)

    # Load or build compound validity mask to exclude HDF5 gap rows.
    # The HDF5 has ~2.3% rows with no matching compound in the DB (PostgreSQL COPY
    # sequence gaps from the bulk import). XGBoost predicts finite scores for all-zero
    # FP vectors, so gap rows get non-zero softmax probability and cause FK violations.
    valid_mask_path = Path("data/fingerprints/valid_mask.npy")
    if valid_mask_path.exists():
        valid_mask = np.load(valid_mask_path, mmap_mode="r")
        logging.info("Loaded validity mask: %d valid rows of %d total", int(valid_mask.sum()), N)
    else:
        logging.info("Building validity mask from DB (one-time, ~45s) ...")
        t_mask = time.time()
        conn = get_db_conn()
        try:
            with conn.cursor("id_cursor") as cur:
                cur.itersize = 500_000
                cur.execute("SELECT id FROM compounds ORDER BY id")
                all_ids = np.fromiter((row[0] for row in cur), dtype=np.int64, count=-1)
        finally:
            conn.close()
        valid_mask = np.zeros(N, dtype=bool)
        hdf5_rows = all_ids - 1
        valid_mask[hdf5_rows[hdf5_rows < N]] = True
        del all_ids, hdf5_rows
        valid_mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(valid_mask_path, valid_mask)
        logging.info(
            "Validity mask built in %.1f s (%d valid, %d gap rows) — saved to %s",
            time.time() - t_mask, int(valid_mask.sum()), int((~valid_mask).sum()), valid_mask_path,
        )

    # Load previously-docked/batched compound IDs — experiment-scoped exclusion so
    # that parallel experiments (e.g. maxmin_init vs random_init) remain independent
    # and can discover the same top hits without cross-contamination.
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT compound_id FROM docking_scores "
                "WHERE target=%s AND experiment_id=%s",
                (cfg.target, exp_id),
            )
            docked_ids = {row[0] for row in cur.fetchall()}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT compound_id FROM al_batches "
                "WHERE target=%s AND experiment_id=%s",
                (cfg.target, exp_id),
            )
            batched_ids = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    excluded_ids = docked_ids | batched_ids
    logging.info(
        "Excluding %d already-docked/batched compounds (%d docked, %d in batches) "
        "[experiment-scoped: %s]",
        len(excluded_ids), len(docked_ids), len(batched_ids), exp_id,
    )

    # Build exclusion mask (0-based HDF5 row = compound_id - 1)
    excluded_mask = np.zeros(N, dtype=bool)
    for cid in excluded_ids:
        idx = cid - 1
        if 0 <= idx < N:
            excluded_mask[idx] = True

    # Negate scores so higher neg-score = better predicted binder
    neg_scores = -predicted_scores.astype(np.float64)

    # Eligibility mask: valid DB compound, not already in this experiment, finite score
    eligible = valid_mask & ~excluded_mask & np.isfinite(neg_scores)
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        logging.error("No eligible compounds available — score file may be corrupt.")
        sys.exit(1)

    if cfg.batch_size > n_eligible:
        logging.warning(
            "Requested batch_size=%d > eligible=%d; reducing.",
            cfg.batch_size, n_eligible,
        )
        batch_size = n_eligible
    else:
        batch_size = cfg.batch_size

    if cfg.temperature == 0:
        # ---------------------------------------------------------------
        # Greedy mode (T=0): deterministic top-K selection by predicted score.
        # Ineligible rows are pushed to -inf so they sort last.
        # ---------------------------------------------------------------
        logging.info("Greedy mode (T=0): selecting top-%d of %d eligible compounds",
                     batch_size, n_eligible)
        sortable = neg_scores.copy()
        sortable[~eligible] = -np.inf
        # argsort(-sortable) ascending → sortable descending; ineligible rows (-inf)
        # become +inf and land at the end.
        selected_rows = np.argsort(-sortable, kind="stable")[:batch_size]
        source_label = "greedy_select"

    else:
        # ---------------------------------------------------------------
        # Boltzmann softmax Thompson sampling (T > 0)
        # ---------------------------------------------------------------
        logits = neg_scores / cfg.temperature
        valid_logits = logits[eligible]
        logits -= valid_logits.max()  # numerically stable shift

        probs = np.exp(logits)
        probs[~eligible] = 0.0
        probs[~np.isfinite(probs)] = 0.0

        total_prob = probs.sum()
        if total_prob == 0:
            logging.error("All softmax probabilities are zero — cannot sample.")
            sys.exit(1)
        probs /= total_prob

        logging.info("Thompson sampling (T=%.3g): %d compounds with non-zero weight",
                     cfg.temperature, int((probs > 0).sum()))

        rng = np.random.default_rng(cfg.seed + al_round)
        selected_rows = rng.choice(N, size=batch_size, replace=False, p=probs)
        source_label = "thompson_sample"

    compound_ids = (selected_rows + 1).tolist()  # HDF5 row → compound_id

    # Report selection quality
    selected_scores = predicted_scores[selected_rows]
    logging.info(
        "Selected %d compounds | pred score range [%.2f, %.2f] kcal/mol (mean %.2f) | source=%s",
        batch_size, selected_scores.min(), selected_scores.max(), selected_scores.mean(),
        source_label,
    )

    # Bulk insert into al_batches
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            rows = [
                (al_round, int(cid), cfg.target, exp_id, source_label)
                for cid in compound_ids
            ]
            execute_values(
                cur,
                "INSERT INTO al_batches (round, compound_id, target, experiment_id, source) "
                "VALUES %s",
                rows,
            )
        conn.commit()
        logging.info(
            "Inserted %d rows into al_batches (round=%d, experiment_id=%s, source=%s)",
            len(rows), al_round, exp_id, source_label,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
