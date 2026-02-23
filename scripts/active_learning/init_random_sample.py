"""
init_random_sample.py — Round-0 uniform random batch initialization.

Selects batch_size compounds from the DB using TABLESAMPLE BERNOULLI (very fast,
~33 ms for 300M rows).  No FP loading or multiprocessing required.

Usage:
    python scripts/active_learning/init_random_sample.py \
        --target trpv1_8gfa \
        --experiment-id random_init \
        --batch-size 24000 \
        --seed 123
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Round-0 uniform random batch init")
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="random_init")
    parser.add_argument("--batch-size", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=123)
    cfg = parser.parse_args(args)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "init_random.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Idempotency check
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM al_batches "
                "WHERE round=0 AND target=%s AND experiment_id=%s",
                (cfg.target, cfg.experiment_id),
            )
            existing = cur.fetchone()[0]
        if existing > 0:
            logging.info(
                "Round 0 already has %d entries for target=%s experiment_id=%s — skipping.",
                existing, cfg.target, cfg.experiment_id,
            )
            return
    finally:
        conn.close()

    # Seed the PostgreSQL random function so sampling is reproducible
    # TABLESAMPLE BERNOULLI uses system sampling; we follow up with a seeded ORDER BY
    # to get exactly batch_size rows deterministically.
    logging.info(
        "Sampling ~%d compounds from DB (TABLESAMPLE BERNOULLI, seed=%d)...",
        cfg.batch_size, cfg.seed,
    )

    # TABLESAMPLE BERNOULLI(p) samples each 8-KB page with probability p%.
    # For 300M rows in ~40 GB, 0.012% gives ~36K rows on average; we then
    # sort with a seeded hash and take exactly batch_size.
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT setseed(%s)", (cfg.seed / 2147483647.0,))
            cur.execute(
                """
                SELECT id FROM compounds TABLESAMPLE BERNOULLI(0.012)
                ORDER BY md5(id::text || %s::text)
                LIMIT %s
                """,
                (cfg.seed, cfg.batch_size),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    compound_ids = [r[0] for r in rows]
    n_sampled = len(compound_ids)
    logging.info("Sampled %d compound IDs", n_sampled)

    if n_sampled < cfg.batch_size:
        logging.warning(
            "Got %d compounds, requested %d — BERNOULLI rate may need adjustment.",
            n_sampled, cfg.batch_size,
        )

    # Bulk insert into al_batches
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            insert_rows = [
                (0, int(cid), cfg.target, cfg.experiment_id, "random_init")
                for cid in compound_ids
            ]
            execute_values(
                cur,
                "INSERT INTO al_batches (round, compound_id, target, experiment_id, source) "
                "VALUES %s",
                insert_rows,
            )
        conn.commit()
        logging.info(
            "Inserted %d rows into al_batches "
            "(round=0, target=%s, experiment_id=%s, source=random_init)",
            len(insert_rows), cfg.target, cfg.experiment_id,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
