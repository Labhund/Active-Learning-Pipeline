"""
watch_docking.py — Live docking progress table per round.

Prints best-1/10/100/1000 scores and progress for each round.
Designed for use with watch:

    watch -n 30 conda run -n chem python analysis/watch_docking.py \
        --target trpv1_8gfa --experiment-id random_init

Or run once:
    python analysis/watch_docking.py --experiment-id random_init
"""

import argparse
import os
from datetime import datetime

import psycopg2

DB_NAME = "analgesics"
DB_USER = "labhund"


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def fetch_progress(target: str, exp_id: str):
    """Returns list of dicts, one per round, with queued/docked/best scores."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            # Queued per round (al_batches)
            cur.execute(
                """
                SELECT round, count(*) as queued
                FROM al_batches
                WHERE target=%s AND experiment_id=%s
                GROUP BY round ORDER BY round
                """,
                (target, exp_id),
            )
            queued = {row[0]: row[1] for row in cur.fetchall()}

            # Docked counts per round
            cur.execute(
                """
                SELECT al_round,
                       count(*) AS docked,
                       count(*) FILTER (WHERE score IS NOT NULL) AS valid,
                       avg(score) FILTER (WHERE score IS NOT NULL) AS mean
                FROM docking_scores
                WHERE target=%s AND experiment_id=%s
                GROUP BY al_round ORDER BY al_round
                """,
                (target, exp_id),
            )
            docked_rows = {row[0]: row for row in cur.fetchall()}

            # Best-N per round via window function
            cur.execute(
                """
                WITH ranked AS (
                    SELECT al_round, score,
                        row_number() OVER (
                            PARTITION BY al_round ORDER BY score ASC
                        ) AS rn
                    FROM docking_scores
                    WHERE target=%s AND experiment_id=%s
                          AND score IS NOT NULL
                )
                SELECT al_round,
                    MIN(score)                                        AS best_1,
                    MAX(score) FILTER (WHERE rn = 10)                 AS best_10,
                    MAX(score) FILTER (WHERE rn = 100)                AS best_100,
                    MAX(score) FILTER (WHERE rn = 1000)               AS best_1000
                FROM ranked
                GROUP BY al_round
                ORDER BY al_round
                """,
                (target, exp_id),
            )
            best_rows = {row[0]: row for row in cur.fetchall()}

    finally:
        conn.close()

    rounds = sorted(set(queued) | set(docked_rows))
    rows = []
    for r in rounds:
        q = queued.get(r, 0)
        dr = docked_rows.get(r)
        br = best_rows.get(r)
        docked = dr[1] if dr else 0
        valid  = dr[2] if dr else 0
        mean   = dr[3] if dr else None
        best1    = br[1] if br else None
        best10   = br[2] if br else None
        best100  = br[3] if br else None
        best1000 = br[4] if br else None
        rows.append(dict(
            round=r, queued=q, docked=docked, valid=valid, mean=mean,
            best1=best1, best10=best10, best100=best100, best1000=best1000,
        ))
    return rows


def fmt(val, digits=2):
    if val is None:
        return "    —   "
    return f"{val:+.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="random_init")
    cfg = parser.parse_args()

    rows = fetch_progress(cfg.target, cfg.experiment_id)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"  {cfg.experiment_id} — {cfg.target}    {now}"
    sep = "─" * 88

    print()
    print(header)
    print(sep)
    print(
        f"  {'Round':>5}  {'Queued':>7}  {'Docked':>7}  {'Valid':>7}  "
        f"{'Pct':>6}  {'Best-1':>8}  {'Best-10':>8}  "
        f"{'Best-100':>9}  {'Best-1000':>10}  {'Mean':>8}"
    )
    print(sep)

    for r in rows:
        pct = f"{100*r['docked']/r['queued']:.1f}%" if r['queued'] else "  —  "
        print(
            f"  {r['round']:>5}  {r['queued']:>7,}  {r['docked']:>7,}  {r['valid']:>7,}  "
            f"  {pct:>5}  {fmt(r['best1']):>8}  {fmt(r['best10']):>8}  "
            f"{fmt(r['best100']):>9}  {fmt(r['best1000']):>10}  {fmt(r['mean']):>8}"
        )

    if not rows:
        print("  (no data yet)")

    print(sep)
    print()


if __name__ == "__main__":
    main()
