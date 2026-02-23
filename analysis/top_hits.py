"""
top_hits.py — 2D structure grid of top-N docked compounds.

Usage:
    python analysis/top_hits.py \
        --target trpv1_8gfa --experiment-id maxmin_init \
        [--n 50] [--mols-per-row 5] \
        [--out analysis/figures/top_hits_trpv1_8gfa.png]
"""

import argparse
import os
from pathlib import Path

import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

DB_NAME = "analgesics"
DB_USER = "labhund"

CAPSAICIN_SCORE = -8.77


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def fetch_top_hits(target: str, exp_id: str, n: int):
    """Fetch top-n hits by docking score (ascending = most negative first)."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ds.compound_id, c.zinc_id, c.smiles_protonated,
                       ds.score, ds.al_round
                FROM   docking_scores ds
                JOIN   compounds c ON ds.compound_id = c.id
                WHERE  ds.target = %s AND ds.experiment_id = %s
                       AND ds.score IS NOT NULL
                ORDER  BY ds.score ASC
                LIMIT  %s
                """,
                (target, exp_id, n),
            )
            return cur.fetchall()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="2D structure grid of top-N docked compounds."
    )
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of top hits to display (default: 50)")
    parser.add_argument("--mols-per-row", type=int, default=5)
    parser.add_argument("--out", default=None)
    cfg = parser.parse_args()

    target = cfg.target
    exp_id = cfg.experiment_id
    n_want = cfg.n

    out_path = Path(cfg.out) if cfg.out else (
        Path("analysis/figures") / f"top_hits_{target}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch extra rows to account for SMILES parse failures
    fetch_n = int(n_want * 1.2) + 10
    print(f"Fetching top-{fetch_n} hits for target={target} exp={exp_id} ...")
    rows = fetch_top_hits(target, exp_id, fetch_n)

    mols, legends = [], []
    n_failed = 0
    for compound_id, zinc_id, smiles, score, al_round in rows:
        if len(mols) >= n_want:
            break
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_failed += 1
            continue
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(f"R{al_round}  {score:.2f} kcal/mol\n{zinc_id}")

    print(f"  {len(mols)} valid structures  ({n_failed} SMILES failures skipped)")

    if not mols:
        print("No valid molecules to display — aborting.")
        return

    # Draw grid
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cfg.mols_per_row,
        subImgSize=(350, 280),
        legends=legends,
        returnPNG=True,
    )
    with open(out_path, "wb") as fh:
        fh.write(img)
    print(f"Saved: {out_path}")

    # Summary table
    print(f"\n{'Rank':>5}  {'Compound':>12}  {'Round':>6}  {'Score':>9}")
    print("-" * 40)
    for i, (compound_id, zinc_id, smiles, score, al_round) in enumerate(rows[:n_want], 1):
        print(f"{i:>5}  {zinc_id:>12}  {al_round:>6}  {score:>9.3f}")


if __name__ == "__main__":
    main()
