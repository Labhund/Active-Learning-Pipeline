"""
scaffold_clusters.py — Murcko scaffold frequency in top-N docked compounds.

Checks for chemotype diversity vs convergence in the top hits.

Two-panel figure:
  Panel 1 (left)  — Horizontal bar chart: top-20 scaffolds by count,
                    coloured by best score (blue = most negative → grey)
  Panel 2 (right) — 2D scaffold grid: top-20 scaffold structures

Usage:
    python analysis/scaffold_clusters.py \
        --target trpv1_8gfa --experiment-id maxmin_init \
        [--n 1000] [--top-scaffolds 20] [--mols-per-row 4] \
        [--out analysis/figures/scaffold_clusters_trpv1_8gfa.png] \
        [--out-csv analysis/scaffold_clusters_trpv1_8gfa.csv]
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def shannon_entropy(counts):
    """Shannon entropy H of a count distribution (nats, then convert to bits)."""
    counts = np.array(counts, dtype=float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def main():
    parser = argparse.ArgumentParser(
        description="Murcko scaffold frequency of top-N docked compounds."
    )
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--top-scaffolds", type=int, default=20)
    parser.add_argument("--mols-per-row", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-csv", default=None)
    cfg = parser.parse_args()

    target = cfg.target
    exp_id = cfg.experiment_id

    out_png = Path(cfg.out) if cfg.out else (
        Path("analysis/figures") / f"scaffold_clusters_{target}.png"
    )
    out_csv = Path(cfg.out_csv) if cfg.out_csv else (
        Path("analysis") / f"scaffold_clusters_{target}.csv"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching top-{cfg.n} hits for target={target} exp={exp_id} ...")
    rows = fetch_top_hits(target, exp_id, cfg.n)

    # Scaffold grouping: scaffold_smi → {count, best_score, worst_score, rounds}
    scaffold_data = defaultdict(lambda: {
        "count": 0, "best_score": 0.0, "worst_score": -999.0, "rounds": set()
    })
    n_parse_fail = 0
    n_scaffold_fail = 0

    for compound_id, zinc_id, smiles, score, al_round in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_parse_fail += 1
            continue
        try:
            scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False
            )
        except Exception:
            n_scaffold_fail += 1
            continue

        d = scaffold_data[scaffold_smi]
        d["count"] += 1
        if score < d["best_score"]:
            d["best_score"] = score
        if score > d["worst_score"]:
            d["worst_score"] = score
        d["rounds"].add(al_round)

    print(f"  SMILES parse failures   : {n_parse_fail}")
    print(f"  Scaffold compute errors : {n_scaffold_fail}")

    # Sort by count descending, then by best_score ascending (most negative)
    sorted_scaffolds = sorted(
        scaffold_data.items(),
        key=lambda x: (-x[1]["count"], x[1]["best_score"]),
    )

    # Write CSV
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scaffold_smiles", "count", "best_score", "worst_score", "rounds"])
        for smi, d in sorted_scaffolds:
            writer.writerow([
                smi, d["count"],
                f"{d['best_score']:.3f}",
                f"{d['worst_score']:.3f}",
                "|".join(str(r) for r in sorted(d["rounds"])),
            ])
    print(f"CSV saved: {out_csv}")

    # Stats
    n_unique = len(scaffold_data)
    all_counts = [d["count"] for _, d in sorted_scaffolds]
    n_singletons = sum(1 for c in all_counts if c == 1)
    pct_singletons = 100.0 * n_singletons / max(n_unique, 1)
    H = shannon_entropy(all_counts)

    print(f"\nScaffold statistics (top-{cfg.n} hits):")
    print(f"  Unique scaffolds : {n_unique}")
    print(f"  Singletons       : {n_singletons}  ({pct_singletons:.1f}%)")
    print(f"  Shannon entropy  : H = {H:.3f} bits")

    # Top-K scaffolds for figure
    top_k = cfg.top_scaffolds
    top_entries = sorted_scaffolds[:top_k]

    # -------------------------------------------------------------------------
    # Figure — Panel 1: horizontal bar chart; Panel 2: scaffold structure grid
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#fafafa")
    ax_bar = fig.add_subplot(1, 2, 1)
    ax_bar.set_facecolor("#fafafa")

    counts_top = [d["count"] for _, d in top_entries]
    best_scores_top = [d["best_score"] for _, d in top_entries]

    # Colour by best score: deep blue (-16) → grey (-12); clamp
    score_min, score_max = -16.0, -12.0
    norm = mcolors.Normalize(vmin=score_min, vmax=score_max)
    cmap = plt.get_cmap("Blues_r")

    bar_colors = [cmap(norm(max(score_min, min(score_max, s)))) for s in best_scores_top]

    y = np.arange(len(top_entries))
    bars = ax_bar.barh(y, counts_top, color=bar_colors, alpha=0.9, height=0.7)
    ax_bar.set_yticks(y)

    # Label bars with count and best score
    for yi, cnt, best in zip(y, counts_top, best_scores_top):
        ax_bar.text(cnt + 0.2, yi, f"{cnt} ({best:.1f})", va="center", fontsize=8)

    # Y-axis labels: scaffold rank
    ax_bar.set_yticklabels([f"Scaffold {i+1}" for i in range(len(top_entries))], fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Count in top hits", fontsize=10)
    ax_bar.set_title(f"Top-{top_k} Murcko scaffolds (count, best score)",
                     fontsize=11, fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_bar, orientation="horizontal", pad=0.12,
                        fraction=0.04, aspect=30)
    cbar.set_label("Best score (kcal/mol)", fontsize=8)

    # Panel 2: scaffold grid
    scaffold_mols, scaffold_legends = [], []
    for i, (smi, d) in enumerate(top_entries):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            # Generic "no scaffold" placeholder
            mol = Chem.MolFromSmiles("C")
        AllChem.Compute2DCoords(mol)
        scaffold_mols.append(mol)
        scaffold_legends.append(f"S{i+1}: N={d['count']}  best={d['best_score']:.1f}")

    # Render scaffold grid as PNG bytes, embed in figure via AxesImage
    ax_grid = fig.add_subplot(1, 2, 2)
    ax_grid.set_facecolor("#fafafa")

    if scaffold_mols:
        grid_png = Draw.MolsToGridImage(
            scaffold_mols,
            molsPerRow=cfg.mols_per_row,
            subImgSize=(300, 240),
            legends=scaffold_legends,
            returnPNG=True,
        )
        import io
        from PIL import Image
        img_pil = Image.open(io.BytesIO(grid_png))
        ax_grid.imshow(img_pil)
        ax_grid.set_title(f"Top-{top_k} scaffold structures", fontsize=11, fontweight="bold")
    else:
        ax_grid.text(0.5, 0.5, "No scaffolds computed", ha="center", va="center",
                     transform=ax_grid.transAxes, fontsize=12)
    ax_grid.axis("off")

    fig.suptitle(
        f"Scaffold clustering — {target}  [{exp_id}]  (top-{cfg.n} hits)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved: {out_png}")

    # Top-10 scaffold table
    print(f"\n{'Rank':>4}  {'Count':>6}  {'Best':>7}  {'Worst':>7}  Rounds  Scaffold")
    print("-" * 75)
    for i, (smi, d) in enumerate(sorted_scaffolds[:10], 1):
        rounds_str = ",".join(str(r) for r in sorted(d["rounds"]))
        smi_trunc = smi[:40] + "…" if len(smi) > 41 else smi
        print(f"{i:>4}  {d['count']:>6}  {d['best_score']:>7.2f}  "
              f"{d['worst_score']:>7.2f}  {rounds_str:<6}  {smi_trunc}")


if __name__ == "__main__":
    main()
