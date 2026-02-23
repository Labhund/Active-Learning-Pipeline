"""
pains_filter.py — PAINS alert filter on top-N docked compounds.

Flags pan-assay interference compounds (PAINS A+B+C) in the top-N hits
and produces a 2-panel figure:
  Panel 1 — Stacked bar per round: clean vs flagged (%)
  Panel 2 — Horizontal bar: top-15 PAINS alert type frequencies

Usage:
    python analysis/pains_filter.py \
        --target trpv1_8gfa --experiment-id maxmin_init \
        [--n 500] \
        [--out-csv analysis/pains_hits_trpv1_8gfa.csv] \
        [--out analysis/figures/pains_filter_trpv1_8gfa.png]
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

DB_NAME = "analgesics"
DB_USER = "labhund"

CAPSAICIN_SCORE = -8.77


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def round_palette(n_rounds: int):
    """Brown-to-orange palette matching the dashboard."""
    import matplotlib.colors as mcolors
    start = np.array(mcolors.to_rgb("#3d0c00"))
    end   = np.array(mcolors.to_rgb("#ffb347"))
    return [start + (end - start) * i / max(n_rounds - 1, 1) for i in range(n_rounds)]


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


def build_pains_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def main():
    parser = argparse.ArgumentParser(
        description="PAINS / Brenk filter on top-N docked compounds."
    )
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out", default=None)
    cfg = parser.parse_args()

    target = cfg.target
    exp_id = cfg.experiment_id

    out_png = Path(cfg.out) if cfg.out else (
        Path("analysis/figures") / f"pains_filter_{target}.png"
    )
    out_csv = Path(cfg.out_csv) if cfg.out_csv else (
        Path("analysis") / f"pains_hits_{target}.csv"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching top-{cfg.n} hits for target={target} exp={exp_id} ...")
    rows = fetch_top_hits(target, exp_id, cfg.n)

    catalog = build_pains_catalog()

    results = []           # (compound_id, zinc_id, score, al_round, pains_flag, alerts)
    alert_counter = Counter()
    n_parse_fail = 0

    for compound_id, zinc_id, smiles, score, al_round in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_parse_fail += 1
            results.append((compound_id, zinc_id, score, al_round, False, "PARSE_FAIL"))
            continue

        matches = catalog.GetMatches(mol)
        if matches:
            alert_names = "; ".join(m.GetDescription() for m in matches)
            for m in matches:
                alert_counter[m.GetDescription()] += 1
            results.append((compound_id, zinc_id, score, al_round, True, alert_names))
        else:
            results.append((compound_id, zinc_id, score, al_round, False, ""))

    # Write CSV
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["compound_id", "zinc_id", "score", "al_round",
                         "pains_flag", "alert_names"])
        for row in results:
            writer.writerow(row)
    print(f"CSV saved: {out_csv}")

    # Stats
    n_total = len(results)
    n_pains = sum(1 for r in results if r[4])
    pct_pains = 100.0 * n_pains / max(n_total, 1)

    # Per-round stats
    by_round = defaultdict(lambda: {"total": 0, "pains": 0})
    for compound_id, zinc_id, score, al_round, flag, alerts in results:
        by_round[al_round]["total"] += 1
        if flag:
            by_round[al_round]["pains"] += 1

    rounds = sorted(by_round.keys())
    n_rounds = len(rounds)
    colors = round_palette(n_rounds)

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#fafafa")
    for ax in axes:
        ax.set_facecolor("#fafafa")

    # Panel 1: Stacked bar per round (% clean vs flagged)
    ax = axes[0]
    bar_w = 0.6
    x = np.arange(n_rounds)

    clean_pcts, pains_pcts = [], []
    for r in rounds:
        total = by_round[r]["total"]
        p = by_round[r]["pains"]
        clean_pcts.append(100.0 * (total - p) / max(total, 1))
        pains_pcts.append(100.0 * p / max(total, 1))

    bars_clean = ax.bar(x, clean_pcts, bar_w, label="Clean", color="#9bc2cf", alpha=0.85)
    bars_pains = ax.bar(x, pains_pcts, bar_w, bottom=clean_pcts,
                        label="PAINS flagged", color="#d45f2e", alpha=0.85)

    # Annotate % PAINS
    for xi, pct in zip(x, pains_pcts):
        if pct > 0:
            ax.text(xi, clean_pcts[x.tolist().index(xi)] + pct + 0.5,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8,
                    color="#d45f2e", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Round {r}" for r in rounds], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of top hits", fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("PAINS flags by round (top-{} hits)".format(cfg.n),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate n per round
    for xi, r in zip(x, rounds):
        ax.text(xi, -4, f"n={by_round[r]['total']}", ha="center",
                va="top", fontsize=7.5, color="#666")

    # Panel 2: Top-15 PAINS alert types
    ax = axes[1]
    top_alerts = alert_counter.most_common(15)
    if top_alerts:
        alert_names_plot = [a[0] for a in top_alerts]
        alert_counts_plot = [a[1] for a in top_alerts]

        y = np.arange(len(alert_names_plot))
        ax.barh(y, alert_counts_plot, color="#d45f2e", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(alert_names_plot, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency in top hits", fontsize=10)
        ax.set_title("Top PAINS alert types", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

        # Annotate counts
        for yi, cnt in zip(y, alert_counts_plot):
            ax.text(cnt + 0.2, yi, str(cnt), va="center", fontsize=8, color="#666")
    else:
        ax.text(0.5, 0.5, "No PAINS alerts found", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="#666")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"PAINS filter — {target}  [{exp_id}]",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved: {out_png}")

    # Summary
    print(f"\nSummary  (top-{cfg.n} hits, target={target}, exp={exp_id})")
    print(f"  Total compounds analysed : {n_total}")
    print(f"  SMILES parse failures    : {n_parse_fail}")
    print(f"  PAINS flagged            : {n_pains}  ({pct_pains:.1f}%)")
    print(f"  Clean                    : {n_total - n_pains}  ({100-pct_pains:.1f}%)")
    if alert_counter:
        print(f"\n  Top-5 alert types:")
        for name, cnt in alert_counter.most_common(5):
            print(f"    {cnt:>4}×  {name}")


if __name__ == "__main__":
    main()
