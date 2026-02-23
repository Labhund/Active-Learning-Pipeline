"""
score_distributions.py — Per-round score distribution and enrichment analysis.

Three-panel figure:
  Panel 1 — Violin plot: docking score distribution per AL round
  Panel 2 — Hit enrichment: % of batch below score thresholds per round
  Panel 3 — Distribution shift: median + IQR over rounds

Usage:
    python analysis/score_distributions.py \
        --target trpv1_8gfa \
        --experiment-id maxmin_init \
        [--out analysis/figures/score_distributions_trpv1_8gfa.png]
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import psycopg2

DB_NAME = "analgesics"
DB_USER = "labhund"

# Reference scores (kcal/mol)
CAPSAICIN_SCORE = -8.77
HIT_THRESHOLDS = [-12.0, -13.0, -14.0, -15.0]
THRESHOLD_COLORS = ["#4393c3", "#2166ac", "#762a83", "#40004b"]
THRESHOLD_LABELS = ["< −12", "< −13", "< −14", "< −15"]


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def fetch_scores_by_round(target: str, exp_id: str):
    """Returns {round: np.array(scores)} for all valid scores."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT al_round, score FROM docking_scores "
                "WHERE target=%s AND experiment_id=%s AND score IS NOT NULL "
                "ORDER BY al_round",
                (target, exp_id),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    from collections import defaultdict
    by_round = defaultdict(list)
    for al_round, score in rows:
        by_round[al_round].append(score)
    return {r: np.array(v) for r, v in sorted(by_round.items())}


def round_palette(n_rounds: int):
    """Brown-to-orange palette matching the dashboard."""
    import matplotlib.colors as mcolors
    start = np.array(mcolors.to_rgb("#3d0c00"))   # dark brown
    end   = np.array(mcolors.to_rgb("#ffb347"))   # light orange
    return [start + (end - start) * i / max(n_rounds - 1, 1) for i in range(n_rounds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--out", default=None)
    cfg = parser.parse_args()

    target = cfg.target
    exp_id = cfg.experiment_id

    print(f"Fetching scores for target={target} experiment_id={exp_id} ...")
    scores_by_round = fetch_scores_by_round(target, exp_id)
    rounds = sorted(scores_by_round.keys())
    n_rounds = len(rounds)
    colors = round_palette(n_rounds)

    out_path = Path(cfg.out) if cfg.out else (
        Path("analysis/figures") / f"score_distributions_{target}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#fafafa")
    axes = fig.subplots(1, 3)
    for ax in axes:
        ax.set_facecolor("#fafafa")

    # -------------------------------------------------------------------------
    # Panel 1: Violin plot
    # -------------------------------------------------------------------------
    ax = axes[0]

    violin_data = [scores_by_round[r] for r in rounds]
    parts = ax.violinplot(
        violin_data,
        positions=rounds,
        widths=0.7,
        showmedians=False,
        showextrema=False,
    )

    for i, (body, color) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(0.85)

    # Overlay box (IQR + median)
    for i, r in enumerate(rounds):
        s = scores_by_round[r]
        p25, p50, p75 = np.percentile(s, [25, 50, 75])
        ax.plot([r, r], [p25, p75], color="white", lw=2.5, solid_capstyle="round", zorder=3)
        ax.scatter(r, p50, color="white", s=40, zorder=4, linewidths=1.2, edgecolors="#444")

    # Reference lines
    ax.axhline(CAPSAICIN_SCORE, color="#888", lw=1.2, ls="--", alpha=0.7)
    ax.text(rounds[-1] + 0.35, CAPSAICIN_SCORE + 0.08, "capsaicin", fontsize=7.5,
            color="#666", va="bottom", ha="right")

    for thresh, tc in zip(HIT_THRESHOLDS[:3], THRESHOLD_COLORS[:3]):
        ax.axhline(thresh, color=tc, lw=0.8, ls=":", alpha=0.55)

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Docking score (kcal/mol)", fontsize=10)
    ax.set_title("Score distribution per round", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="y", color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate n per round
    y_annot = ax.get_ylim()[1]  # bottom of inverted axis = worst score
    for r in rounds:
        ax.text(r, y_annot, f"n={len(scores_by_round[r]):,}", fontsize=7,
                ha="center", va="top", color="#666")

    # -------------------------------------------------------------------------
    # Panel 2: Hit enrichment (% below threshold per round)
    # -------------------------------------------------------------------------
    ax = axes[1]

    for thresh, tc, label in zip(HIT_THRESHOLDS, THRESHOLD_COLORS, THRESHOLD_LABELS):
        rates = [100.0 * (scores_by_round[r] < thresh).sum() / len(scores_by_round[r])
                 for r in rounds]
        ax.plot(rounds, rates, color=tc, marker="o", lw=2, ms=6, label=label)
        # Annotate last point
        ax.annotate(
            f"{rates[-1]:.1f}%",
            xy=(rounds[-1], rates[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=tc, va="center",
        )

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=9)
    ax.set_ylabel("% of batch below threshold", fontsize=10)
    ax.set_xlabel("AL round", fontsize=10)
    ax.set_title("Hit enrichment per round", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.6, loc="upper left")
    ax.grid(color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=0)

    # -------------------------------------------------------------------------
    # Panel 3: Median + IQR shift over rounds
    # -------------------------------------------------------------------------
    ax = axes[2]

    p10s, p25s, p50s, p75s, p90s = [], [], [], [], []
    bests = []
    for r in rounds:
        s = scores_by_round[r]
        p10, p25, p50, p75, p90 = np.percentile(s, [10, 25, 50, 75, 90])
        p10s.append(p10); p25s.append(p25); p50s.append(p50)
        p75s.append(p75); p90s.append(p90)
        bests.append(s.min())

    rounds_arr = np.array(rounds)

    ax.fill_between(rounds_arr, p10s, p90s, alpha=0.15, color="#8B2500", label="P10–P90")
    ax.fill_between(rounds_arr, p25s, p75s, alpha=0.30, color="#8B2500", label="IQR (P25–P75)")
    ax.plot(rounds_arr, p50s, color="#8B2500", lw=2.5, marker="o", ms=6, label="Median")
    ax.plot(rounds_arr, bests, color="#8B2500", lw=1.5, ls="--", marker="^", ms=6,
            alpha=0.7, label="Best")

    ax.axhline(CAPSAICIN_SCORE, color="#888", lw=1.2, ls="--", alpha=0.7)
    ax.text(rounds[-1] + 0.05, CAPSAICIN_SCORE + 0.08, "capsaicin", fontsize=7.5,
            color="#666", va="bottom")

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=9)
    ax.set_ylabel("Docking score (kcal/mol)", fontsize=10)
    ax.set_xlabel("AL round", fontsize=10)
    ax.set_title("Distribution shift over rounds", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9, framealpha=0.6, loc="lower left")
    ax.grid(color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # -------------------------------------------------------------------------
    # Global title + save
    # -------------------------------------------------------------------------
    fig.suptitle(
        f"AL score distributions — {target}  [{exp_id}]",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")

    # Print summary table
    print(f"\n{'Round':>6}  {'N':>6}  {'Best':>7}  {'Median':>8}  {'IQR':>10}  "
          f"{'<-12':>7}  {'<-13':>7}  {'<-14':>7}")
    print("-" * 68)
    for i, r in enumerate(rounds):
        s = scores_by_round[r]
        p25, p50, p75 = np.percentile(s, [25, 50, 75])
        h12 = f"{100*(s < -12).mean():.1f}%"
        h13 = f"{100*(s < -13).mean():.1f}%"
        h14 = f"{100*(s < -14).mean():.1f}%"
        print(f"{r:>6}  {len(s):>6,}  {s.min():>7.2f}  "
              f"{p50:>8.2f}  [{p25:.2f}, {p75:.2f}]  {h12:>7}  {h13:>7}  {h14:>7}")


if __name__ == "__main__":
    main()
