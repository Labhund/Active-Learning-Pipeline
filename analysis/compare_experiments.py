"""
compare_experiments.py — Side-by-side scientific comparison of AL experiments.

4-panel figure (2×2, figsize≈14×10):

  Panel A (top-left): Side-by-side violin plots per round
      12 violins total (2 per round × 6 rounds); orange = maxmin, blue = random.
      Capsaicin reference line at −8.77 kcal/mol.

  Panel B (top-right): Score distribution shift (P10/P50/P90 bands)
      Shaded P10–P90, dashed median, solid mean per experiment.

  Panel C (bottom-left): Hit enrichment curves
      % of batch with score < −13/−14/−15 kcal/mol, both experiments.

  Panel D (bottom-right): Surrogate RMSE convergence
      One line per experiment; secondary Y axis shows ΔBest-1 per round.

Usage:
    python analysis/compare_experiments.py \\
        --target trpv1_8gfa \\
        --experiment-ids maxmin_init random_init \\
        [--out analysis/figures/compare_experiments_trpv1_8gfa.png]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
except ImportError:
    print("ERROR: matplotlib not installed.", file=sys.stderr)
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAPSAICIN_SCORE = -8.77
HIT_THRESHOLDS = [-13.0, -14.0, -15.0]
HIT_COLORS = ["#4393c3", "#2166ac", "#40004b"]
HIT_LABELS = ["< −13", "< −14", "< −15"]
HIT_LINESTYLES = ["-", "--", ":"]

EXP_COLORS = {
    "maxmin_init": "#d94801",   # orange-red (primary)
    "random_init": "#2171b5",   # blue (primary)
}
EXP_FILL = {
    "maxmin_init": "#fd8d3c",
    "random_init": "#6baed6",
}

DB_NAME = "analgesics"
DB_USER = "labhund"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


def fetch_raw_scores(target: str, exp_id: str):
    """Return {al_round: np.array(scores)} for all valid scores."""
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
    return {r: np.array(v, dtype=np.float32) for r, v in sorted(by_round.items())}


def fetch_stats(target: str, exp_id: str):
    """
    Fetch per-round percentiles, mean, best, and hit-rate stats via SQL.
    Returns list of dicts keyed by al_round.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT al_round,
                       percentile_cont(0.10) WITHIN GROUP (ORDER BY score) AS p10,
                       percentile_cont(0.25) WITHIN GROUP (ORDER BY score) AS p25,
                       percentile_cont(0.50) WITHIN GROUP (ORDER BY score) AS p50,
                       percentile_cont(0.75) WITHIN GROUP (ORDER BY score) AS p75,
                       percentile_cont(0.90) WITHIN GROUP (ORDER BY score) AS p90,
                       avg(score) AS mean_score,
                       min(score) AS best1,
                       count(*) FILTER (WHERE score < -13) * 100.0 / count(*) AS pct_13,
                       count(*) FILTER (WHERE score < -14) * 100.0 / count(*) AS pct_14,
                       count(*) FILTER (WHERE score < -15) * 100.0 / count(*) AS pct_15,
                       count(*) AS n
                FROM docking_scores
                WHERE target=%s AND experiment_id=%s AND score IS NOT NULL
                GROUP BY al_round ORDER BY al_round
                """,
                (target, exp_id),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    cols = ["al_round", "p10", "p25", "p50", "p75", "p90",
            "mean_score", "best1", "pct_13", "pct_14", "pct_15", "n"]
    return [{c: v for c, v in zip(cols, row)} for row in rows]


# ---------------------------------------------------------------------------
# Metrics CSV (reuse dashboard pattern)
# ---------------------------------------------------------------------------

def load_metrics_csv(exp_id: str, log_dir: Path):
    """
    Load al_metrics_{exp_id}.csv.
    Returns list of dicts with all columns.
    """
    path = log_dir / f"al_metrics_{exp_id}.csv"
    if not path.exists():
        return []
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


# ---------------------------------------------------------------------------
# Violin helper
# ---------------------------------------------------------------------------

def add_violin(ax, data: np.ndarray, pos: float, color: str,
               width: float = 0.32, alpha: float = 0.80):
    """
    Draw a single violin at position `pos`.
    Returns the Axes for chaining.
    """
    if len(data) < 2:
        return ax
    parts = ax.violinplot(
        [data],
        positions=[pos],
        widths=width,
        showmedians=False,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("white")
        body.set_alpha(alpha)

    # IQR bar + median dot
    p25, p50, p75 = np.percentile(data, [25, 50, 75])
    ax.plot([pos, pos], [p25, p75], color="white", lw=2.2,
            solid_capstyle="round", zorder=3)
    ax.scatter(pos, p50, color="white", s=28, zorder=4,
               linewidths=1.0, edgecolors="#444")
    return ax


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(target, experiment_ids, log_dir, out_path):
    exp_ids = experiment_ids
    n_exp = len(exp_ids)

    # Gather data
    raw_by_exp = {}
    stats_by_exp = {}
    metrics_by_exp = {}

    for exp_id in exp_ids:
        print(f"  Fetching raw scores for {exp_id} ...")
        raw_by_exp[exp_id] = fetch_raw_scores(target, exp_id)
        print(f"  Fetching stats for {exp_id} ...")
        stats_by_exp[exp_id] = fetch_stats(target, exp_id)
        metrics_by_exp[exp_id] = load_metrics_csv(exp_id, log_dir)

    # Determine rounds
    all_rounds = sorted(set(
        r for exp in exp_ids for r in raw_by_exp[exp].keys()
    ))
    n_rounds = len(all_rounds)

    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#fafafa")
    for ax in axes.flat:
        ax.set_facecolor("#fafafa")

    fig.suptitle(
        f"MaxMin vs Random initialization — {target}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # =========================================================================
    # Panel A (top-left): Side-by-side violins
    # =========================================================================
    ax = axes[0, 0]
    offsets = np.linspace(-0.19, 0.19, n_exp) if n_exp > 1 else [0.0]

    for i, exp_id in enumerate(exp_ids):
        color = EXP_COLORS.get(exp_id, f"C{i}")
        for r in all_rounds:
            data = raw_by_exp[exp_id].get(r)
            if data is None or len(data) < 2:
                continue
            add_violin(ax, data, pos=r + offsets[i], color=color, width=0.34)

    ax.axhline(CAPSAICIN_SCORE, color="#888", lw=1.2, ls="--", alpha=0.7)
    ax.text(all_rounds[-1] + 0.5, CAPSAICIN_SCORE + 0.1, "capsaicin",
            fontsize=7.5, color="#666", va="bottom", ha="right")

    for thresh, tc in zip(HIT_THRESHOLDS[:2], HIT_COLORS[:2]):
        ax.axhline(thresh, color=tc, lw=0.8, ls=":", alpha=0.5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=EXP_COLORS.get(e, f"C{i}"), label=e.replace("_", " "), alpha=0.85)
        for i, e in enumerate(exp_ids)
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.7, loc="lower left")

    ax.set_xticks(all_rounds)
    ax.set_xticklabels([f"Round {r}" for r in all_rounds],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Docking score (kcal/mol)", fontsize=10)
    ax.set_title("Score distribution per round (side-by-side)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="y", color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # =========================================================================
    # Panel B (top-right): P10/P50/P90 bands
    # =========================================================================
    ax = axes[0, 1]

    for i, exp_id in enumerate(exp_ids):
        stats = stats_by_exp[exp_id]
        if not stats:
            continue
        rounds_s = [s["al_round"] for s in stats]
        p10 = np.array([s["p10"] for s in stats], dtype=float)
        p50 = np.array([s["p50"] for s in stats], dtype=float)
        p90 = np.array([s["p90"] for s in stats], dtype=float)
        mean = np.array([s["mean_score"] for s in stats], dtype=float)

        color = EXP_COLORS.get(exp_id, f"C{i}")
        fill_color = EXP_FILL.get(exp_id, color)

        ax.fill_between(rounds_s, p10, p90, alpha=0.14, color=fill_color,
                        label=f"{exp_id.replace('_', ' ')} P10–P90")
        ax.plot(rounds_s, p50, lw=2.0, color=color, ls="--",
                marker="o", ms=4, alpha=0.85,
                label=f"{exp_id.replace('_', ' ')} median")
        ax.plot(rounds_s, mean, lw=1.5, color=color, ls="-",
                marker="s", ms=3.5, alpha=0.70)

    ax.axhline(CAPSAICIN_SCORE, color="#888", lw=1.2, ls="--", alpha=0.7)
    ax.text(all_rounds[-1] + 0.05, CAPSAICIN_SCORE + 0.05, "capsaicin",
            fontsize=7.5, color="#666", va="bottom")

    ax.set_xticks(all_rounds)
    ax.set_xticklabels([f"R{r}" for r in all_rounds], fontsize=9)
    ax.set_ylabel("Docking score (kcal/mol)", fontsize=10)
    ax.set_xlabel("AL round", fontsize=10)
    ax.set_title("Score distribution shift (P10/P50/P90 bands)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8, framealpha=0.7, ncol=2, loc="lower left")
    ax.grid(color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # =========================================================================
    # Panel C (bottom-left): Hit enrichment curves
    # =========================================================================
    ax = axes[1, 0]

    # Line style: solid/dashed for experiment, colour for threshold
    exp_linestyles = ["-", "--", ":", "-."]

    for i, exp_id in enumerate(exp_ids):
        stats = stats_by_exp[exp_id]
        if not stats:
            continue
        rounds_s = [s["al_round"] for s in stats]
        ls = exp_linestyles[i % len(exp_linestyles)]

        for thresh_col, tc, label in zip(
            ["pct_13", "pct_14", "pct_15"],
            HIT_COLORS,
            HIT_LABELS,
        ):
            pcts = [float(s[thresh_col]) for s in stats]
            line_label = f"{exp_id.replace('_', ' ')} {label}"
            ax.plot(rounds_s, pcts, lw=1.8, color=tc, ls=ls,
                    marker="o" if i == 0 else "s", ms=5, alpha=0.85,
                    label=line_label)
            # Annotate last value
            if pcts:
                ax.annotate(f"{pcts[-1]:.1f}%",
                            xy=(rounds_s[-1], pcts[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7, color=tc, va="center")

    ax.set_xticks(all_rounds)
    ax.set_xticklabels([f"R{r}" for r in all_rounds], fontsize=9)
    ax.set_ylabel("% of batch below threshold", fontsize=10)
    ax.set_xlabel("AL round", fontsize=10)
    ax.set_title("Hit enrichment per round", fontsize=11, fontweight="bold")

    # Custom legend: threshold colours + experiment line styles
    thresh_patches = [
        mlines.Line2D([], [], color=tc, lw=2, label=lbl)
        for tc, lbl in zip(HIT_COLORS, HIT_LABELS)
    ]
    exp_patches = [
        mlines.Line2D([], [], color="#555", lw=2,
                      ls=exp_linestyles[i], label=e.replace("_", " "))
        for i, e in enumerate(exp_ids)
    ]
    ax.legend(handles=thresh_patches + exp_patches,
              fontsize=7.5, framealpha=0.7, ncol=2, loc="upper left")
    ax.grid(color="#ddd", lw=0.6)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)

    # =========================================================================
    # Panel D (bottom-right): RMSE convergence + ΔBest-1
    # =========================================================================
    ax = axes[1, 1]
    ax2 = ax.twinx()

    for i, exp_id in enumerate(exp_ids):
        rows_m = metrics_by_exp[exp_id]
        if not rows_m:
            continue

        try:
            rounds_m = [int(r["round"]) for r in rows_m]
            rmse_vals = [float(r["val_rmse_kcal_mol"]) for r in rows_m]
            best1_vals = [float(r["best1_kcal_mol"]) for r in rows_m]
        except (KeyError, ValueError):
            continue

        color = EXP_COLORS.get(exp_id, f"C{i}")
        label = exp_id.replace("_", " ")

        # RMSE on left axis
        ax.plot(rounds_m, rmse_vals, "o-", color=color, lw=2.0,
                ms=6, label=label)

        # ΔBest-1 per round on right axis (improvement each round)
        delta_best1 = [None] + [
            best1_vals[j] - best1_vals[j - 1]
            for j in range(1, len(best1_vals))
        ]
        valid = [(r, d) for r, d in zip(rounds_m, delta_best1) if d is not None]
        if valid:
            rx, dy = zip(*valid)
            ax2.plot(rx, dy, color=color, lw=1.2, ls="--",
                     marker="^", ms=4, alpha=0.55,
                     label=f"{label} ΔBest-1")

    ax.set_xticks(all_rounds)
    ax.set_xticklabels([f"R{r}" for r in all_rounds], fontsize=9)
    ax.set_ylabel("Val RMSE (kcal/mol)", fontsize=10)
    ax.set_xlabel("AL round", fontsize=10)
    ax.set_title("Surrogate RMSE + best-score improvement",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.7, loc="upper right")
    ax.grid(color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax2.set_ylabel("ΔBest-1 per round (kcal/mol)", fontsize=9, color="#777")
    ax2.tick_params(axis="y", labelcolor="#777", labelsize=8)
    ax2.axhline(0, color="#ccc", lw=0.8, ls="--")
    ax2.spines[["top"]].set_visible(False)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_comparison_table(experiment_ids, stats_by_exp, metrics_by_exp):
    header = (f"{'Exp':>20}  {'R':>3}  {'N':>7}  {'Best-1':>7}  "
              f"{'Median':>7}  {'Mean':>7}  {'<−14%':>7}  {'<−15%':>7}")
    print("\n" + "=" * len(header))
    print("Score comparison table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for exp_id in experiment_ids:
        stats = stats_by_exp[exp_id]
        for s in stats:
            print(
                f"{exp_id:>20}  {int(s['al_round']):>3}  {int(s['n']):>7,}  "
                f"{float(s['best1']):>7.2f}  {float(s['p50']):>7.2f}  "
                f"{float(s['mean_score']):>7.2f}  "
                f"{float(s['pct_14']):>6.1f}%  {float(s['pct_15']):>6.1f}%"
            )
        print()

    print("\n--- RMSE ---")
    print(f"{'Exp':>20}  {'R':>3}  {'RMSE':>7}  {'Best-1':>7}")
    print("-" * 42)
    for exp_id in experiment_ids:
        for row in metrics_by_exp.get(exp_id, []):
            try:
                print(f"{exp_id:>20}  {int(row['round']):>3}  "
                      f"{float(row['val_rmse_kcal_mol']):>7.4f}  "
                      f"{float(row['best1_kcal_mol']):>7.2f}")
            except (KeyError, ValueError):
                pass
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-ids", nargs="+",
                        default=["maxmin_init", "random_init"])
    parser.add_argument("--log-dir", default="logs/")
    parser.add_argument("--out", default=None)
    cfg = parser.parse_args(args)

    target = cfg.target
    exp_ids = cfg.experiment_ids
    log_dir = Path(cfg.log_dir)
    out_path = (Path(cfg.out) if cfg.out else
                Path(f"analysis/figures/compare_experiments_{target}.png"))

    print(f"Compare experiments: {exp_ids}")
    print(f"Target: {target}")

    # Fetch stats for summary table
    stats_by_exp = {}
    metrics_by_exp = {}
    for exp_id in exp_ids:
        stats_by_exp[exp_id] = fetch_stats(target, exp_id)
        metrics_by_exp[exp_id] = load_metrics_csv(exp_id, log_dir)

    print_comparison_table(exp_ids, stats_by_exp, metrics_by_exp)

    print(f"\nGenerating figure → {out_path} ...")
    make_figure(target, exp_ids, log_dir, out_path)


if __name__ == "__main__":
    main()
