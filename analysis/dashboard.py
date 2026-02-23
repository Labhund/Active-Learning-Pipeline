"""
dashboard.py — Discovery-curve dashboard for AL experiment comparison.

Produces a two-panel figure:
  Panel 1 — Discovery curve (primary comparison)
    X: cumulative molecules docked (within-round from dock_progress CSV,
       between-round reconstructed from DB)
    Y: docking score (kcal/mol)
    4 lines per experiment: best-1, best-10, best-100, best-1000

  Panel 2 — Surrogate quality
    X: round number
    Y: val RMSE (kcal/mol), one line per experiment

Usage:
    python analysis/dashboard.py \
        --target trpv1_8gfa \
        --experiment-ids maxmin_init random_init \
        [--round N]          # restrict to specific round for within-round view
        [--out analysis/figures/dashboard_trpv1_8gfa.png]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
except ImportError:
    print(
        "ERROR: matplotlib not installed. Run: mamba install -n chem matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed.", file=sys.stderr)
    sys.exit(1)

DB_NAME = "analgesics"
DB_USER = "labhund"

# Color palettes: 4 shades per experiment (best-1 darkest, best-1000 lightest)
EXP_COLORS = {
    "maxmin_init": ["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],  # orange-red shades
    "random_init": ["#08306b", "#2171b5", "#6baed6", "#bdd7e7"],  # blue shades
}
# Fallback colors for additional experiments
_FALLBACK_COLORS = [
    ["#00441b", "#238b45", "#74c476", "#c7e9c0"],
    ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"],
]


def get_exp_colors(exp_id: str, idx: int) -> list:
    if exp_id in EXP_COLORS:
        return EXP_COLORS[exp_id]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_progress_csv(exp_id: str, round_num: int, log_dir: Path):
    """
    Load dock_progress_{exp_id}_round{N}.csv.
    Returns list of dicts with keys: docked, best1, best10, best100, best1000.
    Returns empty list if file not found.
    """
    path = log_dir / f"dock_progress_{exp_id}_round{round_num}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                entry = {
                    "docked": int(row["docked"]),
                    "best1": float(row["best1"]) if row["best1"] else None,
                    "best10": float(row["best10"]) if row["best10"] else None,
                    "best100": float(row["best100"]) if row["best100"] else None,
                    "best1000": float(row["best1000"]) if row["best1000"] else None,
                }
                rows.append(entry)
            except (ValueError, KeyError):
                continue
    return rows


def load_db_scores(target: str, exp_id: str):
    """
    Query DB for all valid docking scores, grouped by round.
    Returns dict: {al_round: sorted_scores_array (ascending)}.
    """
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT al_round, score FROM docking_scores
                WHERE target=%s AND experiment_id=%s AND score IS NOT NULL
                ORDER BY al_round, score ASC
                """,
                (target, exp_id),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    by_round = {}
    for round_num, score in rows:
        by_round.setdefault(round_num, []).append(score)

    return {r: np.array(s, dtype=np.float32) for r, s in by_round.items()}


def build_cumulative_curve(scores_by_round: dict):
    """
    Build cumulative best-N discovery curve across rounds.

    Returns:
        cumulative_docked : list of x-values (cumulative compound count at each round end)
        best1, best10, best100, best1000 : lists of y-values per round end
        round_boundaries  : list of cumulative_docked values where a new round starts
    """
    rounds = sorted(scores_by_round.keys())
    cumulative = 0
    cum_docked = []
    best1 = []
    best10 = []
    best100 = []
    best1000 = []
    round_boundaries = [0]

    all_scores_so_far = []

    for r in rounds:
        s = scores_by_round[r]
        all_scores_so_far.extend(s.tolist())
        all_scores_so_far.sort()
        cumulative += len(s)
        cum_docked.append(cumulative)
        round_boundaries.append(cumulative)

        n = len(all_scores_so_far)
        best1.append(all_scores_so_far[0] if n >= 1 else None)
        best10.append(all_scores_so_far[9] if n >= 10 else None)
        best100.append(all_scores_so_far[99] if n >= 100 else None)
        best1000.append(all_scores_so_far[999] if n >= 1000 else None)

    return cum_docked, best1, best10, best100, best1000, round_boundaries[:-1]


def load_surrogate_metrics(target: str, exp_id: str, model_dir: Path):
    """
    Read metrics_{target}_{exp_id}_round{N}.json for all available rounds.
    Returns dict: {round: val_rmse}.
    """
    result = {}
    for path in sorted(model_dir.glob(f"metrics_{target}_{exp_id}_round*.json")):
        try:
            r = int(path.stem.rsplit("round", 1)[1])
            m = json.loads(path.read_text())
            if "val_rmse" in m:
                result[r] = m["val_rmse"]
        except (ValueError, IndexError, KeyError):
            continue
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_within_round_progress(
    ax, exp_id: str, round_num: int, log_dir: Path, colors: list, x_offset: int = 0
):
    """
    Overlay within-round progress from dock_progress CSV onto ax.
    x_offset: cumulative docked at the start of this round (for multi-round plots).
    """
    rows = load_progress_csv(exp_id, round_num, log_dir)
    if not rows:
        return

    x = [r["docked"] + x_offset for r in rows]
    for col, key, label in zip(
        colors,
        ["best1", "best10", "best100", "best1000"],
        ["best-1", "best-10", "best-100", "best-1000"],
    ):
        y = [r[key] for r in rows]
        # Filter None
        valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, color=col, linewidth=1.2, alpha=0.85)


def make_dashboard(
    target: str,
    experiment_ids: list,
    log_dir: Path,
    model_dir: Path,
    focus_round=None,
    out_path: Path = None,
):
    """Build and save the two-panel dashboard figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"AL discovery dashboard — {target}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # ---- Panel 1: Discovery curves ----
    ax1.set_title("Discovery curve (cumulative docked)", fontsize=11)
    ax1.set_xlabel("Cumulative molecules docked")
    ax1.set_ylabel("Docking score (kcal/mol)")

    legend_handles = []
    all_round_boundaries = set()

    for idx, exp_id in enumerate(experiment_ids):
        colors = get_exp_colors(exp_id, idx)
        scores_by_round = load_db_scores(target, exp_id)

        if not scores_by_round:
            continue

        cum_docked, b1, b10, b100, b1000, boundaries = build_cumulative_curve(
            scores_by_round
        )
        all_round_boundaries.update(boundaries)

        # Between-round curve (scatter + step at round endpoints)
        for col, ys, label in zip(
            colors,
            [b1, b10, b100, b1000],
            ["best-1", "best-10", "best-100", "best-1000"],
        ):
            valid = [(x, y) for x, y in zip(cum_docked, ys) if y is not None]
            if not valid:
                continue
            xs, yvs = zip(*valid)
            ax1.plot(
                xs,
                yvs,
                "o--",
                color=col,
                linewidth=1.5,
                markersize=5,
                alpha=0.9,
                zorder=3,
            )

        # Within-round overlay (from CSV)
        offset = 0
        for r in sorted(scores_by_round.keys()):
            plot_within_round_progress(ax1, exp_id, r, log_dir, colors, x_offset=offset)
            offset += len(scores_by_round[r])

        # Legend entry per experiment
        patch = mlines.Line2D(
            [], [], color=colors[0], linewidth=2, label=exp_id.replace("_", " ")
        )
        legend_handles.append(patch)

    # Round boundary lines
    for xb in sorted(all_round_boundaries):
        if xb > 0:
            ax1.axvline(x=xb, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Best-N legend (shared)
    style_handles = [
        mlines.Line2D([], [], color="black", linewidth=2, label="best-1"),
        mlines.Line2D(
            [], [], color="black", linewidth=1.5, label="best-10", alpha=0.75
        ),
        mlines.Line2D(
            [], [], color="black", linewidth=1.0, label="best-100", alpha=0.55
        ),
        mlines.Line2D(
            [], [], color="black", linewidth=0.7, label="best-1000", alpha=0.4
        ),
    ]
    ax1.legend(
        handles=legend_handles + style_handles,
        fontsize=8,
        loc="lower right",
        framealpha=0.8,
    )
    ax1.invert_yaxis()  # more negative = better binder → should trend downward
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ---- Panel 2: Surrogate RMSE ----
    ax2.set_title("Surrogate model quality", fontsize=11)
    ax2.set_xlabel("AL round")
    ax2.set_ylabel("Validation RMSE (kcal/mol)")

    for idx, exp_id in enumerate(experiment_ids):
        colors = get_exp_colors(exp_id, idx)
        rmse_by_round = load_surrogate_metrics(target, exp_id, model_dir)
        if not rmse_by_round:
            continue
        rounds = sorted(rmse_by_round.keys())
        rmses = [rmse_by_round[r] for r in rounds]
        ax2.plot(
            rounds,
            rmses,
            "o-",
            color=colors[0],
            linewidth=1.8,
            markersize=6,
            label=exp_id.replace("_", " "),
        )

    ax2.legend(fontsize=9, framealpha=0.8)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()

    if out_path is None:
        out_path = Path(f"analysis/figures/dashboard_{target}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="AL discovery curve dashboard")
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=["maxmin_init", "random_init"],
        help="One or more experiment_id values to compare",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="If set, restrict within-round progress to this round",
    )
    parser.add_argument("--log-dir", default="logs/")
    parser.add_argument("--model-dir", default="models/")
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: analysis/figures/dashboard_{target}.png)",
    )
    cfg = parser.parse_args(args)

    log_dir = Path(cfg.log_dir)
    model_dir = Path(cfg.model_dir)
    out_path = Path(cfg.out) if cfg.out else None

    make_dashboard(
        target=cfg.target,
        experiment_ids=cfg.experiment_ids,
        log_dir=log_dir,
        model_dir=model_dir,
        focus_round=cfg.round,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
