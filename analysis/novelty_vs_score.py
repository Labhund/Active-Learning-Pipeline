"""
novelty_vs_score.py — Tanimoto novelty vs docking score scatter.

For each round's batch, computes max Tanimoto similarity to the round-0 seed
set vs. docking score. Answers: "Is the surrogate guiding to novel chemistry
or re-sampling the seed set?"

Two-panel figure:
  Panel 1 — Scatter: x = max Tanimoto to R0, y = score. One series per round 1–5.
  Panel 2 — Violin per round: max Tanimoto distribution.

Tanimoto is computed via chunked numpy BLAS matmul (avoids HDF5 scatter reads).
Morgan FPs (radius=4, nBits=4096) generated on-the-fly from SMILES.

Usage:
    python analysis/novelty_vs_score.py \
        --target trpv1_8gfa --experiment-id maxmin_init \
        [--radius 4] [--n-bits 4096] [--chunk-size 1000] \
        [--out analysis/figures/novelty_vs_score_trpv1_8gfa.png]
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

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


def fetch_all_docked(target: str, exp_id: str):
    """Fetch all docked compounds with valid scores."""
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ds.compound_id, c.smiles_protonated, ds.score, ds.al_round
                FROM   docking_scores ds
                JOIN   compounds c ON ds.compound_id = c.id
                WHERE  ds.target = %s AND ds.experiment_id = %s
                       AND ds.score IS NOT NULL
                ORDER  BY ds.al_round, ds.compound_id
                """,
                (target, exp_id),
            )
            return cur.fetchall()
    finally:
        conn.close()


def smiles_to_fp(smiles: str, radius: int, n_bits: int,
                 _gen_cache: dict = {}) -> np.ndarray | None:
    """Convert SMILES to float32 binary Morgan FP vector, or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Cache generator (keyed by params) to avoid re-creating per molecule
    key = (radius, n_bits)
    if key not in _gen_cache:
        _gen_cache[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    gen = _gen_cache[key]
    bv = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return bv


def max_tanimoto_to_ref(query_fps: np.ndarray, ref_fps: np.ndarray,
                        chunk: int = 1000) -> np.ndarray:
    """
    Compute max Tanimoto similarity of each query FP to the reference set.

    Uses chunked BLAS matmul: Tanimoto(q, r) = dot(q, r) / (|q| + |r| - dot(q, r))
    where FPs are binary float32 arrays.

    query_fps : (N, D) float32
    ref_fps   : (M, D) float32
    returns   : (N,) float32  — max similarity to any ref compound
    """
    q_bits = query_fps.sum(axis=1)    # (N,)
    r_bits = ref_fps.sum(axis=1)      # (M,)
    max_sims = np.zeros(len(query_fps), dtype=np.float32)

    for i in range(0, len(query_fps), chunk):
        qc = query_fps[i:i + chunk]              # (chunk, D)
        inter = qc @ ref_fps.T                   # (chunk, M)
        union = q_bits[i:i + chunk, None] + r_bits[None, :] - inter
        max_sims[i:i + chunk] = (inter / np.maximum(union, 1e-8)).max(axis=1)

    return max_sims


def main():
    parser = argparse.ArgumentParser(
        description="Tanimoto novelty vs docking score scatter."
    )
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--n-bits", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--out", default=None)
    cfg = parser.parse_args()

    target = cfg.target
    exp_id = cfg.experiment_id

    out_path = Path(cfg.out) if cfg.out else (
        Path("analysis/figures") / f"novelty_vs_score_{target}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching all docked compounds for target={target} exp={exp_id} ...")
    rows = fetch_all_docked(target, exp_id)
    print(f"  Total rows: {len(rows):,}")

    # Compute Morgan FPs from SMILES
    print(f"Computing Morgan FPs (radius={cfg.radius}, nBits={cfg.n_bits}) ...")
    by_round = {}   # round → list of (score, fp_or_None)

    for compound_id, smiles, score, al_round in rows:
        fp = smiles_to_fp(smiles, cfg.radius, cfg.n_bits)
        by_round.setdefault(al_round, []).append((score, fp))

    rounds = sorted(by_round.keys())
    print(f"  Rounds found: {rounds}")

    if 0 not in by_round:
        print("ERROR: round 0 not found — cannot compute novelty relative to seed set.")
        return

    # Build round-0 reference FP matrix (only valid FPs)
    r0_entries = by_round[0]
    r0_fps_list = [fp for _, fp in r0_entries if fp is not None]
    if not r0_fps_list:
        print("ERROR: no valid FPs for round 0.")
        return
    ref_fps = np.stack(r0_fps_list, axis=0)  # (M, D)
    print(f"  Round-0 reference set: {len(ref_fps):,} compounds")

    # Compute max Tanimoto for each round (including R0 self-similarity for baseline)
    sim_by_round = {}
    score_by_round = {}

    for r in rounds:
        entries = by_round[r]
        fps_valid, scores_valid = [], []
        for score, fp in entries:
            if fp is not None:
                fps_valid.append(fp)
                scores_valid.append(score)

        if not fps_valid:
            continue

        query_fps = np.stack(fps_valid, axis=0)
        scores_arr = np.array(scores_valid, dtype=np.float32)

        print(f"  Computing max Tanimoto for round {r} ({len(query_fps):,} compounds) ...")

        if r == 0:
            # Self-similarity: exclude self-match by using max of all others
            # Compute all-vs-all then mask diagonal
            q_bits = query_fps.sum(axis=1)
            r_bits = ref_fps.sum(axis=1)
            max_sims = np.zeros(len(query_fps), dtype=np.float32)
            chunk = cfg.chunk_size
            for i in range(0, len(query_fps), chunk):
                qc = query_fps[i:i + chunk]
                inter = qc @ ref_fps.T
                union = q_bits[i:i + chunk, None] + r_bits[None, :] - inter
                tani = inter / np.maximum(union, 1e-8)
                # Mask self (diagonal): set to 0 so max excludes self
                for j in range(len(qc)):
                    global_idx = i + j
                    if global_idx < tani.shape[1]:
                        tani[j, global_idx] = 0.0
                max_sims[i:i + chunk] = tani.max(axis=1)
        else:
            max_sims = max_tanimoto_to_ref(query_fps, ref_fps, chunk=cfg.chunk_size)

        sim_by_round[r] = max_sims
        score_by_round[r] = scores_arr

    # Median R0 self-similarity (for vertical dashed reference)
    r0_self_median = float(np.median(sim_by_round[0])) if 0 in sim_by_round else None
    print(f"  R0 self-similarity median: {r0_self_median:.4f}")

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------
    non_zero_rounds = [r for r in rounds if r != 0 and r in sim_by_round]
    if not non_zero_rounds:
        print("No rounds beyond 0 — only R0 self-similarity computed. Exiting.")
        return

    palette_all = round_palette(len(rounds))
    # Map round → color; round 0 gets index 0
    round_to_color = {r: palette_all[i] for i, r in enumerate(rounds)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#fafafa")
    for ax in axes:
        ax.set_facecolor("#fafafa")

    # Panel 1: Scatter x=max_tanimoto, y=score, coloured by round
    ax = axes[0]
    for r in non_zero_rounds:
        sims = sim_by_round[r]
        scores = score_by_round[r]
        color = round_to_color[r]
        ax.scatter(sims, scores, s=2, alpha=0.3, color=color,
                   label=f"Round {r} (n={len(sims):,})", rasterized=True)

    # Reference lines
    ax.axhline(CAPSAICIN_SCORE, color="#888", lw=1.2, ls="--", alpha=0.7)
    ax.text(0.02, CAPSAICIN_SCORE + 0.1, "capsaicin", fontsize=8,
            color="#666", va="bottom", transform=ax.get_yaxis_transform())

    if r0_self_median is not None:
        ax.axvline(r0_self_median, color="#555", lw=1.2, ls=":", alpha=0.8)
        ax.text(r0_self_median + 0.005, ax.get_ylim()[0] if ax.get_ylim()[0] else -16,
                f"R0 self-sim\nmedian={r0_self_median:.3f}", fontsize=7.5,
                color="#555", va="bottom")

    ax.set_xlabel("Max Tanimoto similarity to Round-0 seed set", fontsize=10)
    ax.set_ylabel("Docking score (kcal/mol)", fontsize=10)
    ax.set_title("Chemical novelty vs docking score", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8, framealpha=0.6, markerscale=4, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1)

    # Panel 2: Violin per round (include R0 self-sim as baseline)
    ax = axes[1]
    violin_rounds = ([0] if 0 in sim_by_round else []) + non_zero_rounds
    violin_data = [sim_by_round[r] for r in violin_rounds]
    violin_labels = (
        [f"R0\n(self-sim)"] if 0 in sim_by_round else []
    ) + [f"Round {r}" for r in non_zero_rounds]

    if violin_data:
        parts = ax.violinplot(
            violin_data,
            positions=list(range(len(violin_rounds))),
            widths=0.65,
            showmedians=False,
            showextrema=False,
        )
        for i, (body, r) in enumerate(zip(parts["bodies"], violin_rounds)):
            color = round_to_color[r]
            body.set_facecolor(color)
            body.set_edgecolor("white")
            body.set_alpha(0.85)

        # Overlay IQR and median
        for i, r in enumerate(violin_rounds):
            sims = sim_by_round[r]
            p25, p50, p75 = np.percentile(sims, [25, 50, 75])
            ax.plot([i, i], [p25, p75], color="white", lw=2.5,
                    solid_capstyle="round", zorder=3)
            ax.scatter(i, p50, color="white", s=30, zorder=4,
                       linewidths=1.0, edgecolors="#444")

        if r0_self_median is not None:
            ax.axhline(r0_self_median, color="#555", lw=1.0, ls=":", alpha=0.7)

    ax.set_xticks(list(range(len(violin_rounds))))
    ax.set_xticklabels(violin_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Max Tanimoto to R0 seed set", fontsize=10)
    ax.set_title("Chemical novelty distribution per round", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#ddd", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate median novelty
    for i, r in enumerate(violin_rounds):
        med = np.median(sim_by_round[r])
        ax.text(i, 1.01, f"{med:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#555")

    fig.suptitle(
        f"Chemical novelty vs score — {target}  [{exp_id}]",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")

    # Summary table
    print(f"\n{'Round':>6}  {'N':>6}  {'Median Tani':>12}  {'Mean Tani':>10}  "
          f"{'% > 0.4':>8}  {'% > 0.6':>8}")
    print("-" * 60)
    for r in violin_rounds:
        sims = sim_by_round[r]
        label = "R0 self" if r == 0 else f"R{r}"
        pct04 = 100 * (sims > 0.4).mean()
        pct06 = 100 * (sims > 0.6).mean()
        print(f"{label:>6}  {len(sims):>6,}  {np.median(sims):>12.4f}  "
              f"{sims.mean():>10.4f}  {pct04:>7.1f}%  {pct06:>7.1f}%")


if __name__ == "__main__":
    main()
