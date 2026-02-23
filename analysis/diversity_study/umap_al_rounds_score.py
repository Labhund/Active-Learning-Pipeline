#!/usr/bin/env python3
"""
Interactive UMAP score map — mean/best/count per bin, by AL round.

Slider: R0 – R5 – All
Toggle: Mean | Best | Count  (injected HTML buttons, stateful JS)

Colour (Mean/Best): bright yellow = best binding  ·  dark purple = poor binding
Colour (Count):     yellow = many compounds  ·  dark purple = sparse coverage

Requires cache from umap_al_rounds.py:
    cache/emb_al_{exp_id}_{target}.npy
    cache/al_meta_{exp_id}_{target}.csv
    cache/emb_bg_rev_{target}.npy

Usage:
    python analysis/diversity_study/umap_al_rounds_score.py \\
        --target trpv1_8gfa --experiment-id maxmin_init \\
        [--bins 120] [--min-count 2] [--score-vmax -9.0]
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR  = Path(__file__).resolve().parent
CACHE_DIR   = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
LOG         = logging.getLogger(__name__)
ALL_KEY     = "All"
CAP_SCORE   = -8.77
DIV_ID      = "umap_score_div"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target",        default="trpv1_8gfa")
    p.add_argument("--experiment-id", default="maxmin_init")
    p.add_argument("--bins",          type=int,   default=120)
    p.add_argument("--min-count",     type=int,   default=2,
                   help="Min compounds per bin for mean/best display (default 2)")
    p.add_argument("--bg-subsample",  type=int,   default=30_000)
    p.add_argument("--score-vmin",    type=float, default=None,
                   help="Colorscale lower bound — best score (default: global min)")
    p.add_argument("--score-vmax",    type=float, default=None,
                   help="Colorscale upper bound — worst shown (default: P95 of scores)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out",           type=Path,  default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_al_cache(target, exp_id):
    emb_path  = CACHE_DIR / f"emb_al_{exp_id}_{target}.npy"
    meta_path = CACHE_DIR / f"al_meta_{exp_id}_{target}.csv"
    if not emb_path.exists() or not meta_path.exists():
        LOG.error("Cache missing — run umap_al_rounds.py first:\n  %s\n  %s",
                  emb_path, meta_path)
        sys.exit(1)
    al_emb = np.load(emb_path)
    with open(meta_path, newline="") as f:
        meta = list(csv.DictReader(f))
    scores    = np.array([float(r["score"])   for r in meta], dtype=np.float64)
    al_rounds = np.array([int(r["al_round"])  for r in meta], dtype=np.int32)
    zinc_ids  = [r["zinc_id"] for r in meta]
    return al_emb, scores, al_rounds, zinc_ids


def load_bg(target):
    p = CACHE_DIR / f"emb_bg_rev_{target}.npy"
    if not p.exists():
        LOG.error("Background embeddings not found: %s", p); sys.exit(1)
    return np.load(p)


# ---------------------------------------------------------------------------
# Bin edges
# ---------------------------------------------------------------------------

def shared_edges(emb_bg, al_emb, bins):
    all_x = np.concatenate([emb_bg[:, 0], al_emb[:, 0]])
    all_y = np.concatenate([emb_bg[:, 1], al_emb[:, 1]])
    mx = (all_x.max() - all_x.min()) * 0.02
    my = (all_y.max() - all_y.min()) * 0.02
    xe = np.linspace(all_x.min() - mx, all_x.max() + mx, bins + 1)
    ye = np.linspace(all_y.min() - my, all_y.max() + my, bins + 1)
    return xe, ye


# ---------------------------------------------------------------------------
# Score/count heatmaps (mean, best, count) in one pass
# ---------------------------------------------------------------------------

def compute_bin_stats(emb, scores, xedges, yedges, min_count):
    """
    Returns (mean_s, best_s, count_f) — all shape (n_x_bins, n_y_bins).
    Bins with count < min_count → NaN in mean_s and best_s.
    Count bins with count == 0 → NaN in count_f.
    """
    nx = len(xedges) - 1
    ny = len(yedges) - 1

    ix = np.clip(np.digitize(emb[:, 0], xedges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(emb[:, 1], yedges) - 1, 0, ny - 1)

    sum_s = np.zeros((nx, ny), dtype=np.float64)
    min_s = np.full((nx, ny), np.inf, dtype=np.float64)
    cnt   = np.zeros((nx, ny), dtype=np.int32)

    np.add.at(sum_s, (ix, iy), scores)
    np.minimum.at(min_s, (ix, iy), scores)
    np.add.at(cnt, (ix, iy), 1)

    valid  = cnt >= min_count
    mean_s = np.where(valid, sum_s / np.maximum(cnt, 1), np.nan)
    best_s = np.where(valid, min_s, np.nan)
    count_f = np.where(cnt > 0, cnt.astype(float), np.nan)

    return mean_s, best_s, count_f


def compute_all_slots(al_emb, scores, al_rounds, xedges, yedges, min_count):
    """
    Returns:
        slots    : list of slot keys in slider order  [0, 1, ..., N, ALL_KEY]
        mean_map : dict slot → (nx, ny) float64 mean score
        best_map : dict slot → (nx, ny) float64 best score
        count_map: dict slot → (nx, ny) float64 count
    """
    rounds    = sorted(set(al_rounds.tolist()))
    slots     = rounds + [ALL_KEY]
    mean_map  = {}
    best_map  = {}
    count_map = {}

    for key in slots:
        if key == ALL_KEY:
            mask = np.ones(len(al_emb), dtype=bool)
        else:
            mask = al_rounds == key
        m, b, c = compute_bin_stats(al_emb[mask], scores[mask], xedges, yedges, min_count)
        mean_map[key]  = m
        best_map[key]  = b
        count_map[key] = c

        n_bins = int((~np.isnan(m)).sum())
        sc = scores[mask]
        LOG.info("  Slot %-4s: %6d compounds, %4d bins — mean=%.2f best=%.2f kcal/mol",
                 key, mask.sum(), n_bins, sc.mean(), sc.min())

    return slots, mean_map, best_map, count_map


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def arr_to_json_z(arr2d):
    """
    Transpose numpy (n_x, n_y) → (n_y, n_x) and serialise to list-of-lists.
    NaN → None (becomes null in JSON, rendered as transparent by Plotly).
    Round to 3 dp to limit file size.
    """
    z = arr2d.T   # Plotly Heatmap: z[row=y, col=x]
    return [[None if np.isnan(v) else round(float(v), 3) for v in row]
            for row in z]


# ---------------------------------------------------------------------------
# Build JavaScript post-script
# ---------------------------------------------------------------------------

def make_post_script(slots, mean_map, best_map, count_map,
                     score_vmin, score_vmax, count_vmax,
                     al_rounds, scores):
    """
    Returns JavaScript string that:
      - Embeds all z arrays for all (stat × slot) combinations
      - Adds Mean | Best | Count buttons above the plot
      - Hooks plotly_sliderchange to update heatmap z + colorscale + title
      - Buttons and slider maintain independent state, composed in refreshPlot()
    """
    # ---- pre-serialise all z arrays: [stat_idx][slot_idx] ----
    stat_z = []
    for stat_map in [mean_map, best_map, count_map]:
        slot_z = [arr_to_json_z(stat_map[k]) for k in slots]
        stat_z.append(slot_z)

    # ---- titles: [stat_idx][slot_idx] ----
    def slot_title(key, stat_label):
        if key == ALL_KEY:
            n    = len(al_rounds)
            best = float(scores.min())
            mean = float(scores.mean())
            slabel = "All rounds combined"
        else:
            mask   = al_rounds == key
            n      = int(mask.sum())
            best   = float(scores[mask].min())
            mean   = float(scores[mask].mean())
            slabel = f"Round {key}"
        unit = "kcal/mol" if stat_label != "Count" else "compounds/bin"
        return (
            f"{stat_label} docking score per UMAP bin — {{target}} [{{exp_id}}]<br>"
            f"<sup>{slabel}: n={n:,}  best={best:.2f}  mean={mean:.2f} kcal/mol  |  "
            f"capsaicin ref = {CAP_SCORE} kcal/mol</sup>"
        )

    stat_labels = ["Mean", "Best", "Count"]
    titles = [
        [slot_title(k, lbl) for k in slots]
        for lbl in stat_labels
    ]

    # ---- colorscale configs: one per stat ----
    cs_configs = [
        {"cs": "Plasma",  "rev": True,  "zmin": score_vmin, "zmax": score_vmax,
         "cbTitle": "mean score<br>(kcal/mol)"},
        {"cs": "Plasma",  "rev": True,  "zmin": score_vmin, "zmax": score_vmax,
         "cbTitle": "best score<br>(kcal/mol)"},
        {"cs": "Viridis", "rev": False, "zmin": 0,          "zmax": count_vmax,
         "cbTitle": "compounds<br>per bin"},
    ]

    # ---- serialise to JS ----
    js_z      = json.dumps(stat_z)
    js_titles = json.dumps(titles)
    js_cs     = json.dumps(cs_configs)

    return f"""
// ──────────────────────────────────────────────────────────
//  UMAP score map — combined round slider + stat toggle
// ──────────────────────────────────────────────────────────
(function() {{
    var gd = document.getElementById('{DIV_ID}');

    // Pre-computed data: [statIdx][slotIdx]
    var ALL_Z      = {js_z};
    var ALL_TITLES = {js_titles};
    var ALL_CS     = {js_cs};

    var currentStat = 0;   // 0=mean 1=best 2=count
    var currentSlot = 0;   // 0=R0 … N-1=All

    // ── Update heatmap trace (index 1) ───────────────────
    function refreshPlot() {{
        var cs = ALL_CS[currentStat];
        Plotly.restyle(gd,
            {{
                z: [ALL_Z[currentStat][currentSlot]],
                colorscale:  [cs.cs],
                reversescale:[cs.rev],
                zmin:        [cs.zmin],
                zmax:        [cs.zmax],
                'colorbar.title.text': [cs.cbTitle]
            }},
            [1]   // trace index 1 = heatmap
        );
        Plotly.relayout(gd, {{'title.text': ALL_TITLES[currentStat][currentSlot]}});
    }}

    // ── Slider → update slot ─────────────────────────────
    gd.on('plotly_sliderchange', function(e) {{
        currentSlot = e.slider.active;
        refreshPlot();
    }});

    // ── Inject stat buttons above the plot ───────────────
    var STAT_LABELS = ['Mean', 'Best', 'Count'];
    var wrapper = document.createElement('div');
    wrapper.style.cssText = [
        'text-align: center',
        'padding: 6px 0 2px 0',
        'font-family: Arial, sans-serif'
    ].join('; ');

    var lbl = document.createElement('span');
    lbl.textContent = 'Metric: ';
    lbl.style.cssText = 'color:#aaa; font-size:13px; margin-right:6px;';
    wrapper.appendChild(lbl);

    STAT_LABELS.forEach(function(name, i) {{
        var btn = document.createElement('button');
        btn.id          = 'statbtn_' + i;
        btn.textContent = name;
        btn.style.cssText = [
            'margin: 2px 5px',
            'padding: 5px 16px',
            'border-radius: 5px',
            'cursor: pointer',
            'font-size: 13px',
            'font-weight: ' + (i === 0 ? 'bold' : 'normal'),
            'border: 1px solid #666',
            'background: ' + (i === 0 ? '#ffb347' : '#252545'),
            'color: ' + (i === 0 ? '#111' : '#ccc'),
            'transition: background 0.15s, color 0.15s'
        ].join('; ');
        btn.addEventListener('click', function() {{ setStat(i); }});
        wrapper.appendChild(btn);
    }});

    gd.parentElement.insertBefore(wrapper, gd);

    // ── Stat button click ─────────────────────────────────
    window.setStat = function(idx) {{
        currentStat = idx;
        STAT_LABELS.forEach(function(_, i) {{
            var b = document.getElementById('statbtn_' + i);
            b.style.background  = i === idx ? '#ffb347' : '#252545';
            b.style.color       = i === idx ? '#111'    : '#ccc';
            b.style.fontWeight  = i === idx ? 'bold'    : 'normal';
        }});
        refreshPlot();
    }};

}})();
"""


# ---------------------------------------------------------------------------
# Plotly figure  (single heatmap trace + background scatter)
# ---------------------------------------------------------------------------

def make_figure(emb_bg, al_emb, scores, al_rounds,
                slots, mean_map,
                xedges, yedges,
                score_vmin, score_vmax,
                target, exp_id, bg_subsample, seed):
    import plotly.graph_objects as go

    xcen = 0.5 * (xedges[:-1] + xedges[1:])
    ycen = 0.5 * (yedges[:-1] + yedges[1:])

    traces = []

    # Background scatter
    rng   = np.random.default_rng(seed)
    n_bg  = min(bg_subsample, len(emb_bg))
    idx   = rng.choice(len(emb_bg), size=n_bg, replace=False)
    bg_xy = emb_bg[idx]
    traces.append(go.Scattergl(
        x=bg_xy[:, 0], y=bg_xy[:, 1],
        mode="markers",
        marker=dict(color="rgba(200,200,200,0.06)", size=1.5),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Initial heatmap: mean score for first slot (R0)
    z_init = arr_to_json_z(mean_map[slots[0]])
    traces.append(go.Heatmap(
        x=xcen, y=ycen,
        z=z_init,
        colorscale="Plasma",
        reversescale=True,
        zmin=score_vmin,
        zmax=score_vmax,
        colorbar=dict(
            title=dict(text="mean score<br>(kcal/mol)", side="right",
                       font=dict(color="#cccccc", size=12)),
            tickfont=dict(color="#cccccc", size=10),
            thickness=16, len=0.75,
            bgcolor="rgba(30,30,60,0.6)",
            bordercolor="#555", borderwidth=1,
        ),
        name="score map",
        hovertemplate=(
            "UMAP-1: %{x:.2f}<br>"
            "UMAP-2: %{y:.2f}<br>"
            "Value: %{z:.2f}"
            "<extra></extra>"
        ),
    ))

    # ── Slider (controls round/slot) ──────────────────────────────────────
    def slot_title_init(key):
        if key == ALL_KEY:
            n    = len(al_rounds)
            best = float(scores.min())
            mean = float(scores.mean())
            slabel = "All rounds combined"
        else:
            mask   = al_rounds == key
            n      = int(mask.sum())
            best   = float(scores[mask].min())
            mean   = float(scores[mask].mean())
            slabel = f"Round {key}"
        return (
            f"Mean docking score per UMAP bin — {target} [{exp_id}]<br>"
            f"<sup>{slabel}: n={n:,}  best={best:.2f}  mean={mean:.2f} kcal/mol  |  "
            f"capsaicin ref = {CAP_SCORE} kcal/mol</sup>"
        )

    steps = []
    for i, key in enumerate(slots):
        label = f"R{key}" if key != ALL_KEY else "All"
        steps.append({
            "method": "update",
            "args": [
                {"visible": [True, True]},          # no-op: JS handles z update
                {"title": {"text": slot_title_init(key),
                           "font": {"size": 14, "color": "#e8e8e8"}, "x": 0.5}},
            ],
            "label": label,
        })

    slider_cfg = dict(
        active=0,
        currentvalue=dict(
            prefix="Round: ",
            font=dict(color="#cccccc", size=13),
            visible=True, xanchor="left",
        ),
        steps=steps,
        x=0.05, len=0.9,
        pad={"t": 10, "b": 10},
        bgcolor="#1e1e3a", activebgcolor="#ffb347",
        bordercolor="#555", borderwidth=1,
        font=dict(color="#cccccc", size=11),
    )

    layout = go.Layout(
        title=dict(text=slot_title_init(slots[0]),
                   font=dict(size=14, color="#e8e8e8"), x=0.5),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#0d1117",
        width=1100, height=870,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   showline=False, title="", scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   showline=False, title=""),
        hovermode="closest",
        margin=dict(l=20, r=130, t=90, b=100),
        sliders=[slider_cfg],
    )

    return go.Figure(data=traces, layout=layout)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(al_rounds, scores, slots, mean_map, best_map, count_map,
                  score_vmin, score_vmax, count_vmax, bins, min_count, out_html):
    print(f"\n{'='*68}")
    print(f"  Score map — target / exp derived from cache")
    print(f"  Bins: {bins}×{bins}  min_count={min_count}")
    print(f"  Score range: {score_vmin:.2f} (bright) → {score_vmax:.2f} (dark) kcal/mol")
    print(f"  Count range: 0 → {count_vmax}")
    print(f"{'='*68}")
    print(f"  {'Slot':>5}  {'N':>7}  {'Best':>8}  {'Mean':>8}  "
          f"{'Bins(mean)':>11}  {'Bins(count)':>12}")
    print(f"  {'-'*55}")
    for key in slots:
        m_bins = int((~np.isnan(mean_map[key])).sum())
        c_bins = int((~np.isnan(count_map[key])).sum())
        if key == ALL_KEY:
            n = len(al_rounds); best = scores.min(); mean = scores.mean(); label = "All"
        else:
            mask = al_rounds == key
            n = int(mask.sum()); best = float(scores[mask].min())
            mean = float(scores[mask].mean()); label = f"R{key}"
        print(f"  {label:>5}  {n:>7,}  {best:>8.2f}  {mean:>8.2f}  "
              f"{m_bins:>11,}  {c_bins:>12,}")
    print(f"\n  Output → {out_html}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    args   = parse_args()
    target = args.target
    exp_id = args.experiment_id

    out_html = args.out or (
        FIGURES_DIR / f"umap_al_rounds_score_{target}_{exp_id}.html"
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # Load
    LOG.info("Loading cached AL data (target=%s, exp=%s)…", target, exp_id)
    al_emb, scores, al_rounds, zinc_ids = load_al_cache(target, exp_id)
    LOG.info("  %d compounds, rounds %s", len(al_emb), sorted(set(al_rounds.tolist())))

    LOG.info("Loading background embeddings…")
    emb_bg = load_bg(target)

    # Colorscale limits
    score_vmin = args.score_vmin if args.score_vmin is not None else float(np.floor(scores.min()))
    score_vmax = args.score_vmax if args.score_vmax is not None else float(np.percentile(scores, 95))
    LOG.info("Score colour range: %.2f (bright/best) → %.2f (dark) kcal/mol",
             score_vmin, score_vmax)

    # Bin edges + heatmaps
    LOG.info("Computing %d×%d heatmaps (mean, best, count per slot)…", args.bins, args.bins)
    xedges, yedges = shared_edges(emb_bg, al_emb, args.bins)
    slots, mean_map, best_map, count_map = compute_all_slots(
        al_emb, scores, al_rounds, xedges, yedges, args.min_count
    )

    # Count colorscale upper bound: P99 across all slots
    all_counts = []
    for c in count_map.values():
        v = c[~np.isnan(c)]
        if len(v):
            all_counts.extend(v.tolist())
    count_vmax = int(np.percentile(all_counts, 99)) if all_counts else 100
    LOG.info("Count colour range: 0 → %d (P99)", count_vmax)

    # Build figure
    LOG.info("Building Plotly figure…")
    t0  = time.time()
    fig = make_figure(
        emb_bg, al_emb, scores, al_rounds,
        slots, mean_map, xedges, yedges,
        score_vmin, score_vmax,
        target, exp_id, args.bg_subsample, args.seed,
    )
    LOG.info("  Built in %.1fs", time.time() - t0)

    # Post-script (replaces slot_title placeholders with actual target/exp_id)
    LOG.info("Generating JavaScript post-script…")
    post_script = make_post_script(
        slots, mean_map, best_map, count_map,
        score_vmin, score_vmax, count_vmax,
        al_rounds, scores,
    ).replace("{target}", target).replace("{exp_id}", exp_id)

    # Save
    LOG.info("Writing HTML: %s", out_html)
    t0 = time.time()
    fig.write_html(
        str(out_html),
        include_plotlyjs="cdn",
        div_id=DIV_ID,
        post_script=post_script,
    )
    LOG.info("  Saved in %.1fs  (%.1f MB)", time.time() - t0,
             out_html.stat().st_size / 1e6)

    print_summary(al_rounds, scores, slots, mean_map, best_map, count_map,
                  score_vmin, score_vmax, count_vmax,
                  args.bins, args.min_count, out_html)


if __name__ == "__main__":
    main()
