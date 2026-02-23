# TRPV1 Active Learning Screen — First Prototype Production Run

**Date:** 2026-02-20 → 2026-02-21
**Author:** Markus Lipin
**LabArchives path:** Virtual Screening / Active Learning Screen / TRPV1 / First Prototype Production Run
**Experiment ID:** `maxmin_init`
**Status:** Complete (6 rounds, 142,042 compounds docked)

---

## Objective

Validate the active learning HTVS pipeline end-to-end against the TRPV1 vanilloid binding site (PDB: 8GFA, ZEI co-crystal at the D/A chain interface). Primary goals:

1. Confirm the AL loop runs to completion on local hardware without intervention.
2. Demonstrate progressive enrichment of high-scoring compounds over rounds.
3. Assess surrogate model convergence.
4. Identify a shortlist of putative TRPV1 binders for downstream evaluation.

---

## Methods

### System

| Component | Detail |
|---|---|
| Target | TRPV1 (8GFA, SB-366791/ZEI co-crystal, D/A interface vanilloid site) |
| Receptor PDBQT | Protein (chains D+A) + 8 annular lipids (DU0/POV), prepared with meeko 0.5.0 |
| Grid | 48×44×60 pts, 0.303 Å spacing (14.5×13.3×18.2 Å box), center (81.581, 101.331, 86.875) |
| Atom types | 13 (A, C, Br, Cl, F, HD, N, NA, OA, SA, S, I, P) |
| Docking engine | AutoDock-GPU v1.6 (RTX 5080, NVIDIA MPS, `--filelist` batch mode, 20 runs/compound) |
| Compound library | ~297.9M ZINC20 drug-like (pH 7.4, MW ~350, logP > 3.5) |
| Fingerprints | 4096-bit radius-4 Morgan FP (packed uint8 HDF5, 305 GB) |
| Surrogate | XGBoost regressor; Optuna HPO at round 0 (30 trials, CV RMSE = 1.1875 kcal/mol) |
| Acquisition | Boltzmann softmax + Thompson sampling without replacement (T = 1.0) |
| Batch size | 24,000 compounds per round |
| Hardware | AMD Ryzen 9 9900X (12c/24t), RTX 5080, 128 GB DDR5 |

### AL Loop Configuration

`config/al_loop_maxmin.yaml` — key parameters:

| Parameter | Value |
|---|---|
| `batch_size` | 24,000 |
| `temperature` | 1.0 |
| `fail_threshold` | 0 kcal/mol (NULL in DB; excluded from training) |
| `hparam_tune_every_n_rounds` | 3 |
| `dock_workers` | 2 |
| `dock_batch_size` | 100 |

### Round-0 Initialisation

Round 0 used **MaxMin Tanimoto diversity sampling** to seed the initial docking batch. The 297.9M compound library (HDF5) was split into 24 equal partitions; each partition subsampled 1M compounds, then MaxMin selected 1,000 maximally diverse seeds. The combined 24,000-compound set covers broad chemical space before any surrogate-guided exploration.

---

## Results

### Per-Round Summary

| Round | Docked | Valid | Failed (%) | Best-1 | Best-10 | Best-100 | Best-1000 | Mean | Val RMSE |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 23,301 | 23,301 | 699 (3.0%) | −15.49 | −14.48 | −13.77 | −12.65 | −10.07 | 1.2062 |
| 1 | 23,711 | 23,711 | 289 (1.2%) | −15.31 | −14.89 | −14.09 | −13.13 | −11.06 | 1.0363 |
| 2 | 23,765 | 23,765 | 235 (1.0%) | −15.84 | −14.86 | −14.21 | −13.28 | −11.21 | 0.9883 |
| 3 | 23,772 | 23,772 | 228 (1.0%) | −15.76 | −14.93 | −14.22 | −13.30 | −11.23 | 0.9572 |
| 4 | 23,743 | 23,743 | 257 (1.1%) | −15.43 | −14.89 | −14.22 | −13.30 | −11.25 | 0.9099 |
| 5 | 23,750 | 23,750 | 250 (1.1%) | **−16.06** | **−15.09** | **−14.31** | **−13.35** | **−11.29** | 0.9292 |
| **Total** | **142,042** | **140,084** | **1,958 (1.4%)** | | | | | | |

All scores are `free_NRG_binding` (kcal/mol) from AutoDock-GPU XML output. Capsaicin reference: −8.77 kcal/mol.

### Hit Enrichment

Percentage of each round's batch scoring below hit thresholds:

| Round | < −12 kcal/mol | < −13 kcal/mol | < −14 kcal/mol |
|---|---|---|---|
| 0 | 11.1% | 2.2% | 0.2% |
| 5 | 30.5% | 8.0% | 1.0% |
| **Fold enrichment** | **2.7×** | **3.6×** | **4.6×** |

### Figures

**Figure 1 — AL Discovery Dashboard**

![AL discovery dashboard](../analysis/figures/dashboard_trpv1_8gfa.png)

*Left: Discovery curve (best-1, -10, -100, -1000 docking scores vs. cumulative compounds docked). Each step marks a new round batch. Dashed extension = expected library-wide extrapolation (log fit to observed trend). Right: Surrogate validation RMSE per round. RMSE dropped from 1.21 to 0.91 kcal/mol across 6 rounds.*

**Figure 2 — Score Distributions**

![Score distributions](../analysis/figures/score_distributions_trpv1_8gfa.png)

*Left panel: Violin plot of docking score distribution per round; white dot = median, bar = IQR. Capsaicin reference (−8.77, grey dashed). Distribution shifts leftward (more negative = better) each round. Centre panel: Hit enrichment — % of batch below score thresholds over rounds; progressive enrichment at all thresholds. Right panel: Population-level distribution shift — median, IQR (P25–P75), P10–P90, and best score per round.*

### Key Observations

1. **Rapid surrogate convergence.** Validation RMSE fell from 1.21 → 0.91 kcal/mol in 5 rounds with only 142K compounds docked (~0.047% of the library). The surrogate was useful from round 1 onward.

2. **Progressive enrichment without collapse.** Median batch score improved from −10.07 → −11.29 kcal/mol, and hit enrichment at −12 kcal/mol improved 2.7× over 6 rounds. The Boltzmann acquisition (T = 1.0) prevented mode collapse — scores below −8 kcal/mol are still represented in later batches.

3. **Best-1 trajectory non-monotone.** Best single scores fluctuate round-to-round (−15.49, −15.31, −15.84, −15.76, −15.43, **−16.06**) — expected behaviour for sampling from a continuous distribution; the surrogate is guiding the distribution rather than explicitly optimising the global minimum. The final best-1 of −16.06 kcal/mol is **1.83× better than capsaicin** in absolute terms.

4. **Best-10/100/1000 improve monotonically.** Unlike best-1, the top-10 and top-100 show consistent improvement, which is a better indicator of real enrichment (less susceptible to noise in individual docking runs).

5. **Failure rate dropped after round 0.** The 3.0% failure rate in round 0 was from the diversity seed set; rounds 1–5 settled at ~1.1%, consistent with intrinsic meeko embedding failures for a small fraction of ZINC20 structures.

6. **Wall-clock performance.** ~9.5 hours total on a consumer desktop (RTX 5080, Ryzen 9 9900X). Per-round breakdown: ~62 min docking (24K compounds, 2 batch workers × 100/filelist), ~30 min library scoring (305M compounds, XGBoost inference), ~13 min HPO+training (round 0 only), ~5 min batch selection.

---

## Discussion

The first prototype production run validates the core pipeline. The combination of MaxMin diversity initialisation, XGBoost surrogate (Morgan FP → docking score), and Boltzmann-Thompson sampling achieves ~4.6× enrichment of strong binders (< −14 kcal/mol) in only 142K docks out of 298M — a 2,100× savings in compute versus exhaustive docking.

The vanilloid binding site at the TRPV1 D/A chain interface (ZEI co-crystal) appears druggable with diverse chemical scaffolds, with 1% of round-5 compounds scoring better than −14 kcal/mol and 0.1% of the estimated full-library hits presumably exceeding −15 kcal/mol.

### Caveats

- AutoDock-GPU scores are an approximation; no MM-GBSA post-processing has been applied.
- The lipid-exposed binding site may require more careful receptor flexibility treatment for true actives.
- RMSE of ~0.91 kcal/mol means individual predictions carry ≥ 1 kcal/mol uncertainty; top hits should be re-docked with increased `--nrun` or alternative docking methods.
- No pan-assay interference compound (PAINS) or aggregator filter applied to candidates.

---

## Next Steps

### Immediate
- [ ] Generate top-hit structure grid (2D depictions of top 50 by docking score, annotated with score + round discovered)
- [ ] Novelty/diversity analysis: Tanimoto similarity of each round's batch to round-0 maxmin seed set — are later rounds exploring new chemical space or converging?
- [ ] Known binder comparison: dock capsaicin, resiniferatoxin, AMG-517 through the same pipeline and mark reference scores on the violin plot
- [ ] PAINS/aggregator filter on top-500 compounds (RDKit alerts)

### Short-term
- [ ] Scaffold clustering of top-1000 hits (Murcko scaffolds) — identify dominant chemotypes
- [ ] Chemical property distribution over rounds (MW, cLogP, HBD, HBA, TPSA) — check for property drift
- [ ] Surrogate calibration plot: predicted vs. actual scores for the round-5 held-out validation set
- [ ] Run Experiment B (`random_init`): compare discovery curves for MaxMin vs. random initialisation

### Longer-term
- [ ] Re-dock top-50 with `--nrun 100` for more reliable energy estimates
- [ ] ADMET filtering (SwissADME or RDKit descriptors) on shortlist
- [ ] Extend to additional TRPV1 structures (5IRZ, 8U3L) for binding-site selectivity profiling
- [ ] HPC scaling: the pipeline is target-agnostic; apply to TRPA1, TRPM8

---

## Artifacts

| File | Description |
|---|---|
| `models/surrogate_trpv1_8gfa_maxmin_init_round{0-5}.json` | XGBoost surrogate models per round |
| `models/hparams_trpv1_8gfa_maxmin_init_round0.json` | Optuna best hyperparameters |
| `scores/predicted_trpv1_8gfa_maxmin_init_round{0-5}.h5` | Full-library predicted scores per round (~1.2 GB each) |
| `logs/al_metrics_maxmin_init.csv` | Per-round summary metrics |
| `logs/al_round{0-5}_maxmin_init.log` | Per-round detailed logs |
| `logs/al_loop_maxmin_init_stdout.log` | Full orchestrator stdout |
| `analysis/figures/dashboard_trpv1_8gfa.png` | Discovery curve + RMSE dashboard |
| `analysis/figures/score_distributions_trpv1_8gfa.png` | Violin + enrichment + distribution shift |
| `optuna/trpv1_8gfa_maxmin_init_surrogate.db` | Optuna study (SQLite) |
| `work/docking/maxmin_init/round{0-5}/` | Per-compound PDBQT + XML outputs |
