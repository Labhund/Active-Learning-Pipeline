# Lab Report: MaxMin vs Random Initialization — TRPV1 AL Comparison

**Date:** 2026-02-22
**Target:** TRPV1 (8GFA, vanilloid binding site, ZEI:D:1203)
**Analyst:** Markus / Claude Code

---

## 1. Objective & Experimental Design

**Scientific question:** Does the choice of round-0 initialization strategy — MaxMin
Tanimoto diversity picking versus uniform random sampling — affect the outcome of a
Thompson-sampling active learning campaign?

Both experiments ran 6 rounds with 24,000 compounds per round against the same target
using identical infrastructure. The only differences are:

| Parameter | maxmin_init | random_init |
|---|---|---|
| Round-0 selection | MaxMin Tanimoto diversity (24 parallel batches × 1,000 picks from 500K subsamples) | Uniform random (TABLESAMPLE BERNOULLI, seed 123) |
| Rounds 1–5 selection | Softmax Thompson sampling (T = 1.0) | same |
| Batch size | 24,000 | 24,000 |
| Surrogate | XGBoost (Morgan FP r=4, 4096-bit → score kcal/mol) | same |
| Optuna HPO | Rounds 0, 3 | same |
| Seed | 42 | 123 |
| Total docked | 142,042 | 142,556 |

**Note on Thompson sampling temperature:** Both experiments used `temperature = 1.0`
in `config/al_loop_{exp}.yaml`, i.e., softmax weights `∝ exp(−score / 1.0)`. This is
a moderately exploitative setting — it upweights predicted strong binders but does not
collapse onto a single mode.

**Important implementation detail for MaxMin:** The round-0 diversity picking was
parallelised across 24 CPU workers, each receiving a 500K-compound subsample of the
library. MaxMin was run *within* each worker's subsample, not globally across the full
library. This means the 24,000 seed compounds are the union of 24 independent MaxMin
picks, each optimising diversity within its own batch. The consequences of this are
discussed in Section 5.

---

## 2. Per-Round Metrics (Side-by-Side)

### 2a. Docking scores

| R | MM best-1 | RD best-1 | MM best-10 | RD best-10 | MM best-100 | RD best-100 | MM mean | RD mean |
|---|-----------|-----------|------------|------------|-------------|-------------|---------|---------|
| 0 | **−15.49** | −14.68 | **−14.48** | −14.41 | **−13.77** | −13.56 | −10.07 | −10.27 |
| 1 | −15.31 | **−15.47** | **−14.89** | −14.83 | **−14.09** | −14.07 | −11.06 | −11.16 |
| 2 | **−15.84** | −15.65 | −14.86 | **−14.94** | **−14.21** | −14.21 | −11.21 | −11.25 |
| 3 | **−15.76** | −15.44 | **−14.93** | −14.88 | −14.22 | **−14.24** | −11.23 | −11.27 |
| 4 | **−15.43** | −15.21 | −14.89 | **−14.97** | −14.22 | **−14.35** | −11.25 | **−11.32** |
| 5 | −16.06 | **−16.15** | **−15.09** | −14.82 | **−14.31** | −14.25 | −11.29 | **−11.32** |

MM = maxmin_init, RD = random_init. Bold = higher value in that round.

### 2b. Surrogate model quality (val RMSE, kcal/mol)

| R | MM RMSE | RD RMSE | MM n_train | RD n_train |
|---|---------|---------|------------|------------|
| 0 | 1.2062 | **0.9919** | 20,971 | 21,387 |
| 1 | 1.0363 | **0.9501** | 42,312 | 42,757 |
| 2 | 0.9883 | **0.9092** | 63,707 | 64,161 |
| 3 | 0.9572 | **0.9018** | 85,099 | 85,535 |
| 4 | 0.9099 | **0.8618** | 106,470 | 106,928 |
| 5 | 0.9292 | **0.8458** | 127,842 | 128,306 |

Random initialization produced better surrogate RMSE at every round.

### 2c. Hit enrichment (% of batch with score < threshold)

| R | MM <−13% | RD <−13% | MM <−14% | RD <−14% | MM <−15% | RD <−15% |
|---|----------|----------|----------|----------|----------|----------|
| 0 | 2.1 | 1.6 | 0.2 | 0.1 | 0.0 | 0.0 |
| 1 | 8.2 | 6.1 | 0.5 | 0.5 | 0.0 | 0.0 |
| 2 | 10.0 | 7.6 | 0.8 | 0.8 | 0.0 | 0.0 |
| 3 | 9.7 | 7.6 | 0.8 | 0.8 | 0.0 | 0.0 |
| 4 | 9.7 | 8.2 | 0.8 | 0.9 | 0.0 | 0.0 |
| 5 | **8.0** | **8.4** | **1.0** | **0.9** | **0.0** | **0.0** |

By R5, random's <−13 hit rate (8.4%) slightly exceeds MaxMin's (8.0%). The <−14 and
<−15 rates are negligibly different across all rounds.

---

## 3. Observations

### 3.1 Round-0: MaxMin found a better best compound

MaxMin's round-0 best-1 (−15.49 kcal/mol) is 0.81 kcal/mol more negative than
random's (−14.68 kcal/mol). This is consistent with the hypothesis that a more diverse
seed set increases the probability of sampling a high-affinity scaffold — but this is
a single observation from one replicate and the difference cannot be attributed to the
strategy alone without repeats.

### 3.2 Round-0: Random produced a substantially better surrogate

Random's R0 RMSE (0.99 kcal/mol) is 0.22 kcal/mol lower than MaxMin's (1.21 kcal/mol)
at the same training set size. This gap persists through all 6 rounds (0.08 kcal/mol
at R5). One possible explanation: the MaxMin seed compounds sample the structural
periphery of the library (diverse but atypical), producing a score distribution that is
harder for XGBoost to learn from. Random sampling more closely mirrors the library's
structural distribution, providing a training set that is more representative of what
the surrogate will be asked to predict. This is an observation, not a proven mechanism;
other explanations are possible.

### 3.3 Both experiments converge rapidly via Thompson sampling

The score distributions — mean, median, best-10, best-100 — become statistically
indistinguishable between experiments by round 2. The Thompson sampler (T=1.0) rapidly
overcomes any seed quality advantage or disadvantage and drives both experiments toward
the same high-affinity region of chemical space. This argues that, at this scale
(~6×24K compounds), the initialisation choice has limited long-term influence on which
compounds are discovered.

### 3.4 Final round: no clear winner

At R5, every metric fluctuates within a small range across experiments:

- Best-1: random −16.15 vs MaxMin −16.06 (Δ = 0.09 kcal/mol)
- Best-10: MaxMin −15.09 vs random −14.82 (Δ = 0.27 kcal/mol — the largest persistent gap)
- Mean: random −11.32 vs MaxMin −11.29 (Δ = 0.03 kcal/mol)
- Surrogate RMSE: random consistently better throughout

No single experiment dominates on all metrics. The best-10 advantage of MaxMin and the
RMSE advantage of random may reflect different aspects of what "better" means; they
are not directly comparable.

### 3.5 MaxMin RMSE is non-monotone at R5

MaxMin val RMSE increased from 0.9099 (R4) to 0.9292 (R5), while random continued
to improve monotonically (0.8618 → 0.8458). This worsening is unusual and the cause
is unclear. Possible explanations include a distribution shift in the R5 training
batch, overfitting artefacts, or stochastic variation in the Optuna-tuned
hyperparameters. It is not obviously a pathological failure and may be noise, but it
is worth noting.

### 3.6 MaxMin did not achieve more uniform coverage of chemical space

The UMAP density diagnostic (see `compare_random_trpv1_8gfa.png`) reveals a
counterintuitive result: **MaxMin produces a less uniform coverage of chemical space
than the fresh random baseline**. Specifically, the MaxMin picks show a pronounced
red ridge (≥4× oversampling relative to library density) along the central axis of the
UMAP projection and heavily undersampled blue periphery. The random 24K sample shows a
much more uniform light-blue pattern.

This is likely a consequence of the parallel implementation: MaxMin was run independently
within 24 subsample batches of 500K compounds each, not globally across the full
library. Each worker selected compounds that are maximally diverse *within its own
subsample*. When the 24 sets are pooled, the selections from different workers can be
structurally similar to each other (overlap in the common high-density region of
chemical space). This artefact would not occur with a true global MaxMin pass, which
would be computationally prohibitive at 300M scale.

The quantitative summary from the 6-round combined UMAP:

| Metric | MaxMin (6 rounds) | Random (6 rounds) |
|--------|-------------------|-------------------|
| std(log₂ ratio) | 3.969 | 4.041 |
| IQR(log₂ ratio) | 1.343 | 1.244 |
| Bins ≥4× undersampled (<−2) | 13.9% | 13.4% |
| Bins ≥4× oversampled (>+2) | 2.2% | 2.2% |

Note that after 6 rounds of Thompson sampling, the UMAP coverage statistics are nearly
identical (std 3.97 vs 4.04). The initial structural differences in the seed sets
appear to be washed out by the AL loop.

---

## 4. Molecule and Scaffold Rediscovery

### 4.1 Molecule-level overlap

Of 142,042 compounds docked by MaxMin and 142,556 by random:

- **126 compound IDs appear in both experiments** (0.089% overlap)
- This is consistent with chance: if 142K compounds are drawn from a 300M pool, the
  expected overlap under independent sampling is ~142K²/300M ≈ 67. Observing 126 may
  reflect the Thompson sampler converging toward similar high-value regions from R1+,
  or may be within statistical noise.
- Per-round breakdown: R0=2, R1=6, R2=3, R3=3, R4=5, R5=7. The slight increase in
  later rounds (R4–R5 sharing more than R0) is suggestive of convergence toward the
  same attractive chemical space regions, but the counts are too small to draw
  firm conclusions.

### 4.2 Scaffold-level overlap (Murcko framework)

Murcko scaffolds were computed for all 284,598 docked compounds:

| Metric | Value |
|--------|-------|
| MaxMin unique scaffolds | 121,808 |
| Random unique scaffolds | 119,227 |
| **Shared scaffolds** | **12,232** (10.0% of MaxMin, 10.3% of Random) |
| Union of all scaffolds | 228,803 |

**~10% of scaffolds are shared between the two experiments.** Given that both
experiments dock ~142K compounds from a 300M library (~0.047%), a 10% scaffold overlap
suggests the Thompson sampler is steering both experiments toward overlapping — but by
no means identical — regions of scaffold space. The shared scaffolds likely represent
the most high-affinity chemical series that the surrogate learns to prioritise
regardless of starting point.

### 4.3 Top-hit scaffold overlap

| | MaxMin top-500 | Random top-500 |
|---|---|---|
| Unique scaffolds | 499 | 498 |
| **Shared scaffolds** | **1** (0.2%) | **1** (0.2%) |
| Shared top-50 compound_ids | 0 | 0 |
| Shared top-50 scaffolds | 0 | 0 |

**The top hit series discovered by each experiment are almost entirely distinct.**
Only 1 scaffold appears in both experiments' top-500. The top-50 hits share no
compounds and no scaffolds at all. This is notable: despite docking similar total
numbers of compounds, and despite Thompson sampling converging the score distributions
to similar levels, both experiments are finding different chemical matter at the top.

Scaffold frequencies within each experiment's top-500 are nearly all singletons
(all frequencies are 1–2), consistent with the high scaffold diversity reported in the
maxmin post-screen analysis (996/1000 unique scaffolds). Neither experiment has
collapsed onto a dominant chemotype in the top-500.

This means the two experiments are not simply rediscovering the same compounds with
different efficiency — they are exploring largely complementary hit series. Whether
this reflects genuine scaffold diversity in the TRPV1 vanilloid binding site, or is an
artefact of the small fraction of library explored (<0.1%), is an open question.

---

## 5. Open Questions

- **Is the MaxMin R0 best-1 advantage real?** A 0.81 kcal/mol gap at round 0 is
  plausible but cannot be attributed to the initialization strategy without repeated
  experiments. With a single replicate per condition, this observation cannot be
  distinguished from chance.

- **Why does random produce a consistently better surrogate?** The RMSE gap (0.08–0.22
  kcal/mol) is consistent across all 6 rounds. Whether this is due to the training set
  distribution, the score distribution shape in the seed set, or some other property of
  random vs. diversity picks is not established.

- **Does a globally-computed MaxMin (not batch-parallel) perform differently?** The
  known artefact of the parallel implementation (central ridge in UMAP) could bias the
  seed set in ways that affect both early discovery and surrogate quality. A true global
  MaxMin at 300M scale is computationally expensive but would be a cleaner test of the
  hypothesis.

- **Do the different hit series represent genuinely distinct binding modes?** The near-
  zero top-50 scaffold overlap is striking. Structural clustering and redocking of the
  top hits from both experiments could reveal whether they bind in the same pose or
  sample different sub-pockets within the vanilloid site.

- **How would results change with more rounds?** Six rounds covering 0.047% of the
  library is a very early stage of the AL campaign. The performance gap may widen,
  narrow, or reverse with more rounds — particularly if one surrogate consistently
  guides toward better regions.

- **What caused the MaxMin R5 RMSE increase?** This is unusual and warrants
  investigation. It may be noise, a distribution shift in the Thompson-selected batch,
  or a signal that the MaxMin campaign is beginning to over-exploit a narrow region of
  chemical space at round 5.

- **What is the appropriate evaluation metric?** Best-1 rewards lucky extreme hits;
  best-10 rewards consistent top-hit quality; mean rewards broad enrichment; RMSE
  measures model reliability. The "better" experiment depends on the downstream goal.
  None of these metrics alone answers the scientific question.

---

## 6. Summary

Both initialization strategies produced comparable 6-round outcomes. The Thompson
sampling AL loop (T=1.0) appears to be a stronger determinant of what gets discovered
than the round-0 seed strategy. Some specific observations:

1. **Round-0**: MaxMin found a better best-1 compound. Random trained a better surrogate.
2. **Rounds 1–2**: Gap closes rapidly on all metrics.
3. **Rounds 3–5**: No consistent winner. Metrics alternate between experiments.
4. **RMSE**: Random was consistently better throughout (final 0.85 vs 0.93 kcal/mol).
5. **Chemical space**: MaxMin's parallel implementation produced an unexpected
   non-uniform UMAP coverage with central oversampling. Random better mirrors the
   library density.
6. **Rediscovery**: 10% scaffold overlap across all docked compounds; only 1 shared
   scaffold in each experiment's top-500. Both experiments found largely different
   hit series.
7. **Temperature correction**: Both experiments used T=1.0, not T=0.01 (earlier draft
   was incorrect).

No strong conclusion in favour of either strategy can be drawn from this single-replicate
comparison. Further replicates, longer campaigns, and experimental validation of top
hits are needed.

---

## 7. Figures

| Figure | Description |
|--------|-------------|
| `analysis/figures/dashboard_compare_trpv1_8gfa.png` | Discovery curves + RMSE; both experiments overlaid |
| `analysis/figures/compare_experiments_trpv1_8gfa.png` | 4-panel: violins, P10/P90 bands, hit enrichment, RMSE+ΔBest-1 |
| `analysis/figures/score_distributions_trpv1_8gfa.png` | MaxMin per-round distributions |
| `analysis/figures/score_distributions_trpv1_8gfa_random_init.png` | Random per-round distributions |
| `analysis/figures/novelty_vs_score_trpv1_8gfa.png` | MaxMin novelty vs score scatter |
| `analysis/figures/novelty_vs_score_trpv1_8gfa_random_init.png` | Random novelty vs score scatter |
| `analysis/diversity_study/figures/compare_random_trpv1_8gfa.png` | UMAP density diagnostic: MaxMin 24K vs fresh random 24K (R0 only) |
| `analysis/figures/umap_al_rounds_trpv1_8gfa_maxmin_init.html` | Interactive UMAP: MaxMin rounds in library space |
| `analysis/figures/umap_al_rounds_trpv1_8gfa_random_init.html` | Interactive UMAP: Random rounds in library space |
| `analysis/figures/umap_compare_experiments_trpv1_8gfa.png` | Static 3-panel: density ratios + difference map (6-round combined) |

---

## 8. Data Provenance

| Artifact | Location |
|----------|----------|
| MaxMin config | `config/al_loop_maxmin.yaml` |
| Random config | `config/al_loop_random.yaml` |
| MaxMin metrics | `logs/al_metrics_maxmin_init.csv` |
| Random metrics | `logs/al_metrics_random_init.csv` |
| Scaffold overlap script | `/tmp/scaffold_overlap.py` |
| Comparison figure script | `analysis/compare_experiments.py` |
| UMAP comparison script | `analysis/diversity_study/umap_compare_experiments.py` |
| Prototype run lab report | `documents/labarchives_TRPV1_prototype_run1.md` |
| Post-screen analysis | `documents/labarchives_TRPV1_postscreen_analysis1.md` |
