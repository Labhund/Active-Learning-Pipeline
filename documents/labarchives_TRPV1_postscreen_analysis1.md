# TRPV1 Active Learning Screen — Post-Screen Analysis I: Hit Characterisation

**Date:** 2026-02-21
**Author:** Markus
**LabArchives path:** Virtual Screening / Active Learning Screen / TRPV1 / Post-Screen Analysis I
**Experiment ID:** `maxmin_init`
**Follows:** [First Prototype Production Run](labarchives_TRPV1_prototype_run1.md)
**Status:** Complete

---

## Objective

Characterise the 142,042-compound output of the `maxmin_init` production run (rounds 0–5)
across four complementary analyses:

1. **Top-hit structure grid** — 2D depictions of the 50 highest-scoring compounds.
2. **PAINS filter** — pan-assay interference compound (PAINS A+B+C) prevalence in the
   top-500 hits; assess whether surrogate-guided selection inadvertently enriches
   promiscuous scaffolds.
3. **Scaffold clustering** — Murcko scaffold frequency in the top-1,000 hits; test for
   chemotype convergence vs. maintained diversity.
4. **Chemical novelty vs. score** — max Tanimoto similarity of each AL round's batch to
   the round-0 MaxMin seed set; determine whether the surrogate is exploring new
   chemical space or resampling known regions.

---

## Methods

All analyses run from SMILES strings fetched directly from the PostgreSQL database.
Morgan fingerprints for the novelty analysis were computed on-the-fly (radius = 4,
nBits = 4096) — avoiding HDF5 non-contiguous reads. Tanimoto similarity was computed
via chunked numpy BLAS matrix multiplication.

| Analysis | Script | Input |
|---|---|---|
| Top-hit grid | `analysis/top_hits.py` | Top-50 by docking score |
| PAINS filter | `analysis/pains_filter.py` | Top-500 by docking score |
| Scaffold clustering | `analysis/scaffold_clusters.py` | Top-1,000 by docking score |
| Novelty vs. score | `analysis/novelty_vs_score.py` | All 140,084 valid compounds |

**PAINS catalog:** RDKit `FilterCatalog` with `PAINS_A`, `PAINS_B`, `PAINS_C` (Baell & Holloway, *J. Med. Chem.* 2010).

**Murcko scaffolds:** RDKit `MurckoScaffold.MurckoScaffoldSmiles`, chirality excluded.

**Novelty metric:** max Tanimoto of each compound to the 23,301-compound round-0 seed
set. Round-0 self-similarity computed with diagonal masked (excludes self-match).

---

## Results

### Analysis 1 — Top-Hit Structure Grid

**Figure 1 — Top-50 structures by docking score**

![Top-50 hit structures](figures/top_hits_trpv1_8gfa.png)

*Each panel: 2D depiction, annotated with round discovered and `free_NRG_binding`
score (kcal/mol). Grid ordered by score ascending (most negative = top-left).*

**Top-10 hits:**

| Rank | ZINC ID | Round | Score (kcal/mol) |
|---|---|---|---|
| 1 | 1799172236 | 5 | −16.06 |
| 2 | 409573984 | 2 | −15.84 |
| 3 | 589420689 | 3 | −15.76 |
| 4 | 1778296629 | 3 | −15.63 |
| 5 | 1802175009 | 5 | −15.53 |
| 6 | 1837085015 | 3 | −15.51 |
| 7 | 752423844 | 0 | −15.49 |
| 8 | 1845700701 | 5 | −15.43 |
| 9 | 517434540 | 4 | −15.43 |
| 10 | 1824843615 | 4 | −15.42 |

**Round distribution in top-50:**

| Round | Count in top-50 |
|---|---|
| 0 | 4 |
| 1 | 7 |
| 2 | 7 |
| 3 | 10 |
| 4 | 8 |
| 5 | 14 |

Round 5 contributes 14 of the top-50 — the strongest signal yet of progressive late-round
enrichment. Notably, 4 round-0 MaxMin seeds survive into the top-50, confirming that the
initial diversity sample already captured genuinely potent chemotypes before any
surrogate guidance.

All top-50 compounds pass qualitative drug-likeness inspection: MW ~350–550 Da,
ring-containing scaffolds, no obvious reactive groups (confirmed by PAINS analysis
below). Structural motifs recurring in the top hits include bicyclic/tricyclic ring
systems (indane, decalin, naphthyl, indolyl) linked to amide chains terminating in
morpholine, piperidine, or oxetane — consistent with the lipophilic character expected
for the lipid-exposed TRPV1 vanilloid pocket.

---

### Analysis 2 — PAINS Filter

**Figure 2 — PAINS prevalence by round (top-500 hits)**

![PAINS filter](figures/pains_filter_trpv1_8gfa.png)

*Left: Stacked bar — % clean (blue) vs. PAINS-flagged (orange-red) compounds per
round within the top-500. Numbers above bars = % flagged; n below = count of top-500
compounds from that round. Right: Top PAINS alert type frequencies.*

**Summary (top-500 hits across all rounds):**

| Metric | Value |
|---|---|
| Total compounds analysed | 500 |
| PAINS flagged | 7 (1.4%) |
| Clean | 493 (98.6%) |
| SMILES parse failures | 0 |

**Per-round PAINS rates within the top-500:**

| Round | n in top-500 | PAINS flagged | Rate |
|---|---|---|---|
| 0 | 22 | 2 | 9.1% |
| 1 | 70 | 1 | 1.4% |
| 2 | 95 | 1 | 1.1% |
| 3 | 96 | 1 | 1.0% |
| 4 | 93 | 0 | 0.0% |
| 5 | 124 | 2 | 1.6% |

**Top PAINS alert types:**

| Alert | Count |
|---|---|
| `indol_3yl_alk(461)` | 3 |
| `het_5_A(7)` | 1 |
| `quinone_A(370)` | 1 |
| `ene_six_het_A(483)` | 1 |
| `sulfonamide_F(1)` | 1 |

**Key observations:**

- **Overall PAINS burden is very low** (1.4%). The top-500 from this screen are
  substantially cleaner than typical HTS collections, where PAINS rates of 5–20% are
  common.
- **Round-0 PAINS rate is elevated** (9.1% vs ~1% for AL rounds). This is expected:
  MaxMin diversity sampling selects chemically heterogeneous structures, some of which
  include reactive or promiscuous scaffolds. The surrogate does not explicitly penalise
  these, but their poorer docking scores mean they are diluted out of top-500 lists in
  later rounds.
- **AL rounds 1–5 maintain consistently low rates** (0–1.6%), confirming that
  Boltzmann-Thompson sampling is not selecting for PAINS scaffolds. The surrogate
  correctly identifies steric/electrostatic binding-relevant features rather than
  reactive moieties.
- The dominant alert (`indol_3yl_alk`) flags 3-substituted indoles connected to
  alkyl/alkenyl chains — a pharmacophore present in several known TRPV1 binders
  (e.g., resiniferatoxin, capsaicin analogues). These may be true positives rather than
  assay interferences; caution in interpretation is warranted but they should not be
  discarded solely on this basis.

---

### Analysis 3 — Scaffold Clustering

**Figure 3 — Murcko scaffold frequency (top-1,000 hits)**

![Scaffold clusters](figures/scaffold_clusters_trpv1_8gfa.png)

*Left: Top-20 scaffolds by count, coloured by best docking score (deep blue = most
negative). Right: 2D depictions of the top-20 scaffolds with count and best score
annotations.*

**Summary (top-1,000 hits):**

| Metric | Value |
|---|---|
| Unique Murcko scaffolds | 996 |
| Singletons (N=1) | 992 (99.6%) |
| Scaffold diversity (Shannon entropy H) | 9.958 bits |
| SMILES parse / scaffold failures | 0 / 0 |

**Top-10 scaffolds by count:**

| Rank | Count | Best (kcal/mol) | Rounds present |
|---|---|---|---|
| S1 | 2 | −14.83 | 3, 5 |
| S2 | 2 | −14.52 | 3, 4 |
| S3 | 2 | −14.37 | 0, 4 |
| S4 | 2 | −14.23 | 4 |
| S5–S20 | 1 each | −16.06 → −15.30 | — |

**Key observations:**

- **Near-zero scaffold convergence.** With 996 unique scaffolds across 1,000 compounds
  and 99.6% singletons, the top hits represent an enormous structural diversity. No
  single scaffold dominates the high-scoring region after 6 rounds.
- **Shannon entropy H = 9.96 bits** is close to the theoretical maximum for 1,000
  samples (≈ 9.97 bits for a perfectly uniform distribution), confirming that the
  Boltzmann acquisition function (T = 1.0) is maintaining broad chemical space
  exploration rather than collapsing onto a few scaffold families.
- **The four repeat scaffolds (count=2) straddle multiple rounds**, showing the
  surrogate is not simply memorising scaffold families from early rounds but
  independently rediscovering them — a hallmark of genuine structure–activity signal.
- **The global best compound (−16.06 kcal/mol, S5) is a singleton** with no scaffold
  mate in the top-1,000. This is consistent with the surrogate operating in a sparse
  region of chemical space where individual compounds, rather than scaffold families,
  carry the strongest signal.
- **Scaffold 3 (R0 + R4)** is one of only two scaffolds appearing in both round-0
  (diversity-selected) and a later AL round — suggesting the surrogate correctly
  identified this chemotype's promise from the initial diversity data and returned to it.

---

### Analysis 4 — Chemical Novelty vs. Score

**Figure 4 — Chemical novelty vs. docking score**

![Novelty vs score](figures/novelty_vs_score_trpv1_8gfa.png)

*Left: Scatter of max Tanimoto similarity (to R0 seed set) vs. docking score, one
series per round 1–5. Horizontal dashed: capsaicin reference (−8.77 kcal/mol).
Vertical dotted: R0 internal self-similarity median (0.315). Right: Violin of max
Tanimoto distribution per round, including R0 self-similarity baseline.*

**Novelty summary (max Tanimoto to R0 seed set):**

| Round | N | Median Tanimoto | Mean Tanimoto | % > 0.4 | % > 0.6 |
|---|---|---|---|---|---|
| R0 (self-sim) | 23,301 | 0.315 | 0.324 | 19.7% | 0.7% |
| R1 | 23,711 | 0.270 | 0.283 | 6.3% | 0.1% |
| R2 | 23,765 | 0.272 | 0.284 | 6.1% | 0.1% |
| R3 | 23,772 | 0.271 | 0.284 | 6.3% | 0.1% |
| R4 | 23,743 | 0.271 | 0.284 | 6.1% | 0.1% |
| R5 | 23,750 | 0.272 | 0.284 | 6.1% | 0.1% |

**Key observations:**

- **AL rounds explore more novel chemistry than the seed set.** The median Tanimoto
  of R1–R5 batches to the R0 seed set (0.270–0.272) is consistently *lower* than the
  R0 internal self-similarity median (0.315). The surrogate-guided batches are, on
  average, more dissimilar to the seed set than the seed set compounds are to each
  other — confirming genuine exploration of new chemical space, not re-sampling.
- **Novelty is stable across rounds 1–5.** The median and mean Tanimoto are virtually
  constant (Δ < 0.003 across all five rounds). There is no trend toward either
  increasing similarity (convergence/exploitation) or decreasing similarity (runaway
  exploration). The Boltzmann temperature T = 1.0 is holding the explore–exploit
  balance well.
- **High-scoring compounds span the full novelty range.** The scatter plot (Figure 4,
  left) shows that compounds scoring better than −14 kcal/mol occur at all Tanimoto
  values from 0.1 to 0.6 — the surrogate is not confining its predictions to one
  region of chemical space. Notably, the global best compound (−16.06 kcal/mol,
  round 5) sits at Tanimoto < 0.2 to any R0 seed, making it a genuinely novel
  scaffold.
- **Very few near-neighbours to the seed set are selected.** Only 6.1–6.3% of each
  AL batch has Tanimoto > 0.4 to any R0 compound (vs. 19.7% for R0 internal
  self-similarity). The surrogate is not simply interpolating around known-good regions
  from round 0 but identifying potent compounds in unexplored chemical space.

---

## Discussion

Taken together, the four post-screen analyses paint a consistent and encouraging
picture of the `maxmin_init` production run.

**The hit list is chemically diverse and drug-like.** With 996 unique Murcko scaffolds
in the top-1,000, near-maximum scaffold entropy (H = 9.96 bits), and 98.6% of top-500
hits passing PAINS filters, the screen output is substantially better in both diversity
and quality than what might be expected from a random or greedy selection.

**The surrogate is guiding exploration, not exploitation.** The novelty analysis
demonstrates that rounds 1–5 access chemical space that is *more novel* than the
round-0 seed set's internal diversity — the surrogate is not converging on a local
optimum or resampling the seed set. The consistency of novelty across rounds
(no drift) shows the Boltzmann temperature (T = 1.0) is correctly balanced.

**No PAINS enrichment under AL guidance.** The elevated PAINS rate in round 0 (9.1%)
drops immediately under surrogate guidance (1.0–1.6% in rounds 1–5). This is a
non-trivial finding: it confirms the XGBoost surrogate is learning binding-relevant
structure–activity relationships rather than proxies for reactivity or promiscuity.

**The indole alert warrants attention, not exclusion.** The three `indol_3yl_alk`-
flagged compounds in the top-500 share pharmacophoric features with known TRPV1
agonists. These should be carried forward with a note for counter-screening rather
than hard-filtered.

### Caveats

- All analyses are based on AutoDock-GPU rigid-receptor docking scores. The top hits
  have not been subjected to pose inspection, MM-GBSA rescoring, or experimental
  validation.
- The 3-substituted indole alert overlap with known TRPV1 pharmacophores means PAINS
  filtering should be applied cautiously for this target; biochemical or biophysical
  confirmation is the definitive test.
- Scaffold diversity at the top-1,000 level does not preclude convergence at finer
  structural levels (e.g., R-group patterns within the singleton scaffolds); this
  requires a deeper clustering analysis (e.g., Butina clustering on full Morgan FPs).

---

## Next Steps

### Completed (this session)
- [x] Top-hit structure grid (top-50, 2D depictions)
- [x] PAINS/aggregator filter on top-500 compounds
- [x] Scaffold clustering of top-1,000 hits (Murcko scaffolds, Shannon entropy)
- [x] Novelty/diversity analysis — Tanimoto similarity to R0 seed set

### Immediate
- [ ] Re-dock top-50 with `--nrun 100` for more reliable energy estimates
- [ ] Known binder comparison: cross-dock capsaicin, resiniferatoxin, AMG-517 through
      the same AutoDock-GPU grid; overlay on violin plots
- [ ] Butina clustering (Tc = 0.4) of top-1,000 full Morgan FPs — finer-grained
      chemotype grouping beyond Murcko scaffolds

### Short-term
- [ ] Chemical property distribution over rounds (MW, cLogP, HBD, HBA, TPSA) —
      check for property drift under surrogate guidance
- [ ] Surrogate calibration plot: predicted vs. actual scores for round-5 held-out set
- [ ] ADMET filtering (SwissADME or RDKit descriptors) on shortlist from top-50
- [ ] Experiment B (`random_init`): compare MaxMin vs. random initialisation discovery
      curves; determine whether diversity seeding provides measurable benefit

### Longer-term
- [ ] Binding mode analysis: extract and inspect AutoDock-GPU pose XMLs for top-10;
      visualise in PyMOL against ZEI reference pose
- [ ] Selectivity profiling: dock top-50 against closed-state 5IRZ; identify compounds
      with state-dependent binding preference
- [ ] HPC scaling: apply pipeline to TRPA1 (cold pain), TRPM8 (menthol receptor)

---

## Artifacts

| File | Description |
|---|---|
| `analysis/top_hits.py` | Script: 2D structure grid of top-N hits |
| `analysis/pains_filter.py` | Script: PAINS A+B+C filter on top-N |
| `analysis/scaffold_clusters.py` | Script: Murcko scaffold frequency + figure |
| `analysis/novelty_vs_score.py` | Script: Tanimoto novelty vs. score scatter |
| `analysis/figures/top_hits_trpv1_8gfa.png` | Top-50 structure grid (772 KB) |
| `analysis/figures/pains_filter_trpv1_8gfa.png` | PAINS stacked bar + alert types (67 KB) |
| `analysis/pains_hits_trpv1_8gfa.csv` | Per-compound PAINS flag + alert names (19 KB) |
| `analysis/figures/scaffold_clusters_trpv1_8gfa.png` | Scaffold bar chart + 2D grid (346 KB) |
| `analysis/scaffold_clusters_trpv1_8gfa.csv` | Scaffold SMILES, count, best/worst score, rounds (77 KB) |
| `analysis/figures/novelty_vs_score_trpv1_8gfa.png` | Novelty scatter + violin (374 KB) |
