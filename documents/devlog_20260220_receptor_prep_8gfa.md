# Devlog — TRPV1 8GFA Receptor Preparation & Grid Setup for AL HTVS
**Date:** 2026-02-20
**Author:** Markus / Claude Code (claude-sonnet-4-6)
**Project:** Active Learning HTVS — analgesics / TRPV1
**Session scope:** End-to-end receptor preparation, from structure selection rationale through autogrid4 map generation

---

## 1. Background and Objective

The active learning (AL) HTVS pipeline targets TRPV1 (Transient Receptor Potential Vanilloid 1) for analgesic drug discovery. The pipeline iterates: dock a batch of ~24,000 compounds → train XGBoost surrogate on Morgan FP → docking score → use Thompson/softmax sampling to select the next batch. All docking is via AutoDock-GPU against precomputed autogrid4 maps.

Before the first AL round can run, three things need to exist:
1. A receptor PDBQT file
2. An autogrid4 grid (`.fld` + per-atom-type `.map` files)
3. Correct entries in `config/al_loop.yaml`

The previous config pointed to a placeholder `trpv1_5irz` target and a non-existent receptor file. This session replaces that with a production-ready `trpv1_8gfa` setup.

---

## 2. Structure Selection: Why 8GFA

### Candidate structures
| PDB ID | Description | Resolution |
|--------|-------------|------------|
| 5IRZ   | Apo / closed state (no vanilloid ligand) | 3.0 Å cryo-EM |
| 7MZD   | RTX-bound (open state) | 2.5 Å cryo-EM |
| 8U3L   | Open state | 2.4 Å cryo-EM |
| **8GFA** | **SB-366791 / ZEI co-crystal, 4-fold tetramer** | **3.0 Å cryo-EM** |

8GFA was chosen for the following reasons:

1. **Bound ligand in the vanilloid site.** ZEI (SB-366791) is a known competitive antagonist of the capsaicin/vanilloid binding site — the same pocket being targeted. Its pose in the crystal structure directly defines the grid center and validates that the pocket is well-formed and druggable.

2. **Annular lipid resolution.** 8GFA has cryo-EM-resolved annular lipids (DU0: cholesterol derivative, POV: POPC) sitting around the transmembrane domain. The vanilloid binding site is lipid-exposed (S3-S4 TMD region); including these lipids in the receptor correctly shapes the precomputed desolvation and electrostatic maps so that the outer face of the pocket is not mistakenly treated as aqueous.

3. **All four binding sites are occupied.** 8GFA is a full tetramer (chains A-B-C-D) with four ZEI molecules (one per vanilloid site), providing a complete picture of the binding symmetry.

---

## 3. Binding Site Architecture: D/A Interface

### TRPV1 vanilloid site topology
The capsaicin/vanilloid binding site in TRPV1 is formed at the inter-subunit interface:
- **Principal subunit**: S1, S2, S3, S4 helices provide the main binding cavity walls
- **Adjacent subunit**: S4-S5 linker + S5/S6 provide the intracellular closure

The four binding sites in the tetramer are equivalent by 4-fold symmetry but each involves two adjacent chains.

### Why restrict to chains D + A
The docking target was defined as the D/A interface for two reasons:
1. Computationally: using only 2 of 4 chains halves the receptor preparation complexity and autogrid4 runtime without loss of fidelity (all 4 sites are identical by symmetry).
2. Practically: AutoDock-GPU docks into the specific box defined in the grid; the grid covers one binding site only. Using all four chains in the receptor PDBQT correctly fills the lipid/protein environment around that one site while not adding spurious docking opportunities in the other three sites (which are outside the grid).

---

## 4. Grid Strategy Decision: Restriction vs Post-processing

### The problem
The vanilloid binding site sits adjacent to the S4-S5 linker intracellular void — a secondary pocket that AutoDock-GPU will score favorably if the grid includes it. If that pocket is included:
- AutoDock assigns very negative scores to compounds that bind the S4-S5 linker void
- Those scores enter the XGBoost training set
- Over multiple AL rounds the surrogate learns "compounds with these fingerprint features score very well"
- Thompson sampling concentrates on those compounds
- Rounds 3-5 converge on linker-void binders, not vanilloid-site binders

This is a surrogate collapse failure mode that would invalidate the AL experiment.

### Option A (chosen): Grid restriction
Restrict the grid box so the S4-S5 linker void is outside the searchable volume. AutoDock-GPU cannot place atoms outside the grid, so no compound can earn a score boost from the forbidden pocket.

**Advantages:**
- Clean: cuts off the problem at the source
- No extra code, no extra training, no extra parameters
- AutoDock naturally penalises atoms near the grid boundary (high-energy voxels)

### Option B (rejected): Post-processing + second surrogate
Parse AutoDock XML output for best-pose coordinates, apply a residue-distance filter to flag "thermic contacts", train a second surrogate to classify thermic vs genuine vanilloid binders, filter the batch before Thompson sampling.

**Why rejected:**
- AutoDock-GPU XML format is complex; best-pose atom block is buried in multi-run output
- "Thermic contact" definition requires empirical residue set + distance cutoff — two free parameters, both wrong at the edges
- False negatives still bias the surrogate over many rounds
- Requires schema additions, parsing infrastructure, and a labelled training set for the classifier
- Adds approximately 3× the implementation complexity for an uncertain improvement over grid restriction

**Conclusion:** Grid restriction is the correct engineering choice for a clean first-pass AL experiment.

### Grid sizing principle
- Grid spacing: 0.303 Å (standard AutoDock-GPU resolution)
- Intracellular face: padding ≤ 6 Å beyond ZEI extent → S4-S5 linker Cα should be at or outside the grid edge
- Extracellular/lipid face: slightly more generous — ZEI's aromatic ring points into the lipid-exposed face

---

## 5. ZEI Identification: Which Site is the D/A Interface?

### CIF content
8GFA contains 4 ZEI molecules in the tetramer:

| label_asym_id | auth_asym_id | auth_seq_id | Centroid (x, y, z) |
|---------------|--------------|-------------|---------------------|
| F  | A | 1202 | (105.36, 80.97, 88.36) |
| T  | B | 1204 | (125.45, 105.38, 88.35) |
| DA | C | 1202 | (101.03, 125.45, 88.33) |
| PA | D | 1203 | (80.94, 101.03, 88.33) |

### Computational identification
A Python script (`scripts/utils/extract_receptor_8gfa.py`) was written to identify the D/A interface ZEI by scoring each candidate on combined minimum CA-atom distance to chain D and chain A protein residues. The ZEI geometrically closest to both chains is the one at their interface.

**Result:**

| Rank | Chain | Resseq | Centroid | D_dist | A_dist | Sum |
|------|-------|--------|----------|--------|--------|-----|
| **1** | **D** | **1203** | **(80.94, 101.03, 88.33)** | **5.40** | **8.09** | **13.49** |
| 2 | A | 1202 | (105.36, 80.97, 88.36) | 22.82 | 5.38 | 28.20 |
| 3 | C | 1202 | (101.03, 125.45, 88.33) | 8.16 | 21.42 | 29.59 |
| 4 | B | 1204 | (125.45, 105.38, 88.35) | 21.45 | 22.79 | 44.24 |

**ZEI:D:1203** (label_asym PA) is unambiguously the D/A interface molecule — it is 5.4 Å from chain D and 8.1 Å from chain A, while all others are ≥22 Å from at least one chain.

**Note on plan nomenclature:** The planning document described all 4 ZEI molecules as "labeled chain D in the CIF" — this was imprecise. In the mmCIF file they are distributed across auth_asym_id A, B, C, D. The D/A interface ZEI is the one in auth_asym_id D (not A as might be naively assumed from the interface name).

---

## 6. Grid Center: ADT vs Computed

The extraction script computed the ZEI:D:1203 centroid as **(80.942, 101.025, 88.333)**.

After loading the structure in AutoDock Tools (ADT) for visual inspection and precise pocket centering, the grid center was adjusted to **(81.581, 101.331, 86.875)**. This small shift (≈ 1.4 Å) aligns the center more precisely with the geometric center of the binding cavity as viewed in ADT, accounting for the asymmetry of the pocket walls relative to the ligand centroid.

### Final grid parameters (from ADT)
| Parameter | Value |
|-----------|-------|
| Grid points (x, y, z) | 48 × 44 × 60 |
| Spacing | 0.303 Å |
| Box size | 14.5 × 13.3 × 18.2 Å |
| Center (x, y, z) | 81.581, 101.331, 86.875 |

The tighter box (14–18 Å) vs the initial estimate (24–26 Å) was accepted after visual confirmation that ZEI is comfortably enclosed and the S4-S5 linker is at the intracellular edge of the box.

---

## 7. Receptor Extraction

**Script:** `scripts/utils/extract_receptor_8gfa.py`
**Input:** `targets/trpv1/structures/pdb/8gfa.cif` (4.3 MB mmCIF)

### Extraction logic
1. Parse with BioPython `MMCIFParser`
2. Identify all 4 ZEI residues; compute centroid of each; score by combined CA-distance to chains D and A → select chain D ZEI:1203
3. Collect all DU0 and POV residues within 15 Å of target ZEI centroid (lipid shell)
4. **Output A:** chains D + A protein residues only (no HETATM, no waters) → `trpv1_8gfa_chains_DA.pdb` (1,353 KB)
5. **Output B:** chains D + A protein + 8 annular lipids → `trpv1_8gfa_receptor.pdb` (1,433 KB)

### Lipid shell composition (15 Å cutoff from ZEI:D:1203)
| Residue | Count | Identity |
|---------|-------|----------|
| DU0 | 1 | Cholesterol derivative (C32H52O5, 516 Da) |
| POV | 7 | POPC phosphatidylcholine (C36H72NO8P, 760 Da) |

**Rationale for 15 Å cutoff:** Wider than the grid box (≈ 14–18 Å), so lipids that fall outside the sampling volume still contribute to precomputed desolvation and electrostatic maps at all voxels near the pocket boundary.

---

## 8. Receptor PDBQT Preparation

### Tool: meeko 0.5.0 (`mk_prepare_receptor.py`)

**First attempt: failure — two issues**

Running meeko on the raw extracted PDB (`trpv1_8gfa_receptor.pdb`) produced errors:

```
Error: residue HIS not in residue_params   (×16: 8 HIS per chain × 2 chains)
Error: residue POV not in residue_params   (×7)
Error: residue DU0 not in residue_params   (×1)
```

**Root cause analysis:**
1. **HIS** — meeko 0.5.0 requires protonation-state-specific names: `HIE`, `HID`, or `HIP`. The BioPython-extracted PDB retains `HIS` from the CIF. However, the cryo-EM structure **does include modelled hydrogens**, meaning the tautomer can be read directly from the H-atom positions rather than requiring pKa prediction.
2. **DU0 / POV** — meeko has no residue templates for non-standard lipids. These must be processed separately.

### Fix 1: HIS tautomer assignment from H-atom positions

**Script:** `scripts/utils/preprocess_receptor_pdb.py`

For each HIS residue, the script scans for:
- `HD1` atom present (H on ND1) → assign **HID** (delta-protonated)
- `HE2` atom present (H on NE2) → assign **HIE** (epsilon-protonated)
- Both → **HIP** (charged)
- Neither → **HIE** (default)

**Results (16 HIS residues across chains A and D):**

| Residue | Tautomer | Both chains |
|---------|----------|-------------|
| HIS 207 | HIE | ✓ |
| HIS 233 | HIE | ✓ |
| HIS 290 | HIE | ✓ |
| HIS 321 | **HID** | ✓ |
| HIS 365 | **HID** | ✓ |
| HIS 379 | **HID** | ✓ |
| HIS 411 | HIE | ✓ |
| HIS 533 | HIE | ✓ |

Chains A and D give identical tautomers (expected given 4-fold symmetry and the same modelling procedure for each subunit).

### Fix 2: CYS disulfide identification

Residues CYS 387 and CYS 391 appear in both chains. Pairwise S-S distances were computed:

| Pair | S-S distance |
|------|-------------|
| CYS A:387 – CYS A:391 | **2.03 Å** ← disulfide |
| CYS D:387 – CYS D:391 | **2.03 Å** ← disulfide |
| Inter-chain pairs | 58–61 Å (not bonded) |

CYS 387 ↔ CYS 391 form an **intra-chain disulfide** in both subunits. Both renamed to `CYX`.

### Fix 3: Lipids via OpenBabel

meeko cannot process DU0/POV. The lipids were extracted to a separate PDB (`trpv1_8gfa_lipids.pdb`, 1,027 atoms) and converted to PDBQT using OpenBabel 3.1.0:

```bash
obabel -ipdb trpv1_8gfa_lipids.pdb -opdbqt -O trpv1_8gfa_lipids.pdbqt \
    --partialcharge gasteiger -p 7.4
```

OpenBabel issued warnings about missing element symbols in column 77-78 (a BioPython PDB formatting quirk) but produced a valid PDBQT with 410 atom records and correct AutoDock atom types (C, OA, HD, etc.) and Gasteiger partial charges.

**Important:** OpenBabel writes ligand-format PDBQT with ROOT/BRANCH/ENDBRANCH/TORSDOF records (treating the lipids as rotatable molecules). For receptor PDBQT, only ATOM/HETATM records are needed — the torsion records are irrelevant to autogrid4. ATOM lines were extracted and appended directly to the protein PDBQT.

### meeko run (protein only, fixed PDB)

```bash
mk_prepare_receptor.py \
    --pdb targets/trpv1/structures/pdb/trpv1_8gfa_protein_fixed.pdb \
    -o targets/trpv1/structures/pdbqt/trpv1_8gfa \
    --skip_gpf
```

Output: `targets/trpv1/structures/pdbqt/trpv1_8gfa.pdbqt` (clean pass, no errors)

### Merge: protein + lipid PDBQT

A Python one-liner renumbered lipid atom records sequentially after the last protein atom and appended them to the protein PDBQT.

**Final receptor PDBQT:**
- File: `targets/trpv1/structures/pdbqt/trpv1_8gfa.pdbqt`
- Size: 870 KB
- Total atom records: 10,870 (no duplicate atom numbers)
- AutoDock atom types present: A (992), C (4,984), HD (1,845), N (1,372), NA (23), OA (1,587), P (7), SA (60)

The `P` atoms come from the POPC phosphate groups. The presence of SA confirms the CYX disulfide sulfur atoms were typed correctly.

---

## 9. autogrid4 Map Generation

### GPF (`targets/trpv1/grids/trpv1_8gfa.gpf`)

Key settings:
```
npts          48 44 60
spacing       0.303
gridcenter    81.581 101.331 86.875
receptor_types  A C N NA OA P SA HD      # P added for POPC phosphorus
ligand_types    A C Br Cl F HD N NA OA SA  # halogens added for ZINC library coverage
```

**Note on halogen maps:** Br and Cl were added to `ligand_types` beyond the initial template because the ZINC20 screening library (~300M compounds) contains a significant fraction of halogenated compounds. Without Cl/Br maps, AutoDock-GPU would be unable to score halogenated compounds against this receptor.

**Note on receptor path:** The initial GPF used a relative path (`../../structures/pdbqt/...`) which was one `..` too many. Fixed to an absolute path to avoid ambiguity when autogrid4 is run from the grids directory.

### autogrid4 run

```bash
cd targets/trpv1/grids
autogrid4 -p trpv1_8gfa.gpf -l trpv1_8gfa.glg
```

**Result: Successful Completion** (1.97 s wall / 25.5 s CPU)

Maps generated:
| Map | Min energy | ~Grid points |
|-----|-----------|--------------|
| A (aromatic C) | −0.79 | 202,000 |
| C (aliphatic C) | −0.88 | 202,000 |
| Br | −1.41 | 207,000 |
| Cl | −1.15 | 203,000 |
| F | −0.67 | 200,000 |
| HD | −0.72 | 116,000 |
| N | −1.00 | 200,000 |
| NA | −1.28 | 201,000 |
| OA | −1.41 | 201,000 |
| SA | −1.07 | 206,000 |
| e (electrostatic) | −26.16 | — |
| d (desolvation) | +0.62 | — |

The OA and Br minimum energies (−1.41 kcal/mol) indicate well-defined H-bond acceptor contacts available in the pocket, consistent with the ZEI crystal pose (ZEI has a hydroxyl + methoxy group engaging the S513/T550 region).

---

## 10. Config Changes

`config/al_loop.yaml` updated:
```yaml
# Before:
target: trpv1_5irz
receptor_pdbqt: targets/trpv1/structures/pdbqt/trpv1_receptor.pdbqt
grid_maps:      targets/trpv1/grids/trpv1_5irz.fld

# After:
target: trpv1_8gfa
receptor_pdbqt: targets/trpv1/structures/pdbqt/trpv1_8gfa.pdbqt
grid_maps:      targets/trpv1/grids/trpv1_8gfa.fld
```

No DB schema changes required — `docking_scores.target` and `al_batches.target` store strings and are already target-agnostic.

---

## 11. Files Created / Modified

| File | Role |
|------|------|
| `config/al_loop.yaml` | Updated target/receptor/grid paths |
| `scripts/utils/extract_receptor_8gfa.py` | BioPython CIF extraction + ZEI identification |
| `scripts/utils/preprocess_receptor_pdb.py` | HIS tautomers + CYX disulfide fix + lipid split |
| `scripts/utils/prep_grid_8gfa.sh` | End-to-end shell orchestrator (meeko → autogrid4) |
| `targets/trpv1/structures/pdb/trpv1_8gfa_chains_DA.pdb` | Chains D+A protein only (1,353 KB) |
| `targets/trpv1/structures/pdb/trpv1_8gfa_receptor.pdb` | Protein + lipid shell (1,433 KB) |
| `targets/trpv1/structures/pdb/trpv1_8gfa_protein_fixed.pdb` | Fixed HIS/CYX for meeko input |
| `targets/trpv1/structures/pdb/trpv1_8gfa_lipids.pdb` | Annular lipids for obabel |
| `targets/trpv1/structures/pdbqt/trpv1_8gfa.pdbqt` | Final merged receptor PDBQT (870 KB) |
| `targets/trpv1/grids/trpv1_8gfa.gpf` | autogrid4 parameter file |
| `targets/trpv1/grids/trpv1_8gfa.fld` | Grid field file (for AutoDock-GPU `-M` flag) |
| `targets/trpv1/grids/trpv1_8gfa.*.map` | 10 atom-type maps + elec + dsol |
| `targets/trpv1/grids/trpv1_8gfa.glg` | autogrid4 log |

---

## 12. Remaining Steps Before First AL Round

1. **Visual verification in PyMOL** (not yet done):
   - Load `trpv1_8gfa.pdbqt` + `8gfa.cif`
   - Confirm ZEI:D:1203 is visually inside the grid box
   - Confirm S4-S5 linker Cα atoms (TRPV1 ~residues 580-600) are at or outside the intracellular grid edge
   - Confirm the 8 lipid residues are positioned plausibly around the pocket

2. **Capsaicin smoke-dock:**
   ```bash
   bin/autodock_gpu -M targets/trpv1/grids/trpv1_8gfa.fld \
       -L <capsaicin.pdbqt> -O /tmp/test_dock --nrun 5
   ```
   Expected: score between −8 and −12 kcal/mol. Capsaicin is a ~300 Da aromatic vanilloid; scores outside this range would suggest a grid centering or atom-typing issue.

3. **One-time DB migration** (if not already done):
   ```bash
   source env_db.sh && db-start
   psql -d analgesics -f scripts/utils/create_al_tables.sql
   ```

4. **AL loop smoke test (diversity init only, no docking):**
   ```bash
   conda activate chem
   python scripts/active_learning/al_loop.py \
       --config config/al_loop.yaml --start-round 0 --rounds 1 --init-only
   # Verify:
   psql -d analgesics -c "SELECT count(*) FROM al_batches WHERE round=0;"
   # Expected: 24000
   ```

5. **Start NVIDIA MPS** before first full docking run:
   ```bash
   nvidia-cuda-mps-control -d
   ```

---

## 13. Capsaicin Smoke Dock Validation

**Date:** 2026-02-20
**Purpose:** Confirm the receptor PDBQT and grid produce physically reasonable scores for a known reference binder before committing to the AL loop.

### Ligand preparation

SMILES input: `CC(C)/C=C/CCCCC(=O)NCC1=CC(=C(C=C1)O)OC`

**Protonation state (dimorphite-dl v2.0.2, pH 7.4 ± 0.5):**
```bash
conda run -n chem dimorphite_dl \
    "CC(C)/C=C/CCCCC(=O)NCC1=CC(=C(C=C1)O)OC" \
    --ph_min 7.4 --ph_max 7.4 --precision 0.5
```

Two states returned:
| SMILES | State |
|--------|-------|
| `COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1[O-]` | Phenolate (deprotonated) |
| `COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O` | Phenol (neutral) |

The phenol pKa is ~10; at pH 7.4 the neutral form is overwhelmingly dominant (~99.7%). The phenolate state returned by dimorphite-dl reflects its ±0.5 precision window reaching above 7.9, not a genuine protonation prediction. **Neutral phenol form used for docking.**

Note on dimorphite-dl CLI: the v2.0.2 binary uses a positional `SMI` argument (not `--smiles`), and `--ph_min`/`--ph_max` (not `--min_ph`/`--max_ph`). The binary is at `/home/labhund/mamba_envs/chem/bin/dimorphite_dl` and is only on PATH when the `chem` env is active.

**3D structure:** generated with RDKit ETKDGv3 + MMFF94 optimisation.
**Ligand PDBQT:** prepared with meeko `mk_prepare_ligand.py` (10 rotatable bonds, TORSDOF 10).

### Docking run

```bash
/data2/lipin_lab/markus/analgesics/bin/autodock_gpu \
    --ffile targets/trpv1/grids/trpv1_8gfa.fld \
    --lfile /tmp/capsaicin_dock/capsaicin.pdbqt \
    --resnam /tmp/capsaicin_dock/capsaicin_result \
    --nrun 20 --gbest 1
```

Note: AutoDock-GPU v1.6 uses `--resnam` / `-N` for output prefix (not `-O`).

### Results

**Convergence (stdout):**

| Generation | Best Inter+Intra |
|-----------|-----------------|
| 5  | −12.68 kcal/mol |
| 15 | −13.07 kcal/mol |
| 30 | −13.22 kcal/mol |
| 45 | −13.22 kcal/mol ← converged |

Converged in 45 generations (~410K evaluations), σ = 0.08 kcal/mol for best 10%. GPU time: 0.21 s.

**Binding energy (DLG / XML `free_NRG_binding`):**

| Metric | Value |
|--------|-------|
| Best binding energy | **−8.77 kcal/mol** |
| Mean (20 runs) | −8.39 kcal/mol |
| Worst | −7.97 kcal/mol |
| Spread | 0.80 kcal/mol |

**Clustering (RMSD tolerance 2.0 Å):**

| Cluster | Poses | Best energy |
|---------|-------|-------------|
| 1 | 7 | −8.77 kcal/mol |
| 2 | 11 | −8.68 kcal/mol |
| 3 | 2 | −8.12 kcal/mol |

### Energy decomposition (best run)

| Component | Value |
|-----------|-------|
| `final_intermol_NRG` | −11.75 kcal/mol |
| `internal_ligand_NRG` | −1.37 kcal/mol |
| `torsional_free_NRG` | +2.98 kcal/mol (10 bonds × 0.298) |
| **`free_NRG_binding`** | **−8.77 kcal/mol** |
| `Inter + Intra` (stdout) | −13.22 kcal/mol |

The stdout convergence table reports `Inter + Intra` (the raw optimizer objective). The correct binding energy for DB storage and surrogate training is `free_NRG_binding` from the XML, which applies the torsional reference correction. The ~4.5 kcal/mol gap is expected for a 10-bond ligand.

### dock_batch.py XML parsing — confirmed correct

`dock_batch.py` first seeks `<best_energy>` in the XML; this element does **not exist** in AutoDock-GPU v1.6 output (0 matches). It then falls back to `<free_NRG_binding>`, correctly reading −8.77 kcal/mol. The binding energy stored in `docking_scores` will be the corrected value, which is what the XGBoost surrogate trains on.

### Interpretation

−8.77 kcal/mol for capsaicin is physically consistent with its low-nM EC50 at TRPV1 (~100–300 nM). Two dominant clusters with near-identical energies (−8.77 vs −8.68) and very low spread across 20 runs indicate a well-converged search finding a genuine binding mode. The grid and receptor are validated.

---

## 14. Known Limitations and Future Considerations

- **HIS protonation** was read directly from modelled H positions in the 8GFA cryo-EM structure. This is better than a blanket default but not as rigorous as a pKa calculation (e.g., PropKa or PROPKA). For a first-pass screen this is acceptable; revisit if surrogate score distributions look anomalous.

- **Lipid PDBQT quality:** OpenBabel assigned Gasteiger charges to the lipids and used its default PDBQT atom-type assignment. For the cholesterol derivative (DU0), the spiro ring system is exotic; atom types were not manually validated. This affects desolvation map accuracy at lipid-facing voxels but is unlikely to cause gross scoring errors for compounds that bind the central vanilloid pocket.

- **Single-site docking:** Only the D/A interface site is sampled. This is correct by design (the 4 sites are equivalent) but means the AL loop implicitly assumes complete symmetry. Any asymmetry introduced by membrane/lipid context in a real system is ignored.

- **Grid restriction cutoff:** The S4-S5 linker exclusion was validated by visual inspection in ADT. A more rigorous validation would measure the distance from the grid intracellular face to the S4-S5 linker Cα atoms (residues ~583-591 in TRPV1 UniProt numbering) and confirm all are ≥ 0 Å outside the box. This has not been numerically confirmed yet.

- **Halogen coverage:** Br and Cl maps were added; I (iodine) was omitted (rare in drug-like ZINC subset). If the library scan turns up iodinated compounds in the top-scored cohort, an additional autogrid4 run will be needed.
