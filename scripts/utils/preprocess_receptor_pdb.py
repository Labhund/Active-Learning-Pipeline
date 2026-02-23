#!/usr/bin/env python3
"""
Pre-process trpv1_8gfa_receptor.pdb for meeko receptor preparation.

Fixes required for meeko 0.5.0:
  - HIS → HIE or HID per-residue based on the H atom positions in the PDB
      HD1 present (on ND1) → HID
      HE2 present (on NE2) → HIE
      both present         → HIP (protonated/charged)
      neither present      → HIE (default neutral)
  - CYS 387 / CYS 391  → CYX  (intra-chain disulfide, S-S = 2.03 Å confirmed)

Splits output into two files:
  trpv1_8gfa_protein_fixed.pdb   - chains D+A protein only (for mk_prepare_receptor.py)
  trpv1_8gfa_lipids.pdb          - annular lipids only (for openbabel → PDBQT)

Usage:
    python scripts/utils/preprocess_receptor_pdb.py
"""

from collections import defaultdict
from pathlib import Path

INFILE  = Path("targets/trpv1/structures/pdb/trpv1_8gfa_receptor.pdb")
OUTDIR  = Path("targets/trpv1/structures/pdb")

DISULFIDE_RESNUMS = {387, 391}   # CYS → CYX (intra-chain S-S, 2.03 Å)
LIPID_RESNAMES    = {"DU0", "POV"}

PROTEIN_OUT = OUTDIR / "trpv1_8gfa_protein_fixed.pdb"
LIPIDS_OUT  = OUTDIR / "trpv1_8gfa_lipids.pdb"


def classify_his_residues(lines):
    """
    Scan all ATOM/HETATM lines and return a dict:
        (chain, resnum) → 'HIE' | 'HID' | 'HIP'
    based on which imidazole H atoms are present.
    """
    h_flags = defaultdict(lambda: {"HD1": False, "HE2": False})
    for line in lines:
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        resname = line[17:20].strip()
        if resname != "HIS":
            continue
        atomname = line[12:16].strip()
        chain    = line[21]
        resnum   = int(line[22:26])
        key      = (chain, resnum)
        if atomname == "HD1":
            h_flags[key]["HD1"] = True
        if atomname == "HE2":
            h_flags[key]["HE2"] = True

    result = {}
    for key, flags in h_flags.items():
        if flags["HD1"] and flags["HE2"]:
            result[key] = "HIP"
        elif flags["HD1"]:
            result[key] = "HID"
        else:
            result[key] = "HIE"   # HE2 present or neither (default neutral)
    return result


def main():
    raw_lines = open(INFILE).readlines()

    his_map = classify_his_residues(raw_lines)
    taut_counts = defaultdict(int)

    protein_lines = []
    lipid_lines   = []

    for line in raw_lines:
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            protein_lines.append(line)
            continue

        resname = line[17:20].strip()
        chain   = line[21]
        resnum  = int(line[22:26])

        # Lipids → separate file, no modifications
        if resname in LIPID_RESNAMES:
            lipid_lines.append(line)
            continue

        # Fix HIS tautomer
        if resname == "HIS":
            new_name = his_map.get((chain, resnum), "HIE")
            taut_counts[new_name] += 1
            line = line[:17] + new_name + line[20:]

        # Fix disulfide CYS → CYX
        if resname == "CYS" and resnum in DISULFIDE_RESNUMS:
            line = line[:17] + "CYX" + line[20:]

        protein_lines.append(line)

    with open(PROTEIN_OUT, "w") as fh:
        fh.writelines(protein_lines)

    with open(LIPIDS_OUT, "w") as fh:
        fh.writelines(lipid_lines)
        fh.write("END\n")

    n_prot  = sum(1 for l in protein_lines if l[:6].strip() in ("ATOM", "HETATM"))
    n_lipid = len(lipid_lines)

    print(f"Protein atoms : {n_prot:,}  → {PROTEIN_OUT}")
    print(f"Lipid atoms   : {n_lipid:,}  → {LIPIDS_OUT}")
    print()
    print("HIS tautomer assignments (per chain × residue):")
    # Print per-residue assignments
    for (ch, rn), taut in sorted(his_map.items()):
        print(f"  HIS chain {ch} {rn:4d} → {taut}")
    print()
    print("CYX (disulfide) residues:")
    print("  CYS A 387, CYS A 391  (S-S 2.03 Å)")
    print("  CYS D 387, CYS D 391  (S-S 2.03 Å)")


if __name__ == "__main__":
    main()
