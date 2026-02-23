#!/usr/bin/env python3
"""
Extract receptor subunits and annular lipid shell from 8GFA CIF for docking prep.

Outputs:
  targets/trpv1/structures/pdb/trpv1_8gfa_chains_DA.pdb
      Chains D + A protein residues only (no ligands, no waters, no lipids).
      Feed to prepare_receptor4.py as a clean starting point.

  targets/trpv1/structures/pdb/trpv1_8gfa_receptor.pdb
      Chains D + A protein + annular lipid shell within LIPID_CUTOFF Å of
      the target ZEI centroid. This is the full receptor for autogrid4.

The script automatically determines which of the 4 ZEI molecules sits at the
chain D / chain A interface by finding the ZEI closest (in combined CA distance)
to both chains. Override with --zei-chain if needed.

Usage:
    conda activate chem
    python scripts/utils/extract_receptor_8gfa.py [options]

Options:
    --cif       Path to 8gfa.cif (default: targets/trpv1/structures/pdb/8gfa.cif)
    --outdir    Output directory for PDB files (default: targets/trpv1/structures/pdb)
    --zei-chain Override ZEI chain selection: A, B, C, or D (auth_asym_id)
    --lipid-cutoff  Å cutoff for annular lipid shell (default: 15.0)
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
    from Bio.PDB.vectors import Vector
except ImportError:
    sys.exit("BioPython not found. Run: mamba install -n chem biopython")


LIPID_RESNAMES = {"DU0", "POV"}   # resolved annular lipids in 8GFA
EXCLUDE_RESNAMES = {"HOH", "ZEI", "NA"}  # always strip from receptor


class ProteinSelect(Select):
    """Keep only standard amino-acid residues from specified chains."""

    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.id in self.chain_ids

    def accept_residue(self, residue):
        return residue.resname not in (LIPID_RESNAMES | EXCLUDE_RESNAMES) and \
               residue.id[0] == " "  # id[0]==' ' means standard residue


class ReceptorSelect(Select):
    """Keep protein (chains D+A) + lipid residues near ZEI centroid."""

    def __init__(self, chain_ids, lipid_residues):
        """
        chain_ids    : set of auth_asym_id strings (e.g. {'D', 'A'})
        lipid_residues: set of (chain_id, res_id) tuples to include
        """
        self.chain_ids = set(chain_ids)
        self.lipid_residues = lipid_residues  # set of residue objects or their keys

    def accept_chain(self, chain):
        return True  # we filter at residue level

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        # Include protein residues from selected chains
        if chain_id in self.chain_ids and residue.id[0] == " ":
            return True
        # Include selected lipid residues (any chain)
        res_key = (chain_id, residue.id)
        if res_key in self.lipid_residues:
            return True
        return False


def centroid(residue):
    """Return numpy array centroid of all atoms in a residue."""
    coords = np.array([a.coord for a in residue.get_atoms()])
    return coords.mean(axis=0)


def min_ca_distance(zei_centroid, chain):
    """Minimum distance from ZEI centroid to any CA atom in chain."""
    dists = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if "CA" in res:
            dists.append(np.linalg.norm(zei_centroid - res["CA"].coord))
    return min(dists) if dists else float("inf")


def find_da_interface_zei(model):
    """
    Return the ZEI residue (and its centroid) that sits at the chain D / chain A
    interface. Scores each ZEI by combined minimum CA distance to chains D and A.
    """
    chain_d = model["D"]
    chain_a = model["A"]

    zei_candidates = []
    for chain in model.get_chains():
        for res in chain.get_residues():
            if res.resname == "ZEI":
                c = centroid(res)
                d_dist = min_ca_distance(c, chain_d)
                a_dist = min_ca_distance(c, chain_a)
                score = d_dist + a_dist
                zei_candidates.append((score, chain.id, res, c))

    if not zei_candidates:
        sys.exit("ERROR: No ZEI residues found in the structure.")

    print("\nZEI molecules (ranked by proximity to chain D + chain A):")
    print(f"  {'rank':>4}  {'chain':>5}  {'resseq':>7}  {'centroid (x,y,z)':>30}  {'D_dist':>7}  {'A_dist':>7}  {'sum':>7}")
    for rank, (score, cid, res, c) in enumerate(sorted(zei_candidates)):
        d = min_ca_distance(c, chain_d)
        a = min_ca_distance(c, chain_a)
        print(f"  {rank+1:>4}  {cid:>5}  {res.id[1]:>7}  "
              f"({c[0]:7.2f},{c[1]:7.2f},{c[2]:7.2f})  {d:>7.2f}  {a:>7.2f}  {score:>7.2f}")

    best = sorted(zei_candidates)[0]
    return best[2], best[3]  # residue, centroid


def find_zei_by_chain(model, auth_chain):
    """Return the ZEI residue in the specified chain."""
    if auth_chain not in model:
        sys.exit(f"ERROR: Chain {auth_chain!r} not found in structure.")
    chain = model[auth_chain]
    for res in chain.get_residues():
        if res.resname == "ZEI":
            c = centroid(res)
            return res, c
    sys.exit(f"ERROR: No ZEI residue found in chain {auth_chain!r}.")


def collect_lipid_residues(model, zei_centroid, cutoff):
    """Return set of (chain_id, res_id) for lipid residues within cutoff Å of ZEI centroid."""
    selected = set()
    for chain in model.get_chains():
        for res in chain.get_residues():
            if res.resname not in LIPID_RESNAMES:
                continue
            # distance from ZEI centroid to nearest atom in lipid residue
            min_dist = min(np.linalg.norm(zei_centroid - a.coord) for a in res.get_atoms())
            if min_dist <= cutoff:
                selected.add((chain.id, res.id))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cif",
                        default="targets/trpv1/structures/pdb/8gfa.cif",
                        help="Path to 8gfa.cif")
    parser.add_argument("--outdir",
                        default="targets/trpv1/structures/pdb",
                        help="Output directory for PDB files")
    parser.add_argument("--zei-chain",
                        default=None,
                        choices=["A", "B", "C", "D"],
                        help="Override ZEI selection: use ZEI from this auth_asym_id chain")
    parser.add_argument("--lipid-cutoff",
                        type=float,
                        default=15.0,
                        help="Å cutoff for annular lipid shell (default: 15.0)")
    args = parser.parse_args()

    cif_path = Path(args.cif)
    if not cif_path.exists():
        sys.exit(f"ERROR: CIF not found: {cif_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {cif_path} ...")
    cif_parser = MMCIFParser(QUIET=True)
    structure = cif_parser.get_structure("8gfa", str(cif_path))
    model = structure[0]  # first model

    # Verify chains D and A exist
    for cid in ("D", "A"):
        if cid not in model:
            sys.exit(f"ERROR: Chain {cid!r} not found. Available: "
                     f"{[c.id for c in model.get_chains()]}")

    # Identify target ZEI
    if args.zei_chain:
        zei_res, zei_centroid = find_zei_by_chain(model, args.zei_chain)
        print(f"\nUsing ZEI from chain {args.zei_chain} (user override).")
    else:
        zei_res, zei_centroid = find_da_interface_zei(model)

    zei_chain_id = zei_res.get_parent().id
    print(f"\nTarget ZEI: chain {zei_chain_id}, resseq {zei_res.id[1]}")
    print(f"  Centroid: x={zei_centroid[0]:.3f}  y={zei_centroid[1]:.3f}  z={zei_centroid[2]:.3f}")
    print(f"\n  *** Grid center for autogrid4 GPF ***")
    print(f"      gridcenter {zei_centroid[0]:.3f} {zei_centroid[1]:.3f} {zei_centroid[2]:.3f}")

    # Count protein residues per chain
    for cid in ("D", "A"):
        n = sum(1 for r in model[cid].get_residues() if r.id[0] == " ")
        print(f"Chain {cid}: {n} protein residues")

    # Collect lipid shell
    lipid_keys = collect_lipid_residues(model, zei_centroid, args.lipid_cutoff)
    print(f"\nAnnular lipid shell ({args.lipid_cutoff:.0f} Å cutoff): {len(lipid_keys)} lipid residues selected")

    # Count by type
    from collections import Counter
    by_type = Counter()
    for chain in model.get_chains():
        for res in chain.get_residues():
            if (chain.id, res.id) in lipid_keys:
                by_type[res.resname] += 1
    for rn, cnt in sorted(by_type.items()):
        print(f"  {rn}: {cnt}")

    # Output 1: chains D+A protein only (clean input for prepare_receptor4.py)
    out_protein = outdir / "trpv1_8gfa_chains_DA.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_protein), ProteinSelect(["D", "A"]))
    size = out_protein.stat().st_size
    print(f"\nWrote protein-only PDB: {out_protein} ({size/1024:.0f} KB)")

    # Output 2: chains D+A + lipid shell (for autogrid4)
    out_receptor = outdir / "trpv1_8gfa_receptor.pdb"
    io.save(str(out_receptor), ReceptorSelect(["D", "A"], lipid_keys))
    size = out_receptor.stat().st_size
    print(f"Wrote receptor PDB (protein + lipids): {out_receptor} ({size/1024:.0f} KB)")

    print("\nNext steps:")
    print("  1. Verify in PyMOL: load trpv1_8gfa_chains_DA.pdb + 8gfa.cif,")
    print("     confirm target ZEI is at the chain D / chain A interface.")
    print("  2. Run prepare_receptor4.py on trpv1_8gfa_chains_DA.pdb")
    print("     (use the protein-only file; lipids are added separately or")
    print("      alternatively run prepare_receptor4.py on trpv1_8gfa_receptor.pdb)")
    print("  3. Update GPF with gridcenter printed above.")
    print("  4. Run: bash scripts/utils/prep_grid_8gfa.sh")


if __name__ == "__main__":
    main()
