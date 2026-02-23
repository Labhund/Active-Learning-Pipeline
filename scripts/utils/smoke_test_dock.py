"""
smoke_test_dock.py — Serial end-to-end diagnostic for round-0 docking.

Fetches N compound IDs from al_batches, runs meeko prep one at a time
in-process (so failures are clearly visible), docks each success with
AutoDock-GPU, and prints a summary table.

Usage:
    python scripts/utils/smoke_test_dock.py \
        --target trpv1_8gfa --experiment-id maxmin_init \
        --maps targets/trpv1/grids/trpv1_8gfa.fld \
        --n 100 --nrun 20
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import psycopg2
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
except ImportError:
    print("ERROR: meeko not installed. Run: mamba install -n chem meeko", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
DB_NAME = "analgesics"
DB_USER = "labhund"


def get_db_conn():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        host=os.environ.get("PGHOST", "/tmp"),
    )


# ---------------------------------------------------------------------------
# Prep (in-process, verbose)
# ---------------------------------------------------------------------------
def prep_ligand(compound_id: int, smiles: str, out_pdbqt: Path):
    """
    Attempt full meeko prep.
    Returns (ok: bool, failure_category: str | None, detail: str | None).
    failure_category: 'smiles' | 'embedding' | 'meeko' | None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "smiles", "RDKit could not parse SMILES"

    mol = Chem.AddHs(mol)

    # First embed attempt (ETKDGv3)
    ps = AllChem.ETKDGv3()
    rc = AllChem.EmbedMolecule(mol, ps)
    if rc == -1:
        # Retry with random coordinates
        ps.useRandomCoords = True
        rc = AllChem.EmbedMolecule(mol, ps)
    if rc == -1:
        return False, "embedding", "EmbedMolecule returned -1 (both ETKDGv3 and random coords)"

    # MMFF optimization (non-fatal)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception as e:
        pass  # non-fatal; continue with un-optimised geometry

    # Meeko
    try:
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        if not mol_setups:
            return False, "meeko", "prepare() returned no setups"
        pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return False, "meeko", f"write_string error: {err_msg}"
    except Exception as e:
        return False, "meeko", str(e)

    out_pdbqt.write_text(pdbqt_str)
    return True, None, None


# ---------------------------------------------------------------------------
# Parse AutoDock-GPU XML
# ---------------------------------------------------------------------------
def parse_xml(xml_path: Path, fail_threshold: float):
    """
    Returns (score: float | None, reason: str | None).
    reason is set when score is None.
    """
    if not xml_path.exists():
        return None, "no_xml"
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        energies = root.findall(".//best_energy")
        if not energies:
            energies = root.findall(".//free_NRG_binding")
        if not energies:
            return None, "xml_no_energy_element"
        best = min(float(e.text) for e in energies)
        if best >= fail_threshold:
            return None, f"score_above_threshold ({best:.3f} >= {fail_threshold})"
        return best, None
    except Exception as e:
        return None, f"xml_parse_error: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Serial smoke test for end-to-end docking")
    parser.add_argument("--target", default="trpv1_8gfa")
    parser.add_argument("--experiment-id", default="maxmin_init")
    parser.add_argument("--maps", required=True, help="AutoDock-GPU .fld grid file")
    parser.add_argument("--n", type=int, default=100, help="Number of compounds to test")
    parser.add_argument("--nrun", type=int, default=20, help="AutoDock-GPU nrun")
    parser.add_argument("--autodock-bin", default="bin/autodock_gpu")
    parser.add_argument("--fail-threshold", type=float, default=0.0)
    parser.add_argument("--round", type=int, default=0)
    cfg = parser.parse_args()

    grid_fld = Path(cfg.maps)
    autodock_bin = Path(cfg.autodock_bin)

    if not grid_fld.exists():
        print(f"ERROR: grid map not found: {grid_fld}", file=sys.stderr)
        sys.exit(1)
    if not autodock_bin.exists():
        print(f"ERROR: AutoDock-GPU binary not found: {autodock_bin}", file=sys.stderr)
        sys.exit(1)

    # Fetch compounds from DB
    print(f"Fetching {cfg.n} compounds from al_batches "
          f"(target={cfg.target}, experiment_id={cfg.experiment_id}, round={cfg.round}) …")
    conn = get_db_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ab.compound_id, c.smiles_protonated "
            "FROM al_batches ab "
            "JOIN compounds c ON c.id = ab.compound_id "
            "WHERE ab.round = %s AND ab.target = %s AND ab.experiment_id = %s "
            "ORDER BY ab.compound_id "
            "LIMIT %s",
            (cfg.round, cfg.target, cfg.experiment_id, cfg.n),
        )
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("ERROR: no compounds found in al_batches for these parameters.", file=sys.stderr)
        sys.exit(1)

    print(f"Got {len(rows)} compounds.\n")

    # Counters
    prep_failures = {"smiles": [], "embedding": [], "meeko": []}
    dock_failures = {"no_xml": [], "xml_no_energy_element": [], "xml_parse_error": [],
                     "score_above_threshold": [], "timeout": [], "nonzero_exit": []}
    valid_scores = []
    dock_times = []  # wall-clock seconds per successful dock

    total_t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="smoke_dock_") as tmpdir:
        tmpdir = Path(tmpdir)

        for i, (compound_id, smiles) in enumerate(rows, 1):
            print(f"[{i:3d}/{len(rows)}] cmpd={compound_id}  ", end="", flush=True)

            # --- Stage 1: Prep ---
            pdbqt_path = tmpdir / f"{compound_id}.pdbqt"
            prep_ok, fail_cat, fail_detail = prep_ligand(compound_id, smiles, pdbqt_path)

            if not prep_ok:
                print(f"PREP FAIL ({fail_cat}): {fail_detail}")
                prep_failures[fail_cat].append((compound_id, fail_detail))
                continue

            print("prep OK  ", end="", flush=True)

            # --- Stage 2: Dock ---
            out_prefix = tmpdir / str(compound_id)
            xml_path = Path(str(out_prefix) + ".xml")

            cmd = [
                str(autodock_bin),
                "--ffile", str(grid_fld),
                "--lfile", str(pdbqt_path),
                "--resnam", str(out_prefix),
                "--nrun", str(cfg.nrun),
            ]

            dock_t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                dock_elapsed = time.time() - dock_t0
            except subprocess.TimeoutExpired:
                dock_elapsed = time.time() - dock_t0
                print(f"DOCK FAIL (timeout after {dock_elapsed:.1f}s)")
                dock_failures["timeout"].append(compound_id)
                continue

            if proc.returncode != 0:
                print(f"DOCK FAIL (exit={proc.returncode}, {dock_elapsed:.1f}s)  "
                      f"stderr: {proc.stderr[-200:].strip()}")
                dock_failures["nonzero_exit"].append((compound_id, proc.returncode))
                continue

            # --- Stage 3: Parse XML ---
            score, reason = parse_xml(xml_path, cfg.fail_threshold)

            if score is None:
                # Categorise
                cat = "no_xml"
                for key in dock_failures:
                    if reason and reason.startswith(key):
                        cat = key
                        break
                # fallback for score_above_threshold which has trailing info
                if reason and reason.startswith("score_above_threshold"):
                    cat = "score_above_threshold"
                print(f"DOCK FAIL ({reason}, {dock_elapsed:.1f}s)")
                dock_failures[cat].append((compound_id, reason))
            else:
                valid_scores.append(score)
                dock_times.append(dock_elapsed)
                print(f"score={score:.3f} kcal/mol  ({dock_elapsed:.1f}s)")

            # Print stdout/stderr snippet if NaN warning detected
            if "non finite charge" in proc.stdout or "nan" in proc.stdout.lower():
                print(f"    [NaN warning in stdout] {proc.stdout[:300].strip()}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - total_t0

    print("\n" + "=" * 65)
    print("SMOKE TEST SUMMARY")
    print("=" * 65)
    print(f"  Compounds tested    : {len(rows)}")
    print()

    total_prep_fail = sum(len(v) for v in prep_failures.values())
    print(f"  Prep failures       : {total_prep_fail}")
    for cat, items in prep_failures.items():
        if items:
            print(f"    {cat:20s}: {len(items)}")
            for cid, detail in items[:5]:
                print(f"      cmpd={cid}: {detail}")
            if len(items) > 5:
                print(f"      ... and {len(items) - 5} more")

    print()
    n_prepped = len(rows) - total_prep_fail
    total_dock_fail = sum(len(v) for v in dock_failures.values())
    print(f"  Docked (attempted)  : {n_prepped}")
    print(f"  Dock failures       : {total_dock_fail}")
    for cat, items in dock_failures.items():
        if items:
            print(f"    {cat:30s}: {len(items)}")
            for entry in items[:3]:
                if isinstance(entry, tuple):
                    print(f"      cmpd={entry[0]}: {entry[1]}")
                else:
                    print(f"      cmpd={entry}")
            if len(items) > 3:
                print(f"      ... and {len(items) - 3} more")

    print()
    n_valid = len(valid_scores)
    print(f"  Valid scores        : {n_valid}")
    if valid_scores:
        print(f"    min               : {min(valid_scores):.3f} kcal/mol")
        print(f"    mean              : {sum(valid_scores)/n_valid:.3f} kcal/mol")
        print(f"    max               : {max(valid_scores):.3f} kcal/mol")

    print()
    print(f"  Total wallclock     : {total_elapsed:.1f}s")
    if dock_times:
        mean_dock = sum(dock_times) / len(dock_times)
        print(f"  Mean dock time      : {mean_dock:.1f}s per compound")
        print(f"  Throughput          : {3600/mean_dock:.0f} docks/hour (serial)")
        print(f"  Est. 24K w/ 6 thrd : {24000 * mean_dock / 6 / 3600:.1f}h")

    print("=" * 65)

    # Exit non-zero if all compounds failed (helps CI)
    if n_valid == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
