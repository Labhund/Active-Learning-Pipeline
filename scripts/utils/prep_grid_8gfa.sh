#!/usr/bin/env bash
# Receptor preparation and autogrid4 map generation for TRPV1 8GFA.
#
# Run from the project root after sourcing env_db.sh and activating chem env:
#   source env_db.sh
#   conda activate chem
#   bash scripts/utils/prep_grid_8gfa.sh
#
# Steps:
#   1. Extract chains D+A + lipid shell from 8gfa.cif
#   2. Convert receptor PDB to PDBQT with polar H + Gasteiger charges
#   3. Generate autogrid4 maps (you must edit gridcenter in GPF first)

set -euo pipefail
PROJ="${PROJ_ROOT:-/data2/lipin_lab/markus/analgesics}"
cd "$PROJ"

PDBDIR="targets/trpv1/structures/pdb"
PDBQTDIR="targets/trpv1/structures/pdbqt"
GRIDDIR="targets/trpv1/grids"
GPF="$GRIDDIR/trpv1_8gfa.gpf"
CIF="$PDBDIR/8gfa.cif"
RECEPTOR_PDB="$PDBDIR/trpv1_8gfa_receptor.pdb"
RECEPTOR_PDBQT="$PDBQTDIR/trpv1_8gfa.pdbqt"

# ── Step 1: Extract receptor PDB ──────────────────────────────────────────────
echo "=== Step 1: Extracting receptor PDB from CIF ==="
python scripts/utils/extract_receptor_8gfa.py \
    --cif "$CIF" \
    --outdir "$PDBDIR"

echo ""
echo ">>> ACTION REQUIRED <<<"
echo "    Copy the 'Grid center for autogrid4 GPF' line above and update:"
echo "    $GPF"
echo "    Change the 'gridcenter' line to the correct x y z coordinates."
echo ""
echo "Press Enter when you have updated the gridcenter in $GPF, or Ctrl-C to abort."
read -r _

# Verify gridcenter was updated (check it doesn't still say the placeholder)
if grep -q "# fill in after" "$GPF"; then
    echo "WARNING: GPF still contains the placeholder comment. Continuing anyway."
fi

# ── Step 2: Prepare receptor PDBQT ────────────────────────────────────────────
echo "=== Step 2: Preparing receptor PDBQT ==="

# Try to locate prepare_receptor4.py
PREP4=""
if command -v prepare_receptor4.py &>/dev/null; then
    PREP4="prepare_receptor4.py"
elif command -v prepare_receptor4 &>/dev/null; then
    PREP4="prepare_receptor4"
else
    # Common MGLTools locations
    for loc in \
        "$HOME/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py" \
        "/opt/mgltools/bin/prepare_receptor4.py" \
        "$PROJ/bin/prepare_receptor4.py"; do
        if [ -f "$loc" ]; then
            PREP4="$loc"
            break
        fi
    done
fi

if [ -z "$PREP4" ]; then
    echo ""
    echo "ERROR: prepare_receptor4.py not found."
    echo ""
    echo "Options:"
    echo "  A) Install MGLTools:"
    echo "     conda install -c bioconda mgltools   (or download from scripps.edu)"
    echo "  B) Use meeko (already in chem env):"
    echo "     mk_prepare_receptor.py -i $RECEPTOR_PDB -o $RECEPTOR_PDBQT"
    echo "     NOTE: meeko handles proteins well but may not assign Gasteiger charges"
    echo "     to lipid residues correctly. Check the PDBQT atom type assignments."
    echo "  C) Place prepare_receptor4.py in $PROJ/bin/ and re-run."
    echo ""
    echo "Skipping PDBQT preparation. Generating maps will fail without PDBQT."
    SKIP_MAPS=1
else
    echo "Using: $PREP4"
    mkdir -p "$PDBQTDIR"
    python "$PREP4" \
        -r "$RECEPTOR_PDB" \
        -o "$RECEPTOR_PDBQT" \
        -A hydrogens \
        -U nphs_lps_waters_deleteAltB \
        -v
    if [ -f "$RECEPTOR_PDBQT" ]; then
        SIZE=$(du -h "$RECEPTOR_PDBQT" | cut -f1)
        echo "Receptor PDBQT written: $RECEPTOR_PDBQT ($SIZE)"
    else
        echo "ERROR: PDBQT file not created. Check prepare_receptor4.py output above."
        exit 1
    fi
    SKIP_MAPS=0
fi

# ── Step 3: Generate autogrid4 maps ───────────────────────────────────────────
if [ "${SKIP_MAPS:-0}" -eq 1 ]; then
    echo ""
    echo "=== Step 3 SKIPPED (no PDBQT) ==="
    echo "After fixing PDBQT prep, run:"
    echo "  cd $GRIDDIR && autogrid4 -p trpv1_8gfa.gpf -l trpv1_8gfa.glg"
    exit 1
fi

echo ""
echo "=== Step 3: Running autogrid4 ==="
if ! command -v autogrid4 &>/dev/null; then
    echo "ERROR: autogrid4 not found in PATH."
    echo "Check: $PROJ/bin/autogrid4 or add AutoDockTools to PATH."
    echo "Run manually: cd $GRIDDIR && autogrid4 -p trpv1_8gfa.gpf -l trpv1_8gfa.glg"
    exit 1
fi

cd "$GRIDDIR"
autogrid4 -p trpv1_8gfa.gpf -l trpv1_8gfa.glg
cd "$PROJ"

echo ""
echo "=== autogrid4 complete ==="
if [ -f "$GRIDDIR/trpv1_8gfa.fld" ]; then
    echo "Grid field file: $GRIDDIR/trpv1_8gfa.fld  ✓"
    ls -lh "$GRIDDIR"/trpv1_8gfa.*.map 2>/dev/null | head -20
else
    echo "ERROR: trpv1_8gfa.fld not found — check $GRIDDIR/trpv1_8gfa.glg for errors."
    exit 1
fi

echo ""
echo "=== Verification ==="
echo "1. In PyMOL: load $RECEPTOR_PDBQT and 8gfa.cif"
echo "   Confirm ZEI sits inside the grid box."
echo "   Confirm S4-S5 linker Cα atoms (TRPV1 ~residues 580-600) are at/outside"
echo "   the intracellular grid edge."
echo ""
echo "2. Quick dock test with capsaicin:"
echo "   bin/autodock_gpu -M $GRIDDIR/trpv1_8gfa.fld \\"
echo "       -L <capsaicin.pdbqt> -O /tmp/test_dock --nrun 5"
echo "   Expected: score -8 to -12 kcal/mol"
echo ""
echo "3. Smoke test AL loop:"
echo "   python scripts/active_learning/al_loop.py \\"
echo "       --config config/al_loop.yaml --start-round 0 --rounds 1 --init-only"
