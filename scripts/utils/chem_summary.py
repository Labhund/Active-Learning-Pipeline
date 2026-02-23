import argparse
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def calculate_properties(smiles):
    # Parse the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    # Calculate basic properties
    properties = {
        "SMILES": smiles,
        "Formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        "Mol Weight": f"{Descriptors.MolWt(mol):.2f}",
        "LogP": f"{Descriptors.MolLogP(mol):.2f}",
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "TPSA": f"{Descriptors.TPSA(mol):.2f}",
        "Ring Count": Lipinski.RingCount(mol),
    }
    return properties


def main():
    parser = argparse.ArgumentParser(
        description="Get a summary of chemical properties from a SMILES string."
    )
    parser.add_argument("smiles", help="The SMILES string of the molecule")

    args = parser.parse_args()

    props = calculate_properties(args.smiles)

    if props:
        print(f"\n{' Property ':=^30}")
        for key, value in props.items():
            print(f"{key:15}: {value}")
        print(f"{'':=^30}\n")
    else:
        print(f"Error: Could not parse SMILES string '{args.smiles}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
