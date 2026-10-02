"""
Featurize the ESOL dataset with RDKit molecular descriptors.

Reads data/esol_raw.csv (Compound ID, smiles, measured log solubility),
computes a standard set of physicochemical descriptors for each molecule
with RDKit, and writes data/esol_features.csv ready for model training.
"""

import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

RAW_CSV = os.path.join("data", "esol_raw.csv")
FEATURES_CSV = os.path.join("data", "esol_features.csv")

TARGET_COL = "measured log solubility in mols per litre"

DESCRIPTOR_FUNCS = {
    "mol_weight": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "num_h_donors": Lipinski.NumHDonors,
    "num_h_acceptors": Lipinski.NumHAcceptors,
    "num_rotatable_bonds": Lipinski.NumRotatableBonds,
    "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings,
    "num_rings": rdMolDescriptors.CalcNumRings,
    "num_heavy_atoms": Descriptors.HeavyAtomCount,
    "fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
    "molar_refractivity": Crippen.MolMR,
}


def featurize_smiles(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}


def main():
    df = pd.read_csv(RAW_CSV)
    df.columns = [c.strip() for c in df.columns]

    records = []
    n_failed = 0
    for _, row in df.iterrows():
        feats = featurize_smiles(row["smiles"])
        if feats is None:
            n_failed += 1
            continue
        feats["compound_id"] = row["Compound ID"]
        feats["smiles"] = row["smiles"]
        feats["log_solubility"] = row[TARGET_COL]
        records.append(feats)

    out_df = pd.DataFrame(records)
    cols = ["compound_id", "smiles"] + list(DESCRIPTOR_FUNCS.keys()) + ["log_solubility"]
    out_df = out_df[cols]
    out_df.to_csv(FEATURES_CSV, index=False)

    print(f"Featurized {len(out_df)} / {len(df)} compounds "
          f"({n_failed} SMILES failed to parse)")
    print(f"Wrote -> {FEATURES_CSV}")


if __name__ == "__main__":
    main()
