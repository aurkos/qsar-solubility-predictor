"""
Predict aqueous solubility for a new compound from its SMILES string
using the trained model.

Usage:
    python predict.py "CC(=O)Oc1ccccc1C(=O)O"   # aspirin
"""

import sys

import joblib
import pandas as pd
from rdkit import Chem

from featurize import DESCRIPTOR_FUNCS

MODEL_PATH = "models/solubility_model.joblib"


def predict(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    feats = {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}
    X = pd.DataFrame([feats])[feature_cols]
    return float(model.predict(X)[0])


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    smiles = sys.argv[1]
    pred = predict(smiles)
    print(f"SMILES: {smiles}")
    print(f"Predicted log solubility: {pred:.3f} mol/L")


if __name__ == "__main__":
    main()
