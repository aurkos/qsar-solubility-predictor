"""
Fetch the ESOL (Delaney) aqueous solubility dataset.

ESOL is a well-known cheminformatics/QSAR benchmark: ~1,128 organic
compounds with SMILES and measured log aqueous solubility (log mol/L).
Reference: Delaney, J.S. "ESOL: Estimating Aqueous Solubility Directly
from Molecular Structure." J. Chem. Inf. Comput. Sci. 2004.

Pulls the processed CSV from the DeepChem (MoleculeNet) repository, with
a bundled offline fallback sample so the pipeline runs without network
access.
"""

import os

import requests

URL = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
OUT_PATH = os.path.join("data", "esol_raw.csv")
FALLBACK_PATH = os.path.join("data", "esol_offline_fallback.csv")


def main():
    os.makedirs("data", exist_ok=True)
    try:
        resp = requests.get(URL, timeout=30)
        resp.raise_for_status()
        with open(OUT_PATH, "w") as f:
            f.write(resp.text)
        n_lines = resp.text.count("\n")
        print(f"Downloaded ESOL dataset ({n_lines} lines) -> {OUT_PATH}")
    except Exception as exc:
        print(f"WARNING: could not download dataset ({exc}); using offline fallback.")
        if not os.path.exists(FALLBACK_PATH):
            raise SystemExit("No network and no offline fallback available.")
        with open(FALLBACK_PATH) as f_in, open(OUT_PATH, "w") as f_out:
            f_out.write(f_in.read())
        print(f"Copied offline fallback -> {OUT_PATH}")


if __name__ == "__main__":
    main()
