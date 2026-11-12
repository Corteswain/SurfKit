"""Shared standardisation: identical for SurfPro and the reference databases.

    SMILES -> RDKit mol -> largest (organic) fragment, i.e. counterions removed
           -> domain filter (see below)
           -> neutralised (rdMolStandardize.Uncharger)
           -> canonical isomeric SMILES, canonical SMILES without stereo,
              Morgan fingerprint (radius 2, 2048 bits, packed to 256 bytes)

Domain filter: a database molecule is dropped if its main molecule (largest
fragment) has more heavy atoms than the largest SurfPro main molecule, or
contains an element that no SurfPro main molecule contains. Both limits are
derived from SurfPro.csv by surfpro_domain(); counterions are ignored on both
sides, so e.g. Na or Br that occur in SurfPro only as counterions are not
allowed in a database main molecule.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
FP_RADIUS, FP_BITS = 2, 2048
FP_BYTES = FP_BITS // 8
# Cheap guard before parsing: a few huge PubChem entries made RDKit use 10+ GB
# and got the build OOM-killed. The longest SurfPro SMILES has 319 characters.
MAX_SMILES_LEN = 1000
TOO_LARGE, BAD_ELEMENTS = "too_large", "bad_elements"
_largest = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
_uncharger = rdMolStandardize.Uncharger()
_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def main_molecule(m):
    return _largest.choose(m) if len(Chem.GetMolFrags(m)) > 1 else m


def surfpro_domain(csv_path):
    """{"max_heavy_atoms": int, "elements": [...]} over SurfPro main molecules."""
    import pandas as pd
    heavy, elements = 0, set()
    for smi in pd.read_csv(csv_path)["SMILES"]:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        m = main_molecule(m)
        heavy = max(heavy, m.GetNumHeavyAtoms())
        elements |= {a.GetSymbol() for a in m.GetAtoms()}
    return {"max_heavy_atoms": heavy, "elements": sorted(elements)}


def out_of_domain(m, domain):
    """TOO_LARGE / BAD_ELEMENTS / None for a main molecule."""
    if domain is None:
        return None
    if m.GetNumHeavyAtoms() > domain["max_heavy_atoms"]:
        return TOO_LARGE
    allowed = domain["_allowed"] if "_allowed" in domain else set(domain["elements"])
    if any(a.GetSymbol() not in allowed for a in m.GetAtoms()):
        return BAD_ELEMENTS
    return None


def standardise(smi, domain=None):
    """(parent_smiles, parent_smiles_nostereo, packed_fp); None if RDKit fails;
    TOO_LARGE or BAD_ELEMENTS if outside the domain."""
    if len(smi) > MAX_SMILES_LEN:
        return TOO_LARGE
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        m = main_molecule(m)
        reason = out_of_domain(m, domain)
        if reason:
            return reason
        m = _uncharger.uncharge(m)
        iso = Chem.MolToSmiles(m)
        flat = Chem.MolToSmiles(m, isomericSmiles=False)
        fp = np.packbits(_fpgen.GetFingerprintAsNumPy(m).astype(bool))
    except Exception:          # incl. MemoryError from the worker's memory cap
        return None
    return iso, flat, fp
